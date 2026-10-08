"""
app/hitl.py

El flujo Human-in-the-loop: la pausa obligatoria antes de ejecutar una acción
con efectos secundarios.

Qué se considera crítico, y por qué
-----------------------------------
El orquestador del módulo 6 era de solo lectura: consultaba una base, buscaba
en un índice y calculaba estadísticas. Nada de eso necesita aprobación, porque
nada cambia en el mundo.

Esta entrega le agrega una acción que **sí** cambia algo: `escalar_a_guardia`,
que abre una escalación y despierta a la persona de turno. Tiene costo real
(alguien se levanta a las 3 de la mañana) y no se puede deshacer. Esa es la
definición operativa de crítico que usa este módulo: **efecto externo e
irreversible**, no "el LLM cree que es importante".

La decisión es determinista
---------------------------
`evaluar_criticidad()` no usa LLM. Mira los hallazgos del analista y aplica una
regla: si hay un incidente atípico cuya severidad es crítica, corresponde
escalar. Las razones son las mismas por las que el validador del módulo 6
tampoco usa LLM:

- Un modelo al que se le pregunta "¿esto es crítico?" responde distinto según
  cómo venga redactado el contexto. La compuerta que decide si hace falta
  aprobación humana no puede depender de eso.
- Es testeable sin red: hay tests que verifican que dispara con un atípico
  crítico y que no dispara sin él.

Lo que el LLM sí hace es redactar el motivo que ve la persona que aprueba. Eso
es texto, no control de flujo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.types import interrupt

from app.schemas import PropuestaAccion

log = logging.getLogger("app.hitl")

# Umbral de duración, en minutos, por encima del cual una caída se considera
# grave aunque no esté marcada como crítica. 120 minutos = dos horas de
# servicio afectado.
UMBRAL_MINUTOS_GRAVE = 120

SEVERIDADES_CRITICAS = ("critica",)


def evaluar_criticidad(estado: dict[str, Any]) -> PropuestaAccion | None:
    """
    Decide si el resultado del análisis amerita una acción con efectos.

    Devuelve la propuesta a aprobar, o `None` si no hace falta intervención
    humana y el flujo puede seguir derecho a redactar el informe.

    La regla: hace falta escalar si entre los incidentes atípicos que detectó
    el analista hay alguno con severidad crítica, o alguno que superó el umbral
    de duración. El peor de todos es el que se propone escalar.
    """
    atipicos = _atipicos_detectados(estado)
    if not atipicos:
        return None

    candidatos = [
        incidente
        for incidente in atipicos
        if incidente.get("severidad") in SEVERIDADES_CRITICAS
        or int(incidente.get("minutos_caidos", 0)) >= UMBRAL_MINUTOS_GRAVE
    ]
    if not candidatos:
        log.info("Hay atípicos pero ninguno crítico: no se pide aprobación.")
        return None

    peor = max(candidatos, key=lambda i: int(i.get("minutos_caidos", 0)))
    minutos = int(peor.get("minutos_caidos", 0))
    severidad: Literal["alta", "critica"] = (
        "critica" if peor.get("severidad") in SEVERIDADES_CRITICAS else "alta"
    )

    propuesta = PropuestaAccion(
        accion="escalar_a_guardia",
        motivo=(
            f"El incidente {peor.get('incidente_id', '?')} del servicio "
            f"{peor.get('servicio', '?')} acumuló {minutos} minutos de caída "
            f"({peor.get('veces_la_mediana', '?')}x la mediana) por "
            f"'{peor.get('categoria', '?')}'. Corresponde escalar a la guardia "
            f"para una revisión inmediata."
        ),
        criticidad=severidad,
        efectos=[
            "Abre una escalación con número de seguimiento.",
            "Notifica a la persona de guardia del servicio afectado.",
            "Queda registrada en la auditoría y no se puede deshacer.",
        ],
        parametros={
            "incidente_id": peor.get("incidente_id"),
            "servicio": peor.get("servicio"),
            "categoria": peor.get("categoria"),
            "severidad": peor.get("severidad"),
            "minutos_caidos": minutos,
        },
    )

    log.info(
        "Acción crítica propuesta: escalar %s (%s, %d min).",
        peor.get("incidente_id"), peor.get("servicio"), minutos,
    )
    return propuesta


def _atipicos_detectados(estado: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Los incidentes atípicos que dejó el analista en el estado compartido.

    Se leen de `hallazgos`, el canal estructurado del módulo 6, y no del
    informe en texto: la compuerta tiene que decidir sobre el dato que devolvió
    la herramienta, no sobre cómo el modelo lo contó.
    """
    for hallazgo in estado.get("hallazgos", []):
        herramienta = getattr(hallazgo, "herramienta", None)
        datos = getattr(hallazgo, "datos", None)
        if herramienta == "detectar_atipicos" and isinstance(datos, dict):
            return [a for a in datos.get("atipicos", []) if isinstance(a, dict)]
    return []


# --------------------------------------------------------------------------
# Los nodos del grafo
# --------------------------------------------------------------------------

def nodo_compuerta_hitl(estado: dict[str, Any]) -> dict[str, Any]:
    """
    La pausa. Evalúa la criticidad y, si hace falta, detiene el grafo.

    `interrupt(payload)` corta la ejecución acá mismo y devuelve el control a
    quien invocó el grafo, dejando el estado completo en el checkpointer. El
    `payload` es lo que el worker guarda en Redis y lo que la API le muestra a
    la persona que tiene que aprobar.

    Cuando llega la aprobación, el grafo se retoma con
    `Command(resume=decision)`, y **este nodo se ejecuta de nuevo desde el
    principio**: `interrupt()` no vuelve a pausar, devuelve el valor del
    resume. Por eso todo lo que está antes del `interrupt` tiene que ser
    idempotente — acá lo es, porque solo lee el estado y no escribe nada.
    """
    propuesta = evaluar_criticidad(estado)

    if propuesta is None:
        return {
            "requiere_aprobacion": False,
            "propuesta_accion": None,
            "motivo_hitl": "Ningún hallazgo amerita una acción con efectos secundarios.",
        }

    # Acá se pausa. La segunda vez que pase por esta línea, `interrupt`
    # devuelve lo que se mandó en `Command(resume=...)`.
    decision = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "propuesta": propuesta.model_dump(),
        }
    )

    aprobado = bool(decision.get("aprobado")) if isinstance(decision, dict) else False
    quien = (
        decision.get("aprobado_por", "desconocido")
        if isinstance(decision, dict) else "desconocido"
    )
    comentario = decision.get("comentario", "") if isinstance(decision, dict) else ""

    log.info("Aprobación recibida: %s por %s.", aprobado, quien)

    return {
        "requiere_aprobacion": True,
        "propuesta_accion": propuesta,
        "aprobacion_otorgada": aprobado,
        "aprobado_por": quien,
        "comentario_aprobacion": comentario,
        "motivo_hitl": (
            f"Acción crítica {'aprobada' if aprobado else 'rechazada'} por {quien}."
        ),
    }


def enrutar_desde_compuerta(
    estado: dict[str, Any],
) -> Literal["ejecutar_accion", "sintetizador"]:
    """
    Arista condicional de la compuerta.

    Solo se ejecuta la acción si hubo propuesta **y** aprobación explícita. El
    default de `aprobacion_otorgada` es False, así que un estado incompleto o
    un resume mal formado no dispara la acción: ante la duda, no se escala.
    """
    if estado.get("requiere_aprobacion") and estado.get("aprobacion_otorgada"):
        return "ejecutar_accion"
    return "sintetizador"


async def nodo_ejecutar_accion(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Ejecuta la acción aprobada. Es el único nodo con efectos secundarios.

    El efecto es real y verificable: escribe la escalación en Redis bajo
    `{prefijo}:escalaciones`. No es un `print` que simula haber hecho algo —
    después de aprobar, la escalación existe y se puede listar por la API.
    """
    propuesta = estado.get("propuesta_accion")
    if propuesta is None:
        return {
            "accion_ejecutada": {
                "error": "No hay propuesta que ejecutar.",
            }
        }

    from app.acciones import escalar_a_guardia

    resultado = await escalar_a_guardia(
        incidente_id=propuesta.parametros.get("incidente_id", "?"),
        servicio=propuesta.parametros.get("servicio", "?"),
        motivo=propuesta.motivo,
        aprobado_por=estado.get("aprobado_por", "desconocido"),
    )

    log.info("Acción ejecutada: %s", resultado.get("escalacion_id"))

    return {
        "accion_ejecutada": {
            **resultado,
            "ejecutada_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    }
