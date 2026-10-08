"""
agents/supervisor.py

El supervisor, el validador y el sintetizador: los tres nodos que coordinan a
los especialistas sin hacer el trabajo de dominio.

Reparto de responsabilidades
----------------------------
- **`nodo_supervisor`** (con LLM) decide *quién sigue*. Devuelve salida
  estructurada, así que el ruteo no depende de parsear una frase.
- **`nodo_validador`** (sin LLM) decide *si lo hecho alcanza*. Es Python puro:
  mismo estado, misma respuesta, siempre. Un validador que es un LLM
  preguntándole a otro LLM si estuvo bien agrega una opinión, no una garantía.
- **`nodo_sintetizador`** (con LLM) redacta el informe final a partir de los
  **hallazgos estructurados**, no del historial de mensajes.

Por qué el validador no usa LLM
-------------------------------
La consigna pide un nodo de validación *o* una rúbrica en el prompt del
supervisor. Están las dos cosas, y hacen trabajos distintos a propósito:

La rúbrica del supervisor es su *intención*: el modelo evalúa qué falta y
decide. El validador es la *verificación*: recorre `hallazgos` y comprueba
contra el estado real qué herramientas corrieron efectivamente. Si el
supervisor dice "ya está completo" pero nunca pasó por el analista, el
validador lo detecta y lo devuelve, porque no le cree al modelo: le cree al
estado.

Esa asimetría es la defensa concreta contra el "supervisor infinito" en su
versión más común, que no es el bucle eterno sino el cierre prematuro.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_PASOS_SUPERVISOR, get_llm
from state import DecisionSupervisor, Destino, Rubrica
from utils import texto_de

log = logging.getLogger("agents.supervisor")

# Qué herramienta tiene que haber corrido para que cada criterio se considere
# cumplido. El validador usa este mapa, así que agregar un criterio nuevo es
# agregar una línea acá y un campo en `Rubrica` — no tocar la lógica.
EVIDENCIA_REQUERIDA: dict[str, tuple[str, ...]] = {
    "hay_evidencia_documental": ("buscar_en_documentacion",),
    "hay_datos_cuantitativos": ("listar_incidentes",),
    "hay_analisis_estadistico": ("resumir_incidentes", "detectar_atipicos"),
    "hay_validacion_de_datos": ("validar_registros",),
}

PROMPT_SUPERVISOR = """Sos el SUPERVISOR de un equipo de auditoría técnica. No \
hacés el trabajo: decidís quién lo hace.

Tenés dos especialistas y un validador:

- `investigador`: recupera material. Tiene dos herramientas: búsqueda semántica \
en la documentación técnica del equipo sobre asyncio, y consulta a la bitácora \
de incidentes de producción. NO calcula nada.
- `analista`: procesa los datos que el investigador ya trajo. Calcula métricas, \
detecta valores atípicos y valida la integridad de los registros. NO busca \
información nueva: si lo mandás antes que al investigador, no va a tener qué \
analizar.
- `validador`: verifica que el trabajo esté completo antes de cerrar. Derivá \
acá cuando creas que ya está todo.

RÚBRICA. Antes de decidir, evaluá estas cuatro condiciones mirando los \
hallazgos que ya hay en el estado:

1. `hay_evidencia_documental`: ¿se consultó la documentación técnica?
2. `hay_datos_cuantitativos`: ¿se trajeron incidentes concretos de la bitácora?
3. `hay_analisis_estadistico`: ¿se calcularon métricas sobre esos incidentes?
4. `hay_validacion_de_datos`: ¿se verificó la integridad de los registros?

Reglas de decisión:
- Si faltan 1 o 2, derivá al `investigador`.
- Si faltan 3 o 4, derivá al `analista`.
- Si las cuatro están cumplidas, derivá al `validador`.
- No derivés dos veces seguidas al mismo agente con la misma instrucción: si ya \
trabajó y algo sigue faltando, pedile explícitamente lo que falta.

La `instruccion` que escribís es lo ÚNICO que el especialista va a recibir: no \
ve esta conversación ni tus razonamientos. Tiene que ser autosuficiente y \
concreta."""

PROMPT_SINTETIZADOR = """Sos el REDACTOR final de un informe de auditoría técnica.

Vas a recibir los hallazgos estructurados que produjo el equipo. Tu trabajo es \
escribir el informe para la persona que hizo la pregunta.

Reglas:
- Usá EXCLUSIVAMENTE los números que están en los hallazgos. No redondees ni \
estimes: si un hallazgo dice 1180 minutos, el informe dice 1180 minutos.
- Conectá las dos mitades: qué dice la documentación sobre la causa y qué \
muestran nuestros propios datos. Ahí está el valor del informe.
- Si la validación encontró registros con problemas, mencionalo: afecta la \
confianza en las métricas.
- Estructura sugerida: un párrafo de conclusión al principio, después los \
números que la respaldan, después la recomendación.
- Español rioplatense, concreto, sin relleno. Máximo 350 palabras.
- No inventes recomendaciones que no se desprendan de la documentación \
consultada."""


# --------------------------------------------------------------------------
# Herramientas de lectura del estado
# --------------------------------------------------------------------------

def herramientas_ejecutadas(estado: dict[str, Any]) -> set[str]:
    """Nombres de las herramientas que efectivamente corrieron, según `hallazgos`."""
    return {
        hallazgo.herramienta
        for hallazgo in estado.get("hallazgos", [])
        if not hallazgo.datos.get("error")
    }


def rubrica_verificada(estado: dict[str, Any]) -> Rubrica:
    """
    Arma la rúbrica a partir del estado real, sin preguntarle al modelo.

    Es la versión objetiva de la misma rúbrica que el supervisor completa por
    su cuenta. Comparar las dos es lo que le permite al validador detectar un
    cierre prematuro.
    """
    ejecutadas = herramientas_ejecutadas(estado)
    return Rubrica(
        **{
            criterio: any(h in ejecutadas for h in requeridas)
            for criterio, requeridas in EVIDENCIA_REQUERIDA.items()
        }
    )


def _resumen_de_hallazgos(estado: dict[str, Any]) -> str:
    """
    El estado del trabajo, en texto de tamaño acotado.

    El supervisor ve esto y no los payloads completos: con 20 incidentes y 4
    fragmentos de documentación, pasar los datos crudos serían miles de tokens
    que se repiten en cada vuelta. Los resúmenes son una línea por hallazgo.
    """
    hallazgos = estado.get("hallazgos", [])
    if not hallazgos:
        return "(todavía no hay hallazgos: nadie trabajó)"

    return "\n".join(
        f"- [{h.agente} / {h.herramienta}] {h.resumen}" for h in hallazgos
    )


# --------------------------------------------------------------------------
# Nodo supervisor
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _supervisor_estructurado():
    """El LLM del supervisor, forzado a devolver una `DecisionSupervisor`."""
    return get_llm().with_structured_output(DecisionSupervisor)


async def nodo_supervisor(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Decide qué agente interviene ahora.

    Si se alcanzó `MAX_PASOS_SUPERVISOR`, no consulta al modelo: deriva
    directo al validador. Es el freno semántico contra el supervisor infinito,
    y está antes de la llamada al LLM a propósito — un tope que se evalúa
    *después* de pagar la llamada no ahorra nada.
    """
    pasos = estado.get("pasos_del_supervisor", 0)
    verificada = rubrica_verificada(estado)

    if pasos >= MAX_PASOS_SUPERVISOR:
        log.warning(
            "Tope de %d pasos alcanzado. Se cierra con lo recolectado (falta: %s).",
            MAX_PASOS_SUPERVISOR, ", ".join(verificada.faltantes()) or "nada",
        )
        return {
            "siguiente": "validador",
            "instruccion_actual": "Cerrar con el material disponible.",
            "razonamiento_supervisor": (
                f"Tope de {MAX_PASOS_SUPERVISOR} pasos de supervisión alcanzado; "
                f"se cierra con lo que haya."
            ),
            "rubrica_actual": verificada,
            "pasos_del_supervisor": 1,
            "messages": [
                AIMessage(
                    content=(
                        f"[supervisor] Tope de pasos alcanzado, derivo al validador "
                        f"con lo recolectado."
                    ),
                    name="supervisor",
                )
            ],
        }

    pedido = (
        f"Solicitud del usuario:\n{estado.get('solicitud', '')}\n\n"
        f"Hallazgos acumulados hasta ahora:\n{_resumen_de_hallazgos(estado)}\n\n"
        f"Rúbrica verificada automáticamente contra el estado "
        f"(esto es objetivo, no lo contradigas):\n"
        f"{verificada.model_dump_json(indent=2)}\n\n"
        f"Paso de supervisión {pasos + 1} de {MAX_PASOS_SUPERVISOR}. "
        f"¿Quién interviene ahora?"
    )

    decision: DecisionSupervisor = await _supervisor_estructurado().ainvoke(
        [SystemMessage(PROMPT_SUPERVISOR), HumanMessage(pedido)]
    )

    log.info(
        "Supervisor (paso %d) -> %s | %s",
        pasos + 1, decision.siguiente, decision.razonamiento[:100],
    )

    return {
        "siguiente": decision.siguiente,
        "instruccion_actual": decision.instruccion,
        "razonamiento_supervisor": decision.razonamiento,
        # Se guarda la rúbrica verificada, no la que declaró el modelo: lo que
        # queda en el estado es el hecho comprobable.
        "rubrica_actual": verificada,
        "pasos_del_supervisor": 1,
        "messages": [
            AIMessage(
                content=f"[supervisor -> {decision.siguiente}] {decision.razonamiento}",
                name="supervisor",
            )
        ],
    }


def enrutar_desde_supervisor(
    estado: dict[str, Any],
) -> Literal["investigador", "analista", "validador"]:
    """
    La arista condicional que sale del supervisor.

    El tipo de retorno es un `Literal` con los nombres exactos de los nodos:
    es lo que permite que LangGraph valide el grafo al compilarlo y que un
    nombre mal escrito falle ahí y no en medio de una corrida.
    """
    destino: Destino = estado.get("siguiente", "investigador")
    return destino


# --------------------------------------------------------------------------
# Nodo validador (determinista, sin LLM)
# --------------------------------------------------------------------------

def nodo_validador(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Verifica contra el estado si el trabajo está completo. Sin LLM.

    Dos resultados posibles:

    - **Aprueba**: las cuatro condiciones de la rúbrica se cumplen según las
      herramientas que realmente corrieron.
    - **Rechaza**: algo falta, y devuelve al supervisor diciendo exactamente
      qué. Salvo que se haya agotado el tope de pasos, caso en el que aprueba
      igual y deja constancia: un informe parcial y honesto es mejor que
      ninguno.
    """
    verificada = rubrica_verificada(estado)
    pasos = estado.get("pasos_del_supervisor", 0)
    faltantes = verificada.faltantes()

    if verificada.completa:
        log.info("Validador: aprobado, las cuatro condiciones se cumplen.")
        return {
            "validacion_aprobada": True,
            "motivo_validacion": "Las cuatro condiciones de la rúbrica se cumplen.",
            "rubrica_actual": verificada,
            "messages": [
                AIMessage(
                    content="[validador] Aprobado: rúbrica completa.",
                    name="validador",
                )
            ],
        }

    if pasos >= MAX_PASOS_SUPERVISOR:
        motivo = (
            f"Aprobado por tope de pasos con la rúbrica incompleta. "
            f"Falta: {', '.join(faltantes)}. El informe va a ser parcial."
        )
        log.warning("Validador: %s", motivo)
        return {
            "validacion_aprobada": True,
            "motivo_validacion": motivo,
            "rubrica_actual": verificada,
            "messages": [
                AIMessage(content=f"[validador] {motivo}", name="validador")
            ],
        }

    motivo = f"Rechazado: falta {', '.join(faltantes)}."
    log.info("Validador: %s Vuelve al supervisor.", motivo)
    return {
        "validacion_aprobada": False,
        "motivo_validacion": motivo,
        "rubrica_actual": verificada,
        "messages": [
            AIMessage(
                content=(
                    f"[validador] {motivo} Devuelvo al supervisor para que "
                    f"complete el trabajo."
                ),
                name="validador",
            )
        ],
    }


def enrutar_desde_validador(estado: dict[str, Any]) -> Literal["supervisor", "sintetizador"]:
    """
    La arista condicional que sale del validador.

    Es el "ciclo de refinamiento" de la consigna: si la validación no pasa, el
    flujo vuelve al supervisor para que mande a alguien a completar lo que
    falta, en lugar de responder a medias.
    """
    return "sintetizador" if estado.get("validacion_aprobada") else "supervisor"


# --------------------------------------------------------------------------
# Nodo sintetizador
# --------------------------------------------------------------------------

def _dossier(estado: dict[str, Any]) -> str:
    """
    Los hallazgos estructurados, formateados para el redactor.

    Acá sí se le pasan los payloads completos, porque es el único nodo que
    necesita los números exactos. El resto del grafo trabajó con resúmenes.
    """
    import json

    bloques: list[str] = []
    for hallazgo in estado.get("hallazgos", []):
        bloques.append(
            f"### [{hallazgo.agente} / {hallazgo.herramienta}]\n"
            f"{hallazgo.resumen}\n"
            f"```json\n{json.dumps(hallazgo.datos, ensure_ascii=False, indent=2)}\n```"
        )
    return "\n\n".join(bloques) or "(sin hallazgos)"


async def nodo_sintetizador(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Redacta el informe final a partir de los hallazgos estructurados.

    No lee `messages`. Lee `hallazgos`, que contiene los payloads tal como los
    devolvieron las herramientas. Por eso los números del informe se pueden
    rastrear hasta la fila de la base que los originó.
    """
    advertencia = ""
    if not estado.get("validacion_aprobada", True) or estado.get(
        "rubrica_actual"
    ) and not estado["rubrica_actual"].completa:
        faltantes = (
            estado["rubrica_actual"].faltantes() if estado.get("rubrica_actual") else []
        )
        if faltantes:
            advertencia = (
                f"\n\nADVERTENCIA: este informe es parcial. No se pudo completar: "
                f"{', '.join(faltantes)}. Decilo explícitamente al principio."
            )

    pedido = (
        f"Solicitud del usuario:\n{estado.get('solicitud', '')}\n\n"
        f"Hallazgos del equipo:\n\n{_dossier(estado)}{advertencia}"
    )

    log.info("Sintetizador redactando sobre %d hallazgos.",
             len(estado.get("hallazgos", [])))

    respuesta = await get_llm().ainvoke(
        [SystemMessage(PROMPT_SINTETIZADOR), HumanMessage(pedido)]
    )
    informe = texto_de(respuesta)

    return {
        "informe_final": informe,
        "messages": [AIMessage(content=informe, name="sintetizador")],
    }
