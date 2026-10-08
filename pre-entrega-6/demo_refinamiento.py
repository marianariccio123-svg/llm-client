"""
demo_refinamiento.py

Demuestra el **ciclo de refinamiento**: el validador rechaza un trabajo
incompleto y lo devuelve al supervisor, que manda a completar lo que falta.

Por qué hace falta un script aparte
-----------------------------------
El ciclo de refinamiento es difícil de provocar con el modelo real, y la razón
es una propiedad buscada del diseño: el supervisor recibe en su prompt la
**rúbrica verificada contra el estado**, así que casi nunca declara terminado
un trabajo al que le falta algo. El validador existe igual, porque "casi nunca"
no es "nunca" y porque la verificación no debe depender del criterio del
modelo.

Para mostrar el mecanismo hay que simular un supervisor que se equivoca. Este
script usa el LLM guionado de los tests (`tests/stub_llm.py`) para forzar
exactamente eso: un supervisor que, con una sola herramienta ejecutada, declara
la tarea completa y deriva al validador.

**La traza que genera es determinista y simulada**, y lo dice en su encabezado.
No reemplaza a `traces/flujo-delegacion.log`, que es una corrida real contra la
API; la complementa cubriendo el camino que la corrida real no toma.

Uso:
    python demo_refinamiento.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage

import agents.analyst_agent as analyst
import agents.research_agent as research
import agents.supervisor as supervisor
from config import MAX_PASOS_SUPERVISOR, TRACES_DIR
from graph import compilar, entrada
from state import DecisionSupervisor, Rubrica
from tests.stub_llm import LLMGuionado, mensaje_con_herramientas


def _decision(siguiente: str, razonamiento: str, **rubrica) -> DecisionSupervisor:
    """Una decisión del supervisor, con la rúbrica que el modelo *declara*."""
    base = dict(
        hay_evidencia_documental=False,
        hay_datos_cuantitativos=False,
        hay_analisis_estadistico=False,
        hay_validacion_de_datos=False,
    )
    base.update(rubrica)
    return DecisionSupervisor(
        razonamiento=razonamiento,
        rubrica=Rubrica(**base),
        siguiente=siguiente,
        instruccion=f"Tarea para {siguiente}.",
    )


GUION_ESTRUCTURADO = [
    _decision(
        "investigador",
        "Arranco pidiendo los incidentes.",
    ),
    # Acá está el error deliberado: el supervisor declara las cuatro
    # condiciones cumplidas cuando solo corrió `listar_incidentes`.
    _decision(
        "validador",
        "Ya tengo los incidentes, me parece que con esto alcanza. Cierro.",
        hay_evidencia_documental=True,
        hay_datos_cuantitativos=True,
        hay_analisis_estadistico=True,
        hay_validacion_de_datos=True,
    ),
    # Después del rechazo, corrige el rumbo.
    _decision(
        "investigador",
        "El validador me marcó que falta la evidencia documental. La busco.",
        hay_datos_cuantitativos=True,
    ),
    _decision(
        "analista",
        "Ahora sí tengo documentación y datos; falta el análisis.",
        hay_evidencia_documental=True,
        hay_datos_cuantitativos=True,
    ),
    _decision(
        "validador",
        "Las cuatro condiciones están cumplidas de verdad.",
        hay_evidencia_documental=True,
        hay_datos_cuantitativos=True,
        hay_analisis_estadistico=True,
        hay_validacion_de_datos=True,
    ),
]

GUION_MENSAJES = [
    # 1er paso del investigador: solo trae incidentes (incompleto a propósito).
    mensaje_con_herramientas(("listar_incidentes", {"meses": 12})),
    AIMessage(content="Traje los 20 incidentes del último año."),
    # 2do paso del investigador, después del rechazo: trae la documentación.
    mensaje_con_herramientas(
        ("buscar_en_documentacion", {"consulta": "causas de caídas en asyncio"})
    ),
    AIMessage(content="Ahora sí, traje la documentación técnica."),
    # El analista.
    mensaje_con_herramientas(
        ("resumir_incidentes", {}),
        ("detectar_atipicos", {}),
        ("validar_registros", {}),
    ),
    AIMessage(content="Analizado: domina el bloqueo del event loop."),
    # El sintetizador.
    AIMessage(content="INFORME: (redactado sobre los hallazgos completos)."),
]


def instalar_stub() -> LLMGuionado:
    """Reemplaza el modelo en los tres módulos por el LLM guionado."""
    fake = LLMGuionado(
        respuestas=list(GUION_MENSAJES), estructuradas=list(GUION_ESTRUCTURADO)
    )
    for modulo in (research, analyst, supervisor):
        modulo.get_llm = lambda: fake

    # Sin red: la búsqueda documental va por su ruta local de BM25.
    research._indice_pinecone = lambda: None

    for limpiar in (
        research._agente_react, analyst._agente_react,
        supervisor._supervisor_estructurado, research._bm25_local,
    ):
        limpiar.cache_clear()

    return fake


async def main() -> int:
    instalar_stub()

    lineas: list[str] = [
        "=" * 78,
        "CICLO DE REFINAMIENTO: EL VALIDADOR RECHAZA Y DEVUELVE",
        "=" * 78,
        "",
        "ATENCIÓN: esta traza es DETERMINISTA Y SIMULADA. El modelo está",
        "reemplazado por el LLM guionado de los tests (tests/stub_llm.py), para",
        "forzar un supervisor que declara la tarea completa cuando no lo está.",
        "Con el modelo real esto casi no pasa, porque el supervisor recibe la",
        "rúbrica verificada contra el estado y la sigue — ver",
        "traces/flujo-delegacion.log para la corrida real.",
        "",
        f"tope del supervisor: {MAX_PASOS_SUPERVISOR} pasos",
        f"generada:            {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "-" * 78,
        "",
    ]

    orquestador = compilar()
    estado_final: dict = {}
    # `astream(stream_mode="updates")` entrega el DELTA de cada nodo, no el
    # estado reducido: `pasos_del_supervisor` llega como 1 en cada vuelta
    # porque su reducer es `operator.add`. Para el total hay que contar las
    # visitas acá, o leer el estado final del grafo.
    visitas_al_supervisor = 0

    async for fragmento in orquestador.astream(
        entrada("¿Qué nos está rompiendo producción?"),
        config={"recursion_limit": 30},
        stream_mode="updates",
    ):
        for nodo, actualizacion in fragmento.items():
            estado_final.update(actualizacion)

            if nodo == "supervisor":
                visitas_al_supervisor += 1
                lineas.append(
                    f"SUPERVISOR -> {actualizacion.get('siguiente')}\n"
                    f"    \"{actualizacion.get('razonamiento_supervisor')}\""
                )
                rubrica = actualizacion.get("rubrica_actual")
                if rubrica is not None:
                    faltan = rubrica.faltantes()
                    lineas.append(
                        f"    rúbrica verificada: "
                        f"{'completa' if not faltan else 'falta ' + ', '.join(faltan)}"
                    )

            elif nodo in ("investigador", "analista"):
                hallazgos = actualizacion.get("hallazgos", [])
                lineas.append(f"{nodo.upper()} aportó {len(hallazgos)} hallazgo(s):")
                lineas += [
                    f"    [{h.herramienta}] {h.resumen}" for h in hallazgos
                ]

            elif nodo == "validador":
                marca = (
                    "APROBADO" if actualizacion.get("validacion_aprobada")
                    else "RECHAZADO  <-- acá se cierra el ciclo de refinamiento"
                )
                lineas.append(
                    f"VALIDADOR -> {marca}\n"
                    f"    {actualizacion.get('motivo_validacion')}"
                )

            elif nodo == "sintetizador":
                lineas.append("SINTETIZADOR -> informe final redactado")

            lineas.append("")

    lineas += [
        "-" * 78,
        "QUÉ MOSTRÓ ESTA TRAZA",
        "-" * 78,
        "",
        "1. El supervisor derivó al validador declarando la rúbrica completa,",
        "   cuando solo se había ejecutado `listar_incidentes`.",
        "2. El validador NO le creyó: recorrió los hallazgos del estado, vio que",
        "   faltaba la evidencia documental y el análisis, y devolvió el control",
        "   al supervisor en lugar de dejar pasar un informe a medias.",
        "3. El supervisor corrigió el rumbo y completó el trabajo.",
        "4. Recién con las cuatro condiciones cumplidas de verdad, el validador",
        "   aprobó y el flujo llegó al sintetizador.",
        "",
        f"pasos de supervisión: {visitas_al_supervisor} "
        f"(tope: {MAX_PASOS_SUPERVISOR})",
        f"vueltas al validador: "
        f"{sum(1 for l in lineas if l.startswith('VALIDADOR'))}",
        "",
    ]

    TRACES_DIR.mkdir(exist_ok=True)
    destino = TRACES_DIR / "ciclo-refinamiento.log"
    destino.write_text("\n".join(lineas), encoding="utf-8")

    print("\n".join(lineas))
    print(f"Traza guardada en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
