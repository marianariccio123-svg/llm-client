"""
graph.py

El grafo del orquestador: topología jerárquica con un supervisor que enruta a
dos especialistas, un validador que verifica, y un sintetizador que cierra.

        START
          │
          ▼
    ┌────────────┐
    │ supervisor │◄──────────────────────┐
    └────────────┘                       │
          │  enrutar_desde_supervisor    │
          │  -> Literal[...]             │
     ┌────┼──────────────┐               │
     ▼    ▼              ▼               │
 ┌──────────┐ ┌─────────┐ ┌───────────┐  │
 │investig. │ │analista │ │ validador │  │
 └──────────┘ └─────────┘ └───────────┘  │
     │             │            │        │
     └─────────────┘            │        │
          (vuelven al       enrutar_desde_validador
           supervisor)          │        │
                                │  no aprobado
                                ├────────┘
                                │  aprobado
                                ▼
                        ┌───────────────┐
                        │ sintetizador  │
                        └───────────────┘
                                │
                                ▼
                               END

Los dos ciclos
--------------
1. `supervisor -> especialista -> supervisor`: el ciclo de trabajo. Cada vuelta
   suma hallazgos al estado.
2. `validador -> supervisor`: el ciclo de refinamiento. Si el trabajo no está
   completo, el flujo vuelve en lugar de responder a medias.

Los dos están acotados por `MAX_PASOS_SUPERVISOR`, que se chequea en el nodo
supervisor antes de llamar al modelo, y por `RECURSION_LIMIT` de LangGraph como
red de contención estructural.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador
from agents.supervisor import (
    enrutar_desde_supervisor,
    enrutar_desde_validador,
    nodo_sintetizador,
    nodo_supervisor,
    nodo_validador,
)
from config import CHECKPOINTS_PATH, DATA_DIR, RECURSION_LIMIT
from state import EstadoOrquestador, Hallazgo, Rubrica

log = logging.getLogger("graph")


def construir_grafo() -> StateGraph:
    """
    Arma el `StateGraph` sin compilarlo.

    Separado de la compilación para que los tests (y el renderizado del
    diagrama) puedan compilar sin checkpointer, mientras el CLI compila con
    persistencia.
    """
    grafo = StateGraph(EstadoOrquestador)

    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("validador", nodo_validador)
    grafo.add_node("sintetizador", nodo_sintetizador)

    grafo.add_edge(START, "supervisor")

    # Arista condicional con tipo de retorno Literal: LangGraph valida los
    # destinos al compilar, así que un nombre mal escrito falla acá y no a
    # mitad de una corrida.
    grafo.add_conditional_edges(
        "supervisor",
        enrutar_desde_supervisor,
        ["investigador", "analista", "validador"],
    )

    # Los especialistas siempre devuelven el control al supervisor: ninguno
    # decide quién sigue, que es lo que hace jerárquica a la topología.
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")

    grafo.add_conditional_edges(
        "validador",
        enrutar_desde_validador,
        ["supervisor", "sintetizador"],
    )

    grafo.add_edge("sintetizador", END)

    return grafo


def compilar(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Compila el grafo, con o sin persistencia."""
    return construir_grafo().compile(checkpointer=checkpointer)


def _serializador() -> JsonPlusSerializer:
    """
    Serializador del checkpointer, con los tipos del estado registrados.

    El estado lleva modelos Pydantic propios (`Hallazgo`, `Rubrica`). El
    serializador de LangGraph, por defecto, los deserializa igual pero emite un
    warning por cada uno y avisa que en una versión futura va a bloquearlos —
    momento en el que el orquestador dejaría de poder releer sus propios
    checkpoints.

    Dos detalles que no son obvios:

    - **`with_msgpack_allowlist()` no sirve acá.** Si la allowlist base es la
      permisiva (`True`, el default), ese método devuelve el mismo objeto sin
      tocar nada. Hay que pasar la allowlist al constructor.
    - **Declarar una allowlist explícita no rompe el resto.** Los tipos de
      LangChain (`HumanMessage`, `AIMessage`, ...) están en la lista de tipos
      seguros incorporada, así que siguen deserializándose.

    El par `(módulo, clase)` se deriva de las clases en vez de escribirse a
    mano: si alguien renombra `Hallazgo` o mueve `state.py`, la allowlist lo
    sigue sin que nadie se acuerde de actualizarla.
    """
    return JsonPlusSerializer(
        allowed_msgpack_modules=tuple(
            (clase.__module__, clase.__name__) for clase in (Hallazgo, Rubrica)
        )
    )


@asynccontextmanager
async def abrir_orquestador(
    ruta_checkpoints: str | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """
    Context manager que entrega el grafo compilado con su checkpointer abierto.

    Args:
        ruta_checkpoints: archivo SQLite donde persistir. `":memory:"` da un
            orquestador efímero, útil para demos reproducibles.
    """
    DATA_DIR.mkdir(exist_ok=True)
    ruta = ruta_checkpoints or str(CHECKPOINTS_PATH)

    async with AsyncSqliteSaver.from_conn_string(ruta) as checkpointer:
        checkpointer.serde = _serializador()
        orquestador = compilar(checkpointer)
        log.info("Orquestador listo (checkpoints en %s).", ruta)
        yield orquestador


def entrada(solicitud: str) -> dict[str, Any]:
    """
    El estado inicial para una solicitud.

    `solicitud` se guarda como campo propio además de ir al historial: los
    especialistas no reciben `messages`, así que es de ahí de donde sacan el
    marco de la tarea.
    """
    return {
        "messages": [HumanMessage(solicitud)],
        "solicitud": solicitud,
        "hallazgos": [],
        "pasos_del_supervisor": 0,
        "siguiente": "investigador",
        "instruccion_actual": "",
        "razonamiento_supervisor": "",
        "rubrica_actual": None,
        "validacion_aprobada": False,
        "motivo_validacion": "",
        "informe_final": "",
    }


def config_thread(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    """Config de invocación para un hilo de conversación."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def diagrama_mermaid() -> str:
    """
    El diagrama del grafo en sintaxis Mermaid.

    Se genera desde el grafo compilado y no se escribe a mano: si alguien
    agrega un nodo y se olvida de actualizar el README, el diagrama regenerado
    lo delata. `draw_mermaid()` en vez de `draw_mermaid_png()` porque el texto
    se versiona y se revisa en un diff; un PNG, no.
    """
    return compilar().get_graph().draw_mermaid()
