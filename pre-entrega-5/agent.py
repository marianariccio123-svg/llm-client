"""
agent.py

El grafo del agente: un ciclo ReAct con LangGraph, herramientas asíncronas y
memoria persistente en SQLite.

La topología es mínima a propósito:

        START
          │
          ▼
      ┌────────┐   tools_condition
      │ modelo │ ─────────────────► END   (no pidió herramientas: ya respondió)
      └────────┘
          │  (pidió una o más herramientas)
          ▼
      ┌────────────┐
      │ herramientas│
      └────────────┘
          │
          └──────► vuelve a `modelo`

Ese "vuelve a modelo" es todo el razonamiento cíclico. El modelo ve el
resultado de la herramienta como un mensaje más y decide de nuevo: puede
llamar a otra herramienta, reintentar la misma con argumentos corregidos, o
contestar. Nadie programó esa secuencia — no hay un solo `if` que decida qué
herramienta toca.

Qué permite el checkpointer
---------------------------
`AsyncSqliteSaver` guarda el estado completo del grafo después de cada paso,
indexado por `thread_id`. Dos consecuencias:

- **Memoria entre turnos.** Preguntar "¿y el último?" después de "¿cuánto
  gastó Mariana?" funciona porque el historial de ese thread se levanta de la
  base, no porque se lo vuelva a mandar a mano.
- **Reanudación.** Si el proceso se cae en medio de una cadena de
  herramientas, el estado quedó en disco y el thread se puede retomar.

Se usa `AsyncSqliteSaver` y no el `SqliteSaver` sincrónico que nombra la
consigna porque el grafo entero es asíncrono: el saver sincrónico hace E/S
bloqueante sobre el archivo y congela el event loop en cada checkpoint —
exactamente el error que documenta el corpus de la Pre-entrega 4. Son la misma
clase del mismo paquete (`langgraph-checkpoint-sqlite`), en su versión async.
"""

from __future__ import annotations

import logging
import operator
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import CHECKPOINTS_PATH, DATA_DIR, RECURSION_LIMIT, get_llm
from tools import HERRAMIENTAS

log = logging.getLogger("agent")

PROMPT_SISTEMA = """Sos un asistente que responde preguntas sobre los clientes, \
pedidos y stock de una tienda, consultando la base con las herramientas que tenés \
disponibles.

Cómo trabajar:
- Nunca inventes datos. Si no consultaste la base, no lo sabés.
- Encadená herramientas cuando haga falta: casi ninguna pregunta se responde con \
una sola consulta. Para llegar al detalle de un pedido a partir de un nombre hacen \
falta tres pasos, y está bien que así sea.
- Si una herramienta devuelve un campo "error", leé el resto de la respuesta antes \
de rendirte: casi siempre trae la lista de valores válidos. Reintentá con uno de \
esos en vez de repetir el que falló.
- Si el error dice que algo es ambiguo, NO elijas por tu cuenta: preguntale al \
usuario cuál de las opciones quería y esperá la respuesta.
- Respondé en español rioplatense, con los números concretos que devolvió la base.
- Los montos van en pesos, con separador de miles (ej. $14.500)."""


class EstadoAgente(MessagesState):
    """
    Estado del grafo.

    Hereda de `MessagesState`, que ya trae `messages` con `add_messages` como
    reducer — por eso cada nodo devuelve los mensajes *nuevos* y LangGraph los
    appendea, en lugar de pisar la lista entera.

    Se agrega un contador con `operator.add` como reducer, que es el mismo
    mecanismo aplicado a un int: cada vez que el nodo del modelo devuelve
    `{"turnos_del_modelo": 1}`, LangGraph **suma** en lugar de reemplazar. Sirve
    para leer de un vistazo cuántas vueltas dio el ciclo en un thread.
    """

    turnos_del_modelo: Annotated[int, operator.add]


async def nodo_modelo(estado: EstadoAgente) -> dict[str, Any]:
    """
    Le pasa la conversación al LLM y devuelve lo que conteste.

    Lo único que hace es `ainvoke`. La decisión de llamar o no a una
    herramienta la toma el modelo y viaja dentro de la respuesta, en
    `tool_calls`: este nodo no la inspecciona ni la condiciona.
    """
    mensajes: list[AnyMessage] = [SystemMessage(PROMPT_SISTEMA), *estado["messages"]]
    respuesta = await get_llm().bind_tools(HERRAMIENTAS).ainvoke(mensajes)

    if getattr(respuesta, "tool_calls", None):
        log.info(
            "El modelo pidió %d herramienta(s): %s",
            len(respuesta.tool_calls),
            ", ".join(tc["name"] for tc in respuesta.tool_calls),
        )
    else:
        log.info("El modelo respondió sin pedir herramientas.")

    return {"messages": [respuesta], "turnos_del_modelo": 1}


def construir_grafo() -> StateGraph:
    """
    Arma el `StateGraph` sin compilarlo.

    `tools_condition` es la arista condicional que cierra el ciclo: mira el
    último mensaje y enruta a "tools" si tiene `tool_calls`, o a END si no.
    Es la pieza que reemplaza al `if/else` escrito a mano.
    """
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node("tools", ToolNode(HERRAMIENTAS))

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges("modelo", tools_condition)
    grafo.add_edge("tools", "modelo")  # el retorno que hace el ciclo

    return grafo


@asynccontextmanager
async def abrir_agente(
    ruta_checkpoints: str | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """
    Context manager que entrega el grafo compilado con su checkpointer abierto.

    Es un context manager porque `AsyncSqliteSaver` mantiene una conexión a la
    base: hay que cerrarla. Usarlo así evita el clásico "la última respuesta no
    quedó persistida" cuando el proceso termina antes del flush.

    Args:
        ruta_checkpoints: archivo SQLite donde persistir. `":memory:"` da un
            agente efímero, útil para tests.

    Uso:
        async with abrir_agente() as agente:
            await agente.ainvoke({"messages": [...]}, config=config_thread("demo"))
    """
    DATA_DIR.mkdir(exist_ok=True)
    ruta = ruta_checkpoints or str(CHECKPOINTS_PATH)

    async with AsyncSqliteSaver.from_conn_string(ruta) as checkpointer:
        agente = construir_grafo().compile(checkpointer=checkpointer)
        log.info("Agente listo (checkpoints en %s).", ruta)
        yield agente


def config_thread(thread_id: str, recursion_limit: int = RECURSION_LIMIT) -> dict:
    """
    Config de invocación para un hilo de conversación.

    `thread_id` es la clave de la memoria: dos llamadas con el mismo id
    comparten historial, dos con ids distintos no se ven entre sí.

    `recursion_limit` es el techo de pasos del grafo. Al superarlo, LangGraph
    corta con `GraphRecursionError` en vez de seguir girando — un agente en
    bucle no se cuelga, se detiene y factura una cantidad acotada de llamadas.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def texto_de(mensaje: AnyMessage) -> str:
    """
    Extrae el texto de un mensaje, venga como venga.

    Hace falta porque no todos los proveedores devuelven lo mismo en
    `.content`: OpenAI manda un string pelado, mientras que Gemini y Anthropic
    mandan una lista de bloques tipados (`[{"type": "text", "text": "..."}]`).
    Imprimir el `.content` crudo funcionaría con uno y mostraría el repr de una
    lista con los otros.
    """
    contenido = mensaje.content

    if isinstance(contenido, str):
        return contenido

    if isinstance(contenido, list):
        partes = [
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        ]
        return "\n".join(p for p in partes if p)

    return str(contenido)


async def historial(agente: CompiledStateGraph, thread_id: str) -> list[AnyMessage]:
    """Devuelve los mensajes que el checkpointer tiene guardados para un thread."""
    estado = await agente.aget_state(config_thread(thread_id))
    return estado.values.get("messages", [])
