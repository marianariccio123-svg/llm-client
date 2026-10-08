"""
app/graph.py

El orquestador multi-agente de la Pre-entrega 6, con dos cosas nuevas:
checkpoints en **Redis** y una **pausa de aprobación humana** antes de ejecutar
una acción con efectos secundarios.

Qué se reutiliza y qué se agrega
--------------------------------
Los cinco nodos del módulo 6 se importan tal cual: supervisor, investigador,
analista, validador y sintetizador. No hay una copia de ese código acá, y es
deliberado — una copia divergiría del original en la primera corrección, y
entonces la API estaría corriendo algo distinto de lo que se evaluó.

Lo que cambia es el cableado del final del flujo:

    módulo 6:  validador --aprobado--> sintetizador
    módulo 7:  validador --aprobado--> compuerta_hitl --> [ejecutar_accion] --> sintetizador

El grafo completo:

        START
          │
          ▼
    ┌────────────┐
    │ supervisor │◄──────────────────────┐
    └────────────┘                       │
          │                              │
     ┌────┼──────────────┐               │
     ▼    ▼              ▼               │
 ┌──────────┐ ┌─────────┐ ┌───────────┐  │
 │investig. │ │analista │ │ validador │  │
 └──────────┘ └─────────┘ └───────────┘  │
     │             │            │        │
     └─────────────┘            │  no aprobado
          (vuelven)             ├────────┘
                                │  aprobado
                                ▼
                       ┌──────────────────┐
                       │  compuerta_hitl  │  <-- interrupt() si es crítico
                       └──────────────────┘
                            │          │
                 aprobado   │          │  no hace falta / rechazado
                            ▼          │
                   ┌─────────────────┐ │
                   │ ejecutar_accion │ │
                   └─────────────────┘ │
                            │          │
                            └────┬─────┘
                                 ▼
                        ┌───────────────┐
                        │ sintetizador  │
                        └───────────────┘
                                 │
                                 ▼
                                END

Por qué Redis y no SQLite
-------------------------
El módulo 6 usaba `AsyncSqliteSaver`, que escribe en un archivo local. Eso
alcanza para un CLI, pero no para una API:

- **Varios procesos.** Un `uvicorn --workers 4` son cuatro procesos. Con SQLite
  compiten por el mismo archivo y se bloquean entre sí; con Redis comparten un
  servidor hecho para eso.
- **El estado sobrevive al deploy.** Un job pausado esperando aprobación tiene
  su estado en el checkpointer. Si el checkpointer es un archivo del contenedor,
  el próximo deploy lo borra y la aprobación que llega después no tiene a qué
  volver.
- **La pausa puede durar.** El HITL implica que un grafo quede detenido minutos
  u horas. Ese estado necesita vivir en algún lugar que no sea la memoria del
  worker que lo atendió.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import CLAVE_CHECKPOINTS, TTL_JOB_SEGUNDOS
from app.hitl import enrutar_desde_compuerta, nodo_compuerta_hitl, nodo_ejecutar_accion

# --- Lo que viene del módulo 6, intacto ---
# `app/config.py` ya puso `pre-entrega-6/` en el sys.path al importarse.
from agents.analyst_agent import nodo_analista          # noqa: E402
from agents.research_agent import nodo_investigador     # noqa: E402
from agents.supervisor import (                         # noqa: E402
    enrutar_desde_supervisor,
    nodo_sintetizador,
    nodo_supervisor,
    nodo_validador,
)
from state import EstadoOrquestador                     # noqa: E402

log = logging.getLogger("app.graph")


class EstadoAPI(EstadoOrquestador):
    """
    El estado del módulo 6 más los campos del flujo de aprobación.

    Se extiende en lugar de redefinirse: los reducers de `messages`,
    `hallazgos` y `pasos_del_supervisor` se heredan, así que el
    comportamiento acumulativo sigue siendo el mismo.

    Todos los campos nuevos describen "la última decisión", así que se
    sobreescriben (el comportamiento por defecto) en vez de acumularse.
    """

    requiere_aprobacion: bool
    propuesta_accion: Any           # PropuestaAccion | None
    aprobacion_otorgada: bool
    aprobado_por: str
    comentario_aprobacion: str
    motivo_hitl: str
    accion_ejecutada: dict[str, Any] | None


def enrutar_desde_validador_con_hitl(
    estado: dict[str, Any],
) -> Literal["supervisor", "compuerta_hitl"]:
    """
    Reemplaza la arista del validador del módulo 6.

    La única diferencia con la original es el destino del camino aprobado: en
    vez de ir derecho a redactar, ahora pasa por la compuerta de aprobación.
    El camino de rechazo sigue volviendo al supervisor, así que el ciclo de
    refinamiento del módulo 6 queda intacto.
    """
    return "compuerta_hitl" if estado.get("validacion_aprobada") else "supervisor"


def construir_grafo() -> StateGraph:
    """Arma el `StateGraph` sin compilarlo."""
    grafo = StateGraph(EstadoAPI)

    # Los cinco nodos del módulo 6.
    grafo.add_node("supervisor", nodo_supervisor)
    grafo.add_node("investigador", nodo_investigador)
    grafo.add_node("analista", nodo_analista)
    grafo.add_node("validador", nodo_validador)
    grafo.add_node("sintetizador", nodo_sintetizador)

    # Los dos nuevos.
    grafo.add_node("compuerta_hitl", nodo_compuerta_hitl)
    grafo.add_node("ejecutar_accion", nodo_ejecutar_accion)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar_desde_supervisor,
        ["investigador", "analista", "validador"],
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")

    grafo.add_conditional_edges(
        "validador",
        enrutar_desde_validador_con_hitl,
        ["supervisor", "compuerta_hitl"],
    )
    grafo.add_conditional_edges(
        "compuerta_hitl",
        enrutar_desde_compuerta,
        ["ejecutar_accion", "sintetizador"],
    )
    grafo.add_edge("ejecutar_accion", "sintetizador")
    grafo.add_edge("sintetizador", END)

    return grafo


def compilar(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Compila el grafo, con o sin persistencia."""
    return construir_grafo().compile(checkpointer=checkpointer)


def _serializador():
    """
    El serializador con los tipos del estado registrados.

    Los mismos dos de la Pre-entrega 6 (`Hallazgo`, `Rubrica`) más
    `PropuestaAccion`, que ahora también viaja dentro del estado y por lo tanto
    pasa por el checkpoint. Sin registrarla, el serializador avisa que en una
    versión futura va a bloquearla, y un job pausado esperando aprobación
    dejaría de poder retomarse.

    Tres detalles de este serializador que no son obvios, y que cada uno costó
    un job fallido antes de encontrarse:

    1. **La clase base es `JsonPlusRedisSerializer`**, no el
       `JsonPlusSerializer` genérico de la Pre-entrega 6. El saver de Redis le
       pide métodos propios (`_preprocess_redis_json`, que adapta el payload a
       RedisJSON), así que pisar `serde` con el genérico hace que *todo* job
       falle con un `AttributeError` en la primera escritura de checkpoint.

    2. **Hay que declarar las dos allowlists.** El saver de Redis guarda el
       checkpoint como JSON, no como msgpack, así que `allowed_msgpack_modules`
       sola no alcanza: los modelos vuelven como `dict` y el primer
       `rubrica.completa` revienta. Se pasan las dos porque el serializador usa
       una u otra según el tipo de payload.

    3. **La clave del allowlist es el módulo partido por puntos.**
       `('app', 'schemas', 'PropuestaAccion')`, no
       `('app.schemas', 'PropuestaAccion')`. Con el formato equivocado el
       tipo queda fuera del allowlist en silencio.
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    from app.schemas import PropuestaAccion
    from state import Hallazgo, Rubrica

    permitidos = tuple(
        (*clase.__module__.split("."), clase.__name__)
        for clase in (Hallazgo, Rubrica, PropuestaAccion)
    )
    return JsonPlusRedisSerializer(
        allowed_json_modules=permitidos,
        allowed_msgpack_modules=permitidos,
    )


@asynccontextmanager
async def abrir_orquestador(
    redis_client: Any | None = None,
    redis_url: str | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """
    Entrega el grafo compilado con el checkpointer de Redis abierto.

    Args:
        redis_client: un cliente `redis.asyncio` ya creado. Es la vía
            preferida: así el grafo comparte el pool de la app en lugar de
            abrir uno propio. Un pool por grafo, con el grafo creándose por
            job, sería una fuga de conexiones.
        redis_url: alternativa, para cuando no hay un cliente a mano (scripts).

    Llama a `asetup()`, que crea los índices de RediSearch que usa el saver.
    Es idempotente, así que se puede invocar en cada arranque.
    """
    if redis_client is None and not redis_url:
        raise ValueError("Hace falta `redis_client` o `redis_url`.")

    async with AsyncRedisSaver.from_conn_string(
        redis_url if redis_client is None else None,
        redis_client=redis_client,
        checkpoint_prefix=CLAVE_CHECKPOINTS,
        checkpoint_write_prefix=f"{CLAVE_CHECKPOINTS}_write",
        # Los checkpoints expiran junto con los jobs: una vez que el job se fue,
        # su historial de ejecución no le sirve a nadie y solo ocupa memoria.
        ttl={"default_ttl": TTL_JOB_SEGUNDOS / 60, "refresh_on_read": False},
    ) as checkpointer:
        await checkpointer.asetup()
        checkpointer.serde = _serializador()

        orquestador = compilar(checkpointer)
        log.info("Orquestador listo (checkpoints en Redis, prefijo '%s').",
                 CLAVE_CHECKPOINTS)
        yield orquestador


def entrada(solicitud: str) -> dict[str, Any]:
    """El estado inicial para una solicitud."""
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
        # Campos del flujo HITL.
        "requiere_aprobacion": False,
        "propuesta_accion": None,
        "aprobacion_otorgada": False,
        "aprobado_por": "",
        "comentario_aprobacion": "",
        "motivo_hitl": "",
        "accion_ejecutada": None,
    }


def config_thread(job_id: str, recursion_limit: int = 30) -> dict:
    """
    Config de invocación para un job.

    El `thread_id` es el `job_id`: así el checkpoint de un trabajo se
    encuentra por el mismo identificador que el cliente ya tiene, y retomar un
    job pausado no necesita ninguna tabla de traducción.
    """
    return {
        "configurable": {"thread_id": job_id},
        "recursion_limit": recursion_limit,
    }


def diagrama_mermaid() -> str:
    """El diagrama del grafo, generado desde el grafo compilado."""
    return compilar().get_graph().draw_mermaid()
