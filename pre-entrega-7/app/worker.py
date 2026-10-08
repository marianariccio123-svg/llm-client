"""
app/worker.py

El patrón worker: una cola en memoria y N corrutinas que la consumen.

Por qué una cola y no `BackgroundTasks`
---------------------------------------
`BackgroundTasks` de FastAPI dispara la tarea sin ningún límite: 50 peticiones
simultáneas son 50 grafos corriendo a la vez, 50 veces la cuota del modelo
consumida en paralelo, y una máquina que se queda sin memoria. No hay forma de
aplicar contrapresión ni de saber cuánto trabajo hay pendiente.

Con una `asyncio.Queue` acotada y un número fijo de workers:

- La concurrencia tiene techo (`CANTIDAD_WORKERS`).
- La cola tiene techo (`TAMANIO_COLA`): al llenarse, la API responde 503 en
  lugar de aceptar trabajo que no va a poder atender.
- `queue.qsize()` es una métrica real de saturación, que `/health` expone.

Es el "thread asíncrono simple" que menciona la consigna, con los dos topes que
hacen la diferencia entre encolar y acumular.

Qué garantiza este módulo
-------------------------
**Que ningún job quede colgado.** Es la regla que organiza todo el archivo. Un
cliente que hace polling necesita que el job llegue a un estado terminal o a
ESPERANDO_APROBACION, siempre. Por eso:

- El cuerpo del worker atrapa `BaseException`, no `Exception`: un
  `asyncio.CancelledError` por apagado del proceso también tiene que dejar el
  job en un estado legible.
- Hay un timeout por job, así que un grafo que se cuelga termina en FALLIDO en
  lugar de ocupar un worker para siempre.
- `task_done()` va en un `finally`, para que la cola no quede con un contador
  roto si algo falla en el medio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.types import Command

from app.config import (
    CANTIDAD_WORKERS,
    TAMANIO_COLA,
    TIMEOUT_JOB_SEGUNDOS,
)
from app.graph import config_thread, entrada
from app.jobs import RepositorioJobs, serializar_hallazgos
from app.observability import metadatos_del_job, trazar_job
from app.schemas import DecisionAprobacion, EstadoJob, PropuestaAccion

log = logging.getLogger("app.worker")


class ColaLlena(RuntimeError):
    """La cola está al tope. El llamador debe responder 503."""


class PoolDeWorkers:
    """
    La cola de trabajos y las corrutinas que la atienden.

    Vive mientras vive la app: se arranca en el `lifespan` de FastAPI y se
    apaga ahí mismo. No es un singleton global — se guarda en `app.state`, así
    los tests pueden levantar uno propio sin contaminar a otro.
    """

    def __init__(
        self,
        repositorio: RepositorioJobs,
        orquestador: Any,
        cantidad: int = CANTIDAD_WORKERS,
        tamanio_cola: int = TAMANIO_COLA,
    ) -> None:
        self.repositorio = repositorio
        self.orquestador = orquestador
        self.cantidad = cantidad
        self.cola: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=tamanio_cola)
        self._tareas: list[asyncio.Task] = []

    # --- ciclo de vida ---

    async def arrancar(self) -> None:
        """Levanta las corrutinas consumidoras."""
        for numero in range(self.cantidad):
            tarea = asyncio.create_task(
                self._consumir(numero), name=f"worker-{numero}"
            )
            # Se guarda la referencia: el event loop solo mantiene una
            # referencia débil a las tasks, y una task sin dueño puede ser
            # recolectada y cancelada en silencio.
            self._tareas.append(tarea)
        log.info("%d workers arrancados (cola de hasta %d).",
                 self.cantidad, self.cola.maxsize)

    async def apagar(self) -> None:
        """
        Cancela los workers y espera a que terminen.

        Se cancela en lugar de esperar a que la cola se vacíe: un apagado que
        espera a que terminen 40 jobs pendientes no es un apagado. Los jobs que
        estaban corriendo quedan en FALLIDO por el `except BaseException`, y
        los que estaban en PENDIENTE siguen en Redis con ese estado.
        """
        for tarea in self._tareas:
            tarea.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        self._tareas.clear()
        log.info("Workers apagados.")

    # --- encolado ---

    def encolar(self, job_id: str, comando: Any = None) -> None:
        """
        Pone un job en la cola sin bloquear.

        `put_nowait` y no `await put()` a propósito: esto se llama desde un
        endpoint, y si la cola está llena un `await` dejaría la petición HTTP
        colgada hasta que se libere lugar. Es mejor un 503 inmediato y honesto.

        Args:
            comando: `None` para una ejecución nueva, o un `Command(resume=...)`
                para retomar un job que estaba esperando aprobación.
        """
        try:
            self.cola.put_nowait((job_id, comando))
        except asyncio.QueueFull as e:
            raise ColaLlena(
                f"La cola está al tope ({self.cola.maxsize} trabajos). "
                f"Reintentá en unos segundos."
            ) from e

    @property
    def pendientes(self) -> int:
        return self.cola.qsize()

    @property
    def workers_vivos(self) -> int:
        return sum(1 for t in self._tareas if not t.done())

    # --- consumo ---

    async def _consumir(self, numero: int) -> None:
        """El bucle de un worker. Corre hasta que lo cancelen."""
        log.info("worker-%d esperando trabajo.", numero)
        while True:
            job_id, comando = await self.cola.get()
            try:
                await self._ejecutar_con_red_de_contencion(job_id, comando)
            finally:
                # En el `finally` para que una excepción no deje el contador de
                # la cola desbalanceado y un `join()` colgado para siempre.
                self.cola.task_done()

    async def _ejecutar_con_red_de_contencion(
        self, job_id: str, comando: Any
    ) -> None:
        """
        Corre un job y garantiza que termine en un estado legible.

        Es la red de contención que menciona el docstring del módulo: pase lo
        que pase adentro, el job sale de acá en COMPLETADO, FALLIDO o
        ESPERANDO_APROBACION. Nunca se queda en EJECUTANDO.
        """
        try:
            await asyncio.wait_for(
                ejecutar_job(self.repositorio, self.orquestador, job_id, comando),
                timeout=TIMEOUT_JOB_SEGUNDOS,
            )

        except asyncio.TimeoutError:
            await self._fallar(
                job_id,
                TimeoutError(
                    f"El job excedió los {TIMEOUT_JOB_SEGUNDOS} segundos de "
                    f"ejecución y se abortó."
                ),
            )

        except asyncio.CancelledError:
            # El proceso se está apagando. Se marca el job y se vuelve a lanzar
            # la cancelación: tragársela dejaría la task del worker en un
            # estado inconsistente y colgaría el `gather` del apagado.
            await self._fallar(
                job_id,
                RuntimeError("El worker fue cancelado (apagado del servicio)."),
            )
            raise

        except BaseException as e:  # noqa: BLE001
            # Deliberadamente amplio. Cualquier excepción que no se traduzca a
            # FALLIDO es un cliente haciendo polling para siempre.
            await self._fallar(job_id, e)

    async def _fallar(self, job_id: str, error: BaseException) -> None:
        """Marca el job como FALLIDO, protegiendo también esa escritura."""
        try:
            await self.repositorio.marcar_fallido(job_id, error)
        except Exception:
            # Si Redis tampoco responde no hay nada más que hacer, pero el log
            # tiene que quedar: es la única evidencia que va a haber.
            log.exception(
                "No se pudo marcar el job %s como FALLIDO. Error original: %s",
                job_id, error,
            )


# --------------------------------------------------------------------------
# La ejecución de un job
# --------------------------------------------------------------------------

@trazar_job
async def ejecutar_job(
    repositorio: RepositorioJobs,
    orquestador: Any,
    job_id: str,
    comando: Any = None,
) -> None:
    """
    Corre el grafo para un job y persiste el resultado.

    Sirve para los dos casos:

    - **Ejecución nueva** (`comando=None`): arranca el grafo desde el estado
      inicial.
    - **Retomar tras aprobación** (`comando=Command(resume=...)`): LangGraph
      levanta el estado del checkpoint de Redis y sigue desde el `interrupt()`.

    El `thread_id` es el `job_id` en los dos casos, y es lo que hace que
    retomar funcione sin ninguna tabla intermedia.

    Si el grafo se detiene en un `interrupt`, el resultado trae la clave
    `__interrupt__`: ahí el job pasa a ESPERANDO_APROBACION y el worker lo
    suelta. No se queda esperando a la persona.
    """
    es_reanudacion = comando is not None

    if not es_reanudacion:
        await repositorio.marcar_ejecutando(job_id)
    else:
        log.info("Job %s: retomando tras aprobación.", job_id)
        await repositorio.actualizar(job_id, estado=EstadoJob.EJECUTANDO)

    job = await repositorio.obtener(job_id)
    config = {
        **config_thread(job_id),
        **metadatos_del_job(job_id, "reanudacion" if es_reanudacion else "inicial"),
    }

    carga = comando if es_reanudacion else entrada(job.solicitud)
    resultado = await orquestador.ainvoke(carga, config=config)

    interrupciones = resultado.get("__interrupt__")
    if interrupciones:
        propuesta = _propuesta_de(interrupciones)
        if propuesta is not None:
            await repositorio.marcar_esperando_aprobacion(job_id, propuesta)
            return
        # Un interrupt sin propuesta legible es un bug del grafo, no del
        # cliente: se falla ruidosamente en vez de dejar el job en el aire.
        raise RuntimeError(
            "El grafo se interrumpió sin una propuesta de acción legible. "
            f"Payload: {interrupciones!r:.300}"
        )

    await repositorio.marcar_completado(
        job_id,
        informe=resultado.get("informe_final", ""),
        hallazgos=serializar_hallazgos(resultado.get("hallazgos", [])),
        pasos_del_supervisor=resultado.get("pasos_del_supervisor", 0),
        accion_ejecutada=resultado.get("accion_ejecutada"),
    )


def _propuesta_de(interrupciones: Any) -> PropuestaAccion | None:
    """
    Extrae la `PropuestaAccion` del payload de `__interrupt__`.

    LangGraph devuelve una tupla de objetos `Interrupt`, cada uno con un
    `.value` que es lo que se le pasó a `interrupt()`. Se busca el primero que
    tenga forma de propuesta, en vez de asumir que hay exactamente uno: si
    algún día hay dos nodos que interrumpen, esto sigue funcionando.
    """
    if not isinstance(interrupciones, (list, tuple)):
        interrupciones = [interrupciones]

    for interrupcion in interrupciones:
        valor = getattr(interrupcion, "value", interrupcion)
        if not isinstance(valor, dict):
            continue
        crudo = valor.get("propuesta")
        if isinstance(crudo, dict):
            try:
                return PropuestaAccion.model_validate(crudo)
            except Exception:
                log.warning("Payload de interrupt con forma inesperada: %r", crudo)
    return None


def comando_de_reanudacion(decision: DecisionAprobacion) -> Command:
    """
    Traduce la decisión humana al `Command` que retoma el grafo.

    El diccionario que viaja en `resume` es exactamente lo que
    `nodo_compuerta_hitl` recibe como valor de retorno de `interrupt()`.
    Mantener ese contrato en una función —y no armar el dict a mano en el
    endpoint— es lo que evita que un cambio en el nodo rompa la API en
    silencio.
    """
    return Command(
        resume={
            "aprobado": decision.aprobado,
            "aprobado_por": decision.aprobado_por,
            "comentario": decision.comentario,
        }
    )
