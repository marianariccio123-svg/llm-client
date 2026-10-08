"""
app/jobs.py

El repositorio del estado de los jobs en Redis.

Toda la persistencia del estado pasa por acá. Los endpoints y los workers no
hablan con Redis directamente: piden o guardan `Job`. Eso tiene dos
consecuencias prácticas:

- El formato de las claves y la serialización están en un solo lugar.
- Los tests pueden sustituir el cliente por `fakeredis` sin tocar nada más.

Por qué el estado va en Redis y no en memoria
---------------------------------------------
Un diccionario en memoria funcionaría mientras el proceso viva. Pero la API es
asíncrona: el cliente recibe un `job_id` y vuelve a preguntar después,
posiblemente contra **otra instancia** del proceso detrás de un balanceador, o
contra la misma instancia después de un reinicio. Si el estado vive en memoria,
ese cliente recibe un 404 por un trabajo que sí existió, y no tiene forma de
distinguirlo de un id inventado.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.config import (
    CLAVE_INDICE_JOBS,
    CLAVE_JOB,
    TTL_JOB_SEGUNDOS,
)
from app.schemas import DecisionAprobacion, EstadoJob, Job, PropuestaAccion

log = logging.getLogger("app.jobs")


class JobNoEncontrado(KeyError):
    """No existe un job con ese id (o ya expiró su TTL)."""


class RepositorioJobs:
    """
    Acceso al estado de los jobs.

    Args:
        redis: un cliente `redis.asyncio.Redis` (o `fakeredis.aioredis.FakeRedis`
            en los tests). El repositorio no lo crea ni lo cierra: de eso se
            encarga el ciclo de vida de la app, porque el pool es compartido.
    """

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    # --- lectura ---

    def _clave(self, job_id: str) -> str:
        return f"{CLAVE_JOB}:{job_id}"

    async def obtener(self, job_id: str) -> Job:
        """Devuelve el job, o lanza `JobNoEncontrado`."""
        crudo = await self._redis.get(self._clave(job_id))
        if crudo is None:
            raise JobNoEncontrado(job_id)
        if isinstance(crudo, bytes):
            crudo = crudo.decode("utf-8")
        return Job.model_validate_json(crudo)

    async def existe(self, job_id: str) -> bool:
        return bool(await self._redis.exists(self._clave(job_id)))

    async def listar(self, limite: int = 50) -> list[Job]:
        """
        Los jobs más recientes primero.

        El índice es una lista de Redis a la que se hace `LPUSH`, así que el
        orden sale gratis y no hace falta ordenar por timestamp en Python.
        """
        ids = await self._redis.lrange(CLAVE_INDICE_JOBS, 0, limite - 1)
        jobs: list[Job] = []
        for job_id in ids:
            if isinstance(job_id, bytes):
                job_id = job_id.decode("utf-8")
            try:
                jobs.append(await self.obtener(job_id))
            except JobNoEncontrado:
                # El job expiró por TTL pero su id sigue en el índice. No es un
                # error: se omite y sigue.
                continue
        return jobs

    # --- escritura ---

    async def _guardar(self, job: Job) -> Job:
        """Persiste el job y le refresca el TTL."""
        job.actualizado_en = Job.ahora()
        await self._redis.set(
            self._clave(job.job_id),
            job.model_dump_json(),
            ex=TTL_JOB_SEGUNDOS,
        )
        return job

    async def crear(self, solicitud: str) -> Job:
        """Crea un job en PENDIENTE y lo deja listo para que un worker lo tome."""
        ahora = Job.ahora()
        job = Job(
            job_id=uuid.uuid4().hex[:16],
            estado=EstadoJob.PENDIENTE,
            solicitud=solicitud,
            creado_en=ahora,
            actualizado_en=ahora,
        )
        await self._guardar(job)
        await self._redis.lpush(CLAVE_INDICE_JOBS, job.job_id)
        # El índice se recorta para que no crezca sin límite: los jobs viejos
        # expiran por TTL igual, y un índice infinito solo acumula ids muertos.
        await self._redis.ltrim(CLAVE_INDICE_JOBS, 0, 999)

        log.info("Job %s creado (PENDIENTE).", job.job_id)
        return job

    async def actualizar(self, job_id: str, **campos: Any) -> Job:
        """
        Lee, modifica y vuelve a guardar.

        No es atómico: dos escrituras concurrentes sobre el mismo job podrían
        perder una. En este diseño no pasa, porque un job lo toca un solo
        worker a la vez y el endpoint de aprobación solo escribe cuando el job
        está pausado. Si eso cambiara, haría falta un `WATCH`/`MULTI` o un
        script Lua.
        """
        job = await self.obtener(job_id)
        for clave, valor in campos.items():
            setattr(job, clave, valor)
        return await self._guardar(job)

    # --- transiciones de estado ---

    async def marcar_ejecutando(self, job_id: str) -> Job:
        log.info("Job %s -> EJECUTANDO.", job_id)
        return await self.actualizar(
            job_id, estado=EstadoJob.EJECUTANDO, iniciado_en=Job.ahora()
        )

    async def marcar_esperando_aprobacion(
        self, job_id: str, propuesta: PropuestaAccion
    ) -> Job:
        """
        El grafo se pausó en el nodo HITL.

        Este estado **no** es terminal, pero el worker suelta el job: la
        ejecución se retoma recién cuando llega la aprobación. Si el worker se
        quedara esperando, una aprobación que tarda una hora bloquearía un
        worker una hora.
        """
        log.info("Job %s -> ESPERANDO_APROBACION (%s).", job_id, propuesta.accion)
        return await self.actualizar(
            job_id, estado=EstadoJob.ESPERANDO_APROBACION, propuesta=propuesta
        )

    async def registrar_decision(
        self, job_id: str, decision: DecisionAprobacion
    ) -> Job:
        """Guarda la decisión humana antes de retomar el grafo."""
        log.info(
            "Job %s: %s por %s.",
            job_id, "APROBADO" if decision.aprobado else "RECHAZADO",
            decision.aprobado_por,
        )
        return await self.actualizar(job_id, decision=decision)

    async def marcar_completado(
        self,
        job_id: str,
        informe: str,
        hallazgos: list[dict[str, Any]],
        pasos_del_supervisor: int,
        accion_ejecutada: dict[str, Any] | None = None,
    ) -> Job:
        job = await self.obtener(job_id)
        terminado = Job.ahora()
        log.info("Job %s -> COMPLETADO.", job_id)
        return await self.actualizar(
            job_id,
            estado=EstadoJob.COMPLETADO,
            informe=informe,
            hallazgos=hallazgos,
            pasos_del_supervisor=pasos_del_supervisor,
            accion_ejecutada=accion_ejecutada,
            terminado_en=terminado,
            duracion_segundos=_duracion(job.creado_en, terminado),
        )

    async def marcar_fallido(self, job_id: str, error: BaseException) -> Job:
        """
        Deja el job en FALLIDO con el detalle del error.

        Es el punto más importante de este módulo. Si una excepción en segundo
        plano no se traduce a un estado terminal, el cliente hace polling para
        siempre sobre un job que nadie está ejecutando. Por eso el worker
        envuelve todo en un `try/except BaseException` y llama acá.

        El mensaje se recorta: una traza de SQL o un payload de API dentro del
        campo de error puede filtrar información y además no le sirve a nadie
        por la API. El detalle completo queda en los logs y en la traza.
        """
        job = await self.obtener(job_id)
        terminado = Job.ahora()
        mensaje = str(error).strip() or error.__class__.__name__

        log.error(
            "Job %s -> FALLIDO (%s): %s",
            job_id, type(error).__name__, mensaje, exc_info=error,
        )
        return await self.actualizar(
            job_id,
            estado=EstadoJob.FALLIDO,
            error=mensaje[:500],
            error_tipo=type(error).__name__,
            terminado_en=terminado,
            duracion_segundos=_duracion(job.creado_en, terminado),
        )

    # --- salud ---

    async def ping(self) -> bool:
        """¿Responde Redis? Lo usa `GET /health`."""
        try:
            return bool(await self._redis.ping())
        except Exception as e:
            log.warning("Redis no responde: %s", type(e).__name__)
            return False


def _duracion(desde: str, hasta: str) -> float | None:
    """Segundos entre dos timestamps ISO, redondeados a milisegundos."""
    from datetime import datetime

    try:
        inicio = datetime.fromisoformat(desde)
        fin = datetime.fromisoformat(hasta)
        return round((fin - inicio).total_seconds(), 3)
    except (ValueError, TypeError):
        return None


def serializar_hallazgos(hallazgos: list[Any]) -> list[dict[str, Any]]:
    """
    Convierte los `Hallazgo` del grafo en dicts JSON-serializables.

    El estado del grafo lleva modelos Pydantic de la Pre-entrega 6; el `Job`
    guarda dicts planos. La conversión se hace acá, en el borde, y no en el
    esquema: así `app/` no importa los tipos del módulo 6 más que en el único
    lugar donde se ejecuta el grafo.
    """
    salida: list[dict[str, Any]] = []
    for hallazgo in hallazgos:
        if hasattr(hallazgo, "model_dump"):
            datos = hallazgo.model_dump()
        elif isinstance(hallazgo, dict):
            datos = dict(hallazgo)
        else:
            continue
        # El payload completo puede pesar decenas de KB (los fragmentos de
        # documentación, los 20 incidentes). Para el estado del job alcanza el
        # resumen y la procedencia; el dato crudo está en el checkpoint.
        datos.pop("datos", None)
        salida.append(datos)
    return salida


def json_compacto(valor: Any) -> str:
    """Serializa a JSON sin espacios, para lo que va a Redis."""
    return json.dumps(valor, ensure_ascii=False, separators=(",", ":"))
