"""
app/main.py

La API REST que expone el orquestador multi-agente.

El contrato
-----------
    POST /tasks                  -> 202 + job_id. No espera al agente.
    GET  /tasks/{id}             -> el estado del job. Nunca bloquea.
    POST /tasks/{id}/approve     -> aprueba o rechaza la acción crítica.
    GET  /tasks                  -> los últimos jobs.
    GET  /escalaciones           -> el efecto de las acciones aprobadas.
    GET  /health                 -> Redis, workers, cola, observabilidad.

Por qué ningún endpoint ejecuta el agente
-----------------------------------------
Una corrida del orquestador son entre 8 y 15 llamadas a un modelo: del orden
de un minuto. Resolverla dentro del request significaría un HTTP abierto todo
ese tiempo, un timeout del proxy a mitad de camino y un cliente que no sabe si
el trabajo se hizo o no.

`POST /tasks` escribe el job en Redis, lo pone en la cola y devuelve. Lo que
tarda en responder es lo que tarda Redis: milisegundos. El trabajo pesado lo
hacen los workers, fuera del ciclo de request.

Sobre bloquear el event loop
----------------------------
Todo el camino es asíncrono de punta a punta: `redis.asyncio` para el estado,
`ainvoke` para el grafo, clientes async para los modelos. No hay una sola
llamada sincrónica a red en un endpoint, que es el error que la consigna marca
como el más caro. La regla práctica para quien toque esto después: si una
función nueva hace E/S y no es `async`, va envuelta en
`starlette.concurrency.run_in_threadpool` o no entra.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app import acciones, llm, observability
from app.config import (
    CANTIDAD_WORKERS,
    TAMANIO_COLA,
    require_redis_url,
    setup_logging,
)
from app.graph import abrir_orquestador
from app.jobs import JobNoEncontrado, RepositorioJobs
from app.schemas import (
    DecisionAprobacion,
    EstadoJob,
    RespuestaEstado,
    Salud,
    SolicitudTarea,
    TareaEncolada,
)
from app.worker import ColaLlena, PoolDeWorkers, comando_de_reanudacion

log = logging.getLogger("app.main")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    """
    Arranque y apagado de los recursos compartidos.

    El orden importa en los dos sentidos. Al arrancar: primero la
    observabilidad (tiene que estar activa **antes** de que se cree el primer
    runnable de LangChain, o las trazas se pierden), después Redis, después el
    grafo, después los workers. Al apagar, al revés: primero los workers —para
    que nadie siga usando Redis— y después las conexiones.

    El checkpointer se abre una sola vez para toda la vida de la app. Abrirlo
    por job crearía un pool de conexiones por job.
    """
    setup_logging()

    plataforma = observability.inicializar()
    log.info("Observabilidad: %s", plataforma)

    # Inyecta en los agentes del módulo 6 un modelo con limitador de tasa y
    # reintentos. Va acá, antes de que se construya el grafo, porque los
    # subgrafos ReAct cachean el modelo con el que se arman.
    llm.instalar()

    url = require_redis_url()
    redis_client = aioredis.from_url(
        url,
        decode_responses=True,
        # Un pool acotado: con `CANTIDAD_WORKERS` corrutinas más los endpoints,
        # 20 conexiones sobran. El free tier de Redis Cloud limita a 30.
        max_connections=20,
        health_check_interval=30,
    )

    if not await redis_client.ping():
        raise RuntimeError(f"Redis no responde en {url.split('@')[-1]}")
    log.info("Redis conectado (%s).", url.split("@")[-1])

    # Las acciones con efectos necesitan el cliente para persistir su rastro.
    acciones.configurar_redis(redis_client)

    repositorio = RepositorioJobs(redis_client)

    async with abrir_orquestador(redis_client=redis_client) as orquestador:
        pool = PoolDeWorkers(repositorio, orquestador)
        await pool.arrancar()

        app.state.redis = redis_client
        app.state.repositorio = repositorio
        app.state.orquestador = orquestador
        app.state.pool = pool
        app.state.observabilidad = plataforma

        try:
            yield
        finally:
            await pool.apagar()
            await redis_client.aclose()
            log.info("Recursos liberados.")


app = FastAPI(
    title="API de auditoría multi-agente",
    description=(
        "Expone el orquestador multi-agente de la Pre-entrega 6 como una API "
        "asíncrona, con estado en Redis, trazas en LangSmith y una pausa de "
        "aprobación humana antes de ejecutar acciones con efectos secundarios."
    ),
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


# --------------------------------------------------------------------------
# Dependencias
# --------------------------------------------------------------------------

def get_repositorio(request: Request) -> RepositorioJobs:
    return request.app.state.repositorio


def get_pool(request: Request) -> PoolDeWorkers:
    return request.app.state.pool


# --------------------------------------------------------------------------
# Manejo de errores
# --------------------------------------------------------------------------

@app.exception_handler(JobNoEncontrado)
async def _job_no_encontrado(request: Request, exc: JobNoEncontrado) -> JSONResponse:
    """
    404 con un mensaje que distingue las dos causas posibles.

    Un job puede no existir porque el id es inventado o porque venció su TTL.
    Para el cliente son situaciones distintas, y decirlo evita un ticket de
    soporte.
    """
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "detalle": f"No existe el job '{exc.args[0]}'.",
            "causas_posibles": [
                "El identificador es incorrecto.",
                "El job expiró: los trabajos se conservan 7 días.",
            ],
        },
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.post(
    "/tasks",
    response_model=TareaEncolada,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Encola una auditoría y devuelve su job_id",
)
async def crear_tarea(
    cuerpo: SolicitudTarea,
    repositorio: RepositorioJobs = Depends(get_repositorio),
    pool: PoolDeWorkers = Depends(get_pool),
) -> TareaEncolada:
    """
    Acepta el trabajo y devuelve de inmediato.

    Devuelve 202 y no 201 a propósito: el recurso que se crea es el *job*, no
    el informe. 202 Accepted es exactamente "lo recibí, todavía no lo terminé",
    que es lo que pasa acá.

    Si la cola está llena responde 503 con `Retry-After`. Es preferible a
    aceptar trabajo que no se va a poder atender: un 503 le dice al cliente qué
    hacer, una cola infinita lo deja esperando sin saberlo.
    """
    job = await repositorio.crear(cuerpo.solicitud)

    try:
        pool.encolar(job.job_id)
    except ColaLlena as e:
        # El job queda en Redis en PENDIENTE para no perder el rastro, pero se
        # le avisa al cliente que no fue aceptado.
        await repositorio.marcar_fallido(job.job_id, e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "10"},
        ) from e

    return TareaEncolada(
        job_id=job.job_id,
        estado=job.estado,
        url_estado=f"/tasks/{job.job_id}",
        encolado_en=job.creado_en,
    )


@app.get(
    "/tasks/{job_id}",
    response_model=RespuestaEstado,
    summary="Consulta el estado de un job",
)
async def estado_tarea(
    job_id: str,
    repositorio: RepositorioJobs = Depends(get_repositorio),
) -> RespuestaEstado:
    """
    El estado actual. Es una lectura de Redis: no espera a nada.

    El campo `requiere_aprobacion` es el que le dice al cliente que deje de
    hacer polling y llame a `/approve`. Sin él, un job pausado se ve igual que
    uno que está tardando.
    """
    return RespuestaEstado.desde_job(await repositorio.obtener(job_id))


@app.post(
    "/tasks/{job_id}/approve",
    response_model=RespuestaEstado,
    summary="Aprueba o rechaza la acción crítica de un job pausado",
)
async def aprobar_tarea(
    job_id: str,
    decision: DecisionAprobacion,
    repositorio: RepositorioJobs = Depends(get_repositorio),
    pool: PoolDeWorkers = Depends(get_pool),
) -> RespuestaEstado:
    """
    Resuelve la pausa de aprobación humana y vuelve a encolar el job.

    Rechaza con 409 si el job no está esperando aprobación. Ese chequeo no es
    burocracia: aprobar dos veces el mismo job ejecutaría la acción crítica
    dos veces, y la acción crítica de este sistema despierta a una persona.

    Igual que `POST /tasks`, este endpoint **no** espera a que el grafo
    termine: registra la decisión, encola la reanudación y devuelve. El agente
    todavía tiene que redactar el informe, y eso lleva otra llamada al modelo.
    """
    job = await repositorio.obtener(job_id)

    if job.estado != EstadoJob.ESPERANDO_APROBACION:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"El job está en {job.estado}, no en "
                f"{EstadoJob.ESPERANDO_APROBACION}. "
                f"{'Ya fue resuelto.' if job.estado.es_terminal else 'Todavía está trabajando.'}"
            ),
        )

    job = await repositorio.registrar_decision(job_id, decision)

    try:
        pool.encolar(job_id, comando_de_reanudacion(decision))
    except ColaLlena as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{e} La decisión quedó registrada: reintentá la aprobación.",
            headers={"Retry-After": "10"},
        ) from e

    return RespuestaEstado.desde_job(
        await repositorio.actualizar(job_id, estado=EstadoJob.EJECUTANDO)
    )


@app.get(
    "/tasks",
    response_model=list[RespuestaEstado],
    summary="Los jobs más recientes",
)
async def listar_tareas(
    limite: int = 20,
    repositorio: RepositorioJobs = Depends(get_repositorio),
) -> list[RespuestaEstado]:
    """Útil para la prueba de carga: ver las 5 peticiones de un vistazo."""
    limite = max(1, min(limite, 100))
    return [RespuestaEstado.desde_job(j) for j in await repositorio.listar(limite)]


@app.get(
    "/escalaciones",
    summary="Las escalaciones abiertas por acciones aprobadas",
)
async def listar_escalaciones(limite: int = 50) -> list[dict[str, Any]]:
    """
    El efecto de las acciones críticas aprobadas.

    Existe para que el flujo HITL sea verificable de punta a punta: después de
    aprobar, la escalación está acá. Sin este endpoint, "la acción se ejecutó"
    sería una afirmación del sistema sobre sí mismo.
    """
    return await acciones.listar_escalaciones(max(1, min(limite, 200)))


@app.get("/health", response_model=Salud, summary="Estado del servicio")
async def salud(request: Request) -> Salud:
    """
    Chequeo de salud con las dependencias reales.

    Hace `PING` a Redis en vez de devolver `{"status": "ok"}` fijo: un health
    check que no toca sus dependencias solo confirma que el proceso está vivo,
    que es justo lo que nunca falla.

    Devuelve 200 incluso degradado: quien orquesta decide qué hacer con la
    información, y un 503 acá haría que un balanceador saque de rotación a una
    instancia que todavía puede responder consultas de estado.
    """
    repositorio: RepositorioJobs = request.app.state.repositorio
    pool: PoolDeWorkers = request.app.state.pool

    redis_ok = await repositorio.ping()
    workers = pool.workers_vivos

    problemas: list[str] = []
    if not redis_ok:
        problemas.append("Redis no responde.")
    if workers < CANTIDAD_WORKERS:
        problemas.append(f"Solo {workers} de {CANTIDAD_WORKERS} workers vivos.")
    if pool.pendientes >= TAMANIO_COLA * 0.9:
        problemas.append("La cola está casi llena.")

    return Salud(
        estado="ok" if not problemas else "degradado",
        redis=redis_ok,
        observabilidad=request.app.state.observabilidad,
        jobs_en_cola=pool.pendientes,
        workers=workers,
        detalle=" ".join(problemas),
    )


@app.get("/", include_in_schema=False)
async def raiz() -> dict[str, Any]:
    """Un índice mínimo, para que entrar a la raíz no devuelva 404."""
    return {
        "servicio": "API de auditoría multi-agente",
        "documentacion": "/docs",
        "endpoints": {
            "crear": "POST /tasks",
            "estado": "GET /tasks/{job_id}",
            "aprobar": "POST /tasks/{job_id}/approve",
            "listar": "GET /tasks",
            "escalaciones": "GET /escalaciones",
            "salud": "GET /health",
        },
    }
