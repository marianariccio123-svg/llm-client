"""
tests/test_api.py

Tests de la API. Corren sin Redis real, sin red y sin ninguna API key.

Cómo se sustituye cada dependencia
-----------------------------------
- **Redis** -> `fakeredis.aioredis.FakeRedis`, que implementa el protocolo en
  memoria. Cubre todo lo que usa `RepositorioJobs`: GET, SET con TTL, LPUSH,
  LRANGE, LTRIM, PING.
- **El LLM** -> el `LLMGuionado` de la Pre-entrega 6, una cola de respuestas
  prefabricadas.
- **El checkpointer** -> el de memoria de LangGraph. `fakeredis` no emula
  RediSearch, que es lo que necesita `AsyncRedisSaver`, así que esa pieza no se
  puede testear acá. Es una limitación real y está declarada: la verificación
  de `RedisSaver` se hizo contra una base de Redis Cloud de verdad, y está
  documentada en el README.

Lo que se testea acá es la máquina de estados de los jobs, el contrato HTTP,
el comportamiento de la cola y —sobre todo— que **ningún job quede colgado**.

Correr:
    cd pre-entrega-7
    python -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
# El stub del LLM vive en la Pre-entrega 6 y se reutiliza en vez de copiarse.
sys.path.insert(0, str(RAIZ.parent / "pre-entrega-6"))

import fakeredis.aioredis  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import agents.analyst_agent as analyst  # noqa: E402
import agents.research_agent as research  # noqa: E402
import agents.supervisor as supervisor  # noqa: E402
from app import acciones  # noqa: E402
from app.graph import compilar, entrada  # noqa: E402
from app.hitl import evaluar_criticidad  # noqa: E402
from app.jobs import JobNoEncontrado, RepositorioJobs  # noqa: E402
from app.schemas import DecisionAprobacion, EstadoJob, PropuestaAccion  # noqa: E402
from app.worker import ColaLlena, PoolDeWorkers, comando_de_reanudacion  # noqa: E402
from state import DecisionSupervisor, Hallazgo, Rubrica  # noqa: E402
from tests.stub_llm import LLMGuionado, mensaje_con_herramientas  # noqa: E402

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
async def redis_falso():
    """Un Redis en memoria, limpio por test."""
    cliente = fakeredis.aioredis.FakeRedis(decode_responses=True)
    acciones.configurar_redis(cliente)
    yield cliente
    await cliente.aclose()


@pytest.fixture
async def repositorio(redis_falso):
    return RepositorioJobs(redis_falso)


@pytest.fixture
def stub(monkeypatch):
    """Instala un LLM guionado en los tres módulos de agentes del módulo 6."""

    def _limpiar():
        research._agente_react.cache_clear()
        analyst._agente_react.cache_clear()
        supervisor._supervisor_estructurado.cache_clear()
        research._bm25_local.cache_clear()

    _limpiar()

    def _instalar(respuestas=None, estructuradas=None) -> LLMGuionado:
        fake = LLMGuionado(
            respuestas=list(respuestas or []),
            estructuradas=list(estructuradas or []),
        )
        for modulo in (research, analyst, supervisor):
            monkeypatch.setattr(modulo, "get_llm", lambda: fake)
        # Sin red: la búsqueda documental va por BM25 local.
        monkeypatch.setattr(research, "_indice_pinecone", lambda: None)
        _limpiar()
        return fake

    yield _instalar
    _limpiar()


def _decision(siguiente: str, **rubrica) -> DecisionSupervisor:
    base = dict(
        hay_evidencia_documental=False, hay_datos_cuantitativos=False,
        hay_analisis_estadistico=False, hay_validacion_de_datos=False,
    )
    base.update(rubrica)
    return DecisionSupervisor(
        razonamiento=f"voy a {siguiente}", rubrica=Rubrica(**base),
        siguiente=siguiente, instruccion=f"tarea para {siguiente}",
    )


GUION_COMPLETO = {
    "respuestas": [
        mensaje_con_herramientas(
            ("buscar_en_documentacion", {"consulta": "event loop"}),
            ("listar_incidentes", {"meses": 12}),
        ),
        AIMessage(content="Traje documentación e incidentes."),
        mensaje_con_herramientas(
            ("resumir_incidentes", {}), ("detectar_atipicos", {}),
            ("validar_registros", {}),
        ),
        AIMessage(content="El bloqueo del event loop domina."),
        AIMessage(content="INFORME FINAL de la auditoría."),
    ],
    "estructuradas": [
        _decision("investigador"),
        _decision("analista", hay_evidencia_documental=True, hay_datos_cuantitativos=True),
        _decision("validador", hay_evidencia_documental=True, hay_datos_cuantitativos=True,
                  hay_analisis_estadistico=True, hay_validacion_de_datos=True),
    ],
}


# --------------------------------------------------------------------------
# El repositorio de jobs
# --------------------------------------------------------------------------

async def test_un_job_nace_pendiente(repositorio):
    job = await repositorio.crear("auditá los incidentes de producción")

    assert job.estado == EstadoJob.PENDIENTE
    assert len(job.job_id) == 16
    assert (await repositorio.obtener(job.job_id)).solicitud == job.solicitud


async def test_pedir_un_job_inexistente_no_devuelve_none(repositorio):
    """
    Lanza `JobNoEncontrado` en vez de devolver `None`.

    Un `None` silencioso se propaga hasta un `AttributeError` tres capas más
    arriba; la excepción tiene su manejador y se convierte en un 404 con
    explicación.
    """
    with pytest.raises(JobNoEncontrado):
        await repositorio.obtener("no-existe")


async def test_la_transicion_a_fallido_guarda_el_tipo_de_error(repositorio):
    """
    Es el test más importante del archivo.

    Si una excepción en segundo plano no se traduce a FALLIDO, el cliente hace
    polling para siempre. Se verifica que el estado sea terminal, que el
    mensaje quede y que el tipo se registre aparte, para poder distinguir un
    fallo de infraestructura de uno de dominio sin parsear texto.
    """
    job = await repositorio.crear("una solicitud cualquiera")
    await repositorio.marcar_ejecutando(job.job_id)

    await repositorio.marcar_fallido(job.job_id, ValueError("el modelo no respondió"))

    recuperado = await repositorio.obtener(job.job_id)
    assert recuperado.estado == EstadoJob.FALLIDO
    assert recuperado.estado.es_terminal
    assert "el modelo no respondió" in recuperado.error
    assert recuperado.error_tipo == "ValueError"
    assert recuperado.duracion_segundos is not None


async def test_el_mensaje_de_error_se_recorta(repositorio):
    """Un error gigante no debe inflar el registro ni filtrar payloads enteros."""
    job = await repositorio.crear("una solicitud cualquiera")
    await repositorio.marcar_fallido(job.job_id, RuntimeError("x" * 5000))

    assert len((await repositorio.obtener(job.job_id)).error) <= 500


async def test_el_listado_devuelve_los_mas_nuevos_primero(repositorio):
    for i in range(3):
        await repositorio.crear(f"solicitud número {i}")

    jobs = await repositorio.listar()
    assert [j.solicitud for j in jobs] == [
        "solicitud número 2", "solicitud número 1", "solicitud número 0",
    ]


# --------------------------------------------------------------------------
# La criticidad del HITL
# --------------------------------------------------------------------------

def _hallazgo_atipicos(*incidentes) -> Hallazgo:
    return Hallazgo(
        agente="analista", herramienta="detectar_atipicos", resumen="...",
        datos={"cantidad_atipicos": len(incidentes), "atipicos": list(incidentes)},
    )


async def test_un_atipico_critico_dispara_la_aprobacion():
    estado = {
        "hallazgos": [
            _hallazgo_atipicos({
                "incidente_id": "INC-0214", "servicio": "api-pagos",
                "severidad": "critica", "minutos_caidos": 480,
                "categoria": "cancelacion-mal-manejada", "veces_la_mediana": 12.8,
            })
        ]
    }

    propuesta = evaluar_criticidad(estado)

    assert propuesta is not None
    assert propuesta.accion == "escalar_a_guardia"
    assert propuesta.criticidad == "critica"
    assert propuesta.parametros["incidente_id"] == "INC-0214"
    assert propuesta.efectos, "la propuesta tiene que declarar sus efectos"


async def test_sin_atipicos_no_se_pide_aprobacion():
    """El camino sin HITL: una auditoría tranquila no molesta a nadie."""
    assert evaluar_criticidad({"hallazgos": []}) is None


async def test_un_atipico_leve_no_dispara_la_aprobacion():
    """
    Un atípico de severidad baja y duración corta no amerita despertar a nadie.

    Es el test que evita que la compuerta se vuelva un trámite: si pidiera
    aprobación para todo, la gente aprobaría sin leer y el control dejaría de
    servir.
    """
    estado = {
        "hallazgos": [
            _hallazgo_atipicos({
                "incidente_id": "INC-0001", "servicio": "worker-emails",
                "severidad": "baja", "minutos_caidos": 25,
                "categoria": "tarea-huerfana", "veces_la_mediana": 2.1,
            })
        ]
    }
    assert evaluar_criticidad(estado) is None


async def test_un_atipico_largo_dispara_aunque_no_sea_critico():
    """Dos horas de caída alcanzan, aunque la severidad diga 'alta'."""
    estado = {
        "hallazgos": [
            _hallazgo_atipicos({
                "incidente_id": "INC-0099", "servicio": "api-stock",
                "severidad": "alta", "minutos_caidos": 150,
                "categoria": "timeout-no-configurado", "veces_la_mediana": 4.0,
            })
        ]
    }
    propuesta = evaluar_criticidad(estado)
    assert propuesta is not None
    assert propuesta.criticidad == "alta"


# --------------------------------------------------------------------------
# La acción con efectos
# --------------------------------------------------------------------------

async def test_la_escalacion_se_persiste_de_verdad(redis_falso):
    """
    Después de escalar, la escalación existe y se puede listar.

    El efecto secundario tiene que ser verificable desde afuera: un sistema que
    dice "escalé" sin dejar rastro no se puede auditar.
    """
    resultado = await acciones.escalar_a_guardia(
        incidente_id="INC-0214", servicio="api-pagos",
        motivo="480 minutos de caída", aprobado_por="mariana",
    )

    assert resultado["escalacion_id"].startswith("ESC-")
    assert resultado["aprobado_por"] == "mariana"

    guardadas = await acciones.listar_escalaciones()
    assert len(guardadas) == 1
    assert guardadas[0]["incidente_id"] == "INC-0214"


async def test_escalar_sin_redis_falla_en_vez_de_mentir(monkeypatch):
    """
    Si no se puede dejar rastro, la acción falla.

    Devolver un éxito que no ocurrió sería peor: el informe final diría que se
    escaló y nadie habría sido notificado.
    """
    monkeypatch.setattr(acciones, "_redis", None)

    with pytest.raises(RuntimeError, match="no se puede persistir"):
        await acciones.escalar_a_guardia("INC-1", "api", "motivo", "alguien")


# --------------------------------------------------------------------------
# El pool de workers
# --------------------------------------------------------------------------

async def test_la_cola_llena_se_rechaza_en_vez_de_bloquear(repositorio):
    """
    Al llenarse la cola, `encolar` lanza en lugar de esperar.

    Un `await put()` dejaría la petición HTTP colgada hasta que se libere
    lugar. Es preferible un 503 inmediato, que le dice al cliente qué hacer.
    """
    pool = PoolDeWorkers(repositorio, orquestador=None, cantidad=0, tamanio_cola=2)

    pool.encolar("job-1")
    pool.encolar("job-2")

    with pytest.raises(ColaLlena):
        pool.encolar("job-3")

    assert pool.pendientes == 2


async def test_una_excepcion_del_grafo_deja_el_job_en_fallido(repositorio):
    """
    La red de contención del worker, probada con un grafo que explota.

    Sin ella, el job quedaría en EJECUTANDO para siempre y el cliente haría
    polling eternamente.
    """

    class GrafoQueExplota:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("el proveedor devolvió 500")

    job = await repositorio.crear("una solicitud que va a fallar")
    pool = PoolDeWorkers(repositorio, GrafoQueExplota(), cantidad=1)
    await pool.arrancar()
    try:
        pool.encolar(job.job_id)
        await asyncio.wait_for(pool.cola.join(), timeout=10)
    finally:
        await pool.apagar()

    final = await repositorio.obtener(job.job_id)
    assert final.estado == EstadoJob.FALLIDO
    assert final.error_tipo == "RuntimeError"
    assert "500" in final.error


async def test_un_grafo_colgado_termina_en_fallido(repositorio, monkeypatch):
    """
    El timeout por job también produce un estado terminal.

    Un grafo que nunca responde ocuparía un worker para siempre; con el
    timeout, el job falla y el worker vuelve a la cola.
    """
    import app.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TIMEOUT_JOB_SEGUNDOS", 1)

    class GrafoQueSeCuelga:
        async def ainvoke(self, *a, **k):
            await asyncio.sleep(3600)

    job = await repositorio.crear("una solicitud que se va a colgar")
    pool = PoolDeWorkers(repositorio, GrafoQueSeCuelga(), cantidad=1)
    await pool.arrancar()
    try:
        pool.encolar(job.job_id)
        await asyncio.wait_for(pool.cola.join(), timeout=15)
    finally:
        await pool.apagar()

    final = await repositorio.obtener(job.job_id)
    assert final.estado == EstadoJob.FALLIDO
    assert "TimeoutError" in final.error_tipo


async def test_ningun_job_queda_en_ejecutando(repositorio):
    """
    La invariante del sistema, sobre varios jobs que fallan de formas distintas.

    Cualquier job que salga del worker tiene que estar en un estado que el
    cliente pueda interpretar. EJECUTANDO no es uno de ellos.
    """

    class GrafoCaprichoso:
        def __init__(self):
            self.n = 0

        async def ainvoke(self, *a, **k):
            self.n += 1
            if self.n % 3 == 0:
                raise MemoryError("sin memoria")
            if self.n % 3 == 1:
                raise ValueError("entrada inválida")
            raise KeyError("falta una clave")

    jobs = [await repositorio.crear(f"solicitud número {i}") for i in range(6)]
    pool = PoolDeWorkers(repositorio, GrafoCaprichoso(), cantidad=2)
    await pool.arrancar()
    try:
        for job in jobs:
            pool.encolar(job.job_id)
        await asyncio.wait_for(pool.cola.join(), timeout=20)
    finally:
        await pool.apagar()

    estados = [(await repositorio.obtener(j.job_id)).estado for j in jobs]
    assert all(e == EstadoJob.FALLIDO for e in estados), estados


# --------------------------------------------------------------------------
# El grafo con HITL, de punta a punta
# --------------------------------------------------------------------------

async def test_el_grafo_se_pausa_y_retoma_tras_la_aprobacion(stub, redis_falso):
    """
    El flujo HITL completo, con el checkpointer en memoria.

    Primera invocación: el grafo corre hasta la compuerta y se interrumpe.
    Segunda: se retoma con `Command(resume=...)`, ejecuta la acción y redacta.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    stub(**{
        "respuestas": GUION_COMPLETO["respuestas"][:4] + [
            AIMessage(content="INFORME con la escalación ya hecha."),
        ],
        "estructuradas": GUION_COMPLETO["estructuradas"],
    })

    grafo = compilar(InMemorySaver())
    config = {"configurable": {"thread_id": "test-hitl"}, "recursion_limit": 30}

    primera = await grafo.ainvoke(entrada("auditá producción"), config=config)

    assert "__interrupt__" in primera, "el grafo tenía que pausarse"
    payload = primera["__interrupt__"][0].value
    assert payload["tipo"] == "aprobacion_requerida"
    propuesta = PropuestaAccion.model_validate(payload["propuesta"])
    assert propuesta.accion == "escalar_a_guardia"

    # Nada se ejecutó todavía: ese es el punto de la pausa.
    assert await acciones.listar_escalaciones() == []

    decision = DecisionAprobacion(aprobado=True, aprobado_por="mariana")
    segunda = await grafo.ainvoke(comando_de_reanudacion(decision), config=config)

    assert segunda["accion_ejecutada"]["escalacion_id"].startswith("ESC-")
    assert segunda["informe_final"].startswith("INFORME")
    assert len(await acciones.listar_escalaciones()) == 1


async def test_rechazar_no_ejecuta_la_accion(stub, redis_falso):
    """
    El otro lado del HITL: rechazar significa que no pasa nada.

    Es la mitad que más importa verificar. Que una aprobación funcione lo nota
    cualquiera; que un rechazo **no** ejecute la acción solo se nota cuando ya
    despertaste a alguien de madrugada por error.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    stub(**{
        "respuestas": GUION_COMPLETO["respuestas"][:4] + [
            AIMessage(content="INFORME sin escalación, fue rechazada."),
        ],
        "estructuradas": GUION_COMPLETO["estructuradas"],
    })

    grafo = compilar(InMemorySaver())
    config = {"configurable": {"thread_id": "test-rechazo"}, "recursion_limit": 30}

    await grafo.ainvoke(entrada("auditá producción"), config=config)

    decision = DecisionAprobacion(
        aprobado=False, aprobado_por="mariana", comentario="Ya lo estamos viendo.",
    )
    final = await grafo.ainvoke(comando_de_reanudacion(decision), config=config)

    assert final.get("accion_ejecutada") is None
    assert final["aprobacion_otorgada"] is False
    assert await acciones.listar_escalaciones() == [], (
        "se ejecutó la acción a pesar del rechazo"
    )
    assert final["informe_final"]


async def test_el_worker_deja_el_job_esperando_y_lo_suelta(stub, repositorio):
    """
    El worker no se queda bloqueado esperando a la persona que aprueba.

    Si lo hiciera, una aprobación que tarda una hora ocuparía un worker una
    hora. El job queda en ESPERANDO_APROBACION y el worker vuelve a la cola.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    stub(**GUION_COMPLETO)

    grafo = compilar(InMemorySaver())
    job = await repositorio.crear("auditá los incidentes de producción")

    pool = PoolDeWorkers(repositorio, grafo, cantidad=1)
    await pool.arrancar()
    try:
        pool.encolar(job.job_id)
        await asyncio.wait_for(pool.cola.join(), timeout=30)
    finally:
        await pool.apagar()

    final = await repositorio.obtener(job.job_id)
    assert final.estado == EstadoJob.ESPERANDO_APROBACION
    assert final.propuesta is not None
    assert final.propuesta.accion == "escalar_a_guardia"
    assert not final.estado.es_terminal, "la pausa no es un estado terminal"
