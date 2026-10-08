"""
tests/test_orquestador.py

Tests del orquestador, todos sin red y sin API keys.

Qué se testea y qué no
----------------------
Se testea lo que escribimos nosotros: la topología del grafo, el ruteo, los
reducers del estado, el paso de contexto entre agentes y la aritmética de las
herramientas. **No** se testea si el modelo elige bien, porque eso no es código
nuestro y no es determinista; de eso se encargan las trazas de ejecución.

La división importa: un test que falla acá significa que rompimos algo. Un test
que dependa del criterio del LLM podría fallar sin que nadie haya tocado nada,
y a los dos días nadie le cree.

Cómo se inyecta el stub
-----------------------
Cada módulo hace `from config import get_llm`, así que tiene su propia
referencia a la función: parchear `config.get_llm` no alcanza. El fixture
parchea las tres referencias y además limpia los `@lru_cache` que envuelven a
los subgrafos, porque si no el segundo test reusaría el agente construido con
el stub del primero.

Correr:
    cd pre-entrega-6
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# El paquete vive en la carpeta de arriba y los módulos se importan planos
# (`import graph`, `from state import ...`), igual que cuando corre el CLI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import agents.analyst_agent as analyst  # noqa: E402
import agents.research_agent as research  # noqa: E402
import agents.supervisor as supervisor  # noqa: E402
import graph as grafo_mod  # noqa: E402
from agents.analyst_agent import (  # noqa: E402
    detectar_atipicos,
    resumir_incidentes,
    validar_registros,
)
from agents.supervisor import (  # noqa: E402
    enrutar_desde_supervisor,
    enrutar_desde_validador,
    nodo_validador,
    rubrica_verificada,
)
from config import MAX_PASOS_SUPERVISOR  # noqa: E402
from state import DecisionSupervisor, Hallazgo, Rubrica  # noqa: E402
from tests.stub_llm import LLMGuionado, mensaje_con_herramientas  # noqa: E402

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Datos de prueba
# --------------------------------------------------------------------------

INCIDENTES = [
    {"incidente_id": "INC-0001", "fecha": "2026-01-10", "servicio": "api",
     "categoria": "bloqueo-event-loop", "severidad": "alta",
     "minutos_caidos": 30, "descripcion": "x"},
    {"incidente_id": "INC-0002", "fecha": "2026-02-10", "servicio": "api",
     "categoria": "bloqueo-event-loop", "severidad": "media",
     "minutos_caidos": 40, "descripcion": "x"},
    {"incidente_id": "INC-0003", "fecha": "2026-03-10", "servicio": "worker",
     "categoria": "timeout-no-configurado", "severidad": "baja",
     "minutos_caidos": 20, "descripcion": "x"},
    {"incidente_id": "INC-0004", "fecha": "2026-04-10", "servicio": "api",
     "categoria": "bloqueo-event-loop", "severidad": "critica",
     "minutos_caidos": 50, "descripcion": "x"},
    # El atípico y el registro inválido, como en la base real.
    {"incidente_id": "INC-0005", "fecha": "2026-05-10", "servicio": "pagos",
     "categoria": "cancelacion-mal-manejada", "severidad": "critica",
     "minutos_caidos": 480, "descripcion": "x"},
    {"incidente_id": "INC-0006", "fecha": "2026-06-10", "servicio": "api",
     "categoria": "fuga-de-tareas", "severidad": "media",
     "minutos_caidos": -5, "descripcion": "x"},
]


def estado_con_incidentes() -> dict:
    """Un estado como el que deja el investigador después de trabajar."""
    return {
        "hallazgos": [
            Hallazgo(
                agente="investigador",
                herramienta="listar_incidentes",
                resumen="6 incidentes",
                datos={"cantidad": len(INCIDENTES), "incidentes": INCIDENTES},
            )
        ]
    }


def hallazgo(herramienta: str, agente: str = "investigador", **datos) -> Hallazgo:
    return Hallazgo(
        agente=agente, herramienta=herramienta, resumen="...", datos=datos or {}
    )


# --------------------------------------------------------------------------
# Fixture: inyecta el stub en los tres módulos
# --------------------------------------------------------------------------

@pytest.fixture
def stub(monkeypatch):
    """
    Devuelve una función que instala un `LLMGuionado` en todos los nodos.

    Limpia los `@lru_cache` antes y después: los subgrafos ReAct y el
    supervisor estructurado se construyen una sola vez por proceso, así que sin
    esto el guion de un test se filtraría al siguiente.
    """

    def _limpiar_caches():
        research._agente_react.cache_clear()
        analyst._agente_react.cache_clear()
        supervisor._supervisor_estructurado.cache_clear()
        research._bm25_local.cache_clear()

    _limpiar_caches()

    def _instalar(respuestas=None, estructuradas=None) -> LLMGuionado:
        fake = LLMGuionado(
            respuestas=list(respuestas or []),
            estructuradas=list(estructuradas or []),
        )
        for modulo in (research, analyst, supervisor):
            monkeypatch.setattr(modulo, "get_llm", lambda: fake)

        # Las herramientas se ejecutan de verdad (es lo que queremos: el stub
        # reemplaza al modelo, no al código). Pero `buscar_en_documentacion`
        # consultaría Pinecone y pagaría embeddings, así que se la fuerza a su
        # ruta de respaldo: BM25 sobre los archivos locales del corpus. Sigue
        # siendo la herramienta real, sin una sola llamada de red.
        monkeypatch.setattr(research, "_indice_pinecone", lambda: None)

        _limpiar_caches()
        return fake

    yield _instalar
    _limpiar_caches()


# --------------------------------------------------------------------------
# Topología del grafo
# --------------------------------------------------------------------------

async def test_el_grafo_compila_con_los_cinco_nodos():
    """El grafo se compila sin credenciales ni red."""
    compilado = grafo_mod.compilar()
    nodos = set(compilado.get_graph().nodes)

    for esperado in ("supervisor", "investigador", "analista", "validador",
                     "sintetizador"):
        assert esperado in nodos, f"falta el nodo '{esperado}'"


async def test_el_diagrama_mermaid_refleja_los_ciclos():
    """Los dos ciclos (trabajo y refinamiento) aparecen en el diagrama."""
    diagrama = grafo_mod.diagrama_mermaid()

    assert "investigador --> supervisor" in diagrama
    assert "analista --> supervisor" in diagrama
    assert "validador -.-> supervisor" in diagrama      # ciclo de refinamiento
    assert "validador -.-> sintetizador" in diagrama
    assert "sintetizador --> __end__" in diagrama


# --------------------------------------------------------------------------
# Ruteo (funciones puras, sin LLM)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "destino", ["investigador", "analista", "validador"]
)
async def test_el_ruteo_del_supervisor_sigue_al_estado(destino):
    assert enrutar_desde_supervisor({"siguiente": destino}) == destino


async def test_el_ruteo_sin_decision_arranca_por_el_investigador():
    """Sin datos no se puede analizar, así que el default es ir a buscar."""
    assert enrutar_desde_supervisor({}) == "investigador"


@pytest.mark.parametrize(
    "aprobada,esperado",
    [(True, "sintetizador"), (False, "supervisor")],
)
async def test_el_ruteo_del_validador(aprobada, esperado):
    assert enrutar_desde_validador({"validacion_aprobada": aprobada}) == esperado


# --------------------------------------------------------------------------
# La rúbrica se verifica contra el estado, no contra el modelo
# --------------------------------------------------------------------------

async def test_la_rubrica_arranca_toda_en_falso():
    verificada = rubrica_verificada({"hallazgos": []})

    assert not verificada.completa
    assert len(verificada.faltantes()) == 4


async def test_la_rubrica_se_completa_con_las_cuatro_herramientas():
    estado = {
        "hallazgos": [
            hallazgo("buscar_en_documentacion"),
            hallazgo("listar_incidentes"),
            hallazgo("resumir_incidentes", agente="analista"),
            hallazgo("validar_registros", agente="analista"),
        ]
    }

    assert rubrica_verificada(estado).completa


async def test_una_herramienta_que_fallo_no_cuenta_como_evidencia():
    """
    Un hallazgo con error no cumple el criterio.

    Sin esta regla, un agente que llama a la herramienta correcta y recibe un
    error marcaría el criterio como satisfecho, y el validador aprobaría un
    informe sin datos.
    """
    estado = {"hallazgos": [hallazgo("listar_incidentes", error="la base no existe")]}

    assert not rubrica_verificada(estado).hay_datos_cuantitativos


# --------------------------------------------------------------------------
# Validador
# --------------------------------------------------------------------------

async def test_el_validador_rechaza_y_dice_que_falta():
    resultado = nodo_validador({"hallazgos": [hallazgo("listar_incidentes")],
                                "pasos_del_supervisor": 1})

    assert resultado["validacion_aprobada"] is False
    assert "evidencia documental" in resultado["motivo_validacion"]


async def test_el_validador_aprueba_con_la_rubrica_completa():
    estado = {
        "hallazgos": [
            hallazgo("buscar_en_documentacion"),
            hallazgo("listar_incidentes"),
            hallazgo("detectar_atipicos", agente="analista"),
            hallazgo("validar_registros", agente="analista"),
        ],
        "pasos_del_supervisor": 2,
    }

    assert nodo_validador(estado)["validacion_aprobada"] is True


async def test_el_tope_de_pasos_aprueba_un_informe_parcial():
    """
    Al agotar el presupuesto de pasos, el validador aprueba igual y lo dice.

    Es la decisión de producto detrás del freno: con la rúbrica incompleta y
    sin pasos disponibles, responder parcialmente y declararlo es mejor que no
    responder.
    """
    resultado = nodo_validador(
        {"hallazgos": [], "pasos_del_supervisor": MAX_PASOS_SUPERVISOR}
    )

    assert resultado["validacion_aprobada"] is True
    assert "tope de pasos" in resultado["motivo_validacion"].lower()
    assert "parcial" in resultado["motivo_validacion"].lower()


# --------------------------------------------------------------------------
# Herramientas del analista: leen del estado, no del prompt
# --------------------------------------------------------------------------

async def test_resumir_incidentes_calcula_sobre_el_estado_compartido():
    resultado = resumir_incidentes.invoke({"estado": estado_con_incidentes()})

    assert resultado["total_incidentes"] == 6
    assert resultado["total_minutos_caidos"] == 615          # 30+40+20+50+480-5
    assert resultado["por_categoria"][0]["categoria"] == "bloqueo-event-loop"
    assert resultado["por_categoria"][0]["cantidad"] == 3


async def test_detectar_atipicos_encuentra_la_caida_larga():
    resultado = detectar_atipicos.invoke({"estado": estado_con_incidentes()})

    assert resultado["cantidad_atipicos"] == 1
    assert resultado["atipicos"][0]["incidente_id"] == "INC-0005"
    assert resultado["atipicos"][0]["minutos_caidos"] == 480


async def test_validar_registros_detecta_los_minutos_negativos():
    resultado = validar_registros.invoke({"estado": estado_con_incidentes()})

    assert resultado["invalidos"] == 1
    assert resultado["problemas"][0]["incidente_id"] == "INC-0006"
    assert resultado["problemas"][0]["campo"] == "minutos_caidos"


@pytest.mark.parametrize(
    "herramienta", [resumir_incidentes, detectar_atipicos, validar_registros]
)
async def test_las_herramientas_avisan_cuando_no_hay_datos(herramienta):
    """Sin incidentes en el estado devuelven un error accionable, no una excepción."""
    resultado = herramienta.invoke({"estado": {"hallazgos": []}})

    assert "error" in resultado
    assert "investigador" in resultado["error"]


# --------------------------------------------------------------------------
# El flujo completo, con LLM guionado
# --------------------------------------------------------------------------

def _decision(siguiente: str, **rubrica) -> DecisionSupervisor:
    base = dict(
        hay_evidencia_documental=False,
        hay_datos_cuantitativos=False,
        hay_analisis_estadistico=False,
        hay_validacion_de_datos=False,
    )
    base.update(rubrica)
    return DecisionSupervisor(
        razonamiento=f"voy a {siguiente}",
        rubrica=Rubrica(**base),
        siguiente=siguiente,
        instruccion=f"instrucción para {siguiente}",
    )


async def test_flujo_completo_de_delegacion(stub):
    """
    El camino feliz entero, sin tocar la API.

    Guion: el supervisor manda al investigador, después al analista, después al
    validador; el validador aprueba y el sintetizador redacta.
    """
    fake = stub(
        respuestas=[
            # --- investigador (ciclo ReAct) ---
            mensaje_con_herramientas(
                ("buscar_en_documentacion", {"consulta": "bloqueo del event loop"}),
                ("listar_incidentes", {"meses": 12}),
            ),
            AIMessage(content="Traje documentación e incidentes."),
            # --- analista (ciclo ReAct) ---
            mensaje_con_herramientas(
                ("resumir_incidentes", {}),
                ("detectar_atipicos", {}),
                ("validar_registros", {}),
            ),
            AIMessage(content="El bloqueo del event loop domina."),
            # --- sintetizador ---
            AIMessage(content="INFORME: el 45% de los incidentes son bloqueos."),
        ],
        estructuradas=[
            _decision("investigador"),
            _decision("analista", hay_evidencia_documental=True,
                      hay_datos_cuantitativos=True),
            _decision("validador", hay_evidencia_documental=True,
                      hay_datos_cuantitativos=True, hay_analisis_estadistico=True,
                      hay_validacion_de_datos=True),
        ],
    )

    final = await grafo_mod.compilar().ainvoke(
        grafo_mod.entrada("¿Qué nos está rompiendo producción?"),
        config={"recursion_limit": 25},
    )

    # El orden de delegación es el esperado.
    agentes = [h.agente for h in final["hallazgos"]]
    assert agentes == ["investigador"] * 2 + ["analista"] * 3

    # Las cinco herramientas corrieron y quedaron trazadas en el estado.
    assert {h.herramienta for h in final["hallazgos"]} == {
        "buscar_en_documentacion", "listar_incidentes",
        "resumir_incidentes", "detectar_atipicos", "validar_registros",
    }

    assert final["validacion_aprobada"] is True
    assert final["informe_final"].startswith("INFORME:")
    assert final["pasos_del_supervisor"] == 3
    assert not fake.respuestas, "sobraron respuestas: el grafo llamó de menos"


async def test_el_analista_recibe_los_datos_sin_que_pasen_por_el_prompt(stub):
    """
    El test de la decisión de diseño central.

    El analista calcula sobre 6 incidentes, pero ninguno de esos registros
    aparece en el prompt que recibió: los leyó del estado por `InjectedState`.
    Si alguien algún día "simplifica" eso pasando los datos por el mensaje,
    este test lo detecta.
    """
    fake = stub(
        respuestas=[
            mensaje_con_herramientas(("resumir_incidentes", {})),
            AIMessage(content="Listo el resumen."),
        ]
    )

    estado = {
        "solicitud": "analizá los incidentes",
        "instruccion_actual": "calculá las métricas",
        **estado_con_incidentes(),
    }
    resultado = await analyst.nodo_analista(estado)

    # El cálculo se hizo sobre los 6 incidentes...
    resumen = next(
        h for h in resultado["hallazgos"] if h.herramienta == "resumir_incidentes"
    )
    assert resumen.datos["total_incidentes"] == 6

    # ...y sin embargo ningún id de incidente viajó en el prompt.
    texto_de_los_prompts = "\n".join(
        str(m.content) for llamada in fake.prompts_recibidos for m in llamada
    )
    for incidente in INCIDENTES:
        assert incidente["incidente_id"] not in texto_de_los_prompts


async def test_el_especialista_no_ve_el_historial_de_la_conversacion(stub):
    """
    Contaminación de contexto: el investigador recibe su instrucción, no el chat.

    Se le pone al estado un mensaje que solo existe en el historial. Si
    apareciera en el prompt del especialista, es que le estamos pasando
    `messages` y el prompt va a crecer con cada vuelta del ciclo.
    """
    fake = stub(respuestas=[AIMessage(content="Listo.")])

    estado = {
        "solicitud": "auditá los incidentes",
        "instruccion_actual": "traé los incidentes del último año",
        "messages": [
            HumanMessage("auditá los incidentes"),
            AIMessage(content="SECRETO_DEL_HISTORIAL", name="supervisor"),
        ],
        "hallazgos": [],
    }
    await research.nodo_investigador(estado)

    texto_de_los_prompts = "\n".join(
        str(m.content) for llamada in fake.prompts_recibidos for m in llamada
    )
    assert "SECRETO_DEL_HISTORIAL" not in texto_de_los_prompts
    assert "traé los incidentes del último año" in texto_de_los_prompts


async def test_el_tope_corta_un_supervisor_que_no_cierra_nunca(stub):
    """
    El freno contra el "supervisor infinito".

    El guion simula el peor caso: un supervisor que siempre deriva al
    investigador y nunca da por terminada la tarea. Sin el tope, el grafo
    giraría hasta agotar el `recursion_limit` y terminaría en excepción.
    """
    # Muchas más respuestas de las que debería consumir: si el tope no
    # funcionara, el test no fallaría por falta de guion sino por recursión.
    stub(
        respuestas=[AIMessage(content="Listo.") for _ in range(40)],
        estructuradas=[_decision("investigador") for _ in range(40)],
    )

    final = await grafo_mod.compilar().ainvoke(
        grafo_mod.entrada("pregunta que nunca se da por respondida"),
        config={"recursion_limit": 60},
    )

    # Se frenó en el tope y cerró igual, con el informe marcado como parcial.
    assert final["pasos_del_supervisor"] == MAX_PASOS_SUPERVISOR + 1
    assert final["validacion_aprobada"] is True
    assert "tope de pasos" in final["motivo_validacion"].lower()


async def test_el_informe_final_se_arma_sobre_los_hallazgos(stub):
    """
    El sintetizador lee `hallazgos`, no `messages`.

    Se le da un estado donde el historial dice una cosa y los hallazgos dicen
    otra. El prompt que recibe tiene que contener el dato estructurado.
    """
    fake = stub(respuestas=[AIMessage(content="informe")])

    estado = {
        "solicitud": "resumime la auditoría",
        "messages": [AIMessage(content="charla irrelevante", name="supervisor")],
        "hallazgos": [
            Hallazgo(
                agente="analista",
                herramienta="resumir_incidentes",
                resumen="20 incidentes",
                datos={"total_incidentes": 20, "total_minutos_caidos": 1185},
            )
        ],
        "rubrica_actual": Rubrica(
            hay_evidencia_documental=True, hay_datos_cuantitativos=True,
            hay_analisis_estadistico=True, hay_validacion_de_datos=True,
        ),
        "validacion_aprobada": True,
    }
    await supervisor.nodo_sintetizador(estado)

    prompt = "\n".join(str(m.content) for m in fake.prompts_recibidos[0])
    assert "1185" in prompt
    assert "charla irrelevante" not in prompt


# --------------------------------------------------------------------------
# Persistencia: el estado tiene que sobrevivir al ida y vuelta por SQLite
# --------------------------------------------------------------------------

async def test_el_estado_con_modelos_pydantic_sobrevive_al_checkpointer(
    stub, tmp_path, caplog
):
    """
    `Hallazgo` y `Rubrica` se releen del checkpoint sin perder su tipo.

    El serializador de LangGraph solo deserializa los tipos que están en su
    allowlist: sin registrarlos, hoy devuelve un warning y en una versión
    futura los va a bloquear, con lo que el orquestador no podría releer sus
    propios checkpoints. Este test corre el grafo contra un SQLite real (con el
    LLM guionado, así que no cuesta nada) y verifica que lo que vuelve siguen
    siendo los objetos y no diccionarios pelados.
    """
    import logging

    from graph import abrir_orquestador, config_thread
    from graph import entrada as entrada_inicial

    stub(
        respuestas=[
            # Las dos herramientas del investigador: si falta la documental, el
            # validador rechaza (con razón) y el guion se queda corto.
            mensaje_con_herramientas(
                ("buscar_en_documentacion", {"consulta": "event loop"}),
                ("listar_incidentes", {"meses": 12}),
            ),
            AIMessage(content="Traje los incidentes."),
            mensaje_con_herramientas(
                ("resumir_incidentes", {}), ("validar_registros", {})
            ),
            AIMessage(content="Analizado."),
            AIMessage(content="INFORME final."),
        ],
        estructuradas=[
            _decision("investigador"),
            _decision("analista", hay_datos_cuantitativos=True,
                      hay_evidencia_documental=True),
            _decision("validador", hay_evidencia_documental=True,
                      hay_datos_cuantitativos=True, hay_analisis_estadistico=True,
                      hay_validacion_de_datos=True),
        ],
    )

    ruta = str(tmp_path / "checkpoints.sqlite")

    # El aviso de tipos sin registrar sale por `logging`, NO por `warnings`:
    # capturar el canal equivocado haría pasar este test siempre, que es
    # justo lo que pasó la primera vez que se escribió.
    with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
        async with abrir_orquestador(ruta) as orquestador:
            config = config_thread("test-persistencia")
            await orquestador.ainvoke(entrada_inicial("auditá producción"), config=config)

            # Se relee desde el checkpointer, que es donde ocurre la deserialización.
            recuperado = (await orquestador.aget_state(config)).values

    assert recuperado["hallazgos"], "no quedó ningún hallazgo persistido"
    assert all(isinstance(h, Hallazgo) for h in recuperado["hallazgos"]), (
        "los hallazgos volvieron del checkpoint sin su tipo"
    )
    assert isinstance(recuperado["rubrica_actual"], Rubrica)

    sin_registrar = [
        r.getMessage() for r in caplog.records if "unregistered type" in r.getMessage()
    ]
    assert not sin_registrar, (
        f"el serializador no tiene registrados todos los tipos del estado: "
        f"{sin_registrar}"
    )
