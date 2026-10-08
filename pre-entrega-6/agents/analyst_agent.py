"""
agents/analyst_agent.py

El especialista en **procesar los datos que ya están en el estado**.

La decisión de diseño que define este módulo
--------------------------------------------
Las tres herramientas del analista reciben los datos por `InjectedState`, no
como argumentos que el LLM tenga que escribir. El modelo decide *cuándo* llamar
a `resumir_incidentes`, pero no le dicta los 20 registros: LangGraph los
inyecta desde el estado compartido, exactamente como los devolvió la base.

Por qué importa: la alternativa es que los registros viajen como texto dentro
del prompt. Eso tiene tres costos concretos.

1. **Precisión.** Un LLM que transcribe 20 filas de números se equivoca en
   alguna. Y una media calculada sobre datos mal transcriptos es una media
   mal calculada que *parece* bien.
2. **Contexto.** 20 registros son ~2.000 tokens que se repiten en cada vuelta
   del ciclo. Con `InjectedState` el prompt del analista no crece con el
   tamaño del dataset.
3. **Trazabilidad.** El número del informe final se puede rastrear hasta la
   fila de la base. Si pasó por un prompt, no.

El LLM queda haciendo lo que hace bien —decidir qué análisis corresponde e
interpretar el resultado— y la aritmética la hace Python, que no alucina.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from functools import lru_cache
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field, ValidationError, field_validator

from config import MAX_PASOS_ESPECIALISTA, get_llm
from state import incidentes_recuperados
from utils import extraer_hallazgos, texto_de

log = logging.getLogger("agents.analista")

SEVERIDADES_VALIDAS = ("baja", "media", "alta", "critica")

PROMPT = """Sos el agente ANALISTA de un equipo de auditoría técnica.

Tu responsabilidad es PROCESAR los datos que el investigador ya recuperó. No \
vas a buscar información nueva: no tenés herramientas para eso.

Algo importante sobre tus herramientas: **ya tienen los datos**. Los leen \
directo del estado compartido, así que no necesitás —ni podés— pasarles los \
registros. Llamalas sin inventar argumentos.

Tenés tres:
- `resumir_incidentes`: cuenta y agrupa (por categoría, severidad, servicio) y \
calcula las métricas de minutos caídos.
- `detectar_atipicos`: encuentra los incidentes anormalmente largos por el \
método del rango intercuartílico.
- `validar_registros`: verifica la integridad de los datos y reporta los \
registros que no cumplen el esquema.

Cómo trabajar:
- Usá las tres: un análisis sin validación de datos no es un análisis, es una \
cuenta con suerte.
- Cerrá con una interpretación de TRES O CUATRO ORACIONES: qué dicen los \
números, no cuáles son. Los valores ya quedaron en el estado compartido.
- Si `validar_registros` encuentra problemas, decilo explícitamente: un informe \
que esconde un dato sucio es peor que uno que lo admite."""


# --------------------------------------------------------------------------
# El esquema contra el que se valida
# --------------------------------------------------------------------------

class IncidenteValidado(BaseModel):
    """
    Esquema que debe cumplir cada registro de la bitácora.

    Separado del esquema de la base a propósito: la base acepta cualquier
    entero en `minutos_caidos`, pero una caída de duración negativa no existe.
    Esta clase es la que define qué es un registro *válido*, no qué es un
    registro *almacenable*.
    """

    incidente_id: str = Field(..., pattern=r"^INC-\d{4}$")
    fecha: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    servicio: str = Field(..., min_length=1)
    categoria: str = Field(..., min_length=1)
    severidad: str
    minutos_caidos: int = Field(..., ge=0, description="Una caída no puede durar menos que nada")
    descripcion: str = Field(..., min_length=1)

    @field_validator("severidad")
    @classmethod
    def _severidad_conocida(cls, v: str) -> str:
        if v not in SEVERIDADES_VALIDAS:
            raise ValueError(
                f"severidad '{v}' desconocida; se esperaba una de "
                f"{', '.join(SEVERIDADES_VALIDAS)}"
            )
        return v


# --------------------------------------------------------------------------
# Las herramientas
# --------------------------------------------------------------------------

def _sin_datos() -> dict[str, Any]:
    """Respuesta común cuando el estado todavía no tiene incidentes."""
    return {
        "error": (
            "Todavía no hay incidentes en el estado compartido. El investigador "
            "tiene que ejecutar `listar_incidentes` antes de que yo pueda "
            "analizar nada."
        ),
        "incidentes_disponibles": 0,
    }


@tool
def resumir_incidentes(estado: Annotated[dict, InjectedState]) -> dict[str, Any]:
    """
    Calcula las métricas y las agrupaciones de los incidentes recuperados.

    Esta herramienta NO necesita argumentos: lee los incidentes directamente
    del estado compartido, tal como los devolvió la base. Llamala sin
    parámetros.

    Devuelve:
        {"total_incidentes": 20, "total_minutos_caidos": 1180,
         "metricas_minutos": {"media": 59.0, "mediana": 40.0, "minimo": -5,
                              "maximo": 480, "desvio_estandar": 101.3},
         "por_categoria": [{"categoria": "bloqueo-event-loop", "cantidad": 9,
                            "porcentaje": 45.0, "minutos_totales": 453}],
         "por_severidad": {...}, "por_servicio": {...}}

    Las categorías vienen ordenadas de más a menos frecuente, así que la
    primera es la causa dominante.

    Si el estado todavía no tiene incidentes devuelve {"error": ...}.
    """
    incidentes = incidentes_recuperados(estado)
    if not incidentes:
        return _sin_datos()

    minutos = [int(i.get("minutos_caidos", 0)) for i in incidentes]
    total = len(incidentes)

    conteo_categoria = Counter(i.get("categoria", "?") for i in incidentes)
    minutos_por_categoria: Counter[str] = Counter()
    for incidente in incidentes:
        minutos_por_categoria[incidente.get("categoria", "?")] += int(
            incidente.get("minutos_caidos", 0)
        )

    return {
        "total_incidentes": total,
        "total_minutos_caidos": sum(minutos),
        "metricas_minutos": {
            "media": round(statistics.mean(minutos), 2),
            "mediana": round(statistics.median(minutos), 2),
            "minimo": min(minutos),
            "maximo": max(minutos),
            "desvio_estandar": (
                round(statistics.stdev(minutos), 2) if total > 1 else 0.0
            ),
        },
        "por_categoria": [
            {
                "categoria": categoria,
                "cantidad": cantidad,
                "porcentaje": round(100 * cantidad / total, 1),
                "minutos_totales": minutos_por_categoria[categoria],
            }
            for categoria, cantidad in conteo_categoria.most_common()
        ],
        "por_severidad": dict(
            Counter(i.get("severidad", "?") for i in incidentes).most_common()
        ),
        "por_servicio": dict(
            Counter(i.get("servicio", "?") for i in incidentes).most_common()
        ),
    }


@tool
def detectar_atipicos(
    estado: Annotated[dict, InjectedState], umbral_iqr: float = 1.5
) -> dict[str, Any]:
    """
    Encuentra los incidentes de duración anormalmente alta.

    Lee los incidentes del estado compartido, así que no hace falta pasárselos.
    Usa el método del rango intercuartílico (IQR): es atípico todo valor por
    encima de Q3 + umbral * IQR. Se elige IQR y no desvíos estándar porque la
    media y el desvío los corre el propio atípico, mientras que los cuartiles
    no.

    Args:
        umbral_iqr: multiplicador del IQR. 1.5 es la convención (atípico);
            3.0 marca solo los extremos. Dejalo en 1.5 salvo que te pidan
            otra cosa.

    Devuelve:
        {"metodo": "IQR", "umbral_iqr": 1.5, "q1": 25.0, "q3": 60.0, "iqr": 35.0,
         "limite_superior": 112.5, "cantidad_atipicos": 1,
         "atipicos": [{"incidente_id": "INC-0214", "minutos_caidos": 480,
                       "veces_la_mediana": 12.0, ...}]}

    Si no hay atípicos, `atipicos` viene vacío y eso también es un resultado
    válido para reportar.
    """
    incidentes = incidentes_recuperados(estado)
    if not incidentes:
        return _sin_datos()
    if len(incidentes) < 4:
        return {
            "error": (
                f"Hacen falta al menos 4 incidentes para calcular cuartiles y "
                f"hay {len(incidentes)}."
            ),
            "incidentes_disponibles": len(incidentes),
        }

    minutos = sorted(int(i.get("minutos_caidos", 0)) for i in incidentes)
    cuartiles = statistics.quantiles(minutos, n=4, method="inclusive")
    q1, mediana, q3 = cuartiles[0], cuartiles[1], cuartiles[2]
    iqr = q3 - q1
    limite = q3 + umbral_iqr * iqr

    atipicos = [
        {
            **{k: v for k, v in i.items() if k != "descripcion"},
            "veces_la_mediana": (
                round(int(i["minutos_caidos"]) / mediana, 1) if mediana else None
            ),
        }
        for i in incidentes
        if int(i.get("minutos_caidos", 0)) > limite
    ]

    return {
        "metodo": "IQR",
        "umbral_iqr": umbral_iqr,
        "q1": round(q1, 2),
        "mediana": round(mediana, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "limite_superior": round(limite, 2),
        "cantidad_atipicos": len(atipicos),
        "atipicos": sorted(
            atipicos, key=lambda a: a["minutos_caidos"], reverse=True
        ),
    }


@tool
def validar_registros(estado: Annotated[dict, InjectedState]) -> dict[str, Any]:
    """
    Verifica que los incidentes recuperados cumplan el esquema esperado.

    Lee los registros del estado compartido; no hace falta pasárselos. Valida
    cada uno contra un modelo Pydantic: formato del id y de la fecha, severidad
    dentro del vocabulario conocido, campos no vacíos, y minutos de caída no
    negativos.

    Devuelve:
        {"registros_evaluados": 20, "validos": 19, "invalidos": 1,
         "tasa_de_validez": 95.0,
         "problemas": [{"incidente_id": "INC-0231", "campo": "minutos_caidos",
                        "problema": "Input should be greater than or equal to 0",
                        "valor": -5}]}

    Una tasa menor a 100 significa que hay datos sucios: decilo en el informe,
    porque afecta la confianza en todas las métricas calculadas sobre ellos.
    """
    incidentes = incidentes_recuperados(estado)
    if not incidentes:
        return _sin_datos()

    problemas: list[dict[str, Any]] = []
    validos = 0

    for incidente in incidentes:
        try:
            IncidenteValidado.model_validate(incidente)
            validos += 1
        except ValidationError as e:
            for error in e.errors():
                campo = ".".join(str(p) for p in error["loc"]) or "(registro)"
                problemas.append(
                    {
                        "incidente_id": incidente.get("incidente_id", "?"),
                        "campo": campo,
                        "problema": error["msg"],
                        "valor": incidente.get(campo),
                    }
                )

    total = len(incidentes)
    return {
        "registros_evaluados": total,
        "validos": validos,
        "invalidos": total - validos,
        "tasa_de_validez": round(100 * validos / total, 1),
        "problemas": problemas,
    }


HERRAMIENTAS = [resumir_incidentes, detectar_atipicos, validar_registros]


# --------------------------------------------------------------------------
# El nodo del grafo
# --------------------------------------------------------------------------

class EstadoAnalista(AgentState):
    """
    Estado del subgrafo del analista.

    Extiende el `AgentState` de `create_react_agent` con `hallazgos`. Sin este
    campo el subgrafo descartaría la lista al entrar y `InjectedState` le
    inyectaría a las herramientas un estado sin datos: todas devolverían
    "todavía no hay incidentes" aunque el investigador ya los hubiera traído.
    """

    hallazgos: list[Any]


@lru_cache(maxsize=1)
def _agente_react():
    """El subgrafo ReAct del analista, con sus herramientas acotadas."""
    return create_react_agent(
        get_llm(), HERRAMIENTAS, prompt=PROMPT, state_schema=EstadoAnalista
    )


async def nodo_analista(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Corre al analista con la instrucción puntual del supervisor.

    Igual que el investigador, no recibe `estado["messages"]`. Pero hay una
    diferencia: el subgrafo del analista **sí** necesita el estado completo,
    porque sus herramientas leen `hallazgos` por `InjectedState`. Por eso se le
    pasan los hallazgos junto con el mensaje, mientras el historial de
    deliberación queda afuera.
    """
    instruccion = estado.get("instruccion_actual", "") or estado.get("solicitud", "")
    disponibles = len(incidentes_recuperados(estado))

    pedido = (
        f"Solicitud original del usuario (como marco):\n{estado.get('solicitud', '')}\n\n"
        f"Tu tarea puntual ahora:\n{instruccion}\n\n"
        f"Hay {disponibles} incidentes cargados en el estado compartido, listos "
        f"para que tus herramientas los lean."
    )

    log.info("Analista trabajando sobre %d incidentes: %s", disponibles, instruccion[:90])

    resultado = await _agente_react().ainvoke(
        {
            "messages": [HumanMessage(pedido)],
            # Los hallazgos viajan al subgrafo para que InjectedState los
            # encuentre. Es el canal estructurado, no el de texto.
            "hallazgos": estado.get("hallazgos", []),
        },
        config={"recursion_limit": MAX_PASOS_ESPECIALISTA * 2},
    )

    mensajes = resultado["messages"]
    hallazgos = extraer_hallazgos(mensajes, "analista")
    informe = mensajes[-1]

    log.info("Analista produjo %d hallazgo(s).", len(hallazgos))

    return {
        "messages": [
            HumanMessage(content=f"[analista] {texto_de(informe)}", name="analista")
        ],
        "hallazgos": hallazgos,
    }
