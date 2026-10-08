"""
app/observability.py

La capa de observabilidad: envía una traza por ejecución a LangSmith (o a
Arize Phoenix, si se configura).

Qué se instrumenta y qué no hace falta instrumentar
---------------------------------------------------
LangChain y LangGraph ya emiten eventos de callback en cada paso: cada llamada
al modelo, cada herramienta, cada nodo del grafo. Con `LANGSMITH_TRACING=true`
esos eventos se exportan solos, sin decorar nada. Poner un decorador en cada
nodo sería duplicar spans.

Lo que **sí** hace falta es un span raíz por job. Sin él, una corrida aparece
en el dashboard como varios árboles sueltos —uno por invocación del grafo— y
un job que se pausó para aprobación y se retomó después queda partido en dos
trazas sin relación visible. `@traceable` sobre la función que ejecuta el job
los agrupa bajo un nombre y unos metadatos comunes.

Sobre el costo por ejecución
----------------------------
No se calcula acá. LangSmith lo deriva de los tokens de entrada y salida que
ya vienen en los eventos del modelo, usando su tabla de precios por modelo.
Por eso importa que `LANGSMITH_PROJECT` esté seteado: el costo y los
percentiles de latencia se agregan **por proyecto**, y si las corridas caen en
el proyecto "default" quedan mezcladas con cualquier otra cosa.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Callable

from app.config import PLATAFORMA_TRAZAS, PROYECTO_TRAZAS, hay_observabilidad

log = logging.getLogger("app.observability")


@lru_cache(maxsize=1)
def inicializar() -> str:
    """
    Activa la exportación de trazas. Devuelve una descripción de lo que quedó.

    Es idempotente (`@lru_cache`) porque la llaman el arranque de la app y
    también los scripts: inicializar dos veces el exportador de OpenTelemetry
    deja spans duplicados.

    **No falla si no hay credenciales.** Una API que no arranca porque el
    dashboard de métricas no está configurado es una API que cambió una
    dependencia blanda por una dura. Se avisa por log y se sigue.
    """
    if not hay_observabilidad():
        log.warning(
            "Observabilidad DESACTIVADA: falta la credencial de %s. "
            "La API funciona igual, pero no se van a ver trazas en el dashboard.",
            PLATAFORMA_TRAZAS,
        )
        return "desactivada"

    if PLATAFORMA_TRAZAS == "langsmith":
        return _inicializar_langsmith()
    if PLATAFORMA_TRAZAS == "phoenix":
        return _inicializar_phoenix()

    log.warning("Plataforma de trazas '%s' desconocida.", PLATAFORMA_TRAZAS)
    return "desconocida"


def _inicializar_langsmith() -> str:
    """
    Configura LangSmith por variables de entorno.

    Se setean acá y no solo en el `.env` por una razón: `LANGSMITH_TRACING`
    tiene que estar en el entorno **antes** de que LangChain cree el primer
    runnable, y el orden de imports de una app FastAPI no es obvio. Hacerlo
    explícito en el arranque elimina la clase de bug donde las trazas aparecen
    o no según qué módulo se importó primero.
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_PROJECT"] = PROYECTO_TRAZAS

    # Compatibilidad: algunas versiones de langchain leen los nombres viejos.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = PROYECTO_TRAZAS

    log.info("Trazas -> LangSmith, proyecto '%s'.", PROYECTO_TRAZAS)
    return f"langsmith:{PROYECTO_TRAZAS}"


def _inicializar_phoenix() -> str:
    """
    Configura Arize Phoenix vía OpenInference sobre OTLP.

    El instrumentador de LangChain cubre LangGraph, porque los nodos del grafo
    se ejecutan como runnables de LangChain.

    Nota de entorno: el paquete del **servidor** (`arize-phoenix`) no compila en
    Python 3.14 — una de sus dependencias en Rust falla al construirse. El
    cliente (`arize-phoenix-otel`) sí, así que esta rama sirve apuntando a un
    Phoenix hosteado (Phoenix Cloud) o a uno corriendo en otro entorno.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError as e:
        log.error(
            "Faltan paquetes para Phoenix (%s). Instalá: "
            "pip install arize-phoenix-otel openinference-instrumentation-langchain",
            e.name,
        )
        return "desactivada"

    proveedor = register(
        project_name=PROYECTO_TRAZAS,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", ""),
        auto_instrument=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=proveedor)

    log.info("Trazas -> Phoenix, proyecto '%s'.", PROYECTO_TRAZAS)
    return f"phoenix:{PROYECTO_TRAZAS}"


def estado() -> str:
    """Qué plataforma está activa. Lo informa `GET /health`."""
    return inicializar()


def trazar_job(funcion: Callable) -> Callable:
    """
    Decora la ejecución de un job para que sea un span raíz.

    Si LangSmith no está disponible devuelve la función sin tocar, de modo que
    el código del worker no tiene que preguntarse si la observabilidad está
    encendida.
    """
    if not hay_observabilidad() or PLATAFORMA_TRAZAS != "langsmith":
        return funcion

    try:
        from langsmith import traceable
    except ImportError:
        log.warning("`langsmith` no está instalado: no hay span raíz por job.")
        return funcion

    return traceable(
        run_type="chain",
        name="auditoria_job",
        # Los metadatos permiten filtrar en el dashboard: sin esto, para
        # separar las 5 peticiones de la prueba de carga del resto habría que
        # ir por timestamp.
        metadata={"componente": "worker", "entrega": "pre-entrega-7"},
    )(funcion)


def metadatos_del_job(job_id: str, etiqueta: str = "") -> dict[str, Any]:
    """
    Los metadatos que se le pasan al grafo para que aparezcan en la traza.

    `job_id` es el que permite cruzar una traza del dashboard con el registro
    de Redis. Es la diferencia entre "una de las corridas falló" y "falló el
    job 4f2a1c, acá está su estado".
    """
    metadatos: dict[str, Any] = {"job_id": job_id}
    if etiqueta:
        metadatos["etiqueta"] = etiqueta
    return {
        "metadata": metadatos,
        "run_name": f"auditoria[{job_id}]",
        "tags": ["pre-entrega-7", etiqueta] if etiqueta else ["pre-entrega-7"],
    }
