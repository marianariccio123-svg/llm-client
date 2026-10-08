"""
app/llm.py

El modelo de lenguaje para el entorno de API, con limitador de tasa y
reintentos.

El problema que resuelve
------------------------
El orquestador de la Pre-entrega 6 construía su modelo asumiendo un uso de a
una consulta por vez: un CLI, una persona esperando. Bajo la API eso cambia.
Cinco peticiones concurrentes son cinco grafos corriendo a la vez, y cada grafo
hace entre 8 y 15 llamadas al proveedor. Son del orden de 50 llamadas en un
minuto.

El tier gratuito de Gemini permite unas pocas por minuto. La primera prueba de
carga de esta entrega terminó con **3 de 5 jobs en FALLIDO por 429
RESOURCE_EXHAUSTED**. El sistema se comportó bien —los tres fallos quedaron
registrados en Redis con su error, que es exactamente lo que la consigna
pide— pero un informe que no se produce no le sirve a nadie.

Un límite de tasa del proveedor no es un caso excepcional: es una condición
permanente de cualquier API que dependa de un tercero. La respuesta correcta no
es bajar la concurrencia de la API hasta que deje de pasar, sino **acomodar el
ritmo de las llamadas al límite del proveedor** y reintentar lo que igual
rebote.

Las dos piezas
--------------
1. **`InMemoryRateLimiter`** reparte las llamadas en el tiempo. Es un token
   bucket compartido por todas las corrutinas del proceso: no importa cuántos
   grafos corran en paralelo, entre todos no superan la tasa configurada. Es
   preventivo — evita el 429 en lugar de recuperarse de él.
2. **`max_retries`** con backoff exponencial, para lo que igual rebote. El
   limitador trabaja con la tasa promedio; una ráfaga puntual puede pasarse
   igual.

Por qué se define acá y no se toca la Pre-entrega 6
---------------------------------------------------
Los agentes del módulo 6 obtienen su modelo con `from config import get_llm`.
Esta entrega **no modifica** ese archivo: inyecta su propio `get_llm` en los
módulos de agentes durante el arranque de la app (ver `instalar()`).

Es inyección de dependencias en el punto de composición, que es donde
corresponde: el módulo 6 define *qué* hacen los agentes, y el entorno que los
ejecuta decide *con qué modelo*. Además mantiene la entrega anterior intacta
mientras está en corrección.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

log = logging.getLogger("app.llm")

# Llamadas por segundo al proveedor, sumando todos los grafos del proceso.
# 0.5 = una cada dos segundos. Es deliberadamente conservador: el free tier de
# Gemini permite muy pocas por minuto, y pasarse cuesta un job entero, mientras
# que quedarse corto solo cuesta tiempo.
LLAMADAS_POR_SEGUNDO = float(os.getenv("LLM_LLAMADAS_POR_SEGUNDO", "0.5"))

# Cuántas llamadas puede acumular el bucket para una ráfaga. Con 3 se permite
# que un grafo que estuvo esperando arranque sus primeros pasos de corrido, sin
# dejar que cinco grafos disparen quince llamadas juntas.
RAFAGA_MAXIMA = float(os.getenv("LLM_RAFAGA_MAXIMA", "3"))

# Reintentos ante 429 o 5xx. El cliente de LangChain usa backoff exponencial.
MAX_REINTENTOS = int(os.getenv("LLM_MAX_REINTENTOS", "6"))

# Modelo concreto por defecto para Gemini, en lugar del alias `-latest` que usa
# la Pre-entrega 6.
#
# La razón es la atribución de costos, que es una de las dos métricas que hay
# que capturar. LangSmith deriva el costo de los tokens usando una tabla de
# precios indexada por nombre de modelo, y esa tabla tiene entradas para los
# nombres concretos (`gemini-3.1-flash-lite`) pero no para los alias
# (`gemini-flash-lite-latest`). Con el alias, las trazas llegan con los tokens
# bien contados y el costo en blanco.
#
# Es además lo correcto para un entorno de producción por otro motivo: un alias
# cambia de modelo sin aviso, y con él cambian la latencia, la calidad y el
# precio. Un servicio que necesita métricas comparables entre semanas fija la
# versión.
MODELO_GEMINI_API = os.getenv("MODELO_GEMINI_API", "gemini-3.1-flash-lite")


@lru_cache(maxsize=1)
def _limitador():
    """
    El token bucket, compartido por todo el proceso.

    `@lru_cache` no es una optimización acá: es lo que hace que el límite sea
    global. Un limitador por modelo daría tantos presupuestos como instancias,
    que es lo mismo que no tener límite.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter

    log.info(
        "Limitador de tasa: %.2f llamadas/s (ráfaga %.0f), %d reintentos.",
        LLAMADAS_POR_SEGUNDO, RAFAGA_MAXIMA, MAX_REINTENTOS,
    )
    return InMemoryRateLimiter(
        requests_per_second=LLAMADAS_POR_SEGUNDO,
        # Cada cuánto el bucket revisa si hay token disponible. 0.1s es lo bastante
        # fino para no agregar latencia perceptible.
        check_every_n_seconds=0.1,
        max_bucket_size=RAFAGA_MAXIMA,
    )


@lru_cache(maxsize=1)
def get_llm():
    """
    El modelo para el entorno de API.

    Es el mismo modelo y el mismo proveedor que usa el módulo 6 —se lee la
    misma configuración— con el limitador y los reintentos agregados.
    """
    # `config` es el de la Pre-entrega 6: ya está en el sys.path.
    import config as config_m6

    proveedor = config_m6.get_provider()

    # CHAT_MODEL del entorno gana siempre; si no está y el proveedor es Gemini,
    # se usa el modelo concreto en vez del alias del módulo 6.
    modelo = os.getenv("CHAT_MODEL", "").strip()
    if not modelo:
        modelo = (
            MODELO_GEMINI_API if proveedor == "gemini"
            else config_m6.get_model_name(proveedor)
        )

    log.info("LLM de la API: %s / %s (con limitador).", proveedor, modelo)

    comunes = {
        "temperature": 0,
        "timeout": 90,
        "max_retries": MAX_REINTENTOS,
        "rate_limiter": _limitador(),
    }

    if proveedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=modelo,
            google_api_key=config_m6._require_key("GEMINI_API_KEY"),
            **comunes,
        )

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=modelo,
            api_key=config_m6._require_key("OPENAI_API_KEY"),
            **comunes,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=modelo,
        api_key=config_m6._require_key("ANTHROPIC_API_KEY"),
        **comunes,
    )


def instalar() -> None:
    """
    Inyecta este modelo en los agentes del módulo 6.

    Los módulos de agentes hicieron `from config import get_llm`, así que cada
    uno tiene su **propia referencia** a esa función: reemplazarla en `config`
    no alcanza. Hay que pisar el nombre en cada módulo.

    También se limpian los `@lru_cache` que envuelven a los subgrafos: si un
    agente ya se construyó con el modelo anterior, seguiría usándolo.

    Se llama una sola vez, en el arranque de la app, antes de que entre el
    primer job.
    """
    import agents.analyst_agent as analyst
    import agents.research_agent as research
    import agents.supervisor as supervisor

    for modulo in (research, analyst, supervisor):
        modulo.get_llm = get_llm

    for limpiar in (
        research._agente_react,
        analyst._agente_react,
        supervisor._supervisor_estructurado,
    ):
        limpiar.cache_clear()

    log.info("Modelo con limitador inyectado en los agentes del módulo 6.")
