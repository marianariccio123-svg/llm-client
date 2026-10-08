"""
config.py

Configuración de la Pre-entrega 6 (orquestador multi-agente).

Mismo criterio que las entregas 3, 4 y 5: **todo perezoso** (`@lru_cache`).
Importar este módulo no instancia ningún cliente ni valida ninguna API key, así
que los tests pueden importar el grafo entero sin credenciales y sin red.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

load_dotenv(REPO_ROOT / ".env")

# --- Rutas ---
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_PATH = DATA_DIR / "checkpoints.sqlite"
TRACES_DIR = BASE_DIR / "traces"
DOCS_DIR = BASE_DIR / "docs"

# El corpus de documentación técnica vive en la Pre-entrega 4. Se lo consulta
# desde acá en vez de duplicar los 8 archivos: la consigna propone justamente
# reusar el vector DB de las entregas anteriores.
CORPUS_PRE_ENTREGA_4 = REPO_ROOT / "pre-entrega-4" / "data"

# --- Frenos del ciclo de supervisión ---
# Dos topes independientes contra el "supervisor infinito":
#
# 1. MAX_PASOS_SUPERVISOR es el freno *semántico*: al llegar al tope, el
#    supervisor deja de poder derivar y el grafo va directo a sintetizar con lo
#    que haya. Se responde con lo recolectado en lugar de no responder.
# 2. RECURSION_LIMIT es el freno *estructural* de LangGraph: si algo falla de
#    un modo que ni siquiera deja contar pasos, corta con GraphRecursionError.
#
# El primero produce una respuesta degradada; el segundo, un error. Hacen falta
# los dos: uno protege la experiencia, el otro protege la factura.
MAX_PASOS_SUPERVISOR = int(os.getenv("MAX_PASOS_SUPERVISOR", "6"))
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "25"))

# Tope de iteraciones internas de cada especialista (su propio ciclo ReAct).
MAX_PASOS_ESPECIALISTA = int(os.getenv("MAX_PASOS_ESPECIALISTA", "6"))

# --- Modelos por proveedor ---
MODELS = {
    "gemini": "gemini-flash-lite-latest",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el formato de logs. Se llama desde los scripts, no al importar."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in (
        "httpx", "urllib3", "google_genai", "langchain_google_genai", "pinecone",
    ):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_provider() -> str:
    """Devuelve el proveedor activo, validándolo."""
    provider = os.getenv("PROVIDER", "gemini").strip().lower()
    if provider not in MODELS:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. Opciones: {', '.join(MODELS)}."
        )
    return provider


def get_model_name(provider: str | None = None) -> str:
    """Modelo a usar, pisable con CHAT_MODEL si se agota la cuota del default."""
    provider = provider or get_provider()
    return os.getenv("CHAT_MODEL", "").strip() or MODELS[provider]


def _require_key(env_var: str) -> str:
    """Lee una API key del entorno y falla con un mensaje claro si falta."""
    key = os.getenv(env_var)
    if not key or not key.strip():
        raise RuntimeError(
            f"Falta la variable de entorno {env_var}. "
            f"Cargala en el .env de la raíz del repo antes de correr el orquestador."
        )
    return key.strip()


@lru_cache(maxsize=1)
def get_llm():
    """
    Modelo de chat del proveedor activo.

    `temperature=0` es obligatorio en un orquestador: el supervisor decide el
    ruteo, y un ruteo no determinista haría que la misma pregunta siga caminos
    distintos entre corridas. La traza que se entrega dejaría de representar al
    sistema.
    """
    provider = get_provider()
    modelo = get_model_name(provider)
    logging.getLogger(__name__).info("Inicializando LLM (%s / %s)...", provider, modelo)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=modelo,
            google_api_key=_require_key("GEMINI_API_KEY"),
            temperature=0,
            timeout=90,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=modelo,
            api_key=_require_key("OPENAI_API_KEY"),
            temperature=0,
            timeout=90,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=modelo,
        api_key=_require_key("ANTHROPIC_API_KEY"),
        temperature=0,
        timeout=90,
    )
