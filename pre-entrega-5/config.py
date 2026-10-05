"""
config.py

Configuración de la Pre-entrega 5 (agente de razonamiento cíclico con LangGraph).

Mismo criterio que las entregas 3 y 4: **todo perezoso** (`@lru_cache`).
Importar este módulo no instancia ningún cliente ni valida ninguna API key.
`import agent` funciona sin credenciales; el error aparece recién cuando se
intenta usar el modelo, y dice qué variable falta.
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
# Base del checkpointer: acá LangGraph persiste el estado de cada thread_id.
CHECKPOINTS_PATH = DATA_DIR / "checkpoints.sqlite"
TRACES_DIR = BASE_DIR / "traces"

# --- Límites del ciclo ---
# Techo de pasos del grafo. Sin esto, un modelo que se obstina en llamar a la
# misma herramienta una y otra vez corre para siempre y factura cada vuelta.
# 10 pasos alcanzan de sobra: el escenario más largo de esta entrega usa 8
# (4 turnos del modelo + 3 de herramientas + el nodo final).
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "10"))

# --- Modelos por proveedor ---
MODELS = {
    "gemini": "gemini-flash-latest",
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
    for ruidoso in ("httpx", "urllib3", "google_genai", "langchain_google_genai"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_provider() -> str:
    """Devuelve el proveedor activo, validándolo."""
    provider = os.getenv("PROVIDER", "gemini").strip().lower()
    if provider not in MODELS:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. Opciones: {', '.join(MODELS)}."
        )
    return provider


def get_model_name(provider: str) -> str:
    """Modelo a usar, pisable con CHAT_MODEL si se agota la cuota del default."""
    return os.getenv("CHAT_MODEL", "").strip() or MODELS[provider]


def _require_key(env_var: str) -> str:
    """Lee una API key del entorno y falla con un mensaje claro si falta."""
    key = os.getenv(env_var)
    if not key or not key.strip():
        raise RuntimeError(
            f"Falta la variable de entorno {env_var}. "
            f"Cargala en el .env de la raíz del repo antes de correr el agente."
        )
    return key.strip()


@lru_cache(maxsize=1)
def get_llm():
    """
    Modelo de chat del proveedor activo.

    `temperature=0` no es opcional acá: un agente que elige herramientas tiene
    que ser reproducible. Con temperatura alta, la misma pregunta puede llamar
    a `buscar_pedidos` una vez y a `detalle_pedido` la siguiente, y entonces la
    traza que se entrega deja de representar lo que hace el sistema.
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
