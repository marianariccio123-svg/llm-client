"""
config.py

Configuración central de la Pre-entrega 3 (RAG local con LangChain + ChromaDB).

Punto clave: TODO se construye de forma perezosa (`@lru_cache`). Importar este
módulo no instancia ningún cliente ni valida ninguna API key: eso recién pasa
la primera vez que alguien pide el modelo o los embeddings de verdad. Así un
`import` nunca revienta por falta de credenciales.

El proveedor se elige con la variable de entorno PROVIDER (gemini | openai |
anthropic), igual que en la Pre-entrega 1 y la Pre-entrega 2. El LLM y los
embeddings se resuelven SIEMPRE con el mismo proveedor, porque mezclarlos
rompería el retriever: un índice creado con embeddings de un proveedor no se
puede consultar con los de otro (distinta dimensión y distinto espacio
vectorial).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# El .env vive en la raíz del repo, un nivel arriba de esta carpeta.
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

load_dotenv(REPO_ROOT / ".env")

# --- Rutas ---
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "asyncio_docs"

# --- Parámetros de chunking e indexado ---
# La consigna pide chunks de 500 tokens con 50 de overlap. Se miden en TOKENS
# reales vía tiktoken, no en caracteres.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TIKTOKEN_ENCODING = "cl100k_base"

# --- Parámetros de recuperación ---
TOP_K = 4  # entre 3 y 5, como pide la consigna

# --- Modelos por proveedor ---
# Solo se prueba end-to-end con Gemini (es la única API key con cupo en este
# proyecto). Las otras ramas existen y son correctas, pero no están ejercitadas.
MODELS = {
    "gemini": {
        "chat": "gemini-flash-latest",
        "embeddings": "models/gemini-embedding-001",
    },
    "openai": {
        "chat": "gpt-4o-mini",
        "embeddings": "text-embedding-3-small",
    },
    "anthropic": {
        # Anthropic no publica un modelo de embeddings propio, así que esta
        # rama usa el chat de Anthropic + embeddings de OpenAI. Requiere las
        # DOS API keys.
        "chat": "claude-3-5-haiku-latest",
        "embeddings": "text-embedding-3-small",
    },
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el formato de logs. Se llama desde los scripts, no al importar."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chroma y httpx son ruidosos y tapan los logs útiles del pipeline.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)


def get_chat_model_name(provider: str) -> str:
    """
    Nombre del modelo de chat a usar.

    Se puede pisar el default con la variable de entorno CHAT_MODEL. Sirve
    cuando el modelo por defecto se queda sin cupo diario en el free tier
    (por ejemplo `gemini-flash-latest`, que permite 20 requests por día):
    con `CHAT_MODEL=gemini-flash-lite-latest` se sigue trabajando contra otro
    modelo, que tiene su propia cuota.
    """
    override = os.getenv("CHAT_MODEL", "").strip()
    return override or MODELS[provider]["chat"]


def get_provider() -> str:
    """Devuelve el proveedor activo, validando que esté soportado."""
    provider = os.getenv("PROVIDER", "gemini").strip().lower()
    if provider not in MODELS:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. "
            f"Opciones válidas: {', '.join(MODELS)}."
        )
    return provider


def _require_key(env_var: str) -> str:
    """Lee una API key del entorno y falla con un mensaje claro si falta."""
    key = os.getenv(env_var)
    if not key or not key.strip():
        raise RuntimeError(
            f"Falta la variable de entorno {env_var}. "
            f"Cargala en el .env de la raíz del repo antes de correr el pipeline."
        )
    return key.strip()


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Modelo de embeddings del proveedor activo. Se instancia una sola vez, en
    el primer uso real (no al importar el módulo).
    """
    provider = get_provider()
    cfg = MODELS[provider]
    logging.getLogger(__name__).info(
        "Inicializando embeddings (%s / %s)...", provider, cfg["embeddings"]
    )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=cfg["embeddings"],
            google_api_key=_require_key("GEMINI_API_KEY"),
        )

    # openai y anthropic comparten el modelo de embeddings de OpenAI.
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg["embeddings"],
        api_key=_require_key("OPENAI_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Modelo de chat del proveedor activo, también instanciado de forma perezosa.
    temperature=0 para que las respuestas sean reproducibles y se apeguen al
    contexto recuperado.
    """
    provider = get_provider()
    modelo = get_chat_model_name(provider)
    logging.getLogger(__name__).info(
        "Inicializando LLM (%s / %s)...", provider, modelo
    )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=modelo,
            google_api_key=_require_key("GEMINI_API_KEY"),
            temperature=0,
            timeout=60,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=modelo,
            api_key=_require_key("OPENAI_API_KEY"),
            temperature=0,
            timeout=60,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=modelo,
        api_key=_require_key("ANTHROPIC_API_KEY"),
        temperature=0,
        timeout=60,
    )
