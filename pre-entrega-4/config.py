"""
config.py

Configuración central de la Pre-entrega 4 (RAG escalable en la nube con
Pinecone + recuperador híbrido).

Mantiene el criterio de la Pre-entrega 3: **todo se construye de forma
perezosa** (`@lru_cache`). Importar este módulo no instancia ningún cliente ni
valida ninguna API key; eso recién pasa en el primer uso real. Así un `import`
nunca revienta por falta de credenciales y los tests/imports son baratos.

Diferencia importante respecto de la Pre-entrega 3: acá la dimensión del
embedding es un contrato con el índice remoto. Un índice de Pinecone se crea
con una dimensión fija y no se puede cambiar; si el modelo de embeddings emite
vectores de otro tamaño, todos los upserts fallan. Por eso `EMBEDDING_DIM` es
una constante única de la que dependen tanto la creación del índice como el
modelo de embeddings.
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
# Corpus de chunks en texto plano. BM25 es un algoritmo léxico local: necesita
# los documentos en memoria, no sirve el índice vectorial remoto. Se escribe
# durante la ingesta para que el retriever no tenga que re-descargar nada.
BM25_CORPUS_PATH = BASE_DIR / "bm25_corpus.jsonl"
GOLDEN_SET_PATH = BASE_DIR / "golden_set.json"

# --- Índice de Pinecone ---
INDEX_NAME = os.getenv("INDEX_NAME", "asyncio-docs").strip()
# Namespace: particiona el índice lógicamente. Sin namespace, todos los
# corpus de todos los inquilinos comparten el mismo espacio y la búsqueda se
# vuelve ruidosa. Un índice, varios namespaces.
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "python-asyncio").strip()
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()
METRIC = "cosine"

# Dimensión del vector. 1536 es la nativa de `text-embedding-3-small` (OpenAI);
# `gemini-embedding-001` se trunca a la misma dimensión vía
# `output_dimensionality`, así que el MISMO índice sirve para los dos
# proveedores. Cambiar este número obliga a recrear el índice.
EMBEDDING_DIM = 1536

# --- Chunking ---
# La consigna sugiere 500-800 tokens. Se usa el piso del rango (500/60) porque
# el corpus de ejemplo son 8 documentos cortos: con 800 tokens quedarían ~10
# chunks en total y la evaluación Precision@5 perdería sentido estadístico.
# Ambos valores son configurables por entorno.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
TIKTOKEN_ENCODING = "cl100k_base"

# --- Recuperación ---
TOP_K = 5  # la consigna evalúa Precision@5 / Recall@5
# Pesos del EnsembleRetriever: [BM25, denso]. Se le da más peso al vectorial
# porque cubre paráfrasis; BM25 aporta los nombres propios exactos
# (`TaskGroup`, `run_in_executor`, `AsyncMock`) que el embedding diluye.
# Configurable por entorno para poder barrer valores sin tocar el código:
#     ENSEMBLE_WEIGHTS=0.2,0.8 python evaluate.py --modo hibrido
def _leer_pesos() -> tuple[float, float]:
    crudo = os.getenv("ENSEMBLE_WEIGHTS", "0.3,0.7").strip()
    try:
        bm25, denso = (float(x) for x in crudo.split(","))
    except ValueError as e:
        raise ValueError(
            f"ENSEMBLE_WEIGHTS='{crudo}' es inválido. Se espera 'peso_bm25,peso_denso', "
            f"por ejemplo '0.3,0.7'."
        ) from e
    return bm25, denso


ENSEMBLE_WEIGHTS = _leer_pesos()

# --- Modelos por proveedor ---
MODELS = {
    "gemini": {
        "chat": "gemini-flash-latest",
        "embeddings": "models/gemini-embedding-001",
    },
    "openai": {
        "chat": "gpt-4o-mini",
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
    for ruidoso in ("httpx", "urllib3", "pinecone", "langchain_google_genai"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def get_provider() -> str:
    """Devuelve el proveedor activo de embeddings/chat, validándolo."""
    provider = os.getenv("PROVIDER", "gemini").strip().lower()
    if provider not in MODELS:
        raise ValueError(
            f"Proveedor '{provider}' no soportado en esta entrega. "
            f"Opciones válidas: {', '.join(MODELS)}."
        )
    return provider


def get_chat_model_name(provider: str) -> str:
    """Modelo de chat, pisable con CHAT_MODEL (útil si se agota la cuota diaria)."""
    return os.getenv("CHAT_MODEL", "").strip() or MODELS[provider]["chat"]


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
    Modelo de embeddings del proveedor activo, fijado a `EMBEDDING_DIM`.

    Los dos proveedores emiten vectores de 1536 dimensiones, así que el índice
    de Pinecone es compatible con ambos (aunque NO se pueden mezclar en el
    mismo namespace: son espacios vectoriales distintos).
    """
    provider = get_provider()
    cfg = MODELS[provider]
    logging.getLogger(__name__).info(
        "Inicializando embeddings (%s / %s, dim=%d)...",
        provider, cfg["embeddings"], EMBEDDING_DIM,
    )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=cfg["embeddings"],
            google_api_key=_require_key("GEMINI_API_KEY"),
            output_dimensionality=EMBEDDING_DIM,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg["embeddings"],
        api_key=_require_key("OPENAI_API_KEY"),
        dimensions=EMBEDDING_DIM,
    )


@lru_cache(maxsize=1)
def get_chat_model():
    """Modelo de chat del proveedor activo. temperature=0 para reproducibilidad."""
    provider = get_provider()
    modelo = get_chat_model_name(provider)
    logging.getLogger(__name__).info("Inicializando LLM (%s / %s)...", provider, modelo)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=modelo,
            google_api_key=_require_key("GEMINI_API_KEY"),
            temperature=0,
            timeout=60,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=modelo,
        api_key=_require_key("OPENAI_API_KEY"),
        temperature=0,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_pinecone():
    """Cliente de Pinecone, instanciado una sola vez y recién en el primer uso."""
    from pinecone import Pinecone

    logging.getLogger(__name__).info("Conectando con Pinecone...")
    return Pinecone(api_key=_require_key("PINECONE_API_KEY"))


@lru_cache(maxsize=1)
def get_index():
    """
    Handle del índice de Pinecone. Falla con un mensaje accionable si el índice
    todavía no existe, en vez de con un 404 crudo del SDK.
    """
    pc = get_pinecone()
    existentes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existentes:
        raise RuntimeError(
            f"El índice '{INDEX_NAME}' no existe en tu proyecto de Pinecone "
            f"(hay: {existentes or 'ninguno'}). Corré primero:\n"
            f"    python setup_index.py"
        )
    return pc.Index(INDEX_NAME)
