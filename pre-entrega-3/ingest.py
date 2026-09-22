"""
ingest.py

Script de ingesta: carga los documentos de `data/`, los parte en chunks de 500
tokens con 50 de overlap, los vectoriza con el proveedor configurado y los
persiste en ChromaDB (`./vectorstore`).

Si el vectorstore ya tiene datos indexados, no reprocesa nada salvo que se
pase `--force` (o se llame a `ingest(force=True)`).

Uso:
    python ingest.py
    python ingest.py --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    TIKTOKEN_ENCODING,
    VECTORSTORE_DIR,
    get_embeddings,
    get_provider,
    setup_logging,
)

logger = logging.getLogger(__name__)

EXTENSIONES = (".txt", ".md")


def cargar_documentos() -> List[Document]:
    """Lee todos los .txt/.md de data/ y los devuelve como Documents."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {DATA_DIR}")

    rutas = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    if not rutas:
        raise FileNotFoundError(
            f"No se encontró ningún archivo {'/'.join(EXTENSIONES)} en {DATA_DIR}."
        )

    documentos = []
    for ruta in rutas:
        texto = ruta.read_text(encoding="utf-8")
        documentos.append(
            Document(page_content=texto, metadata={"source": ruta.name})
        )
        logger.info("Cargado: %s (%d caracteres)", ruta.name, len(texto))

    logger.info("Total de documentos cargados: %d", len(documentos))
    return documentos


def dividir_en_chunks(documentos: List[Document]) -> List[Document]:
    """
    Parte los documentos en chunks midiendo el tamaño en TOKENS reales
    (tiktoken), no en caracteres: 500 tokens con 50 de overlap.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)

    # Numerar los chunks dentro de cada archivo ayuda a leer las referencias.
    por_fuente: dict[str, int] = {}
    for chunk in chunks:
        fuente = chunk.metadata.get("source", "desconocido")
        indice = por_fuente.get(fuente, 0)
        chunk.metadata["chunk"] = indice
        por_fuente[fuente] = indice + 1

    logger.info(
        "Se generaron %d chunks (chunk_size=%d tokens, overlap=%d tokens).",
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    for fuente, cantidad in por_fuente.items():
        logger.info("  %-35s -> %d chunks", fuente, cantidad)
    return chunks


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Abre (o crea) la colección persistente de Chroma. Perezoso: recién acá se
    construye el modelo de embeddings.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


def contar_indexados(store: Chroma) -> int:
    """Cuántos vectores hay ya en la colección."""
    try:
        return store._collection.count()
    except Exception:  # pragma: no cover - fallback si cambia la API interna
        return len(store.get(limit=1).get("ids", []))


def ingest(force: bool = False) -> int:
    """
    Ejecuta la ingesta completa. Devuelve la cantidad de chunks indexados
    (o los que ya estaban, si se saltea el reprocesado).
    """
    logger.info("Proveedor activo: %s", get_provider())
    logger.info("Vectorstore: %s", VECTORSTORE_DIR)

    if force and VECTORSTORE_DIR.exists():
        logger.warning("--force: borrando el vectorstore existente...")
        get_vectorstore.cache_clear()
        shutil.rmtree(VECTORSTORE_DIR)

    store = get_vectorstore()
    ya_indexados = contar_indexados(store)

    if ya_indexados > 0 and not force:
        logger.info(
            "El vectorstore ya tiene %d chunks indexados. No se reprocesa nada "
            "(usá --force para reindexar de cero).",
            ya_indexados,
        )
        return ya_indexados

    documentos = cargar_documentos()
    chunks = dividir_en_chunks(documentos)

    logger.info("Vectorizando e indexando %d chunks...", len(chunks))
    store.add_documents(chunks)

    total = contar_indexados(store)
    logger.info("Ingesta terminada. Chunks indexados en la colección: %d", total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de documentos al vectorstore.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borra el vectorstore y reindexa todo de cero.",
    )
    args = parser.parse_args()

    setup_logging()
    ingest(force=args.force)


if __name__ == "__main__":
    main()
