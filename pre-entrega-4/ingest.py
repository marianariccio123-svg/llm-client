"""
ingest.py

Pipeline de ingesta a Pinecone.

Flujo:
    data/*.md  ->  front matter + secciones  ->  RecursiveCharacterTextSplitter
    ->  embeddings (1536 dim)  ->  upsert al índice Serverless, en un namespace

Tres decisiones que vale la pena mirar:

- **El texto viaja en la metadata.** Pinecone guarda `text` junto al vector, así
  que una búsqueda devuelve el contenido listo para armar el prompt. No hace
  falta una segunda base de datos para resolver id -> texto.
- **El chunking se mide en tokens reales.** `RecursiveCharacterTextSplitter`
  parte por separadores semánticos (párrafo, línea, oración) pero cuenta el
  tamaño con `tiktoken`, no con `len()`. Un chunk de 500 caracteres y uno de
  500 tokens no se parecen en nada.
- **Las páginas son reales.** Cada encabezado de nivel 2 del documento es una
  sección, y el chunk hereda el número de la sección donde empieza. Así la
  metadata `pagina` apunta a algo verificable y no a un número inventado.

Uso:
    python ingest.py               # ingesta incremental (salta si ya hay datos)
    python ingest.py --recreate    # vacía el namespace y re-indexa desde cero
    python ingest.py --dry-run     # arma los chunks y los muestra, sin subir nada
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    BM25_CORPUS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_NAME,
    NAMESPACE,
    TIKTOKEN_ENCODING,
    get_embeddings,
    get_index,
    setup_logging,
)
from schemas import ChunkMetadata

log = logging.getLogger("ingest")

UPSERT_BATCH = 100  # tamaño de lote para el upsert; Pinecone recomienda <= 100

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECCION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def parsear_front_matter(texto: str) -> tuple[dict[str, str], str]:
    """
    Extrae el bloque YAML simple del encabezado. Es un parser mínimo a
    propósito (solo `clave: valor`): el corpus lo controlamos nosotros y no
    justifica sumar PyYAML como dependencia.
    """
    match = FRONT_MATTER_RE.match(texto)
    if not match:
        return {}, texto
    meta: dict[str, str] = {}
    for linea in match.group(1).splitlines():
        if ":" in linea:
            clave, _, valor = linea.partition(":")
            meta[clave.strip()] = valor.strip()
    return meta, texto[match.end():]


def offsets_de_secciones(cuerpo: str) -> list[int]:
    """
    Posiciones de carácter donde arranca cada encabezado de nivel 2.

    Sirve para traducir "este chunk empieza en el carácter N" a un número de
    página verificable. Si el documento no tiene encabezados, hay una sola
    página que arranca en 0.
    """
    offsets = [m.start() for m in SECCION_RE.finditer(cuerpo)]
    return offsets or [0]


def pagina_de_offset(offsets: list[int], posicion: int) -> int:
    """Número de sección (1-based) a la que pertenece un offset del documento."""
    pagina = 1
    for numero, inicio in enumerate(offsets, start=1):
        if posicion >= inicio:
            pagina = numero
        else:
            break
    return pagina


def construir_splitter() -> RecursiveCharacterTextSplitter:
    """
    Splitter recursivo que cuenta en tokens reales.

    `from_tiktoken_encoder` mantiene la lógica de separadores de
    `RecursiveCharacterTextSplitter` (prueba los separadores en orden y baja al
    siguiente solo cuando el pedazo sigue siendo demasiado grande) pero usa el
    tokenizador para medir, que es lo que realmente le importa al modelo de
    embeddings.

    El primer separador es `\\n## `, el encabezado de sección. Así el corte
    natural cae en un límite temático, y solo si una sección sola ya excede el
    presupuesto se baja a párrafo, línea y oración. Lo importante es que el
    splitter **empaqueta** secciones cortas juntas hasta llegar a CHUNK_SIZE en
    vez de emitir una por sección: cortar en cada `##` daría chunks de ~120
    tokens, demasiado chicos para retener contexto.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )


def cargar_chunks() -> list[ChunkMetadata]:
    """Lee `data/`, la parte en chunks y devuelve la metadata ya validada."""
    archivos = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.txt"))
    if not archivos:
        raise RuntimeError(f"No hay documentos en {DATA_DIR}. Agregá .md o .txt.")

    splitter = construir_splitter()
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    chunks: list[ChunkMetadata] = []

    for archivo in archivos:
        crudo = archivo.read_text(encoding="utf-8")
        meta, cuerpo = parsear_front_matter(crudo)

        doc_id = meta.get("doc_id") or archivo.stem
        titulo = meta.get("titulo") or archivo.stem.replace("-", " ").title()
        categoria = meta.get("categoria", "sin-categoria")

        offsets = offsets_de_secciones(cuerpo)
        cursor = 0  # avanza con los chunks para no re-encontrar el mismo texto

        indice_en_doc = 0
        for fragmento in splitter.split_text(cuerpo):
            fragmento = fragmento.strip()
            if not fragmento:
                continue

            # Ubicar el chunk en el documento original da la página real.
            posicion = cuerpo.find(fragmento[:80], cursor)
            if posicion == -1:  # el splitter normalizó algún espacio
                posicion = cursor
            cursor = posicion + 1

            chunks.append(
                ChunkMetadata(
                    doc_id=doc_id,
                    fuente=archivo.name,
                    titulo=titulo,
                    categoria=categoria,
                    pagina=pagina_de_offset(offsets, posicion),
                    chunk_index=indice_en_doc,
                    n_tokens=len(encoding.encode(fragmento)),
                    text=fragmento,
                )
            )
            indice_en_doc += 1

        log.info("%-32s -> %2d chunks", archivo.name, indice_en_doc)

    return chunks


def guardar_corpus_bm25(chunks: list[ChunkMetadata]) -> None:
    """
    Persiste los chunks en JSONL para el recuperador léxico.

    BM25 no es un servicio: es un cálculo sobre el corpus entero, que tiene que
    estar en memoria. Guardarlo acá evita que `rag.py` dependa de volver a leer
    y re-parsear `data/`, y garantiza que el índice léxico y el vectorial
    contengan exactamente los mismos chunks.
    """
    with BM25_CORPUS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    log.info("Corpus léxico escrito en %s (%d chunks).", BM25_CORPUS_PATH.name, len(chunks))


def vectores_en_namespace(index) -> int:
    stats = index.describe_index_stats()
    return (stats.get("namespaces", {}) or {}).get(NAMESPACE, {}).get("vector_count", 0)


def upsert(chunks: list[ChunkMetadata], index) -> None:
    """Genera los embeddings y los sube en lotes al namespace configurado."""
    embeddings = get_embeddings()
    textos = [c.text for c in chunks]

    log.info("Generando %d embeddings...", len(textos))
    vectores = embeddings.embed_documents(textos)

    # Chequeo barato que ahorra el error más caro de esta entrega.
    dim_indice = index.describe_index_stats()["dimension"]
    dim_real = len(vectores[0])
    if dim_real != dim_indice:
        raise RuntimeError(
            f"El modelo devolvió vectores de {dim_real} dimensiones y el índice "
            f"espera {dim_indice}. Revisá EMBEDDING_DIM en config.py, o recreá "
            f"el índice con: python setup_index.py --delete && python setup_index.py"
        )

    payload = [
        {
            "id": f"{c.doc_id}#{c.chunk_index}",
            "values": vector,
            "metadata": c.model_dump(),
        }
        for c, vector in zip(chunks, vectores)
    ]

    for inicio in range(0, len(payload), UPSERT_BATCH):
        lote = payload[inicio:inicio + UPSERT_BATCH]
        index.upsert(vectors=lote, namespace=NAMESPACE)
        log.info("Upsert %d/%d vectores.",
                 min(inicio + UPSERT_BATCH, len(payload)), len(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true",
                        help="vacía el namespace y re-indexa desde cero")
    parser.add_argument("--dry-run", action="store_true",
                        help="arma los chunks y los muestra, sin llamar a ninguna API")
    args = parser.parse_args()

    setup_logging()

    chunks = cargar_chunks()
    tokens = [c.n_tokens for c in chunks]
    log.info(
        "Total: %d chunks | tokens por chunk: min=%d, media=%d, max=%d",
        len(chunks), min(tokens), sum(tokens) // len(tokens), max(tokens),
    )

    # El corpus léxico se escribe siempre: es local, barato y no toca la red.
    # Esto deja el modo BM25 utilizable incluso en --dry-run, sin Pinecone.
    guardar_corpus_bm25(chunks)

    if args.dry_run:
        print("\n--- DRY RUN: primeros 3 chunks ---")
        for c in chunks[:3]:
            print(f"\n[{c.doc_id}#{c.chunk_index}] {c.titulo} / pág. {c.pagina} "
                  f"/ {c.categoria} / {c.n_tokens} tokens")
            print(c.text[:220].replace("\n", " ") + "...")
        print(f"\n(no se subió nada a Pinecone; {len(chunks)} chunks en total)")
        return 0

    index = get_index()

    if args.recreate:
        log.warning("Vaciando el namespace '%s'...", NAMESPACE)
        try:
            index.delete(delete_all=True, namespace=NAMESPACE)
        except Exception as e:  # el namespace puede no existir todavía
            log.info("Nada que borrar (%s).", type(e).__name__)
    else:
        ya_hay = vectores_en_namespace(index)
        if ya_hay >= len(chunks):
            log.info(
                "El namespace '%s' ya tiene %d vectores (>= %d chunks). "
                "No se re-indexa nada. Usá --recreate para forzar.",
                NAMESPACE, ya_hay, len(chunks),
            )
            return 0

    upsert(chunks, index)

    log.info(
        "Ingesta terminada: %d vectores en el índice '%s', namespace '%s'.",
        len(chunks), INDEX_NAME, NAMESPACE,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("ingest").error("%s", e)
        sys.exit(1)
