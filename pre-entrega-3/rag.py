"""
rag.py

Cadena RAG: recuperación semántica sobre ChromaDB + generación con el LLM
configurado, devolviendo un objeto Pydantic validado.

Todo se arma de forma perezosa: importar este módulo no crea clientes, no lee
API keys y no abre el vectorstore. La cadena se construye en la primera
llamada real a `get_rag_response()`.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from config import TOP_K, VECTORSTORE_DIR, get_chat_model, get_provider
from schemas import RAGResponse, Referencia

logger = logging.getLogger(__name__)

# Cuánto texto de cada fragmento se guarda en las referencias de la respuesta.
LARGO_REFERENCIA = 300


SYSTEM_PROMPT = """\
Sos un asistente técnico que responde ÚNICAMENTE con la información contenida \
en el CONTEXTO que se te entrega. El contexto son fragmentos recuperados de una \
documentación interna sobre programación asíncrona en Python.

Reglas estrictas:
1. Respondé usando exclusivamente lo que dice el CONTEXTO. No uses conocimiento \
previo propio ni completes con suposiciones.
2. Si la respuesta a la pregunta NO está en el CONTEXTO, no la inventes: \
respondé literalmente "No tengo esa información en los documentos disponibles." \
y marcá encontrado_en_contexto en false.
3. Si la respuesta SÍ está en el contexto, respondé de forma concreta y técnica \
(2 a 5 oraciones), citando los nombres de funciones o excepciones tal como \
aparecen, y marcá encontrado_en_contexto en true.
4. No menciones que estás leyendo un "contexto" ni describas el proceso de \
recuperación: respondé directamente la pregunta.
"""

HUMAN_PROMPT = """\
CONTEXTO:
{contexto}

PREGUNTA: {pregunta}
"""


class _RespuestaLLM(BaseModel):
    """Salida estructurada que se le pide al modelo (uso interno de la cadena)."""

    respuesta: str = Field(
        ...,
        description="Respuesta a la pregunta, basada solo en el contexto recuperado.",
    )
    encontrado_en_contexto: bool = Field(
        ...,
        description=(
            "true si el contexto contenía la información necesaria; "
            "false si hubo que responder que no se tiene esa información."
        ),
    )


def _formatear_contexto(documentos: List[Document]) -> str:
    """Arma el bloque de contexto numerando cada fragmento con su fuente."""
    if not documentos:
        return "(sin fragmentos recuperados)"

    partes = []
    for i, doc in enumerate(documentos, start=1):
        fuente = doc.metadata.get("source", "desconocido")
        partes.append(f"[Fragmento {i} — fuente: {fuente}]\n{doc.page_content}")
    return "\n\n".join(partes)


@lru_cache(maxsize=1)
def get_retriever():
    """
    Retriever sobre la colección persistida de Chroma, con top_k fragmentos.
    Falla con un mensaje claro si todavía no se corrió la ingesta.
    """
    if not VECTORSTORE_DIR.exists():
        raise RuntimeError(
            f"No existe el vectorstore en {VECTORSTORE_DIR}. "
            f"Corré primero: python ingest.py"
        )

    # Import local para no arrastrar chromadb en un import trivial del módulo.
    from ingest import contar_indexados, get_vectorstore

    store = get_vectorstore()
    total = contar_indexados(store)
    if total == 0:
        raise RuntimeError(
            "El vectorstore está vacío. Corré primero: python ingest.py"
        )

    logger.info(
        "Retriever listo sobre %d chunks indexados (top_k=%d).", total, TOP_K
    )
    return store.as_retriever(search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def get_rag_chain():
    """
    Ensambla la cadena LCEL completa:

        pregunta -> retriever -> contexto -> prompt -> LLM estructurado

    Devuelve un dict con los documentos recuperados y la salida del modelo,
    para poder construir las referencias a partir de las fuentes reales.
    """
    logger.info("Construyendo la cadena RAG (proveedor: %s)...", get_provider())

    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    llm_estructurado = get_chat_model().with_structured_output(_RespuestaLLM)

    generacion = (prompt | llm_estructurado).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return (
        RunnableParallel(
            documentos=itemgetter("pregunta") | retriever,
            pregunta=itemgetter("pregunta"),
        )
        | RunnablePassthrough.assign(
            contexto=lambda x: _formatear_contexto(x["documentos"])
        )
        | RunnablePassthrough.assign(salida=generacion)
    )


def _construir_referencias(documentos: List[Document]) -> List[Referencia]:
    """Convierte los documentos recuperados en referencias citables."""
    referencias = []
    for doc in documentos:
        texto = " ".join(doc.page_content.split())
        if len(texto) > LARGO_REFERENCIA:
            texto = texto[:LARGO_REFERENCIA].rstrip() + "..."
        fuente = doc.metadata.get("source", "desconocido")
        if "chunk" in doc.metadata:
            fuente = f"{fuente}#chunk{doc.metadata['chunk']}"
        referencias.append(Referencia(fuente=fuente, fragmento=texto))
    return referencias


async def get_rag_response(query: str) -> RAGResponse:
    """
    Responde una pregunta con el pipeline RAG completo, de forma asíncrona.

    Recupera los fragmentos más relevantes del vectorstore, se los pasa al LLM
    como contexto y devuelve un `RAGResponse` validado. Si la información no
    está en los documentos, el modelo lo declara explícitamente en vez de
    alucinar una respuesta.
    """
    logger.info("Pregunta recibida: %r", query)

    chain = get_rag_chain()
    resultado = await chain.ainvoke({"pregunta": query})

    documentos: List[Document] = resultado["documentos"]
    salida: _RespuestaLLM = resultado["salida"]

    fuentes = [d.metadata.get("source", "desconocido") for d in documentos]
    logger.info("Recuperados %d fragmentos de: %s", len(documentos), ", ".join(fuentes))
    logger.info(
        "Generación lista (encontrado_en_contexto=%s).", salida.encontrado_en_contexto
    )

    # Si el modelo dice que la info no está, no tiene sentido citar fuentes:
    # los fragmentos recuperados no respaldan la respuesta.
    referencias = (
        _construir_referencias(documentos) if salida.encontrado_en_contexto else []
    )

    return RAGResponse(
        respuesta=salida.respuesta,
        referencias=referencias,
        encontrado_en_contexto=salida.encontrado_en_contexto,
    )
