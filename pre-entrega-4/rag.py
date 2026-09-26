"""
rag.py

Recuperador híbrido y sistema RAG.

El núcleo es `RAGSystem`, que encapsula un `EnsembleRetriever` de LangChain
combinando dos recuperadores que fallan de maneras distintas:

- **Denso (Pinecone).** Entiende paráfrasis: "¿cómo evito que se me cuelgue el
  programa?" recupera la sección de código bloqueante aunque no comparta una
  sola palabra con ella. A cambio, diluye los identificadores exactos: para el
  embedding, `TaskGroup` y `gather` viven casi en el mismo punto.
- **Léxico (BM25).** Hace lo contrario. Si buscás `run_in_executor`, lo
  encuentra literal, con su rareza ponderada por IDF. Pero una pregunta bien
  formulada sin las palabras exactas del documento le pasa por al lado.

En documentación técnica, donde las consultas mezclan lenguaje natural con
nombres propios de la API, la unión de los dos rinde mejor que cualquiera solo.
`evaluate.py` lo mide en vez de asumirlo.

Nota de implementación: se usa el **SDK nativo de Pinecone** envuelto en un
`BaseRetriever` propio en lugar de `PineconeVectorStore`. La razón es
concreta: `langchain-pinecone` fija `numpy<2` y no tiene release compatible con
Python 3.14, que es el intérprete de este repo. El SDK nativo es la alternativa
que la propia consigna contempla, y el wrapper de 30 líneas deja el retriever
100% compatible con `EnsembleRetriever`.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Iterable, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import (
    BM25_CORPUS_PATH,
    ENSEMBLE_WEIGHTS,
    NAMESPACE,
    TOP_K,
    get_chat_model,
    get_embeddings,
    get_index,
)
from schemas import DocumentoRecuperado, RAGResponse

log = logging.getLogger("rag")

Modo = Literal["hibrido", "denso", "bm25"]

SIN_CONTEXTO = "No tengo esa información en los documentos disponibles."

# Tokens "de código": permite puntos y guiones bajos, así `asyncio.to_thread` y
# `run_in_executor` sobreviven enteros en lugar de partirse en pedazos que BM25
# consideraría palabras comunes. Incluye los acentos y la ñ: el corpus está en
# español y sin ellos "función" se partiría en "funci" + "n".
_CARACTER = r"[a-záéíóúüñ0-9_]"
TOKEN_RE = re.compile(rf"{_CARACTER}+(?:\.{_CARACTER}+)*")


def tokenizar(texto: str) -> list[str]:
    """
    Tokenizador para BM25.

    El default de `BM25Retriever` es `str.split`, que deja la puntuación pegada
    (`gather(),` no matchea `gather`). Este normaliza a minúsculas, conserva
    los identificadores con punto y además los indexa por sus partes, para que
    `asyncio.to_thread` también responda a la consulta `to_thread`.
    """
    tokens: list[str] = []
    for token in TOKEN_RE.findall(texto.lower()):
        tokens.append(token)
        if "." in token:
            tokens.extend(parte for parte in token.split(".") if parte)
    return tokens


def cargar_documentos_locales() -> list[Document]:
    """Lee el JSONL que dejó la ingesta y lo convierte en `Document`s."""
    if not BM25_CORPUS_PATH.exists():
        raise RuntimeError(
            f"Falta {BM25_CORPUS_PATH.name}, que genera la ingesta. "
            f"Corré primero: python ingest.py"
        )
    documentos: list[Document] = []
    with BM25_CORPUS_PATH.open(encoding="utf-8") as f:
        for linea in f:
            registro = json.loads(linea)
            texto = registro.pop("text")
            documentos.append(Document(page_content=texto, metadata=registro))
    return documentos


class PineconeRetriever(BaseRetriever):
    """
    Retriever denso sobre un índice Serverless de Pinecone, vía SDK nativo.

    Implementa la interfaz de LangChain (`BaseRetriever`), así que se enchufa
    directo en un `EnsembleRetriever` como cualquier otro.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    embeddings: Any
    namespace: str = NAMESPACE
    k: int = TOP_K
    # Filtro de metadata opcional, p.ej. {"categoria": {"$eq": "testing"}}.
    filtro: dict[str, Any] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        respuesta = self.index.query(
            vector=vector,
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
            filter=self.filtro,
        )

        documentos: list[Document] = []
        for match in respuesta.get("matches", []):
            metadata = dict(match.get("metadata") or {})
            texto = metadata.pop("text", "")
            metadata["score"] = match.get("score")
            metadata["_id"] = match.get("id")
            documentos.append(Document(page_content=texto, metadata=metadata))
        return documentos


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un asistente técnico que responde ÚNICAMENTE con la información "
        "del CONTEXTO que te pasan.\n"
        "Reglas estrictas:\n"
        "1. No uses conocimiento propio ni completes con lo que suponés.\n"
        "2. Si el contexto no alcanza para responder, contestá exactamente: "
        f"'{SIN_CONTEXTO}'\n"
        "3. Respondé en español, de forma concreta y breve (máximo 6 oraciones).\n"
        "4. No inventes nombres de archivos ni de funciones que no estén en el "
        "contexto.",
    ),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])


class RAGSystem:
    """
    Sistema RAG completo: recuperación híbrida + generación con citas.

    Todo es perezoso: construir un `RAGSystem` no toca la red. El índice, los
    embeddings y el LLM se resuelven en la primera consulta real.

    Parámetros
    ----------
    modo:
        `"hibrido"` (default) combina BM25 y Pinecone con un `EnsembleRetriever`.
        `"denso"` y `"bm25"` aíslan cada mitad; existen para que `evaluate.py`
        pueda comparar los tres y mostrar que el híbrido aporta algo.
    k:
        Cantidad de documentos a devolver (5 por consigna).
    """

    def __init__(self, modo: Modo = "hibrido", k: int = TOP_K) -> None:
        if modo not in ("hibrido", "denso", "bm25"):
            raise ValueError(f"Modo '{modo}' inválido: usá hibrido | denso | bm25.")
        self.modo: Modo = modo
        self.k = k
        self._retriever = None

    # --- construcción perezosa ---

    def _construir_bm25(self):
        from langchain_community.retrievers import BM25Retriever

        documentos = cargar_documentos_locales()
        retriever = BM25Retriever.from_documents(documentos, preprocess_func=tokenizar)
        retriever.k = self.k
        log.info("BM25 listo sobre %d chunks locales.", len(documentos))
        return retriever

    def _construir_denso(self) -> PineconeRetriever:
        return PineconeRetriever(
            index=get_index(),
            embeddings=get_embeddings(),
            namespace=NAMESPACE,
            k=self.k,
        )

    @property
    def retriever(self):
        """El recuperador configurado, construido una sola vez."""
        if self._retriever is not None:
            return self._retriever

        if self.modo == "bm25":
            self._retriever = self._construir_bm25()
        elif self.modo == "denso":
            self._retriever = self._construir_denso()
        else:
            from langchain_classic.retrievers import EnsembleRetriever

            # EnsembleRetriever fusiona con Reciprocal Rank Fusion: cada
            # documento suma peso_i / (60 + rank_i) por cada lista donde
            # aparece. Lo interesante es que NO compara los scores crudos de
            # BM25 con las similitudes coseno de Pinecone, que viven en escalas
            # incomparables; compara posiciones. Por eso un documento que sale
            # 2º en las dos listas le gana a uno que sale 1º en una sola.
            self._retriever = EnsembleRetriever(
                retrievers=[self._construir_bm25(), self._construir_denso()],
                weights=list(ENSEMBLE_WEIGHTS),
            )
            log.info("EnsembleRetriever listo (pesos BM25/denso = %s).",
                     ENSEMBLE_WEIGHTS)

        return self._retriever

    # --- recuperación ---

    def retrieve(self, consulta: str) -> list[Document]:
        """
        Devuelve los top-k documentos para la consulta.

        `EnsembleRetriever` fusiona las dos listas y devuelve la unión
        reordenada, que puede tener más de k elementos; el recorte final a k
        se hace acá.
        """
        documentos = self.retriever.invoke(consulta)
        return documentos[: self.k]

    async def aretrieve(self, consulta: str) -> list[Document]:
        """Versión asíncrona de `retrieve`."""
        documentos = await self.retriever.ainvoke(consulta)
        return documentos[: self.k]

    # --- generación ---

    @staticmethod
    def _formatear_contexto(documentos: Iterable[Document]) -> str:
        bloques = []
        for i, doc in enumerate(documentos, start=1):
            meta = doc.metadata
            bloques.append(
                f"[{i}] {meta.get('titulo', '?')} "
                f"(fuente: {meta.get('fuente', '?')}, pág. {meta.get('pagina', '?')})\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(bloques)

    @staticmethod
    def _a_referencias(documentos: Iterable[Document], modo: Modo) -> list[DocumentoRecuperado]:
        origen = {"hibrido": "hibrido", "denso": "denso", "bm25": "bm25"}[modo]
        return [
            DocumentoRecuperado(
                doc_id=d.metadata.get("doc_id", "?"),
                titulo=d.metadata.get("titulo", "?"),
                fuente=d.metadata.get("fuente", "?"),
                categoria=d.metadata.get("categoria", "?"),
                pagina=int(d.metadata.get("pagina", 1)),
                fragmento=d.page_content[:300].strip(),
                origen=origen,
            )
            for d in documentos
        ]

    async def answer(self, pregunta: str) -> RAGResponse:
        """
        Recupera, genera y devuelve una `RAGResponse` validada.

        Las `referencias` salen de la metadata de los documentos recuperados,
        no del texto del modelo: el LLM no tiene forma de inventarlas.
        """
        documentos = await self.aretrieve(pregunta)

        if not documentos:
            return RAGResponse(
                pregunta=pregunta,
                respuesta=SIN_CONTEXTO,
                referencias=[],
                encontrado_en_contexto=False,
            )

        cadena = PROMPT | get_chat_model() | StrOutputParser()
        texto = (await cadena.ainvoke({
            "contexto": self._formatear_contexto(documentos),
            "pregunta": pregunta,
        })).strip()

        encontrado = SIN_CONTEXTO.rstrip(".").lower() not in texto.lower()

        return RAGResponse(
            pregunta=pregunta,
            respuesta=texto,
            # Si el modelo dijo que no hay información, mostrar "referencias"
            # sería engañoso: esos chunks no respaldan nada.
            referencias=self._a_referencias(documentos, self.modo) if encontrado else [],
            encontrado_en_contexto=encontrado,
        )


@lru_cache(maxsize=3)
def get_rag_system(modo: Modo = "hibrido", k: int = TOP_K) -> RAGSystem:
    """Instancia cacheada por modo, para no reconstruir BM25 en cada consulta."""
    return RAGSystem(modo=modo, k=k)
