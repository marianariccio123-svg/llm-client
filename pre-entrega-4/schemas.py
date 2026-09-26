"""
schemas.py

Modelos Pydantic de la Pre-entrega 4. Cumplen dos funciones distintas:

1. `ChunkMetadata` define el **contrato de la metadata** que viaja a Pinecone.
   Validarlo antes del upsert evita el clásico "subí 300 vectores y recién
   ahora me doy cuenta de que la mitad no tiene `categoria`", que en un índice
   remoto se paga caro: hay que borrar y re-embeber todo.
2. `DocumentoRecuperado` y `RAGResponse` describen la salida del sistema. Las
   referencias se construyen a partir de la metadata que devuelve el
   recuperador, nunca a partir del texto del LLM, así que no se pueden
   alucinar.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """Metadata avanzada que se guarda junto a cada vector en Pinecone."""

    doc_id: str = Field(..., description="Identificador estable del documento fuente")
    fuente: str = Field(..., description="Nombre del archivo de origen")
    titulo: str = Field(..., description="Título humano del documento")
    categoria: str = Field(..., description="Etiqueta temática para filtrado")
    pagina: int = Field(..., ge=1, description="Nº de sección/página dentro del documento")
    chunk_index: int = Field(..., ge=0, description="Posición del chunk dentro del documento")
    n_tokens: int = Field(..., gt=0, description="Tamaño real del chunk en tokens")
    # El texto viaja DENTRO de la metadata de Pinecone a propósito: así una
    # búsqueda devuelve el contenido listo para armar el prompt, sin una
    # segunda consulta a una base relacional.
    text: str = Field(..., min_length=1, description="Texto original del chunk")

    @field_validator("text")
    @classmethod
    def _limite_de_pinecone(cls, v: str) -> str:
        # Pinecone limita la metadata de cada vector a 40 KB. Un chunk de 500
        # tokens está muy por debajo, pero conviene fallar acá y no en el
        # upsert, que rechaza el batch entero.
        if len(v.encode("utf-8")) > 38_000:
            raise ValueError(
                "El chunk supera el límite de metadata de Pinecone (40 KB). "
                "Bajá CHUNK_SIZE."
            )
        return v


class DocumentoRecuperado(BaseModel):
    """Un chunk devuelto por el recuperador, con su procedencia."""

    doc_id: str
    titulo: str
    fuente: str
    categoria: str
    pagina: int
    fragmento: str = Field(..., description="Texto del chunk (recortado para mostrar)")
    origen: Literal["hibrido", "denso", "bm25"] = "hibrido"


class RAGResponse(BaseModel):
    """Respuesta final del sistema, con sus referencias verificables."""

    pregunta: str
    respuesta: str
    referencias: list[DocumentoRecuperado] = Field(default_factory=list)
    encontrado_en_contexto: bool = Field(
        ...,
        description="False cuando el modelo declaró que el corpus no cubre la pregunta",
    )


class PreguntaGolden(BaseModel):
    """Una entrada del Golden Set de evaluación."""

    pregunta: str
    documento_id_esperado: str = Field(
        ..., description="doc_id que DEBE aparecer en el top-k (base de Recall@k)"
    )
    documentos_relevantes: list[str] = Field(
        default_factory=list,
        description=(
            "doc_ids que cuentan como útiles para esta pregunta. Base de "
            "Precision@k. Siempre incluye a `documento_id_esperado`."
        ),
    )

    @field_validator("documentos_relevantes")
    @classmethod
    def _incluye_el_esperado(cls, v: list[str], info) -> list[str]:
        esperado = info.data.get("documento_id_esperado")
        if esperado and esperado not in v:
            v = [esperado, *v]
        return v


class MetricasModo(BaseModel):
    """Resultado agregado de evaluar un modo de recuperación."""

    modo: str
    k: int
    recall_at_k: float = Field(..., ge=0.0, le=1.0)
    precision_at_k: float = Field(..., ge=0.0, le=1.0)
    precision_normalizada: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Precision@k dividida por la máxima alcanzable dado cuántos chunks "
            "relevantes existen realmente en el índice."
        ),
    )
    mrr: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Mean Reciprocal Rank: 1/posición del documento esperado, promediado. "
            "Métrica extra: con un corpus chico Recall@5 satura en 1.0 y deja de "
            "distinguir modos, mientras que MRR sí premia traerlo primero."
        ),
    )
    preguntas: int = Field(..., gt=0)
