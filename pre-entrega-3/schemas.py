"""
schemas.py

Modelos Pydantic de salida del sistema RAG.
"""

from typing import List

from pydantic import BaseModel, Field


class Referencia(BaseModel):
    """Un fragmento recuperado que se usó como contexto para responder."""

    fuente: str = Field(
        ...,
        description="Archivo del que proviene el fragmento (ej: 02-tasks-y-concurrencia.md).",
    )
    fragmento: str = Field(
        ...,
        description="Extracto del texto recuperado que respalda la respuesta.",
    )


class RAGResponse(BaseModel):
    """
    Respuesta final del pipeline RAG, ya validada.

    `referencias` se completa con los documentos que devolvió el retriever, no
    con lo que el modelo dice haber usado: así las citas no se pueden alucinar.
    """

    respuesta: str = Field(
        ...,
        min_length=1,
        description="Respuesta generada a partir del contexto recuperado.",
    )
    referencias: List[Referencia] = Field(
        default_factory=list,
        description="Fragmentos fuente recuperados del vectorstore.",
    )
    encontrado_en_contexto: bool = Field(
        ...,
        description=(
            "False cuando el modelo indicó que la respuesta no está en los "
            "documentos indexados (pregunta fuera de dominio)."
        ),
    )
