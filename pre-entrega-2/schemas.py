"""
schemas.py

Define la estructura de salida validada que el pipeline debe producir
a partir de un texto técnico sin procesar.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class NivelCriticidad(str, Enum):
    """Nivel de criticidad detectado en el texto técnico."""
    baja = "baja"
    media = "media"
    alta = "alta"


class TechExtraction(BaseModel):
    """
    Resultado validado de extraer entidades técnicas de un texto
    (por ejemplo, una descripción de arquitectura o un log de error).
    """

    tecnologias: List[str] = Field(
        ...,
        min_length=1,
        description="Lista de tecnologías mencionadas en el texto (ej: FastAPI, Redis, PostgreSQL).",
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description="Qué tan crítico es el problema o componente descrito: baja, media o alta.",
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=10,
        description="Resumen breve y técnico de la situación descrita en el texto.",
    )