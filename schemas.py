"""
schemas.py

Define los "moldes" de datos que usa todo el proyecto.
Pydantic valida automáticamente que los datos tengan la forma correcta
antes de mandarlos a cualquier API.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Role(str, Enum):
    """Roles posibles de un mensaje en la conversación."""
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """Un único mensaje dentro de una conversación."""
    role: Role
    content: str


class ModelConfig(BaseModel):
    """
    Configuración para una llamada al modelo.
    Pydantic valida rangos y tipos automáticamente.
    """
    model: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    """Respuesta estandarizada, sin importar qué proveedor la generó."""
    content: str
    provider: str
    model: str
    finish_reason: Optional[str] = None