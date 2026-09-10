"""
base_client.py

Clase base abstracta que define el "contrato" que todo cliente LLM
debe cumplir, sin importar el proveedor.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """
    Interfaz común para todos los clientes de LLM.
    OpenAIClient, AnthropicClient y GeminiClient heredan de acá.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """
        Genera una respuesta completa (no streaming).
        Debe ser implementado por cada proveedor.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """
        Genera la respuesta en streaming: va emitiendo fragmentos
        de texto (tokens) a medida que llegan, en vez de esperar
        toda la respuesta completa.
        """
        raise NotImplementedError
        yield  # nunca se ejecuta, pero le indica a Python que esto es un generador