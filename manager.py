"""
manager.py

AsyncLLMManager: selecciona y crea el cliente correcto (OpenAI, Anthropic o Gemini)
según una variable de configuración, y expone una interfaz única para generar
respuestas, sin que el resto del código necesite saber qué proveedor hay detrás.
"""

import os
from typing import AsyncGenerator, List

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse
from openai_client import OpenAIClient
from anthropic_client import AnthropicClient
from gemini_client import GeminiClient


class AsyncLLMManager:
    """Punto de entrada único para hablar con cualquier proveedor de LLM."""

    # Mapea el nombre del proveedor -> (clase que lo implementa, nombre de la env var con su API key)
    _PROVIDERS = {
        "openai": (OpenAIClient, "OPENAI_API_KEY"),
        "anthropic": (AnthropicClient, "ANTHROPIC_API_KEY"),
        "gemini": (GeminiClient, "GEMINI_API_KEY"),
    }

    def __init__(self, provider: str, config: ModelConfig):
        provider = provider.lower()
        if provider not in self._PROVIDERS:
            raise ValueError(
                f"Proveedor '{provider}' no soportado. "
                f"Opciones válidas: {list(self._PROVIDERS.keys())}"
            )

        client_class, env_var_name = self._PROVIDERS[provider]
        api_key = os.getenv(env_var_name)
        if not api_key:
            raise ValueError(
                f"Falta la variable de entorno {env_var_name} en el archivo .env"
            )

        self.provider = provider
        self._client: BaseLLMClient = client_class(config, api_key)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Respuesta completa, delegando al cliente concreto ya instanciado."""
        return await self._client.generate(messages)

    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Streaming, delegando al cliente concreto ya instanciado."""
        async for chunk in self._client.generate_stream(messages):
            yield chunk