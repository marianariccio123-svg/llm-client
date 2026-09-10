"""
openai_client.py

Implementación del cliente para OpenAI usando el SDK oficial async (AsyncOpenAI).
"""

from typing import AsyncGenerator, List

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    """Cliente para modelos de OpenAI (GPT-4o, GPT-4, etc.)."""

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)

    def _to_openai_messages(self, messages: List[ChatMessage]) -> list[dict]:
        """Convierte nuestros ChatMessage (Pydantic) al formato que espera OpenAI."""
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (sin streaming), con reintentos ante fallos temporales."""
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=self._to_openai_messages(messages),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                choice = response.choices[0]
                return ModelResponse(
                    content=choice.message.content or "",
                    provider="openai",
                    model=self.config.model,
                    finish_reason=choice.finish_reason,
                )
            except RateLimitError as e:
                last_error = f"Límite de tasa excedido: {e}"
            except APIConnectionError as e:
                last_error = f"Error de conexión: {e}"
            except APIError as e:
                # Ej: API key inválida. Reintentar no lo va a arreglar, devolvemos ya.
                return ModelResponse(
                    content="",
                    provider="openai",
                    model=self.config.model,
                    finish_reason=f"error: {e}",
                )

        return ModelResponse(
            content="",
            provider="openai",
            model=self.config.model,
            finish_reason=f"error: {last_error}",
        )

    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Genera la respuesta token por token, a medida que llega de la API."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self._to_openai_messages(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"[Error de streaming: {e}]"