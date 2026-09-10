"""
anthropic_client.py

Implementación del cliente para Anthropic usando el SDK oficial async (AsyncAnthropic).
"""

from typing import AsyncGenerator, List, Optional

from anthropic import AsyncAnthropic, APIError, RateLimitError, APIConnectionError

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Role


class AnthropicClient(BaseLLMClient):
    """Cliente para modelos de Anthropic (Claude)."""

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=api_key)

    def _split_system_and_messages(
        self, messages: List[ChatMessage]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Anthropic pide el system prompt aparte, no como un mensaje más.
        Esta función separa: (system_prompt, resto_de_mensajes_en_formato_dict)
        """
        system_prompt = None
        chat_messages = []
        for m in messages:
            if m.role == Role.system:
                system_prompt = m.content
            else:
                chat_messages.append({"role": m.role.value, "content": m.content})
        return system_prompt, chat_messages

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (sin streaming), con reintentos ante fallos temporales."""
        system_prompt, chat_messages = self._split_system_and_messages(messages)
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt or "",
                    messages=chat_messages,
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                return ModelResponse(
                    content=text,
                    provider="anthropic",
                    model=self.config.model,
                    finish_reason=response.stop_reason,
                )
            except RateLimitError as e:
                last_error = f"Límite de tasa excedido: {e}"
            except APIConnectionError as e:
                last_error = f"Error de conexión: {e}"
            except APIError as e:
                return ModelResponse(
                    content="",
                    provider="anthropic",
                    model=self.config.model,
                    finish_reason=f"error: {e}",
                )

        return ModelResponse(
            content="",
            provider="anthropic",
            model=self.config.model,
            finish_reason=f"error: {last_error}",
        )

    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Genera la respuesta token por token, a medida que llega de la API."""
        system_prompt, chat_messages = self._split_system_and_messages(messages)
        try:
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=chat_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"[Error de streaming: {e}]"