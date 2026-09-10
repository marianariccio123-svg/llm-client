"""
gemini_client.py

Implementación del cliente para Gemini usando el SDK oficial async (google-genai).
Este es el cliente que realmente vamos a poder probar con una API key gratuita.
"""

import asyncio
from typing import AsyncGenerator, List, Optional

from google import genai
from google.genai import types, errors

from base_client import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse, Role


class GeminiClient(BaseLLMClient):
    """Cliente para modelos de Google Gemini."""

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = genai.Client(api_key=api_key)

    def _to_gemini_contents(
        self, messages: List[ChatMessage]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Gemini pide el system prompt aparte (no como mensaje) y usa
        'model' en vez de 'assistant' como nombre de rol.
        """
        system_prompt = None
        contents = []
        for m in messages:
            if m.role == Role.system:
                system_prompt = m.content
            else:
                gemini_role = "model" if m.role == Role.assistant else "user"
                contents.append({"role": gemini_role, "parts": [{"text": m.content}]})
        return system_prompt, contents

    def _build_config(self, system_prompt: Optional[str]) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    @staticmethod
    async def _backoff_wait(attempt: int) -> None:
        """Espera con backoff exponencial: 1s, 2s, 4s, 8s..."""
        await asyncio.sleep(2 ** attempt)

    @staticmethod
    def _is_daily_quota_exhausted(e: Exception) -> bool:
        """
        Distingue un límite DIARIO agotado (no tiene sentido reintentar,
        no se libera en segundos) de un error temporal de tasa/saturación
        (sí conviene reintentar).
        """
        return "PerDay" in str(e)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (sin streaming), con reintentos ante fallos temporales."""
        system_prompt, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(system_prompt)
        max_retries = 3
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=gen_config,
                )
                finish_reason = None
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)

                return ModelResponse(
                    content=response.text or "",
                    provider="gemini",
                    model=self.config.model,
                    finish_reason=finish_reason,
                )
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code == 429 and self._is_daily_quota_exhausted(e):
                    # Límite diario agotado: reintentar no sirve, fallamos rápido.
                    return ModelResponse(
                        content="",
                        provider="gemini",
                        model=self.config.model,
                        finish_reason=f"error: cuota diaria agotada: {e}",
                    )
                elif code in (429, 503):
                    # Saturación temporal o rate-limit de corto plazo: sí conviene reintentar.
                    last_error = f"Error temporal ({code}): {e}"
                    if attempt < max_retries:
                        await self._backoff_wait(attempt)
                else:
                    # Ej: API key inválida (401/403). Reintentar no lo va a arreglar.
                    return ModelResponse(
                        content="",
                        provider="gemini",
                        model=self.config.model,
                        finish_reason=f"error: {e}",
                    )
            except Exception as e:
                last_error = f"Error inesperado: {e}"
                if attempt < max_retries:
                    await self._backoff_wait(attempt)

        return ModelResponse(
            content="",
            provider="gemini",
            model=self.config.model,
            finish_reason=f"error: {last_error}",
        )

    async def generate_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """
        Genera la respuesta token por token, a medida que llega de la API.
        Reintenta con backoff exponencial solo ante fallos temporales
        (no ante límite diario agotado).
        """
        system_prompt, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(system_prompt)
        max_retries = 3
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=gen_config,
                )
                async for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return  # terminó bien
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code == 429 and self._is_daily_quota_exhausted(e):
                    yield f"[Error: cuota diaria agotada: {e}]"
                    return
                elif code in (429, 503):
                    last_error = f"Error temporal ({code}): {e}"
                    if attempt < max_retries:
                        await self._backoff_wait(attempt)
                else:
                    yield f"[Error: {e}]"
                    return
            except Exception as e:
                last_error = f"Error inesperado: {e}"
                if attempt < max_retries:
                    await self._backoff_wait(attempt)

        yield f"[Error de streaming tras {max_retries} reintentos: {last_error}]"