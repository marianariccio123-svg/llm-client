"""
main.py

Script de validación: prueba el cliente en modo normal y en modo streaming.
"""

import asyncio
from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Role

load_dotenv()  # carga las variables del archivo .env


async def main():
    config = ModelConfig(
        model="gemini-flash-lite-latest",
        temperature=0.7,
        max_tokens=2048,
    )

    manager = AsyncLLMManager(provider="gemini", config=config)

    messages = [
        ChatMessage(
            role=Role.system,
            content=(
                "Respondé en un máximo de 2 oraciones cortas, en texto plano, "
                "sin usar Markdown, sin títulos ni listas."
            ),
        ),
        ChatMessage(role=Role.user, content="¿Qué es la entropía?"),
    ]

    # --- Modo normal (respuesta completa) ---
    print("=== MODO NORMAL ===")
    response = await manager.generate(messages)
    print(f"[{response.provider} | {response.model}]")
    print(response.content)
    print(f"(finish_reason: {response.finish_reason})\n")

    # --- Modo streaming (fragmento por fragmento) ---
    print("=== MODO STREAMING ===")
    async for chunk in manager.generate_stream(messages):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())