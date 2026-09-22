"""
main.py

Script de prueba del sistema RAG. Corre dos casos:

  CASO 1 — pregunta respondible: la respuesta está en los documentos indexados.
  CASO 2 — pregunta trampa: el tema no aparece en ningún documento, así que el
           sistema debe decir que no tiene esa información en vez de alucinar.

Si el vectorstore está vacío, corre la ingesta automáticamente antes de probar.

Uso:
    python main.py
    python main.py "tu propia pregunta"
"""

from __future__ import annotations

import asyncio
import sys

from config import setup_logging
from rag import get_rag_response

PREGUNTA_RESPONDIBLE = (
    "¿Qué diferencia hay entre asyncio.gather y asyncio.TaskGroup cuando una "
    "de las tareas falla?"
)

PREGUNTA_TRAMPA = (
    "¿Cuál es el precio mensual del plan empresarial de AWS Lambda y qué "
    "descuento tiene si se paga por año?"
)


def _imprimir_encabezado(titulo: str) -> None:
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


async def probar(titulo: str, pregunta: str) -> None:
    _imprimir_encabezado(titulo)
    print(f"Pregunta: {pregunta}\n")

    respuesta = await get_rag_response(pregunta)

    print("\n--- Salida JSON (RAGResponse) ---")
    print(respuesta.model_dump_json(indent=2))


async def main() -> None:
    setup_logging()

    # Asegura que haya algo indexado; si ya lo hay, no reprocesa nada.
    from ingest import ingest

    ingest()

    if len(sys.argv) > 1:
        await probar("PREGUNTA PERSONALIZADA", " ".join(sys.argv[1:]))
        return

    await probar(
        "CASO 1 — Pregunta respondible (la info está en los documentos)",
        PREGUNTA_RESPONDIBLE,
    )
    await probar(
        "CASO 2 — Pregunta trampa (fuera de los documentos indexados)",
        PREGUNTA_TRAMPA,
    )

    _imprimir_encabezado("RESUMEN")
    print(
        "Caso 1 debe responder con contenido técnico y referencias no vacías.\n"
        "Caso 2 debe responder que no tiene esa información, con "
        "encontrado_en_contexto=false y referencias vacías."
    )


if __name__ == "__main__":
    asyncio.run(main())
