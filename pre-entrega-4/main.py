"""
main.py

Demo del sistema RAG: hace una consulta (o varias) contra el índice de
Pinecone usando el recuperador híbrido y muestra la respuesta con sus citas.

Uso:
    python main.py                                  # corre las preguntas de ejemplo
    python main.py "¿Qué es el event loop?"         # una pregunta puntual
    python main.py --modo bm25 "run_in_executor"    # aísla el recuperador léxico
    python main.py --solo-recuperar "TaskGroup"     # muestra los chunks, sin LLM
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from config import TOP_K, setup_logging
from rag import RAGSystem

log = logging.getLogger("main")

PREGUNTAS_DEMO = [
    "¿Cómo ejecuto una función bloqueante sin congelar el programa?",
    "¿Qué pasa si varias tareas de un TaskGroup fallan al mismo tiempo?",
    # Fuera del corpus a propósito: el sistema tiene que decir que no sabe,
    # en vez de inventar. Es la prueba de que el prompt lo tiene agarrado.
    "¿Cuál es la capital de Australia?",
]


def mostrar_documentos(documentos) -> None:
    print(f"\n{len(documentos)} fragmentos recuperados:\n")
    for i, doc in enumerate(documentos, start=1):
        meta = doc.metadata
        score = meta.get("score")
        extra = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        print(f"[{i}] {meta.get('titulo')} — pág. {meta.get('pagina')} "
              f"({meta.get('fuente')}, categoría: {meta.get('categoria')}{extra})")
        print(f"    {doc.page_content[:200].strip().replace(chr(10), ' ')}...\n")


async def responder(sistema: RAGSystem, pregunta: str) -> None:
    print("\n" + "=" * 70)
    print(f"PREGUNTA: {pregunta}")
    print("=" * 70)

    respuesta = await sistema.answer(pregunta)

    print(f"\n{respuesta.respuesta}\n")

    if respuesta.referencias:
        print("Referencias:")
        for ref in respuesta.referencias:
            print(f"  - {ref.titulo} (pág. {ref.pagina}, {ref.fuente}) "
                  f"[{ref.categoria}]")
    else:
        print("(sin referencias: la pregunta no está cubierta por el corpus)")

    print(f"\nencontrado_en_contexto = {respuesta.encontrado_en_contexto}")


async def main_async(args: argparse.Namespace) -> int:
    sistema = RAGSystem(modo=args.modo, k=args.k)

    preguntas = [args.pregunta] if args.pregunta else PREGUNTAS_DEMO

    for pregunta in preguntas:
        if args.solo_recuperar:
            print("\n" + "=" * 70)
            print(f"CONSULTA: {pregunta}   [modo: {args.modo}]")
            print("=" * 70)
            mostrar_documentos(await sistema.aretrieve(pregunta))
        else:
            await responder(sistema, pregunta)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pregunta", nargs="?", help="pregunta a responder")
    parser.add_argument("--modo", choices=("hibrido", "denso", "bm25"),
                        default="hibrido", help="recuperador a usar")
    parser.add_argument("--k", type=int, default=TOP_K, help="documentos a recuperar")
    parser.add_argument("--solo-recuperar", action="store_true",
                        help="muestra los chunks recuperados sin llamar al LLM")
    args = parser.parse_args()

    setup_logging()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("main").error("%s", e)
        sys.exit(1)
