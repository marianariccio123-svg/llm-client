"""
main.py

CLI del agente. Mantiene una conversación sobre un `thread_id`, que es lo que
le da memoria entre turnos y entre ejecuciones del programa.

Uso:
    python main.py                                  # chat interactivo
    python main.py "¿Cuántos pedidos tiene Mariana Riccio?"
    python main.py --thread soporte-42 "¿y el último?"
    python main.py --thread soporte-42 --historial   # ver lo que recuerda
    python main.py --verboso "..."                   # mostrar el ciclo ReAct

Los checkpoints se guardan en `data/checkpoints.sqlite`, así que un
`--thread soporte-42` de hoy sigue existiendo mañana.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent import abrir_agente, config_thread, historial, texto_de
from config import RECURSION_LIMIT, setup_logging

log = logging.getLogger("main")


def mostrar_ciclo(mensajes: list, desde: int) -> None:
    """Imprime el ciclo ReAct de los mensajes nuevos (de `desde` en adelante)."""
    for mensaje in mensajes[desde:]:
        if isinstance(mensaje, AIMessage):
            for llamada in getattr(mensaje, "tool_calls", None) or []:
                args = ", ".join(f"{k}={v!r}" for k, v in llamada["args"].items())
                print(f"  -> herramienta: {llamada['name']}({args})")
        elif isinstance(mensaje, ToolMessage):
            contenido = str(mensaje.content)
            if len(contenido) > 200:
                contenido = contenido[:197] + "..."
            print(f"  <- {mensaje.name}: {contenido}")


async def preguntar(agente, thread_id: str, pregunta: str, verboso: bool) -> str:
    """Manda un turno al agente y devuelve su respuesta final."""
    previos = len(await historial(agente, thread_id))

    try:
        estado = await agente.ainvoke(
            {"messages": [HumanMessage(pregunta)]},
            config=config_thread(thread_id),
        )
    except GraphRecursionError:
        # El techo existe justamente para esto: el agente entró en bucle y el
        # grafo corta en vez de seguir facturando llamadas a la API.
        return (
            f"[El agente superó el límite de {RECURSION_LIMIT} pasos sin llegar a "
            f"una respuesta. Probá reformulando la pregunta, o subí "
            f"RECURSION_LIMIT en el .env si el caso realmente la necesita.]"
        )

    if verboso:
        mostrar_ciclo(estado["messages"], previos)

    return texto_de(estado["messages"][-1])


async def modo_interactivo(agente, thread_id: str, verboso: bool) -> None:
    print(f"Agente de pedidos — thread_id: {thread_id}")
    print("Escribí tu pregunta, o 'salir' para terminar.\n")

    while True:
        try:
            pregunta = input("vos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pregunta:
            continue
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break

        respuesta = await preguntar(agente, thread_id, pregunta, verboso)
        print(f"\nagente> {respuesta}\n")


async def main_async(args: argparse.Namespace) -> int:
    async with abrir_agente() as agente:
        if args.historial:
            mensajes = await historial(agente, args.thread)
            if not mensajes:
                print(f"El thread '{args.thread}' no tiene nada guardado todavía.")
                return 0
            print(f"Historial del thread '{args.thread}' ({len(mensajes)} mensajes):\n")
            for mensaje in mensajes:
                etiqueta = {
                    "HumanMessage": "usuario",
                    "AIMessage": "agente",
                    "ToolMessage": "tool",
                }.get(mensaje.__class__.__name__, "otro")

                texto = texto_de(mensaje)
                if not texto:
                    # Un AIMessage que solo pide herramientas no tiene texto.
                    # Mostrar su `.content` crudo imprimiría un "[]" inútil;
                    # lo que importa de ese mensaje son las llamadas.
                    llamadas = getattr(mensaje, "tool_calls", None) or []
                    texto = (
                        "(pide herramientas: "
                        + ", ".join(f"{c['name']}({c['args']})" for c in llamadas)
                        + ")"
                    ) if llamadas else "(sin contenido)"

                print(f"[{etiqueta:>7}] {texto[:160].replace(chr(10), ' ')}")
            return 0

        if args.pregunta:
            respuesta = await preguntar(agente, args.thread, args.pregunta, args.verboso)
            print(f"\n{respuesta}\n")
            return 0

        await modo_interactivo(agente, args.thread, args.verboso)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pregunta", nargs="?", help="pregunta puntual")
    parser.add_argument("--thread", default="cli", help="thread_id de la conversación")
    parser.add_argument("--verboso", action="store_true",
                        help="mostrar las llamadas a herramientas")
    parser.add_argument("--historial", action="store_true",
                        help="mostrar lo que el agente recuerda del thread")
    args = parser.parse_args()

    setup_logging(logging.WARNING if not args.verboso else logging.INFO)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("main").error("%s", e)
        sys.exit(1)
