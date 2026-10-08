"""
main.py

CLI del orquestador. Corre una solicitud por el grafo completo y muestra el
flujo de delegación paso a paso.

Uso:
    python main.py                                  # la consulta de demostración
    python main.py "¿Qué nos está rompiendo producción?"
    python main.py --paso-a-paso                    # ver cada nodo a medida que corre
    python main.py --guardar-traza                  # escribir traces/
    python main.py --diagrama                       # imprimir el Mermaid del grafo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from langgraph.errors import GraphRecursionError

from config import (
    MAX_PASOS_SUPERVISOR,
    RECURSION_LIMIT,
    TRACES_DIR,
    get_model_name,
    get_provider,
    setup_logging,
)
from graph import abrir_orquestador, config_thread, diagrama_mermaid, entrada

log = logging.getLogger("main")

# Necesita los dos especialistas y una síntesis: la documentación explica la
# causa, los datos propios dicen cuánto nos cuesta. Ningún agente solo la cierra.
CONSULTA_DEMO = (
    "Nuestros servicios asíncronos se vienen cayendo seguido. "
    "¿Qué dice nuestra documentación técnica sobre las causas más probables, "
    "y qué muestran nuestros propios incidentes del último año? "
    "Quiero saber cuál es la causa dominante y si hay algún caso fuera de lo normal."
)


def imprimir_cabecera(solicitud: str) -> None:
    print("=" * 78)
    print("ORQUESTADOR MULTI-AGENTE")
    print("=" * 78)
    print(f"proveedor:       {get_provider()} / {get_model_name()}")
    print(f"tope supervisor: {MAX_PASOS_SUPERVISOR} pasos")
    print(f"recursion_limit: {RECURSION_LIMIT}")
    print(f"\nSolicitud:\n  {solicitud}\n")
    print("-" * 78)


def describir_paso(nodo: str, actualizacion: dict[str, Any]) -> str:
    """Una línea de qué hizo un nodo, para el seguimiento en vivo."""
    if nodo == "supervisor":
        destino = actualizacion.get("siguiente", "?")
        razon = (actualizacion.get("razonamiento_supervisor") or "")[:90]
        return f"SUPERVISOR -> {destino}\n    {razon}"

    if nodo in ("investigador", "analista"):
        hallazgos = actualizacion.get("hallazgos", [])
        lineas = [f"{nodo.upper()} aportó {len(hallazgos)} hallazgo(s):"]
        lineas += [f"    [{h.herramienta}] {h.resumen}" for h in hallazgos]
        return "\n".join(lineas)

    if nodo == "validador":
        marca = "APROBADO" if actualizacion.get("validacion_aprobada") else "RECHAZADO"
        return f"VALIDADOR -> {marca}\n    {actualizacion.get('motivo_validacion', '')}"

    if nodo == "sintetizador":
        return "SINTETIZADOR -> informe final redactado"

    return f"{nodo}"


async def correr(
    solicitud: str,
    thread_id: str,
    paso_a_paso: bool,
    efimero: bool,
) -> dict[str, Any] | None:
    """Corre el grafo y devuelve el estado final, o None si se cortó por recursión."""
    ruta = ":memory:" if efimero else None

    async with abrir_orquestador(ruta) as orquestador:
        config = config_thread(thread_id)

        try:
            if paso_a_paso:
                # `astream` con stream_mode="updates" entrega lo que devolvió
                # cada nodo a medida que termina: es la forma de ver la
                # delegación en vivo en lugar de esperar al final.
                async for fragmento in orquestador.astream(
                    entrada(solicitud), config=config, stream_mode="updates"
                ):
                    for nodo, actualizacion in fragmento.items():
                        print(f"\n> {describir_paso(nodo, actualizacion)}")
                return await _estado_final(orquestador, config)

            return await orquestador.ainvoke(entrada(solicitud), config=config)

        except GraphRecursionError:
            log.error(
                "Se superó el recursion_limit de %d. El tope semántico de %d pasos "
                "de supervisión debería haber cortado antes: revisá "
                "MAX_PASOS_SUPERVISOR.",
                RECURSION_LIMIT, MAX_PASOS_SUPERVISOR,
            )
            return None


async def _estado_final(orquestador, config) -> dict[str, Any]:
    estado = await orquestador.aget_state(config)
    return estado.values


def imprimir_resultado(final: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("INFORME FINAL")
    print("=" * 78)
    print(f"\n{final.get('informe_final', '(sin informe)')}\n")

    print("-" * 78)
    rubrica = final.get("rubrica_actual")
    print(f"pasos de supervisión:  {final.get('pasos_del_supervisor', 0)}")
    print(f"hallazgos acumulados:  {len(final.get('hallazgos', []))}")
    print(f"validación:            {final.get('motivo_validacion', '?')}")
    if rubrica is not None:
        faltantes = rubrica.faltantes()
        print(f"rúbrica:               "
              f"{'completa' if not faltantes else 'falta ' + ', '.join(faltantes)}")

    print("\nTrazabilidad (qué agente aportó qué):")
    for h in final.get("hallazgos", []):
        print(f"  [{h.agente:<13} / {h.herramienta:<24}] {h.resumen}")
    print()


def guardar_traza(
    solicitud: str, final: dict[str, Any], nombre: str = "flujo-delegacion"
) -> tuple[str, str]:
    """Escribe la traza del flujo de delegación en JSON y en log."""
    TRACES_DIR.mkdir(exist_ok=True)

    rubrica = final.get("rubrica_actual")
    traza = {
        "solicitud": solicitud,
        "proveedor": get_provider(),
        "modelo": get_model_name(),
        "max_pasos_supervisor": MAX_PASOS_SUPERVISOR,
        "recursion_limit": RECURSION_LIMIT,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pasos_del_supervisor": final.get("pasos_del_supervisor", 0),
        "validacion_aprobada": final.get("validacion_aprobada"),
        "motivo_validacion": final.get("motivo_validacion"),
        "rubrica_final": rubrica.model_dump() if rubrica is not None else None,
        "delegacion": [
            {
                "orden": i,
                "agente": h.agente,
                "herramienta": h.herramienta,
                "resumen": h.resumen,
                "momento": h.momento,
                "datos": h.datos,
            }
            for i, h in enumerate(final.get("hallazgos", []), start=1)
        ],
        "conversacion": [
            {
                "autor": getattr(m, "name", None) or m.__class__.__name__,
                "contenido": str(m.content)[:2000],
            }
            for m in final.get("messages", [])
        ],
        "informe_final": final.get("informe_final", ""),
    }

    ruta_json = TRACES_DIR / f"{nombre}.json"
    ruta_json.write_text(
        json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lineas = [
        "=" * 78,
        "FLUJO DE DELEGACIÓN DEL ORQUESTADOR MULTI-AGENTE",
        "=" * 78,
        "",
        f"solicitud:       {solicitud}",
        f"modelo:          {traza['proveedor']} / {traza['modelo']}",
        f"tope supervisor: {traza['max_pasos_supervisor']} pasos",
        f"ejecutado:       {traza['timestamp']}",
        "",
        "-" * 78,
        "DELEGACIÓN (en orden)",
        "-" * 78,
    ]
    for paso in traza["delegacion"]:
        lineas.append(
            f"{paso['orden']}. [{paso['agente']} / {paso['herramienta']}] "
            f"{paso['resumen']}"
        )

    lineas += [
        "",
        "-" * 78,
        "CONVERSACIÓN ENTRE AGENTES",
        "-" * 78,
    ]
    for turno in traza["conversacion"]:
        lineas.append(f"\n[{turno['autor']}]")
        lineas.append(turno["contenido"][:700])

    lineas += [
        "",
        "-" * 78,
        "RÚBRICA FINAL",
        "-" * 78,
        json.dumps(traza["rubrica_final"], ensure_ascii=False, indent=2),
        f"\nvalidación: {traza['motivo_validacion']}",
        "",
        "-" * 78,
        "INFORME FINAL",
        "-" * 78,
        "",
        traza["informe_final"],
        "",
    ]

    ruta_log = TRACES_DIR / f"{nombre}.log"
    ruta_log.write_text("\n".join(lineas), encoding="utf-8")

    return str(ruta_json), str(ruta_log)


async def main_async(args: argparse.Namespace) -> int:
    solicitud = args.solicitud or CONSULTA_DEMO

    imprimir_cabecera(solicitud)

    final = await correr(solicitud, args.thread, args.paso_a_paso, args.efimero)
    if final is None:
        return 1

    imprimir_resultado(final)

    if args.guardar_traza:
        ruta_json, ruta_log = guardar_traza(solicitud, final, args.nombre_traza)
        print(f"Traza guardada en:\n  {ruta_json}\n  {ruta_log}\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solicitud", nargs="?", help="la consulta a procesar")
    parser.add_argument("--thread", default="orquestador", help="thread_id")
    parser.add_argument("--paso-a-paso", action="store_true",
                        help="mostrar cada nodo a medida que corre")
    parser.add_argument("--guardar-traza", action="store_true",
                        help="escribir la traza en traces/")
    parser.add_argument("--nombre-traza", default="flujo-delegacion",
                        help="nombre base de los archivos de traza")
    parser.add_argument("--efimero", action="store_true",
                        help="usar checkpointer en memoria (no persiste)")
    parser.add_argument("--diagrama", action="store_true",
                        help="imprimir el diagrama Mermaid y salir")
    args = parser.parse_args()

    if args.diagrama:
        print(diagrama_mermaid())
        return 0

    setup_logging()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("main").error("%s", e)
        sys.exit(1)
