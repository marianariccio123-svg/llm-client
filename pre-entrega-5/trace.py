"""
trace.py

Corre escenarios guionados contra el agente y guarda la traza del ciclo ReAct
en `traces/`, en dos formatos: un `.json` completo (para inspeccionar) y un
`.log` legible (para leer de corrido).

Cada escenario es una conversación entera sobre un mismo `thread_id`, así que
los turnos siguientes **dependen** de que el checkpointer haya guardado los
anteriores: si la persistencia estuviera rota, el escenario de memoria fallaría
en voz alta en vez de pasar desapercibido.

Uso:
    python trace.py                      # corre los cuatro escenarios
    python trace.py --escenario memoria  # corre uno solo
    python trace.py --listar             # muestra los escenarios disponibles
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import abrir_agente, config_thread, texto_de
from config import RECURSION_LIMIT, TRACES_DIR, get_model_name, get_provider, setup_logging

log = logging.getLogger("trace")


@dataclass
class Escenario:
    """Una conversación guionada, con lo que se espera que demuestre."""

    nombre: str
    titulo: str
    demuestra: str
    thread_id: str
    turnos: list[str] = field(default_factory=list)


ESCENARIOS: list[Escenario] = [
    Escenario(
        nombre="multipaso",
        titulo="Razonamiento multi-paso: tres herramientas encadenadas",
        demuestra=(
            "El agente parte de un nombre y necesita llegar al detalle de un "
            "pedido. Ninguna herramienta sola lo resuelve: tiene que encadenar "
            "buscar_cliente -> buscar_pedidos -> detalle_pedido. La secuencia "
            "no está programada en ningún lado; la arma el modelo leyendo los "
            "docstrings."
        ),
        thread_id="demo-multipaso",
        turnos=[
            "¿Cuánto gastó en total Mariana Riccio y qué productos tenía su último pedido?",
        ],
    ),
    Escenario(
        nombre="memoria",
        titulo="Resiliencia de estado: el agente recuerda el turno anterior",
        demuestra=(
            "Dos turnos sobre el mismo thread_id. El segundo ('¿y de ese "
            "pedido, hay stock?') no menciona ni al cliente ni al pedido: solo "
            "se entiende con el historial, que el agente levanta del "
            "checkpointer de SQLite."
        ),
        thread_id="demo-memoria",
        turnos=[
            "¿Cuántos pedidos tiene Mariana Riccio y cuánto gastó?",
            "¿Y de su último pedido, qué productos eran?",
            "¿Queda stock de alguno de esos dos?",
        ],
    ),
    Escenario(
        nombre="reintento",
        titulo="Ciclo de retorno: la herramienta falla y el agente se corrige",
        demuestra=(
            "El usuario da un SKU que no existe. `consultar_stock` devuelve un "
            "error que incluye el catálogo válido, y el agente reintenta solo "
            "con el código correcto en vez de rendirse. Es el ciclo de retorno "
            "en su forma más pura: el error es información, no un cierre."
        ),
        thread_id="demo-reintento",
        turnos=[
            "¿Cuánto stock queda del teclado? El código creo que es TECLADO-87.",
        ],
    ),
    Escenario(
        nombre="ambiguedad",
        titulo="Pedido de aclaración: el agente no adivina",
        demuestra=(
            "Hay dos clientes llamados Juan. La herramienta devuelve un error "
            "de ambigüedad con los dos candidatos, y el agente pregunta en vez "
            "de elegir uno al azar. En el segundo turno el usuario aclara y el "
            "agente retoma el hilo."
        ),
        thread_id="demo-ambiguedad",
        turnos=[
            "¿Cuántos pedidos tiene Juan?",
            "El de Córdoba.",
        ],
    ),
]


def serializar(mensaje: Any) -> dict[str, Any]:
    """Convierte un mensaje de LangChain en un dict JSON-serializable."""
    base: dict[str, Any] = {"tipo": mensaje.__class__.__name__}

    if isinstance(mensaje, HumanMessage):
        base["rol"] = "usuario"
        base["contenido"] = texto_de(mensaje)

    elif isinstance(mensaje, AIMessage):
        base["rol"] = "agente"
        base["contenido"] = texto_de(mensaje)
        llamadas = getattr(mensaje, "tool_calls", None) or []
        if llamadas:
            base["tool_calls"] = [
                {"herramienta": c["name"], "argumentos": c["args"], "id": c.get("id")}
                for c in llamadas
            ]

    elif isinstance(mensaje, ToolMessage):
        base["rol"] = "herramienta"
        base["herramienta"] = mensaje.name
        base["tool_call_id"] = mensaje.tool_call_id
        # El contenido de una ToolMessage es el dict de la herramienta
        # serializado a string. Se reparsea para que el JSON de la traza quede
        # anidado y legible en vez de ser un string con comillas escapadas.
        try:
            base["resultado"] = json.loads(mensaje.content)
        except (json.JSONDecodeError, TypeError):
            base["resultado"] = str(mensaje.content)

    else:
        base["rol"] = "otro"
        base["contenido"] = texto_de(mensaje)

    return base


def renderizar_log(traza: dict[str, Any]) -> str:
    """Arma la versión legible de la traza, con el formato ReAct de la consigna."""
    lineas: list[str] = [
        "=" * 78,
        traza["titulo"],
        "=" * 78,
        "",
        f"escenario:       {traza['escenario']}",
        f"thread_id:       {traza['thread_id']}",
        f"modelo:          {traza['proveedor']} / {traza['modelo']}",
        f"recursion_limit: {traza['recursion_limit']}",
        f"ejecutado:       {traza['timestamp']}",
        "",
        "Qué demuestra:",
    ]
    lineas += [f"  {linea}" for linea in _envolver(traza["demuestra"], 74)]
    lineas += ["", "-" * 78, ""]

    # Para describir con precisión por qué el agente dejó de pedir herramientas
    # hace falta saber si lo último que vio fue un error.
    ultimo_resultado_fue_error = False

    for mensaje in traza["mensajes"]:
        rol = mensaje["rol"]

        if rol == "usuario":
            lineas += ["", f'Usuario: "{mensaje["contenido"]}"']

        elif rol == "agente":
            for llamada in mensaje.get("tool_calls", []):
                args = ", ".join(f"{k}={v!r}" for k, v in llamada["argumentos"].items())
                lineas.append(
                    f"  -> El agente decide usar la herramienta: "
                    f"{llamada['herramienta']}({args})"
                )
            if mensaje["contenido"]:
                if mensaje.get("tool_calls"):
                    lineas.append(f"  -> El agente razona: {mensaje['contenido']}")
                else:
                    razonamiento = (
                        "la herramienta no alcanzó -> le pide una aclaración al usuario."
                        if ultimo_resultado_fue_error
                        else "ya tiene los datos -> responde."
                    )
                    lineas += [f"  -> El agente razona: {razonamiento}", ""]
                    lineas.append(f"Respuesta: {mensaje['contenido']}")

        elif rol == "herramienta":
            hubo_error = (
                isinstance(mensaje["resultado"], dict) and "error" in mensaje["resultado"]
            )
            ultimo_resultado_fue_error = hubo_error

            resultado = json.dumps(mensaje["resultado"], ensure_ascii=False)
            if len(resultado) > 320:
                resultado = resultado[:317] + "..."
            lineas.append(
                f"  -> La herramienta devuelve{' [ERROR]' if hubo_error else ''}: {resultado}"
            )

    lineas += [
        "",
        "-" * 78,
        f"Turnos del modelo (vueltas del ciclo): {traza['turnos_del_modelo']}",
        f"Llamadas a herramientas:               {traza['llamadas_a_herramientas']}",
        f"Mensajes totales en el estado:         {len(traza['mensajes'])}",
        "",
    ]
    return "\n".join(lineas)


def _envolver(texto: str, ancho: int) -> list[str]:
    """Corta un párrafo en líneas de a lo sumo `ancho` caracteres."""
    palabras, lineas, actual = texto.split(), [], ""
    for palabra in palabras:
        if len(actual) + len(palabra) + 1 > ancho:
            lineas.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    if actual:
        lineas.append(actual)
    return lineas


async def correr_escenario(escenario: Escenario) -> dict[str, Any]:
    """Ejecuta todos los turnos de un escenario y devuelve la traza completa."""
    log.info("--- Escenario '%s' (thread_id=%s) ---", escenario.nombre, escenario.thread_id)

    # Cada escenario usa su propia base efímera: si reusara el archivo de
    # checkpoints, una corrida anterior del mismo thread_id contaminaría el
    # historial y la traza dejaría de ser reproducible.
    async with abrir_agente(":memory:") as agente:
        estado: dict[str, Any] = {}
        for i, turno in enumerate(escenario.turnos, start=1):
            log.info("Turno %d/%d: %s", i, len(escenario.turnos), turno)
            estado = await agente.ainvoke(
                {"messages": [HumanMessage(turno)]},
                config=config_thread(escenario.thread_id),
            )

    mensajes = [serializar(m) for m in estado["messages"]]
    llamadas = sum(len(m.get("tool_calls", [])) for m in mensajes)

    return {
        "escenario": escenario.nombre,
        "titulo": escenario.titulo,
        "demuestra": escenario.demuestra,
        "thread_id": escenario.thread_id,
        "proveedor": get_provider(),
        "modelo": get_model_name(get_provider()),
        "recursion_limit": RECURSION_LIMIT,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turnos_del_usuario": escenario.turnos,
        "turnos_del_modelo": estado.get("turnos_del_modelo", 0),
        "llamadas_a_herramientas": llamadas,
        "mensajes": mensajes,
    }


def guardar(traza: dict[str, Any]) -> tuple[str, str]:
    """Escribe la traza en JSON y en log. Devuelve las dos rutas."""
    TRACES_DIR.mkdir(exist_ok=True)

    ruta_json = TRACES_DIR / f"{traza['escenario']}.json"
    ruta_log = TRACES_DIR / f"{traza['escenario']}.log"

    ruta_json.write_text(
        json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ruta_log.write_text(renderizar_log(traza), encoding="utf-8")

    return str(ruta_json), str(ruta_log)


async def main_async(args: argparse.Namespace) -> int:
    elegidos = ESCENARIOS
    if args.escenario:
        elegidos = [e for e in ESCENARIOS if e.nombre == args.escenario]
        if not elegidos:
            disponibles = ", ".join(e.nombre for e in ESCENARIOS)
            log.error("No existe el escenario '%s'. Hay: %s", args.escenario, disponibles)
            return 1

    resumen: list[tuple[str, int, int]] = []

    for escenario in elegidos:
        traza = await correr_escenario(escenario)
        ruta_json, ruta_log = guardar(traza)
        log.info("Traza guardada en %s y %s", ruta_json, ruta_log)
        resumen.append((
            escenario.nombre,
            traza["turnos_del_modelo"],
            traza["llamadas_a_herramientas"],
        ))

    print("\n" + "=" * 62)
    print("TRAZAS GENERADAS")
    print("=" * 62)
    print(f"{'escenario':<14} {'turnos del modelo':>19} {'llamadas a tools':>18}")
    print("-" * 62)
    for nombre, turnos, llamadas in resumen:
        print(f"{nombre:<14} {turnos:>19} {llamadas:>18}")
    print("-" * 62)
    print(f"Archivos en {TRACES_DIR}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escenario", help="correr un escenario puntual")
    parser.add_argument("--listar", action="store_true", help="listar escenarios")
    parser.add_argument("--solo-logs", action="store_true",
                        help="rehacer los .log desde los .json ya guardados, sin "
                             "llamar al modelo")
    args = parser.parse_args()

    if args.listar:
        for e in ESCENARIOS:
            print(f"{e.nombre:<14} {e.titulo}")
        return 0

    if args.solo_logs:
        # Re-renderiza la parte legible sin gastar una sola llamada a la API.
        # Sirve cuando se ajusta el formato del log y las trazas ya existen.
        for ruta in sorted(TRACES_DIR.glob("*.json")):
            traza = json.loads(ruta.read_text(encoding="utf-8"))
            destino = TRACES_DIR / f"{traza['escenario']}.log"
            destino.write_text(renderizar_log(traza), encoding="utf-8")
            print(f"re-renderizado: {destino.name}")
        return 0

    setup_logging()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("trace").error("%s", e)
        sys.exit(1)
