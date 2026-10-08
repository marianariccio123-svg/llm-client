"""
scripts/percentiles_langsmith.py

Calcula los percentiles de latencia sobre las trazas que LangSmith ya tiene
registradas.

Por qué hace falta
------------------
La consigna pide capturar la **latencia p95** del dashboard. El panel
**Trace Latency** de LangSmith grafica percentiles de latencia, pero en su
versión actual dibuja **P50 y P99**: no hay selector de P95.

Este script no mide nada por su cuenta ni abre una medición paralela. Lee **las
mismas trazas** que alimentan ese gráfico, usando la API de LangSmith, y les
calcula el percentil que falta.

La verificación de que es el mismo conjunto de datos es que el P50 que calcula
acá coincide con el que muestra el panel. Si no coincidiera, alguno de los dos
estaría mirando otra cosa.

Uso:
    python scripts/percentiles_langsmith.py
    python scripts/percentiles_langsmith.py --proyecto otro-proyecto
    python scripts/percentiles_langsmith.py --json percentiles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROYECTO_TRAZAS  # noqa: E402

NOMBRE_RAIZ = "auditoria_job"


def percentil(valores: list[float], p: float) -> float | None:
    """
    El percentil p por interpolación lineal.

    Es el mismo método que usa `prueba_de_carga.py`, para que los dos números
    sean comparables. Con pocas muestras la interpolación importa: tomar "el
    valor en la posición 0.95*n" devolvería el máximo y sobreestimaría.
    """
    if not valores:
        return None
    if len(valores) == 1:
        return round(valores[0], 2)

    ordenados = sorted(valores)
    posicion = (len(ordenados) - 1) * p
    inferior = int(posicion)
    superior = min(inferior + 1, len(ordenados) - 1)
    peso = posicion - inferior
    return round(ordenados[inferior] * (1 - peso) + ordenados[superior] * peso, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proyecto", default=PROYECTO_TRAZAS,
                        help="proyecto de LangSmith a consultar")
    parser.add_argument("--limite", type=int, default=100,
                        help="máximo de ejecuciones a traer (tope de la API: 100)")
    parser.add_argument("--json", help="archivo donde guardar el resultado")
    args = parser.parse_args()

    try:
        from langsmith import Client
    except ImportError:
        print("Falta el paquete `langsmith`: pip install langsmith")
        return 1

    cliente = Client()

    try:
        ejecuciones = [
            r
            for r in cliente.list_runs(
                project_name=args.proyecto,
                filter=f'eq(name, "{NOMBRE_RAIZ}")',
                limit=min(args.limite, 100),
            )
            # Solo las raíces: cada una es un job completo. Incluir los spans
            # hijos mezclaría la latencia de una llamada al modelo con la de
            # una ejecución entera.
            if r.parent_run_id is None and r.end_time
        ]
    except Exception as e:
        print(f"No se pudo consultar LangSmith: {type(e).__name__}: {e}")
        print("Revisá que LANGSMITH_API_KEY esté en el .env de la raíz.")
        return 1

    if not ejecuciones:
        print(f"No hay ejecuciones '{NOMBRE_RAIZ}' en el proyecto '{args.proyecto}'.")
        print("Corré primero: python scripts/prueba_de_carga.py --aprobar")
        return 1

    latencias = sorted(
        (r.end_time - r.start_time).total_seconds() for r in ejecuciones
    )

    resultado = {
        "proyecto": args.proyecto,
        "ejecuciones": len(latencias),
        "p50": percentil(latencias, 0.50),
        "p95": percentil(latencias, 0.95),
        "p99": percentil(latencias, 0.99),
        "min": round(min(latencias), 2),
        "max": round(max(latencias), 2),
        "latencias_segundos": [round(x, 2) for x in latencias],
    }

    print("=" * 66)
    print(f"PERCENTILES DE LATENCIA — proyecto '{args.proyecto}'")
    print("=" * 66)
    print(f"ejecuciones '{NOMBRE_RAIZ}' medidas: {resultado['ejecuciones']}")
    print()
    print(f"  P50  {resultado['p50']:>8.2f} s    <- comparalo con el panel del dashboard")
    print(f"  P95  {resultado['p95']:>8.2f} s    <- el que pide la consigna")
    print(f"  P99  {resultado['p99']:>8.2f} s")
    print(f"  min  {resultado['min']:>8.2f} s")
    print(f"  max  {resultado['max']:>8.2f} s")
    print()
    print("El P50 tiene que coincidir con el del grafico 'Trace Latency' de")
    print("LangSmith. Si coincide, los dos estan mirando las mismas trazas y el")
    print("P95 de arriba es el percentil que al panel le falta dibujar.")
    print()

    if args.json:
        Path(args.json).write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Guardado en {args.json}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
