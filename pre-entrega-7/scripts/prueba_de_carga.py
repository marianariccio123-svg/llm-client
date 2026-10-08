"""
scripts/prueba_de_carga.py

Lanza N peticiones concurrentes contra la API y mide qué pasa.

Para qué sirve
-------------
La consigna pide 5 peticiones concurrentes y dos métricas del dashboard: el
costo por ejecución y la latencia p95. Este script produce la carga y además
calcula sus propias métricas del lado del cliente.

Tener las dos mediciones no es redundante, miden cosas distintas:

- **El p95 del cliente** es de punta a punta: incluye la espera en la cola.
  Con 3 workers y 5 peticiones, dos esperan su turno, y eso se ve acá.
- **El p95 del dashboard** es el de la ejecución del grafo: empieza cuando el
  worker toma el trabajo. No incluye la cola.

Si los dos números son parecidos, la cola no está siendo un cuello de botella.
Si el del cliente es mucho mayor, el sistema está saturado aunque cada corrida
individual sea rápida. Esa comparación es lo que hace que la captura signifique
algo en vez de ser un número suelto.

Uso:
    python scripts/prueba_de_carga.py
    python scripts/prueba_de_carga.py --peticiones 10 --url http://127.0.0.1:8000
    python scripts/prueba_de_carga.py --aprobar      # resuelve las pausas HITL
    python scripts/prueba_de_carga.py --json resultados.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

# Cinco consultas distintas: todas necesitan a los dos especialistas, pero
# preguntan cosas diferentes. Usar la misma cinco veces mediría el caché del
# proveedor tanto como el sistema.
CONSULTAS = [
    "Nuestros servicios asincronicos se vienen cayendo seguido. Que dice la "
    "documentacion tecnica sobre las causas y que muestran nuestros incidentes "
    "del ultimo anio?",

    "Cual es la causa dominante de nuestras caidas de produccion y que "
    "recomienda la documentacion para evitarla?",

    "Hay algun incidente fuera de lo normal en la bitacora? Analizalo y decime "
    "que dice la documentacion sobre ese tipo de falla.",

    "Que servicio es el mas problematico segun nuestros incidentes y cuanto "
    "tiempo de caida acumulo? Respalda la explicacion con la documentacion.",

    "Cuantos minutos de caida acumulamos y que proporcion corresponde a "
    "bloqueos del event loop? Que dice la documentacion al respecto?",
]

TERMINALES = {"COMPLETADO", "FALLIDO"}


async def una_peticion(
    cliente: httpx.AsyncClient,
    base: str,
    solicitud: str,
    numero: int,
    aprobar: bool,
    intervalo_polling: float,
) -> dict[str, Any]:
    """
    Manda una petición y hace polling hasta que el job termina.

    Devuelve las mediciones de esa petición. No lanza excepciones hacia
    arriba: un fallo es un resultado más de la prueba de carga, y abortar todo
    porque una falló perdería la medición de las otras cuatro.
    """
    medicion: dict[str, Any] = {
        "numero": numero,
        "solicitud": solicitud[:70] + "...",
        "estado": None,
        "job_id": None,
        "ms_encolado": None,
        "segundos_total": None,
        "paso_por_hitl": False,
        "error": None,
    }

    inicio = time.perf_counter()
    try:
        t0 = time.perf_counter()
        respuesta = await cliente.post(f"{base}/tasks", json={"solicitud": solicitud})
        medicion["ms_encolado"] = round((time.perf_counter() - t0) * 1000, 1)

        if respuesta.status_code != 202:
            medicion["estado"] = f"HTTP_{respuesta.status_code}"
            medicion["error"] = respuesta.text[:200]
            return medicion

        job_id = respuesta.json()["job_id"]
        medicion["job_id"] = job_id

        while True:
            await asyncio.sleep(intervalo_polling)
            estado = (await cliente.get(f"{base}/tasks/{job_id}")).json()

            if estado["estado"] == "ESPERANDO_APROBACION":
                medicion["paso_por_hitl"] = True
                if not aprobar:
                    # Sin --aprobar el job queda pausado a propósito: es el
                    # estado que hay que mostrar en la captura del HITL.
                    medicion["estado"] = "ESPERANDO_APROBACION"
                    break
                await cliente.post(
                    f"{base}/tasks/{job_id}/approve",
                    json={
                        "aprobado": True,
                        "aprobado_por": "prueba-de-carga",
                        "comentario": f"Aprobación automática de la petición {numero}.",
                    },
                )
                continue

            if estado["estado"] in TERMINALES:
                medicion["estado"] = estado["estado"]
                medicion["error"] = estado.get("error")
                break

    except Exception as e:  # noqa: BLE001
        medicion["estado"] = "ERROR_CLIENTE"
        medicion["error"] = f"{type(e).__name__}: {e}"

    medicion["segundos_total"] = round(time.perf_counter() - inicio, 2)
    return medicion


def percentil(valores: list[float], p: float) -> float | None:
    """
    El percentil p de una lista, por interpolación lineal.

    Con 5 muestras el p95 cae entre las dos más altas, así que la
    interpolación importa: tomar "el valor en la posición 0.95*n" daría
    directamente el máximo y sobreestimaría.
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


def resumir(mediciones: list[dict[str, Any]], segundos_totales: float) -> dict[str, Any]:
    """Agrega las mediciones individuales."""
    completados = [m for m in mediciones if m["estado"] == "COMPLETADO"]
    duraciones = [m["segundos_total"] for m in completados if m["segundos_total"]]
    encolados = [m["ms_encolado"] for m in mediciones if m["ms_encolado"]]

    return {
        "peticiones": len(mediciones),
        "completadas": len(completados),
        "fallidas": sum(1 for m in mediciones if m["estado"] == "FALLIDO"),
        "pausadas_en_hitl": sum(1 for m in mediciones if m["paso_por_hitl"]),
        "segundos_de_la_corrida": round(segundos_totales, 2),
        "latencia_extremo_a_extremo": {
            "p50": percentil(duraciones, 0.50),
            "p95": percentil(duraciones, 0.95),
            "min": round(min(duraciones), 2) if duraciones else None,
            "max": round(max(duraciones), 2) if duraciones else None,
            "media": round(statistics.mean(duraciones), 2) if duraciones else None,
        },
        "ms_hasta_el_202": {
            "p50": percentil(encolados, 0.50),
            "p95": percentil(encolados, 0.95),
            "max": round(max(encolados), 1) if encolados else None,
        },
    }


def imprimir(mediciones: list[dict[str, Any]], resumen: dict[str, Any], base: str) -> None:
    print("\n" + "=" * 78)
    print("RESULTADO DE LA PRUEBA DE CARGA")
    print("=" * 78)
    print(f"{'#':>2}  {'estado':<22} {'202 en':>9} {'total':>9}  hitl")
    print("-" * 78)
    for m in sorted(mediciones, key=lambda x: x["numero"]):
        ms = f"{m['ms_encolado']} ms" if m["ms_encolado"] else "—"
        total = f"{m['segundos_total']} s" if m["segundos_total"] else "—"
        print(f"{m['numero']:>2}  {str(m['estado']):<22} {ms:>9} {total:>9}  "
              f"{'sí' if m['paso_por_hitl'] else 'no'}")
        if m["error"]:
            print(f"    error: {str(m['error'])[:110]}")

    lat = resumen["latencia_extremo_a_extremo"]
    enc = resumen["ms_hasta_el_202"]
    print("-" * 78)
    print(f"completadas: {resumen['completadas']}/{resumen['peticiones']}   "
          f"fallidas: {resumen['fallidas']}   "
          f"pausadas en HITL: {resumen['pausadas_en_hitl']}")
    print(f"corrida completa: {resumen['segundos_de_la_corrida']} s")
    print()
    print("Latencia extremo a extremo (incluye espera en cola):")
    print(f"  p50={lat['p50']} s   p95={lat['p95']} s   min={lat['min']} s   max={lat['max']} s")
    print("Tiempo hasta el 202 (la API no bloquea):")
    print(f"  p50={enc['p50']} ms   p95={enc['p95']} ms   max={enc['max']} ms")
    print()
    print("Ahora, en el dashboard de LangSmith:")
    print(f"  1. Abrí el proyecto y filtrá por los últimos {resumen['peticiones']} runs.")
    print("  2. Capturá el COSTO POR EJECUCIÓN (columna Total Cost / Cost).")
    print("  3. Capturá la LATENCIA P95 (en el panel de métricas del proyecto).")
    print("  4. Guardá las dos capturas en screenshots/.")
    print(f"\n  (la API corrió en {base})")
    print()


async def main_async(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as cliente:
        try:
            salud = (await cliente.get(f"{base}/health")).json()
        except Exception as e:
            print(f"No se pudo contactar la API en {base}: {type(e).__name__}: {e}")
            print("¿Está levantada? uvicorn app.main:app --reload")
            return 1

        print(f"API: {base}")
        print(f"  estado={salud['estado']}  redis={salud['redis']}  "
              f"workers={salud['workers']}  observabilidad={salud['observabilidad']}")
        if not salud["redis"]:
            print("  ADVERTENCIA: Redis no responde; los jobs van a fallar.")
        if salud["observabilidad"] == "desactivada":
            print("  ADVERTENCIA: sin observabilidad no va a haber trazas que capturar.")

        consultas = [CONSULTAS[i % len(CONSULTAS)] for i in range(args.peticiones)]
        print(f"\nLanzando {args.peticiones} peticiones concurrentes...\n")

        inicio = time.perf_counter()
        # `gather` las dispara todas a la vez: es lo que las hace concurrentes
        # de verdad y no un bucle secuencial disfrazado.
        mediciones = await asyncio.gather(
            *(
                una_peticion(cliente, base, consulta, numero, args.aprobar,
                             args.intervalo)
                for numero, consulta in enumerate(consultas, start=1)
            )
        )
        segundos = time.perf_counter() - inicio

    resumen = resumir(list(mediciones), segundos)
    imprimir(list(mediciones), resumen, base)

    if args.json:
        salida = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url": base,
            "resumen": resumen,
            "mediciones": list(mediciones),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados en {args.json}\n")

    return 0 if resumen["fallidas"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base de la API")
    parser.add_argument("--peticiones", type=int, default=5,
                        help="cuántas lanzar en paralelo (la consigna pide 5)")
    parser.add_argument("--aprobar", action="store_true",
                        help="aprobar automáticamente las pausas HITL")
    parser.add_argument("--intervalo", type=float, default=2.0,
                        help="segundos entre consultas de estado")
    parser.add_argument("--json", help="archivo donde guardar los resultados")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
