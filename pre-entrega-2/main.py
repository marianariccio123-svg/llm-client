"""
main.py

Script de prueba del pipeline de extracción de entidades técnicas.
Prueba un caso normal (texto claro) y un caso ambiguo (para estresar
el validador y la lógica de reintentos).
"""

import asyncio

from chain import process_text


TEXTO_NORMAL = (
    "Tenemos una API en FastAPI con Redis para cache y PostgreSQL para "
    "persistencia. Bajo carga alta, el pool de conexiones a Postgres se "
    "agota y la API empieza a devolver errores 500."
)

TEXTO_AMBIGUO = (
    "El sistema tuvo un problemita ayer, no sé bien qué pasó pero ya "
    "parece que se solucionó solo."
)


async def main():
    print("=" * 60)
    print("CASO 1: texto claro y técnico")
    print("=" * 60)
    resultado_normal = await process_text(TEXTO_NORMAL)
    if resultado_normal:
        print(resultado_normal.model_dump_json(indent=2))
    else:
        print("El pipeline no pudo extraer datos válidos.")

    print()
    print("=" * 60)
    print("CASO 2: texto ambiguo, sin tecnologías explícitas (prueba de estrés)")
    print("=" * 60)
    resultado_ambiguo = await process_text(TEXTO_AMBIGUO)
    if resultado_ambiguo:
        print(resultado_ambiguo.model_dump_json(indent=2))
    else:
        print(
            "El pipeline no pudo extraer datos válidos de este texto "
            "(esperable: es ambiguo y no menciona tecnologías concretas)."
        )


if __name__ == "__main__":
    asyncio.run(main())