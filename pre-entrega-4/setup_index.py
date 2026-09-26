"""
setup_index.py

Inicialización de la infraestructura: crea el índice Serverless de Pinecone si
no existe y, si ya existe, **verifica que la dimensión coincida** con la que
usa el proyecto.

Esa verificación es el punto del script. El error más caro de esta entrega es
el mismatch de dimensiones: un índice creado a 768 acepta la conexión sin
quejarse y recién falla en el upsert, después de haber pagado los embeddings de
todo el corpus. Acá se detecta antes de gastar una sola llamada al modelo.

Uso:
    python setup_index.py            # crea si falta, verifica si existe
    python setup_index.py --info     # solo informa el estado actual
    python setup_index.py --delete   # borra el índice (pide confirmación)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    METRIC,
    NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    get_pinecone,
    setup_logging,
)

log = logging.getLogger("setup_index")


def indice_existe(pc) -> bool:
    return INDEX_NAME in [i["name"] for i in pc.list_indexes()]


def crear_indice(pc) -> None:
    """Crea el índice Serverless y espera a que esté listo para recibir datos."""
    from pinecone import ServerlessSpec

    log.info(
        "Creando índice Serverless '%s' (dim=%d, metric=%s, %s/%s)...",
        INDEX_NAME, EMBEDDING_DIM, METRIC, PINECONE_CLOUD, PINECONE_REGION,
    )
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

    # create_index vuelve enseguida, pero el índice tarda unos segundos en
    # aceptar escrituras. Sin esta espera el primer upsert falla.
    for _ in range(60):
        if pc.describe_index(INDEX_NAME).status.get("ready"):
            log.info("Índice listo.")
            return
        time.sleep(2)
    raise RuntimeError(
        f"El índice '{INDEX_NAME}' no quedó listo después de 2 minutos. "
        f"Revisá el estado en https://app.pinecone.io"
    )


def verificar_dimension(pc) -> None:
    """Falla ruidosamente si el índice existente no es compatible."""
    desc = pc.describe_index(INDEX_NAME)
    dim_real = desc.dimension
    metric_real = desc.metric

    if dim_real != EMBEDDING_DIM:
        raise RuntimeError(
            f"MISMATCH DE DIMENSIONES.\n"
            f"  El índice '{INDEX_NAME}' fue creado con dimensión {dim_real}, "
            f"pero este proyecto genera embeddings de {EMBEDDING_DIM}.\n"
            f"  La dimensión de un índice no se puede cambiar. Tenés dos salidas:\n"
            f"    1) usar otro nombre:  INDEX_NAME=asyncio-docs-1536 en el .env\n"
            f"    2) borrar y recrear:  python setup_index.py --delete && python setup_index.py"
        )

    if metric_real != METRIC:
        log.warning(
            "El índice usa la métrica '%s' y el proyecto asume '%s'. "
            "Los scores van a ser comparables igual, pero el orden puede variar.",
            metric_real, METRIC,
        )

    log.info("Índice '%s' ya existe y es compatible (dim=%d, metric=%s).",
             INDEX_NAME, dim_real, metric_real)


def mostrar_info(pc) -> None:
    """Imprime el estado del índice y el conteo de vectores por namespace."""
    if not indice_existe(pc):
        print(f"El índice '{INDEX_NAME}' no existe todavía. Corré: python setup_index.py")
        return

    desc = pc.describe_index(INDEX_NAME)
    stats = pc.Index(INDEX_NAME).describe_index_stats()

    print(f"\nÍndice:      {INDEX_NAME}")
    print(f"Dimensión:   {desc.dimension}")
    print(f"Métrica:     {desc.metric}")
    print(f"Host:        {desc.host}")
    print(f"Estado:      {'listo' if desc.status.get('ready') else 'inicializando'}")
    print(f"Vectores:    {stats.get('total_vector_count', 0)} en total")

    namespaces = stats.get("namespaces", {}) or {}
    if namespaces:
        print("Namespaces:")
        for nombre, datos in sorted(namespaces.items()):
            marca = "  <- activo" if nombre == NAMESPACE else ""
            print(f"  - {nombre or '(default)'}: {datos['vector_count']} vectores{marca}")
    else:
        print("Namespaces:  (ninguno, el índice está vacío)")
    print()


def borrar_indice(pc) -> None:
    if not indice_existe(pc):
        print(f"El índice '{INDEX_NAME}' no existe; no hay nada que borrar.")
        return
    respuesta = input(
        f"Vas a BORRAR el índice '{INDEX_NAME}' y todos sus vectores. "
        f"Escribí el nombre del índice para confirmar: "
    ).strip()
    if respuesta != INDEX_NAME:
        print("Cancelado.")
        return
    pc.delete_index(INDEX_NAME)
    print(f"Índice '{INDEX_NAME}' borrado.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", action="store_true", help="solo mostrar el estado")
    parser.add_argument("--delete", action="store_true", help="borrar el índice")
    args = parser.parse_args()

    setup_logging()
    pc = get_pinecone()

    if args.delete:
        borrar_indice(pc)
        return 0

    if args.info:
        mostrar_info(pc)
        return 0

    if indice_existe(pc):
        verificar_dimension(pc)
    else:
        crear_indice(pc)

    mostrar_info(pc)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("setup_index").error("%s", e)
        sys.exit(1)
