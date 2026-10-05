"""
seed_db.py

Crea y puebla la base SQLite que consultan las herramientas del agente.

Es una base **real**, no un diccionario en memoria: las herramientas abren
conexiones con `aiosqlite` y ejecutan SQL de verdad. Importa para esta entrega
porque el agente tiene que lidiar con lo que una base devuelve en serio —cero
filas, varias filas, un id que no existe— y no con un `dict.get()` que siempre
sale bien.

Los datos están armados para que el agente **no pueda** resolver las preguntas
de un solo salto:

- Nadie conoce su `cliente_id`: hay que buscarlo por nombre primero.
- Los totales están en `pedidos`, pero el detalle está en `items`.
- Hay dos clientes que se llaman "Juan", así que una búsqueda por nombre de
  pila es ambigua a propósito y obliga al agente a pedir una aclaración.

Uso:
    python seed_db.py           # crea data/tienda.db (no pisa si ya existe)
    python seed_db.py --force   # la recrea desde cero
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "tienda.db"

ESQUEMA = """
CREATE TABLE clientes (
    cliente_id INTEGER PRIMARY KEY,
    nombre     TEXT NOT NULL,
    email      TEXT NOT NULL,
    ciudad     TEXT NOT NULL
);

CREATE TABLE pedidos (
    pedido_id  TEXT PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(cliente_id),
    fecha      TEXT NOT NULL,
    total      REAL NOT NULL,
    estado     TEXT NOT NULL
);

CREATE TABLE productos (
    sku       TEXT PRIMARY KEY,
    nombre    TEXT NOT NULL,
    precio    REAL NOT NULL,
    stock     INTEGER NOT NULL
);

CREATE TABLE items (
    pedido_id TEXT NOT NULL REFERENCES pedidos(pedido_id),
    sku       TEXT NOT NULL REFERENCES productos(sku),
    cantidad  INTEGER NOT NULL,
    PRIMARY KEY (pedido_id, sku)
);
"""

CLIENTES = [
    (101, "Soledad Ferreyra", "soledad.f@example.com", "Rosario"),
    (102, "Mariana Riccio", "mariana.r@example.com", "Buenos Aires"),
    # Dos "Juan" a propósito: hacen que buscar por nombre de pila sea ambiguo.
    (103, "Juan Pérez", "juan.perez@example.com", "Córdoba"),
    (104, "Juan Perazzo", "juan.perazzo@example.com", "La Plata"),
    (105, "Ignacio Bordón", "nacho.b@example.com", "Mendoza"),
]

PRODUCTOS = [
    ("TEC-MEC-87", "Teclado mecánico 87 teclas", 3500.0, 12),
    ("MOU-ERG-01", "Mouse ergonómico inalámbrico", 1000.0, 0),
    ("MON-27-4K", "Monitor 27 pulgadas 4K", 4200.0, 3),
    ("HUB-USBC-7", "Hub USB-C de 7 puertos", 1800.0, 25),
    ("SIL-ERG-02", "Silla ergonómica", 4000.0, 1),
    ("CAB-HDMI-2", "Cable HDMI 2.1 de 2 metros", 800.0, 48),
]

# Cliente 102 tiene 3 pedidos por 14500 en total, igual que el ejemplo de la
# consigna. El último (P-1027) es el que se consulta en la traza multi-paso.
PEDIDOS = [
    ("P-1001", 102, "2025-03-14", 4200.0, "entregado"),
    ("P-1014", 102, "2025-06-02", 5800.0, "entregado"),
    ("P-1027", 102, "2025-09-21", 4500.0, "en camino"),
    ("P-1003", 101, "2025-03-28", 1800.0, "entregado"),
    ("P-1019", 103, "2025-07-11", 8200.0, "entregado"),
    ("P-1022", 104, "2025-08-05", 800.0, "cancelado"),
]

# Los items de cada pedido suman exactamente el `total` de la tabla `pedidos`.
# Que la base sea internamente consistente no es decorativo: si el detalle no
# cerrara con el total, el agente podría "razonar" sobre una contradicción y no
# se sabría si el error es del modelo o de los datos.
ITEMS = [
    ("P-1001", "MON-27-4K", 1),                       # 4200
    ("P-1014", "SIL-ERG-02", 1), ("P-1014", "HUB-USBC-7", 1),   # 4000 + 1800 = 5800
    ("P-1027", "TEC-MEC-87", 1), ("P-1027", "MOU-ERG-01", 1),   # 3500 + 1000 = 4500
    ("P-1003", "HUB-USBC-7", 1),                      # 1800
    ("P-1019", "MON-27-4K", 1), ("P-1019", "SIL-ERG-02", 1),    # 4200 + 4000 = 8200
    ("P-1022", "CAB-HDMI-2", 1),                      # 800
]


def crear(force: bool = False) -> Path:
    """Crea la base. Devuelve la ruta. Con `force`, borra la existente."""
    DATA_DIR.mkdir(exist_ok=True)

    if DB_PATH.exists():
        if not force:
            print(f"{DB_PATH} ya existe. Usá --force para recrearla.")
            return DB_PATH
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(ESQUEMA)
        con.executemany("INSERT INTO clientes VALUES (?, ?, ?, ?)", CLIENTES)
        con.executemany("INSERT INTO productos VALUES (?, ?, ?, ?)", PRODUCTOS)
        con.executemany("INSERT INTO pedidos VALUES (?, ?, ?, ?, ?)", PEDIDOS)
        con.executemany("INSERT INTO items VALUES (?, ?, ?)", ITEMS)
        con.commit()
    finally:
        con.close()

    print(f"Base creada en {DB_PATH}")
    print(f"  {len(CLIENTES)} clientes, {len(PRODUCTOS)} productos, "
          f"{len(PEDIDOS)} pedidos, {len(ITEMS)} items")
    return DB_PATH


def asegurar_db() -> Path:
    """Crea la base si falta. La llaman las herramientas para no fallar en seco."""
    if not DB_PATH.exists():
        crear()
    return DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recrear desde cero")
    args = parser.parse_args()
    crear(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
