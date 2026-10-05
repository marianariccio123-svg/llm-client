"""
tools.py

Las herramientas del agente: cuatro operaciones asíncronas sobre la base
SQLite de `seed_db.py`.

Sobre los docstrings
--------------------
El LLM elige qué herramienta llamar leyendo **únicamente** el nombre, la firma
y el docstring. No ve el cuerpo de la función. Por eso cada docstring dice tres
cosas explícitamente:

1. Qué hace y cuándo conviene usarla.
2. Qué devuelve, con un ejemplo concreto de la forma del resultado.
3. **Con qué otra herramienta se encadena.** Esto es lo que hace que el agente
   razone en varios pasos solo: si `buscar_pedidos` avisa que necesita un
   `cliente_id` y que ese id lo da `buscar_cliente`, el modelo arma la cadena
   sin que nadie se la programe.

Sobre los errores
-----------------
Ninguna herramienta lanza excepciones hacia el grafo. Devuelven un dict con
`"error"` y, cuando se puede, **opciones concretas para reintentar**: los
nombres que sí existen, los ids válidos, los SKUs disponibles. Esa es la
diferencia entre un agente que se queda trabado y uno que se corrige: un
`ValueError` pelado no le dice al modelo qué hacer después, pero
`{"error": "...", "coincidencias": [...]}` sí.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
from langchain_core.tools import tool

from seed_db import asegurar_db


async def _consultar(sql: str, parametros: tuple = ()) -> list[dict[str, Any]]:
    """Ejecuta una consulta y devuelve las filas como diccionarios."""
    ruta = asegurar_db()
    async with aiosqlite.connect(ruta) as con:
        con.row_factory = aiosqlite.Row
        async with con.execute(sql, parametros) as cursor:
            filas = await cursor.fetchall()
    return [dict(fila) for fila in filas]


@tool
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """
    Busca un cliente por nombre (o parte del nombre) y devuelve su cliente_id.

    Usala SIEMPRE como primer paso cuando el usuario menciona a un cliente por
    su nombre, porque el resto de las herramientas necesitan el cliente_id
    numérico y el usuario casi nunca lo sabe.

    La búsqueda no distingue mayúsculas y acepta coincidencias parciales:
    "mariana" encuentra a "Mariana Riccio".

    Args:
        nombre: nombre o apellido del cliente, completo o parcial.

    Devuelve, si hay exactamente una coincidencia:
        {"cliente_id": 102, "nombre": "Mariana Riccio",
         "email": "mariana.r@example.com", "ciudad": "Buenos Aires"}

    Si no encuentra a nadie devuelve {"error": ..., "clientes_disponibles": [...]}
    con la lista completa de nombres, para que puedas reintentar con uno válido.

    Si encuentra VARIOS clientes devuelve {"error": ..., "coincidencias": [...]}
    con todos los candidatos. En ese caso NO adivines: preguntale al usuario a
    cuál se refería.
    """
    filas = await _consultar(
        "SELECT cliente_id, nombre, email, ciudad FROM clientes "
        "WHERE lower(nombre) LIKE lower(?)",
        (f"%{nombre}%",),
    )

    if not filas:
        todos = await _consultar("SELECT nombre FROM clientes ORDER BY nombre")
        return {
            "error": f"No existe ningún cliente que coincida con '{nombre}'.",
            "clientes_disponibles": [f["nombre"] for f in todos],
        }

    if len(filas) > 1:
        return {
            "error": (
                f"'{nombre}' es ambiguo: hay {len(filas)} clientes que coinciden. "
                f"Preguntale al usuario a cuál se refiere antes de seguir."
            ),
            "coincidencias": filas,
        }

    return filas[0]


@tool
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """
    Devuelve todos los pedidos de un cliente, con la cantidad y el total gastado.

    Necesita el cliente_id numérico. Si solo tenés el nombre del cliente,
    llamá primero a `buscar_cliente` para obtenerlo.

    Args:
        cliente_id: identificador numérico del cliente (ej. 102).

    Devuelve:
        {"cliente_id": 102, "cantidad_de_pedidos": 3, "total_gastado": 14500.0,
         "pedidos": [{"pedido_id": "P-1027", "fecha": "2025-09-21",
                      "total": 4500.0, "estado": "en camino"}, ...]}

    Los pedidos vienen ordenados de más reciente a más antiguo, así que el
    primero de la lista es el último pedido del cliente.

    Esta herramienta devuelve los TOTALES de cada pedido, no qué productos
    tenía adentro. Para eso usá `detalle_pedido` con el pedido_id.

    Si el cliente_id no existe devuelve {"error": ..., "cliente_ids_validos": [...]}.
    """
    existe = await _consultar(
        "SELECT nombre FROM clientes WHERE cliente_id = ?", (cliente_id,)
    )
    if not existe:
        validos = await _consultar("SELECT cliente_id, nombre FROM clientes")
        return {
            "error": f"No existe el cliente_id {cliente_id}.",
            "cliente_ids_validos": validos,
        }

    pedidos = await _consultar(
        "SELECT pedido_id, fecha, total, estado FROM pedidos "
        "WHERE cliente_id = ? ORDER BY fecha DESC",
        (cliente_id,),
    )

    return {
        "cliente_id": cliente_id,
        "nombre": existe[0]["nombre"],
        "cantidad_de_pedidos": len(pedidos),
        "total_gastado": round(sum(p["total"] for p in pedidos), 2),
        "pedidos": pedidos,
    }


@tool
async def detalle_pedido(pedido_id: str) -> dict[str, Any]:
    """
    Devuelve qué productos contiene un pedido puntual, con cantidades y precios.

    Usala cuando el usuario pregunta QUÉ compró alguien, no cuánto gastó.
    Necesita el pedido_id (formato "P-1027"), que sale de `buscar_pedidos`.

    Args:
        pedido_id: identificador del pedido, con el formato "P-1027".

    Devuelve:
        {"pedido_id": "P-1027", "fecha": "2025-09-21", "estado": "en camino",
         "total": 4500.0,
         "items": [{"sku": "TEC-MEC-87", "producto": "Teclado mecánico 87 teclas",
                    "cantidad": 1, "precio_unitario": 3500.0, "subtotal": 3500.0}, ...]}

    Cada item incluye su `sku`, que es lo que necesita `consultar_stock` si
    después hace falta saber si queda mercadería disponible.

    Si el pedido no existe devuelve {"error": ...}.
    """
    cabecera = await _consultar(
        "SELECT pedido_id, cliente_id, fecha, total, estado FROM pedidos "
        "WHERE upper(pedido_id) = upper(?)",
        (pedido_id,),
    )
    if not cabecera:
        return {
            "error": f"No existe el pedido '{pedido_id}'. "
                     f"Verificá el id con `buscar_pedidos`; el formato es 'P-1027'."
        }

    items = await _consultar(
        "SELECT i.sku, p.nombre AS producto, i.cantidad, p.precio AS precio_unitario, "
        "       (i.cantidad * p.precio) AS subtotal "
        "FROM items i JOIN productos p ON p.sku = i.sku "
        "WHERE upper(i.pedido_id) = upper(?)",
        (pedido_id,),
    )

    return {**cabecera[0], "items": items}


@tool
async def consultar_stock(sku: str) -> dict[str, Any]:
    """
    Dice cuántas unidades quedan en stock de un producto, buscándolo por SKU.

    Usala cuando el usuario pregunta por disponibilidad, reposición o si se
    puede volver a comprar algo. Los SKUs salen de `detalle_pedido`.

    Args:
        sku: código del producto, con el formato "TEC-MEC-87".

    Devuelve:
        {"sku": "TEC-MEC-87", "producto": "Teclado mecánico 87 teclas",
         "precio": 3500.0, "stock": 12, "disponible": true}

    `disponible` es false cuando el stock es 0: en ese caso el producto existe
    pero no se puede reponer.

    Si el SKU no existe devuelve {"error": ..., "skus_validos": [...]} con todos
    los códigos del catálogo, para que puedas reintentar con uno correcto.
    """
    filas = await _consultar(
        "SELECT sku, nombre AS producto, precio, stock FROM productos "
        "WHERE upper(sku) = upper(?)",
        (sku,),
    )
    if not filas:
        catalogo = await _consultar("SELECT sku, nombre FROM productos ORDER BY sku")
        return {
            "error": f"No existe el SKU '{sku}'.",
            "skus_validos": catalogo,
        }

    producto = filas[0]
    return {**producto, "disponible": producto["stock"] > 0}


# El agente recibe esta lista tal cual en `bind_tools`.
HERRAMIENTAS = [buscar_cliente, buscar_pedidos, detalle_pedido, consultar_stock]
