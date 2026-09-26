---
doc_id: timeouts-y-cancelacion
titulo: Timeouts y cancelación
categoria: control-de-flujo
fuente_original: https://docs.python.org/3/library/asyncio-task.html#timeouts
---

## asyncio.timeout

`asyncio.timeout(delay)` (Python 3.11+) es un context manager que cancela todo
lo que esté adentro si se pasa del tiempo. Es la forma preferida porque abarca
un bloque entero, no una sola llamada.

```python
try:
    async with asyncio.timeout(5):
        datos = await pedir(url)
        await guardar(datos)
except TimeoutError:
    print("el bloque tardó más de 5 segundos")
```

También existe `asyncio.timeout_at(when)` para un instante absoluto del reloj
del loop, y el timeout se puede reprogramar en caliente con
`cm.reschedule(nuevo_deadline)`.

## asyncio.wait_for

`asyncio.wait_for(aw, timeout)` es la versión clásica, para un único awaitable.
Al vencer el plazo cancela la tarea interna, espera a que la cancelación termine
y recién ahí lanza `TimeoutError`.

Desde Python 3.11 `asyncio.TimeoutError` es un alias de `TimeoutError`, el
builtin. En código nuevo se atrapa el builtin.

## CancelledError no es un error común

La cancelación se implementa lanzando `asyncio.CancelledError` dentro de la
tarea. Desde Python 3.8 hereda de `BaseException`, no de `Exception`, justamente
para que un `except Exception` genérico no se la coma sin querer.

La regla es: se puede atrapar para limpiar, pero **hay que volver a lanzarla**.

```python
try:
    await trabajo_largo()
except asyncio.CancelledError:
    await limpiar()
    raise          # imprescindible: nunca te tragues la cancelación
```

Tragarse un `CancelledError` deja la tarea en un estado inconsistente y puede
colgar el `TaskGroup` o el `asyncio.run()` que la esperaba.

## shield: proteger una sección crítica

`asyncio.shield(aw)` evita que la cancelación externa llegue al awaitable
envuelto. Sirve para operaciones que no se pueden dejar a medias, como confirmar
una transacción o cerrar un archivo.

```python
await asyncio.shield(confirmar_pago(id_pago))
```

Ojo: el `shield` protege la operación interna, pero el `await` de afuera sí
recibe la cancelación. La operación protegida sigue corriendo en segundo plano.

## Limpieza con finally

Para código que debe correr sí o sí (cerrar conexiones, liberar locks) se usa
`try/finally`, que se ejecuta también cuando la tarea es cancelada. Si la
limpieza es asíncrona y puede tardar, conviene envolverla en su propio
`asyncio.shield` o `asyncio.timeout`.
