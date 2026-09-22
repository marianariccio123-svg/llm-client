# Timeouts, cancelación y limpieza

## Cancelación cooperativa

`task.cancel()` no mata la tarea: agenda el lanzamiento de una excepción
`asyncio.CancelledError` en el próximo punto de `await` de esa tarea. Si la
tarea está corriendo un bucle de CPU sin `await`, la cancelación nunca llega a
efectuarse.

Desde Python 3.8, `CancelledError` hereda de `BaseException`, no de `Exception`.
Por eso un `except Exception:` genérico ya no la intercepta por accidente, pero
un `except BaseException:` sí, y eso rompe el mecanismo de cancelación.

Si una corrutina necesita limpiar recursos al ser cancelada, debe volver a
lanzar la excepción después de limpiar:

```python
try:
    await operacion_larga()
except asyncio.CancelledError:
    await conexion.close()
    raise
```

Tragarse el `CancelledError` sin relanzarlo hace que la tarea aparezca como
completada normalmente y deja el `TaskGroup` o el `asyncio.timeout` que la
cancelaba en un estado inconsistente.

## asyncio.timeout vs. wait_for

`asyncio.wait_for(coro, timeout=5)` envuelve la corrutina, la cancela al vencer
el plazo y lanza `asyncio.TimeoutError`. Funciona pero solo aplica a una
corrutina.

Desde Python 3.11 se prefiere el context manager `asyncio.timeout`, que impone
un plazo a todo un bloque de código:

```python
async with asyncio.timeout(5):
    datos = await cliente.get(url)
    await guardar(datos)
```

Al vencer el plazo, el bloque se cancela y se lanza `TimeoutError`. Existe
también `asyncio.timeout_at(deadline)` para un plazo absoluto medido con el
reloj del loop (`loop.time()`), útil para propagar un deadline único a través
de varias llamadas encadenadas.

## Blindar una tarea contra la cancelación

`asyncio.shield(coro)` evita que la cancelación del código que espera se
propague a la corrutina protegida. Es útil para operaciones que no deben
interrumpirse a la mitad, como confirmar una transacción. Ojo: si el event loop
se cierra, la tarea blindada muere igual.

## Apagado ordenado

Al recibir SIGTERM, el patrón correcto es cancelar las tareas pendientes y
esperarlas con `return_exceptions=True` antes de cerrar el loop:

```python
tareas = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
for t in tareas:
    t.cancel()
await asyncio.gather(*tareas, return_exceptions=True)
```
