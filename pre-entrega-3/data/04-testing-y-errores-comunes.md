# Testing de código asíncrono y errores frecuentes

## Ejecutar tests asíncronos

`pytest` no ejecuta corrutinas de test por sí solo: las marca como *skipped* o
directamente emite un warning de "coroutine was never awaited". Hace falta el
plugin `pytest-asyncio`. Con `asyncio_mode = auto` en `pytest.ini` o
`pyproject.toml`, cualquier test declarado con `async def` se ejecuta sin
necesidad de decorarlo con `@pytest.mark.asyncio`.

Cada test corre por defecto en un event loop nuevo. Si un fixture crea un
recurso ligado a un loop (por ejemplo un pool de conexiones) y tiene scope
`session`, ese recurso va a quedar atado a un loop ya cerrado y fallará con
`RuntimeError: Event loop is closed`. La solución es alinear el scope del
fixture del loop con el del recurso.

## Mockear corrutinas

`unittest.mock.MagicMock` devuelve un `MagicMock` común cuando se lo llama, y
hacerle `await` falla con `TypeError: object MagicMock can't be used in 'await'
expression`. Hay que usar `AsyncMock`, que devuelve una corrutina. Desde Python
3.8, `patch()` detecta automáticamente que el objeto parcheado es una función
asíncrona y usa `AsyncMock` sin configuración extra.

## Errores frecuentes

**`RuntimeWarning: coroutine 'foo' was never awaited`**: se llamó a la corrutina
sin `await` ni `create_task`. El trabajo nunca se ejecutó.

**`RuntimeError: This event loop is already running`**: se llamó a
`asyncio.run()` o `loop.run_until_complete()` dentro de un loop que ya estaba
corriendo, algo típico en Jupyter. En un notebook hay que usar `await` directo,
porque las celdas ya corren dentro de un loop.

**`RuntimeError: Task attached to a different loop`**: un objeto creado en un
loop (un `asyncio.Lock`, una `Queue`, un cliente HTTP) se usó desde otro. Las
primitivas de sincronización de asyncio deben crearse dentro del mismo loop que
las consume.

**Sesión HTTP sin cerrar**: un `aiohttp.ClientSession` o un `httpx.AsyncClient`
que no se cierra deja conexiones abiertas y emite "Unclosed client session" al
salir. La solución es usarlos como context manager asíncrono (`async with`) o
llamar a `await cliente.aclose()`.

## Medir concurrencia real

Un test útil para verificar que las llamadas realmente van en paralelo es medir
el tiempo total: si diez llamadas de 1 segundo tardan ~10 segundos, el código es
secuencial aunque use `async def`. La causa habitual es hacer `await` dentro de
un `for` en vez de juntar las corrutinas y pasarlas a `gather` o a un TaskGroup.
