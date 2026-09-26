---
doc_id: testing-asincrono
titulo: Testing de código asíncrono
categoria: testing
fuente_original: https://docs.python.org/3/library/unittest.html#unittest.IsolatedAsyncioTestCase
---

## pytest-asyncio

`pytest` no sabe ejecutar corrutinas por su cuenta: sin un plugin, un test
`async def` se saltea con un warning y **pasa sin haber corrido nada**, que es
la peor clase de falso positivo.

```bash
pip install pytest pytest-asyncio
```

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
```

Con `asyncio_mode = auto` cualquier test `async def` se ejecuta sin decorador.
En modo `strict` (el default) hay que marcar cada uno con
`@pytest.mark.asyncio`.

## Fixtures asíncronas

Una fixture que necesita `await` se declara con `@pytest_asyncio.fixture`:

```python
@pytest_asyncio.fixture
async def cliente():
    c = await abrir_cliente()
    yield c
    await c.cerrar()
```

El código después del `yield` corre como teardown, también asíncrono.

## unittest.IsolatedAsyncioTestCase

La biblioteca estándar trae su propia opción, sin dependencias externas:

```python
import unittest

class TestApi(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cliente = await abrir_cliente()

    async def test_responde(self) -> None:
        self.assertEqual(await self.cliente.ping(), "pong")

    async def asyncTearDown(self) -> None:
        await self.cliente.cerrar()
```

Cada test corre en su propio event loop, así que no hay estado filtrado entre
casos.

## Mockear corrutinas

`unittest.mock.MagicMock` devuelve un `MagicMock`, no un awaitable: usarlo para
una función `async` rompe con `TypeError: object MagicMock can't be used in
'await' expression`. La clase correcta es `AsyncMock`.

```python
from unittest.mock import AsyncMock, patch

@patch("modulo.cliente.get", new_callable=AsyncMock)
async def test_llama_a_la_api(mock_get):
    mock_get.return_value = {"ok": True}
    assert await consultar() == {"ok": True}
    mock_get.assert_awaited_once()
```

`AsyncMock` agrega aserciones propias: `assert_awaited()`,
`assert_awaited_once_with()` y `await_count`.

## Testear timeouts sin esperar de verdad

Un test que realmente duerme 30 segundos es un test que nadie va a correr. Las
salidas son inyectar el timeout como parámetro y usar un valor chico, o
reemplazar la operación lenta por un `AsyncMock` con `side_effect=TimeoutError`.
