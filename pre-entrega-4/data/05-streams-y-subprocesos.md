---
doc_id: streams-y-subprocesos
titulo: Streams de red y subprocesos
categoria: entrada-salida
fuente_original: https://docs.python.org/3/library/asyncio-stream.html
---

## Streams: sockets sin protocolos ni transports

La API de streams es la capa de alto nivel para trabajar con sockets. Devuelve
un par `(reader, writer)` y evita tener que escribir clases `Protocol` y manejar
`Transport` a mano.

```python
reader, writer = await asyncio.open_connection("example.com", 80)
writer.write(b"GET / HTTP/1.0\r\n\r\n")
await writer.drain()
datos = await reader.read(-1)
writer.close()
await writer.wait_closed()
```

`await writer.drain()` es obligatorio después de escribir: aplica contrapresión
y espera a que el buffer de salida se vacíe. Olvidarlo hace que la memoria del
proceso crezca sin control cuando el otro extremo lee más lento de lo que
escribimos.

## Métodos de lectura

- `read(n)` devuelve hasta `n` bytes, o lo que haya disponible.
- `readline()` lee hasta el `\n` inclusive.
- `readexactly(n)` lee exactamente `n` bytes o lanza `IncompleteReadError`.
- `readuntil(sep)` lee hasta el separador indicado.

## Servidores con start_server

`asyncio.start_server(handler, host, port)` levanta un servidor TCP y llama al
handler, una corrutina, por cada conexión entrante.

```python
async def manejar(reader, writer) -> None:
    datos = await reader.readline()
    writer.write(datos.upper())
    await writer.drain()
    writer.close()

async def main() -> None:
    server = await asyncio.start_server(manejar, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()
```

## Subprocesos asíncronos

`asyncio.create_subprocess_exec()` y `create_subprocess_shell()` lanzan procesos
externos sin bloquear el loop.

```python
proc = await asyncio.create_subprocess_exec(
    "git", "status", "--short",
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
salida, error = await proc.communicate()
print(proc.returncode, salida.decode())
```

Usá `create_subprocess_exec` con argumentos separados y no `create_subprocess_shell`
con una cadena armada por concatenación: lo segundo abre la puerta a inyección
de comandos.

Para leer la salida a medida que se produce se usa `proc.stdout.readline()` en
un bucle, en lugar de `communicate()`, que espera a que el proceso termine.
