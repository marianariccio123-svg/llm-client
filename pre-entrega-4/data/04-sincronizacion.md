---
doc_id: sincronizacion
titulo: Primitivas de sincronización y colas
categoria: concurrencia
fuente_original: https://docs.python.org/3/library/asyncio-sync.html
---

## Por qué hacen falta si hay un solo hilo

Aunque `asyncio` corre en un único hilo, cada `await` es un punto donde otra
tarea puede tomar el control. Si una sección de código lee y después escribe un
estado compartido con un `await` en el medio, otra tarea puede colarse entre
las dos operaciones. Eso es una condición de carrera, sin hilos.

Las primitivas de `asyncio` tienen la misma interfaz que las de `threading`,
pero **no son intercambiables**: hay que usar siempre las asíncronas, porque las
de `threading` bloquean el event loop.

## Lock

`asyncio.Lock` garantiza que una sola tarea a la vez entre a la sección
crítica.

```python
lock = asyncio.Lock()

async def incrementar() -> None:
    async with lock:
        actual = await leer_contador()
        await guardar_contador(actual + 1)
```

No es reentrante: si una tarea que ya tiene el lock intenta tomarlo otra vez,
se cuelga para siempre (deadlock).

## Semaphore: limitar concurrencia

`asyncio.Semaphore(n)` permite hasta `n` tareas simultáneas. Es la herramienta
estándar para no saturar una API externa con cientos de pedidos a la vez.

```python
sem = asyncio.Semaphore(5)        # máximo 5 requests en vuelo

async def pedir(url: str):
    async with sem:
        return await cliente.get(url)
```

`asyncio.BoundedSemaphore` además lanza `ValueError` si se libera más veces de
las que se adquirió, lo que ayuda a detectar bugs.

## Event y Condition

`asyncio.Event` es una bandera de un solo sentido: las tareas hacen
`await evento.wait()` y se despiertan todas cuando alguien llama a
`evento.set()`. Sirve para señalizar "ya está listo el arranque" o
"hay que apagar todo".

`asyncio.Condition` combina un lock con la espera por un predicado: una tarea
hace `await cond.wait_for(lambda: hay_datos)` y otra la despierta con
`cond.notify()`.

## asyncio.Queue: el patrón productor-consumidor

`asyncio.Queue` es una cola FIFO segura entre tareas, con `maxsize` para aplicar
contrapresión: si la cola se llena, el productor se bloquea hasta que haya
lugar.

```python
cola: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

async def productor() -> None:
    for item in items:
        await cola.put(item)

async def consumidor() -> None:
    while True:
        item = await cola.get()
        try:
            await procesar(item)
        finally:
            cola.task_done()
```

`await cola.join()` espera a que todos los ítems hayan sido marcados con
`task_done()`. Las variantes `asyncio.LifoQueue` y `asyncio.PriorityQueue`
cambian el orden de salida.
