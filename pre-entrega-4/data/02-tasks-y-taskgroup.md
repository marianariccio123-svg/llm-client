---
doc_id: tasks-y-taskgroup
titulo: Tasks, gather y TaskGroup
categoria: concurrencia
fuente_original: https://docs.python.org/3/library/asyncio-task.html
---

## Corrutina vs Task

Llamar a una función `async` no ejecuta nada: devuelve un objeto corrutina
inerte. Recién se ejecuta cuando alguien la espera con `await` o la programa en
el loop con `asyncio.create_task()`.

La diferencia es concurrencia: `await coro()` corre una cosa y espera; 
`asyncio.create_task(coro())` la agenda de inmediato y sigue, así que varias
tareas progresan entrelazadas.

```python
tarea = asyncio.create_task(descargar(url))   # arranca ya
otra_cosa()                                   # sigue en paralelo lógico
resultado = await tarea                       # recién acá se espera
```

## asyncio.gather

`asyncio.gather(*aws)` espera varios awaitables y devuelve los resultados en el
mismo orden en que fueron pasados, sin importar en qué orden terminaron.

Con `return_exceptions=False` (el default), la primera excepción se propaga de
inmediato, pero **las demás tareas siguen corriendo**: quedan huérfanas si nadie
las cancela. Con `return_exceptions=True` las excepciones vuelven como valores
dentro de la lista de resultados y hay que revisarlas a mano.

## TaskGroup: la forma moderna

`asyncio.TaskGroup` (Python 3.11+) da concurrencia estructurada: el bloque
`async with` no sale hasta que todas sus tareas terminaron, y si una falla, las
hermanas se cancelan automáticamente.

```python
async def main() -> None:
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(pedir("/a"))
        t2 = tg.create_task(pedir("/b"))
    print(t1.result(), t2.result())
```

Es la opción por defecto en código nuevo: no deja tareas huérfanas y el alcance
de vida de cada tarea es visible en el código.

## ExceptionGroup y except*

Si varias tareas de un `TaskGroup` fallan, el grupo lanza un `ExceptionGroup`
que las envuelve a todas. Se atrapa con la sintaxis `except*`:

```python
try:
    async with asyncio.TaskGroup() as tg:
        ...
except* ValueError as eg:
    for err in eg.exceptions:
        print("falló:", err)
```

Un `except ValueError` común **no** atrapa un `ExceptionGroup`: hace falta el
asterisco.

## as_completed y wait

`asyncio.as_completed(aws)` entrega los resultados a medida que van terminando,
útil para mostrar progreso. `asyncio.wait(tasks, return_when=FIRST_COMPLETED)`
devuelve dos conjuntos, `done` y `pending`, y deja la cancelación de los
pendientes en tus manos.
