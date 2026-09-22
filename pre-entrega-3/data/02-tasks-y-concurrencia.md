# Tareas, gather y TaskGroup

## Corrutina vs. Task

Llamar a una corrutina (`foo()`) no ejecuta nada: devuelve un objeto corrutina
inerte. Recién cuando se la espera con `await` o se la envuelve en una Task se
agenda en el event loop. `asyncio.create_task(coro)` la envuelve y la agenda
inmediatamente, permitiendo que corra concurrentemente con el código que sigue.

Una Task guardada solo en una variable local puede ser recolectada por el
garbage collector antes de terminar, porque el event loop solo mantiene una
referencia débil a ella. La práctica recomendada es guardar las tareas en un
`set` a nivel de módulo y descartarlas en el callback de finalización:

```python
_background_tasks = set()

task = asyncio.create_task(worker())
_background_tasks.add(task)
task.add_done_callback(_background_tasks.discard)
```

## asyncio.gather

`asyncio.gather(*coros)` ejecuta varias corrutinas concurrentemente y devuelve
la lista de resultados en el mismo orden en que se pasaron los argumentos, sin
importar el orden en que terminaron.

Por defecto, si una de las corrutinas lanza una excepción, `gather` propaga esa
excepción inmediatamente pero **no cancela** las demás, que quedan corriendo en
segundo plano. Con `return_exceptions=True` las excepciones se devuelven como
elementos de la lista de resultados en lugar de propagarse.

## TaskGroup

Desde Python 3.11 existe `asyncio.TaskGroup`, la alternativa recomendada frente
a `gather` para concurrencia estructurada:

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(fetch(1))
    tg.create_task(fetch(2))
```

A diferencia de `gather`, si una tarea del grupo falla, el TaskGroup cancela
automáticamente todas las tareas hermanas y, al salir del bloque `async with`,
lanza un `ExceptionGroup` que agrupa todos los errores ocurridos. Ese
`ExceptionGroup` se captura con la sintaxis `except*`.

## Límite de concurrencia

Lanzar 10.000 tareas a la vez contra una API la va a tirar abajo. El patrón
estándar es acotar con un semáforo:

```python
sem = asyncio.Semaphore(20)

async def limitado(url):
    async with sem:
        return await fetch(url)
```
