---
doc_id: event-loop
titulo: El event loop de asyncio
categoria: fundamentos
fuente_original: https://docs.python.org/3/library/asyncio-eventloop.html
---

## Qué es el event loop

El event loop es el núcleo de `asyncio`: un bucle de un solo hilo que mantiene
una cola de tareas listas para ejecutarse y las va corriendo de a una. Cuando
una corrutina llega a un `await` sobre una operación de E/S, devuelve el control
al loop, que aprovecha ese hueco para avanzar otra tarea. No hay paralelismo
real de CPU: hay *entrelazado* de esperas.

Por eso `asyncio` brilla en trabajo limitado por E/S (llamadas HTTP, consultas a
base de datos, lectura de sockets) y no aporta nada en trabajo limitado por CPU.

## Arrancar el loop con asyncio.run

`asyncio.run(coro)` es la única forma recomendada de arrancar un programa
asíncrono desde código síncrono. Crea un event loop nuevo, ejecuta la corrutina
hasta el final, cancela las tareas pendientes, cierra los generadores
asíncronos y finalmente cierra el loop.

```python
import asyncio

async def main() -> None:
    print("hola")

asyncio.run(main())
```

Llamar a `asyncio.run()` cuando ya hay un loop corriendo lanza
`RuntimeError: asyncio.run() cannot be called from a running event loop`. Dentro
de código que ya es asíncrono se usa `await` directamente.

## get_running_loop y get_event_loop

`asyncio.get_running_loop()` devuelve el loop activo y falla con `RuntimeError`
si no hay ninguno. Es la función correcta desde código asíncrono.

`asyncio.get_event_loop()` está obsoleta fuera de un loop en ejecución: en
versiones modernas emite `DeprecationWarning` y en el futuro va a lanzar error.
No la uses en código nuevo.

## Trabajo bloqueante con run_in_executor

Una llamada bloqueante (`time.sleep`, `requests.get`, un cálculo pesado,
lectura de disco sincrónica) congela el loop entero: ninguna otra tarea avanza
mientras dure. La salida es delegarla a un pool de hilos o procesos.

```python
import asyncio, time

def tarea_lenta() -> str:
    time.sleep(3)          # bloqueante a propósito
    return "listo"

async def main() -> None:
    resultado = await asyncio.to_thread(tarea_lenta)
    print(resultado)
```

`asyncio.to_thread()` es el atajo moderno; por debajo usa
`loop.run_in_executor(None, func)`. Para trabajo de CPU conviene un
`ProcessPoolExecutor` explícito, porque los hilos no esquivan el GIL.
