---
doc_id: rendimiento-y-debugging
titulo: Rendimiento y depuración
categoria: performance
fuente_original: https://docs.python.org/3/library/asyncio-dev.html#debug-mode
---

## Modo debug

El modo debug de asyncio activa chequeos caros pero muy informativos: avisa
cuando una corrutina nunca fue esperada, cuando una tarea se destruye estando
pendiente, y cuando un callback tarda demasiado en ejecutarse.

Se enciende de tres maneras equivalentes:

```bash
PYTHONASYNCIODEBUG=1 python app.py
python -X dev app.py
```

```python
asyncio.run(main(), debug=True)
```

## Slow callback warnings

Con el modo debug activo, cualquier callback o paso de corrutina que tarde más
de 100 ms aparece en el log:

```
Executing <Task ...> took 0.512 seconds
```

Ese mensaje es la forma más directa de encontrar código bloqueante escondido.
El umbral se ajusta con `loop.slow_callback_duration = 0.05`.

## Medir de verdad

`time.perf_counter()` alrededor de un bloque alcanza para comparar dos
implementaciones. Para un perfil completo, `yappi` entiende corrutinas y
atribuye el tiempo por tarea, cosa que `cProfile` no hace bien en código
asíncrono.

`asyncio.all_tasks()` devuelve las tareas vivas del loop y sirve para detectar
fugas: si ese número crece indefinidamente, hay tareas que nunca terminan.

## uvloop

`uvloop` es una implementación alternativa del event loop basada en libuv,
entre dos y cuatro veces más rápida que la de la biblioteca estándar en trabajo
de red intensivo. Se instala con `pip install uvloop` y se activa en una línea:

```python
import asyncio, uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```

No está disponible en Windows. En ese caso, `WindowsSelectorEventLoopPolicy`
suele dar mejor compatibilidad con librerías de terceros que el
`ProactorEventLoop` que viene por defecto.

## Cuándo asyncio no es la respuesta

Si el cuello de botella es CPU (procesar imágenes, entrenar un modelo, calcular
hashes), `asyncio` no ayuda: un solo hilo sigue siendo un solo hilo. La
herramienta correcta ahí es `multiprocessing` o un `ProcessPoolExecutor`. La
regla práctica: `asyncio` para esperar, procesos para calcular.
