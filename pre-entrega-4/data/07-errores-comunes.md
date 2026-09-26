---
doc_id: errores-comunes
titulo: Errores comunes en asyncio
categoria: troubleshooting
fuente_original: https://docs.python.org/3/library/asyncio-dev.html
---

## "coroutine was never awaited"

`RuntimeWarning: coroutine 'x' was never awaited` significa que se llamó a una
función `async` y se descartó el objeto devuelto sin esperarlo. El cuerpo de la
corrutina nunca se ejecutó.

```python
guardar(datos)          # mal: no hace nada
await guardar(datos)    # bien
```

Como es un *warning* y no un error, el programa sigue andando y el bug pasa
desapercibido. Conviene correr los tests con `-W error::RuntimeWarning`.

## Bloquear el event loop

La causa número uno de que un programa async ande lento. Cualquier llamada
sincrónica pesada dentro de una corrutina congela todas las demás tareas:

- `time.sleep()` en lugar de `await asyncio.sleep()`
- `requests.get()` en lugar de un cliente asíncrono como `httpx.AsyncClient`
- lectura de archivos grandes con `open()`
- serialización de un JSON enorme, hashing, compresión

La solución es usar la variante asíncrona de la librería, o delegar la llamada
con `await asyncio.to_thread(func, *args)`.

## Tareas fire-and-forget que desaparecen

El event loop guarda solo una **referencia débil** a las tareas. Si no se
conserva el objeto devuelto por `create_task()`, el recolector de basura puede
destruirla a mitad de camino y la tarea se cancela silenciosamente.

```python
tareas: set[asyncio.Task] = set()

def lanzar(coro) -> None:
    t = asyncio.create_task(coro)
    tareas.add(t)
    t.add_done_callback(tareas.discard)
```

La alternativa mejor es no hacer fire-and-forget: usar un `TaskGroup`.

## Excepciones que nunca se ven

Si una tarea falla y nadie hace `await` sobre ella ni llama a `.result()`, la
excepción queda guardada en el objeto `Task` y solo aparece como
`Task exception was never retrieved` cuando el recolector la destruye — a veces
mucho después, a veces nunca.

## Mezclar asyncio con threading

Llamar a una corrutina desde otro hilo no funciona. Para eso está
`asyncio.run_coroutine_threadsafe(coro, loop)`, que devuelve un
`concurrent.futures.Future`. En el sentido contrario, para avisarle al loop
desde otro hilo, se usa `loop.call_soon_threadsafe(callback)`.

## await dentro de un bucle: concurrencia que no lo es

```python
for url in urls:
    resultados.append(await pedir(url))    # secuencial, uno por uno
```

Esto no gana nada frente a código sincrónico. La versión concurrente agrupa las
corrutinas y las espera juntas con `asyncio.gather()` o un `TaskGroup`.
