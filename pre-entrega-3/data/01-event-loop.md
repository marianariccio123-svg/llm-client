# El event loop de asyncio

El event loop es el núcleo de `asyncio`. Es un bucle de despacho de tareas de
un solo hilo que mantiene una cola de callbacks listos para ejecutarse y un
conjunto de descriptores de archivo vigilados por el selector del sistema
operativo (`epoll` en Linux, `kqueue` en macOS, IOCP en Windows).

## Cómo se obtiene el loop

Desde Python 3.10 en adelante, la forma recomendada de arrancar un programa
asíncrono es `asyncio.run(main())`. Esa función crea un event loop nuevo,
ejecuta la corrutina hasta que termina, cancela las tareas pendientes,
cierra los generadores asíncronos y finalmente cierra el loop.

Llamar a `asyncio.get_event_loop()` desde código síncrono cuando no hay un
loop corriendo está deprecado desde Python 3.12 y emite un `DeprecationWarning`.
Dentro de código asíncrono la función correcta es `asyncio.get_running_loop()`,
que lanza `RuntimeError` si no hay loop activo.

## Política de un solo hilo

El event loop no es thread-safe. Para agendar trabajo en el loop desde otro
hilo hay que usar `loop.call_soon_threadsafe(callback)` o
`asyncio.run_coroutine_threadsafe(coro, loop)`, que devuelve un
`concurrent.futures.Future`.

## Bloquear el loop es el error más caro

Cualquier llamada bloqueante dentro de una corrutina congela el loop entero:
`time.sleep()`, `requests.get()`, lecturas de archivo con `open()`, o un
cálculo de CPU largo. El síntoma típico es que la latencia p99 del servicio
se dispara aunque el uso de CPU sea bajo.

La solución para trabajo bloqueante de I/O es `asyncio.to_thread(func, *args)`,
que lo delega a un `ThreadPoolExecutor`. Para trabajo intensivo de CPU hay que
usar `loop.run_in_executor(ProcessPoolExecutor(), func, *args)`, porque el GIL
impide que los hilos aprovechen varios núcleos.

## Modo debug

Activar el modo debug con `asyncio.run(main(), debug=True)` o la variable de
entorno `PYTHONASYNCIODEBUG=1` hace que el loop registre un warning cuando un
callback tarda más de 100 ms en ejecutarse. Ese umbral se ajusta con
`loop.slow_callback_duration = 0.5`.
