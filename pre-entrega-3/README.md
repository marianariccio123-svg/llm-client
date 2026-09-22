# Pre-entrega 3 — RAG local con LangChain + ChromaDB

Sistema de **Recuperación Semántica Local (RAG)**: indexa una documentación
propia en una base vectorial persistente y responde preguntas usando
**únicamente** esos documentos, citando los fragmentos que respaldan cada
respuesta. Si la pregunta se sale del corpus, el sistema lo dice en vez de
inventar.

El corpus de ejemplo es documentación técnica sobre **programación asíncrona en
Python** (`asyncio`): event loop, tasks y `TaskGroup`, timeouts y cancelación,
y testing de código asíncrono.

## Qué hace

1. **Ingesta** (`ingest.py`): carga los `.md`/`.txt` de `data/`, los parte en
   chunks de **500 tokens con 50 de overlap** (medidos en tokens reales con
   `tiktoken`, no en caracteres), los vectoriza y los persiste en ChromaDB
   bajo `vectorstore/`. Si ya hay datos indexados, no reprocesa nada.
2. **Recuperación** (`rag.py`): busca por similitud semántica los **4**
   fragmentos más relevantes (`top_k=4`).
3. **Generación**: arma un prompt con esos fragmentos como contexto y le pide
   al LLM que responda **solo** con esa información. El prompt le ordena
   explícitamente responder *"No tengo esa información en los documentos
   disponibles."* cuando la respuesta no está en el contexto.
4. **Salida validada**: `get_rag_response()` es `async`, usa `.ainvoke()` y
   devuelve un modelo Pydantic `RAGResponse` con `respuesta`, `referencias` y
   `encontrado_en_contexto`.

### Decisiones de diseño

- **Lazy loading en todo.** Ni el LLM, ni los embeddings, ni el vectorstore, ni
  la cadena se construyen al importar un módulo: todo va detrás de funciones con
  `@lru_cache` que se resuelven en el primer uso real. Un `import rag` funciona
  aunque no haya ninguna API key cargada; el error recién aparece (y con un
  mensaje claro) cuando se intenta usar el modelo de verdad.
- **Las referencias no se pueden alucinar.** El LLM devuelve solo el texto de la
  respuesta y un flag; la lista de `referencias` se arma en código, a partir de
  los documentos que efectivamente devolvió el retriever.
- **LLM y embeddings siempre del mismo proveedor.** Un índice creado con
  embeddings de un proveedor no se puede consultar con los de otro (distinta
  dimensión y distinto espacio vectorial), así que `PROVIDER` gobierna los dos.

## Estructura de archivos

```
pre-entrega-3/
├── data/                            # Corpus a indexar
│   ├── 01-event-loop.md
│   ├── 02-tasks-y-concurrencia.md
│   ├── 03-timeouts-y-cancelacion.md
│   └── 04-testing-y-errores-comunes.md
├── vectorstore/                     # ChromaDB persistido (generado, en .gitignore)
├── config.py                        # Proveedor, modelos, rutas y constructores lazy
├── schemas.py                       # RAGResponse y Referencia (Pydantic)
├── ingest.py                        # Carga, chunking, vectorización y persistencia
├── rag.py                           # Retriever + cadena LCEL + get_rag_response()
├── main.py                          # Script de prueba (2 casos)
├── requirements.txt
└── README.md
```

## Configuración

El `.env` está en la **raíz del repo** (un nivel arriba de esta carpeta) y se
comparte con la Pre-entrega 1 y la Pre-entrega 2. No hace falta crear uno nuevo.

| Variable | Obligatoria | Descripción |
|---|---|---|
| `PROVIDER` | No (default `gemini`) | `gemini`, `openai` o `anthropic`. Gobierna LLM **y** embeddings. |
| `GEMINI_API_KEY` | Sí, si `PROVIDER=gemini` | API key de Google AI Studio. |
| `OPENAI_API_KEY` | Sí, si `PROVIDER=openai` o `anthropic` | Anthropic no tiene embeddings propios: esa rama usa los de OpenAI. |
| `ANTHROPIC_API_KEY` | Sí, si `PROVIDER=anthropic` | |
| `CHAT_MODEL` | No | Pisa el modelo de chat por defecto. Útil si se agota la cuota diaria del free tier. |

Modelos por defecto:

| Proveedor | Chat | Embeddings |
|---|---|---|
| `gemini` | `gemini-flash-latest` | `models/gemini-embedding-001` |
| `openai` | `gpt-4o-mini` | `text-embedding-3-small` |
| `anthropic` | `claude-3-5-haiku-latest` | `text-embedding-3-small` (de OpenAI) |

> **Nota sobre proveedores.** La consigna menciona OpenAI/Anthropic. El código
> para esos proveedores está implementado, pero en este proyecto solo hay cupo
> en la API de Gemini, así que la prueba end-to-end se hizo y se verificó con
> Gemini. Es la misma convención que en las Pre-entregas 1 y 2.

## Cómo instalar

Desde la raíz del repo, con el venv ya existente:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r pre-entrega-3\requirements.txt
```

## Cómo correr la ingesta

```powershell
cd pre-entrega-3
python ingest.py
```

Salida real de la primera corrida:

```
15:28:45 [INFO] __main__: Proveedor activo: gemini
15:28:45 [INFO] __main__: Vectorstore: C:\Coder House\llm-client\pre-entrega-3\vectorstore
15:28:45 [INFO] config: Inicializando embeddings (gemini / models/gemini-embedding-001)...
15:28:50 [INFO] __main__: Cargado: 01-event-loop.md (2052 caracteres)
15:28:50 [INFO] __main__: Cargado: 02-tasks-y-concurrencia.md (2130 caracteres)
15:28:50 [INFO] __main__: Cargado: 03-timeouts-y-cancelacion.md (2296 caracteres)
15:28:50 [INFO] __main__: Cargado: 04-testing-y-errores-comunes.md (2474 caracteres)
15:28:50 [INFO] __main__: Total de documentos cargados: 4
15:29:07 [INFO] __main__: Se generaron 8 chunks (chunk_size=500 tokens, overlap=50 tokens).
15:29:07 [INFO] __main__:   01-event-loop.md                    -> 2 chunks
15:29:07 [INFO] __main__:   02-tasks-y-concurrencia.md          -> 2 chunks
15:29:07 [INFO] __main__:   03-timeouts-y-cancelacion.md        -> 2 chunks
15:29:07 [INFO] __main__:   04-testing-y-errores-comunes.md     -> 2 chunks
15:29:07 [INFO] __main__: Vectorizando e indexando 8 chunks...
15:29:08 [INFO] __main__: Ingesta terminada. Chunks indexados en la colección: 8
```

En la segunda corrida detecta que ya hay datos y no reprocesa:

```
15:38:49 [INFO] ingest: El vectorstore ya tiene 8 chunks indexados. No se reprocesa
nada (usá --force para reindexar de cero).
```

Para reindexar de cero (por ejemplo después de editar `data/` o de cambiar de
proveedor de embeddings):

```powershell
python ingest.py --force
```

## Cómo correr las pruebas

```powershell
cd pre-entrega-3
python main.py
```

`main.py` corre la ingesta si hace falta y después prueba dos casos: una
pregunta cuya respuesta está en los documentos y una **pregunta trampa** sobre
un tema que no aparece en ningún documento.

También acepta una pregunta propia por línea de comandos:

```powershell
python main.py "¿Cómo se limita la concurrencia con un semáforo?"
```

## Ejemplo de salida real

Corrida verificada con `PROVIDER=gemini` y `CHAT_MODEL=gemini-flash-lite-latest`.

### Caso 1 — pregunta respondible

**Pregunta:** *¿Qué diferencia hay entre asyncio.gather y asyncio.TaskGroup
cuando una de las tareas falla?*

```json
{
  "respuesta": "Por defecto, si una corrutina en asyncio.gather falla, se propaga la excepción inmediatamente pero no cancela las demás tareas que siguen corriendo en segundo plano, a menos que se use return_exceptions=True. En cambio, asyncio.TaskGroup cancela automáticamente todas las tareas hermanas si una de ellas falla y al salir del bloque async with lanza un ExceptionGroup que agrupa todos los errores.",
  "referencias": [
    {
      "fuente": "02-tasks-y-concurrencia.md#chunk0",
      "fragmento": "# Tareas, gather y TaskGroup ## Corrutina vs. Task Llamar a una corrutina (`foo()`) no ejecuta nada: devuelve un objeto corrutina inerte. Recién cuando se la espera con `await` o se la envuelve en una Task se agenda en el event loop. `asyncio.create_task(coro)` la envuelve y la agenda inmediatamente..."
    },
    {
      "fuente": "03-timeouts-y-cancelacion.md#chunk0",
      "fragmento": "# Timeouts, cancelación y limpieza ## Cancelación cooperativa `task.cancel()` no mata la tarea: agenda el lanzamiento de una excepción `asyncio.CancelledError` en el próximo punto de `await` de esa tarea. Si la tarea está corriendo un bucle de CPU sin `await`, la cancelación nunca llega a efectuarse..."
    },
    {
      "fuente": "04-testing-y-errores-comunes.md#chunk1",
      "fragmento": "**`RuntimeError: Task attached to a different loop`**: un objeto creado en un loop (un `asyncio.Lock`, una `Queue`, un cliente HTTP) se usó desde otro. Las primitivas de sincronización de asyncio deben crearse dentro del mismo loop que las consume. **Sesión HTTP sin cerrar**: un `aiohttp.ClientSessi..."
    },
    {
      "fuente": "03-timeouts-y-cancelacion.md#chunk1",
      "fragmento": "## Blindar una tarea contra la cancelación `asyncio.shield(coro)` evita que la cancelación del código que espera se propague a la corrutina protegida. Es útil para operaciones que no deben interrumpirse a la mitad, como confirmar una transacción. Ojo: si el event loop se cierra, la tarea blindada mu..."
    }
  ],
  "encontrado_en_contexto": true
}
```

La respuesta es exacta y coincide con lo que dice `02-tasks-y-concurrencia.md`.

### Caso 2 — pregunta trampa

**Pregunta:** *¿Cuál es el precio mensual del plan empresarial de AWS Lambda y
qué descuento tiene si se paga por año?*

```json
{
  "respuesta": "No tengo esa información en los documentos disponibles.",
  "referencias": [],
  "encontrado_en_contexto": false
}
```

El retriever igual devolvió 4 fragmentos (siempre devuelve los `top_k` más
cercanos, por lejanos que estén), pero el modelo detectó que ninguno responde la
pregunta y no alucinó un precio. Como `encontrado_en_contexto` es `false`, el
código además vacía las referencias: no tiene sentido citar fuentes que no
respaldan nada.

## Limitaciones conocidas

- **Cuota del free tier de Gemini.** `gemini-flash-latest` permite 20 requests
  por día en el plan gratuito y se agota rápido si se itera. Por eso existe la
  variable `CHAT_MODEL`: la corrida documentada arriba usa
  `gemini-flash-lite-latest`, que tiene su propia cuota. Con cuota disponible,
  el default (`gemini-flash-latest`) funciona igual.
- **El retriever siempre devuelve `top_k` fragmentos**, sin umbral de similitud
  mínima. Filtrar la irrelevancia queda enteramente a cargo del prompt. Un
  `similarity_score_threshold` o un reranker harían el filtro más robusto.
- **El "no lo sé" depende del modelo.** El flag `encontrado_en_contexto` lo
  decide el LLM. Con `temperature=0` y un prompt explícito el comportamiento es
  estable en las pruebas, pero no es una garantía formal: un modelo más chico o
  una pregunta parcialmente cubierta por el corpus pueden dar falsos positivos.
- **Búsqueda solo densa.** No hay búsqueda híbrida (BM25 + vectorial), así que
  las consultas por término exacto poco frecuente pueden fallar.
- **Sin memoria conversacional.** Cada llamada a `get_rag_response()` es
  independiente: no resuelve preguntas de seguimiento del tipo *"¿y eso cómo se
  testea?"*, porque no reescribe la consulta con el historial.
- **Ingesta sin detección de cambios.** El chequeo de persistencia es "¿hay algo
  indexado?", no un hash por archivo. Si se edita un documento de `data/`, hay
  que reindexar con `--force`.
- **Cambiar de proveedor obliga a reindexar.** Los vectores de Gemini y de
  OpenAI no son compatibles entre sí; hay que correr `python ingest.py --force`
  después de cambiar `PROVIDER`.
- **Corpus chico.** Son 8 chunks en total, así que con `top_k=4` cada consulta
  recupera la mitad del corpus. Con un corpus real (cientos de chunks) la
  calidad de la recuperación pesaría mucho más que acá.
