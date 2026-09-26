# Pre-entrega 4 — RAG escalable en la nube con Pinecone

Módulo de recuperación escalable: indexa un corpus técnico en un índice
**Pinecone Serverless** con metadata avanzada, lo consulta con un **recuperador
híbrido** (BM25 léxico + búsqueda vectorial, combinados con `EnsembleRetriever`)
y **mide** la calidad de esa recuperación con Precision@5 y Recall@5 sobre un
Golden Set.

Es la continuación natural de la Pre-entrega 3: allá el índice era ChromaDB
local y la búsqueda puramente semántica; acá el índice vive en la nube, es
multi-namespace, y la recuperación deja de ser un acto de fe porque hay números
que la respaldan.

El corpus de ejemplo es documentación técnica sobre **programación asíncrona en
Python** (`asyncio`): 8 documentos que cubren event loop, tasks y `TaskGroup`,
timeouts y cancelación, primitivas de sincronización, streams y subprocesos,
testing, errores comunes, y rendimiento.

---

## Qué hace

1. **Infraestructura** (`setup_index.py`): crea el índice Serverless si no
   existe y, si existe, verifica que la dimensión coincida antes de que se
   gaste un solo embedding.
2. **Ingesta** (`ingest.py`): lee los `.md` de `data/`, los parte con
   `RecursiveCharacterTextSplitter` midiendo en **tokens reales** (`tiktoken`),
   genera embeddings de **1536 dimensiones** y los sube a Pinecone con metadata
   validada por Pydantic: `doc_id`, `fuente`, `titulo`, `categoria`, `pagina`,
   `chunk_index`, `n_tokens` y el **texto original**.
3. **Recuperación híbrida** (`rag.py`): la clase `RAGSystem` encapsula un
   `EnsembleRetriever` que fusiona BM25 y Pinecone, y devuelve los top-5.
4. **Generación con citas**: `RAGSystem.answer()` es `async`, arma el prompt con
   los fragmentos recuperados y devuelve un `RAGResponse` de Pydantic. Si la
   pregunta no está en el corpus, el sistema lo dice en vez de inventar.
5. **Evaluación** (`evaluate.py`): corre un Golden Set de 5 preguntas contra los
   tres modos (híbrido, denso, BM25) e imprime Recall@5, Precision@5 y MRR.

---

## Decisiones de diseño

### Por qué un recuperador híbrido

Los dos recuperadores fallan de maneras **complementarias**:

| | Fuerte en | Débil en |
|---|---|---|
| **Denso** (Pinecone) | paráfrasis, sinónimos, preguntas en lenguaje natural | identificadores exactos: para el embedding `TaskGroup` y `gather` viven casi en el mismo punto |
| **Léxico** (BM25) | nombres propios raros: `run_in_executor`, `AsyncMock`, `CancelledError` (IDF alto) | preguntas que no comparten vocabulario con el documento |

En documentación técnica las consultas mezclan las dos cosas ("¿qué pasa si
varias tareas de un **TaskGroup** fallan?"), así que la expectativa es que la
unión rinda mejor que cualquiera sola. `evaluate.py` existe para **medirlo en
lugar de asumirlo** — y en este corpus la medición dice que no: ver
[Métricas](#métricas).

`EnsembleRetriever` fusiona con **Reciprocal Rank Fusion**: cada documento suma
`peso_i / (60 + rank_i)` por cada lista donde aparece. Lo importante es que
compara **posiciones**, no scores crudos — los scores de BM25 y las similitudes
coseno de Pinecone viven en escalas incomparables y sumarlos directamente no
significaría nada.

### El texto viaja dentro de la metadata de Pinecone

Cada vector lleva su propio `text` en la metadata. Así una búsqueda devuelve el
contenido listo para armar el prompt, sin una segunda consulta a una base
relacional para resolver `id -> texto`. `ChunkMetadata` valida que el chunk no
supere el límite de 40 KB de metadata por vector **antes** del upsert, porque
Pinecone rechaza el batch entero y no el ítem que falla.

### La dimensión es un contrato, y se verifica

Un índice de Pinecone se crea con una dimensión fija que **no se puede cambiar**.
Si el modelo emite vectores de otro tamaño, el índice acepta la conexión sin
quejarse y recién falla en el upsert — después de haber pagado los embeddings de
todo el corpus.

Por eso `EMBEDDING_DIM = 1536` es una constante única de la que dependen tanto
la creación del índice como el modelo de embeddings, y `setup_index.py` corta
con un mensaje accionable si encuentra un índice incompatible:

```
MISMATCH DE DIMENSIONES.
  El índice 'asyncio-docs' fue creado con dimensión 768, pero este proyecto
  genera embeddings de 1536.
  La dimensión de un índice no se puede cambiar. Tenés dos salidas:
    1) usar otro nombre:  INDEX_NAME=asyncio-docs-1536 en el .env
    2) borrar y recrear:  python setup_index.py --delete && python setup_index.py
```

### 1536 dimensiones con Gemini

`text-embedding-3-small` de OpenAI es nativamente de 1536. `gemini-embedding-001`
emite 3072 pero acepta truncarse vía `output_dimensionality`, así que se lo fija
en 1536 también. Resultado: **el mismo índice sirve para los dos proveedores**
(aunque no se pueden mezclar en el mismo namespace — son espacios vectoriales
distintos).

El default es Gemini porque su API key gratuita es la que tiene cupo en este
proyecto. Con `PROVIDER=openai` en el `.env` el pipeline corre igual, sin tocar
una línea de código.

### Namespaces

Todo se escribe en el namespace `python-asyncio`, no en el default. Un índice
sin namespaces mezcla todos los corpus en el mismo espacio de búsqueda: los
resultados se vuelven ruidosos y la consulta más lenta. Con namespaces, agregar
un segundo corpus (digamos documentación de `httpx`) es cambiar una variable de
entorno, no crear otro índice.

### Chunking: 500 tokens, cortando por sección

`RecursiveCharacterTextSplitter.from_tiktoken_encoder()` combina lo mejor de dos
mundos: la lógica de separadores del splitter recursivo, pero midiendo el tamaño
con el **tokenizador real** y no con `len()`. Un chunk de 500 caracteres y uno
de 500 tokens no se parecen en nada.

El primer separador es `\n## `, el encabezado de sección, así el corte natural
cae en un límite temático. El detalle importante es que el splitter
**empaqueta** secciones cortas hasta llenar el presupuesto en vez de emitir una
por sección: cortar en cada `##` daba 39 chunks de ~125 tokens, demasiado
chicos para retener contexto. Con el empaquetado quedan **16 chunks de 306
tokens en promedio** (máx. 497), dentro del rango que recomienda la consigna.

### Por qué el SDK nativo de Pinecone y no `PineconeVectorStore`

`langchain-pinecone` fija `numpy<2` y no tiene ninguna release compatible con
Python 3.14, que es el intérprete de este repo (pip intenta compilar numpy
1.26.4 desde el código fuente y falla). La consigna contempla explícitamente
las dos opciones, así que se usa el **SDK nativo** envuelto en un
`BaseRetriever` propio de 30 líneas (`PineconeRetriever` en `rag.py`), que queda
100% compatible con `EnsembleRetriever` como cualquier otro retriever de
LangChain.

### Tokenizador propio para BM25

El default de `BM25Retriever` es `str.split`, que deja la puntuación pegada:
`gather(),` no matchea la consulta `gather`. El tokenizador de `rag.py`
normaliza a minúsculas, conserva enteros los identificadores con punto
(`asyncio.to_thread`) y además los indexa por sus partes, para que la consulta
`to_thread` también los encuentre.

### Carga perezosa, igual que en la Pre-entrega 3

Nada se construye al importar: ni el cliente de Pinecone, ni los embeddings, ni
el LLM, ni el retriever. Todo va detrás de `@lru_cache` y se resuelve en el
primer uso real. Un `import rag` funciona sin una sola API key cargada; el error
aparece recién cuando se intenta usar el modelo, y con un mensaje que dice qué
falta.

---

## Salida real del sistema

`python main.py`, recortado a las partes que importan:

```
======================================================================
PREGUNTA: ¿Cómo ejecuto una función bloqueante sin congelar el programa?
======================================================================

Para ejecutar una llamada bloqueante sin congelar el event loop, la solución
es delegarla a un pool de hilos o procesos.

La forma moderna recomendada es usar `await asyncio.to_thread(func, *args)`,
que por debajo invoca `loop.run_in_executor(None, func)`. Si se trata de
trabajo de CPU, conviene recurrir explícitamente a un `ProcessPoolExecutor`,
ya que los hilos no esquivan el GIL.

Referencias:
  - El event loop de asyncio (pág. 4, 01-event-loop.md) [fundamentos]
  - El event loop de asyncio (pág. 1, 01-event-loop.md) [fundamentos]
  - Errores comunes en asyncio (pág. 1, 07-errores-comunes.md) [troubleshooting]
  - Tasks, gather y TaskGroup (pág. 1, 02-tasks-y-taskgroup.md) [concurrencia]
  - Timeouts y cancelación (pág. 4, 03-timeouts-y-cancelacion.md) [control-de-flujo]

encontrado_en_contexto = True

======================================================================
PREGUNTA: ¿Qué pasa si varias tareas de un TaskGroup fallan al mismo tiempo?
======================================================================

Si varias tareas de un `TaskGroup` fallan, el grupo lanza un `ExceptionGroup`
que las envuelve a todas. Además, cuando una tarea falla, las tareas hermanas
se cancelan automáticamente. Para atrapar este `ExceptionGroup` es necesario
utilizar la sintaxis `except*`, ya que un `except` común no lo atrapa.

Referencias:
  - Tasks, gather y TaskGroup (pág. 4, 02-tasks-y-taskgroup.md) [concurrencia]
  - Tasks, gather y TaskGroup (pág. 1, 02-tasks-y-taskgroup.md) [concurrencia]
  - Timeouts y cancelación (pág. 1, 03-timeouts-y-cancelacion.md) [control-de-flujo]
  ...

encontrado_en_contexto = True

======================================================================
PREGUNTA: ¿Cuál es la capital de Australia?
======================================================================

No tengo esa información en los documentos disponibles.

(sin referencias: la pregunta no está cubierta por el corpus)

encontrado_en_contexto = False
```

La tercera pregunta está en el set de demo **a propósito**: es la prueba de que
el sistema se niega a contestar fuera del corpus en lugar de inventar, y de que
en ese caso no muestra referencias — citar fragmentos que no respaldan nada
sería peor que no citar.

---

## Estructura

```
pre-entrega-4/
├── config.py           # configuración central + clientes perezosos (@lru_cache)
├── schemas.py          # contratos Pydantic: ChunkMetadata, RAGResponse, métricas
├── setup_index.py      # crea/verifica el índice Serverless de Pinecone
├── ingest.py           # pipeline de ingesta: chunking -> embeddings -> upsert
├── rag.py              # PineconeRetriever + RAGSystem (EnsembleRetriever)
├── evaluate.py         # Precision@5, Recall@5 y MRR sobre el Golden Set
├── golden_set.json     # 5 preguntas con su documento fuente conocido
├── main.py             # demo: consulta el sistema y muestra respuesta + citas
├── data/               # corpus: 8 documentos .md con front matter
├── requirements.txt
└── README.md
```

---

## Cómo replicar el índice

### 1. Dependencias

```bash
# desde la raíz del repo
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source venv/bin/activate       # Mac/Linux

pip install -r pre-entrega-4/requirements.txt
```

> **Nota para Python 3.14**: si `pip` intenta compilar `numpy 1.26.4` y falla,
> es porque alguna dependencia transitiva pide `numpy<2`. Este proyecto no usa
> `langchain-pinecone` justamente por eso; con el `requirements.txt` de esta
> carpeta no debería pasar.

### 2. Variables de entorno

En el `.env` de la **raíz del repo** (ver `.env.example`):

```bash
PINECONE_API_KEY=pcsk_...          # gratis en https://app.pinecone.io
GEMINI_API_KEY=...                 # gratis en https://aistudio.google.com/apikey
INDEX_NAME=asyncio-docs
PINECONE_NAMESPACE=python-asyncio
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PROVIDER=gemini                    # o "openai" si cargás OPENAI_API_KEY
```

El plan **Starter** de Pinecone es gratuito y alcanza de sobra: permite un
índice Serverless, que es exactamente lo que usa este proyecto. La región
`us-east-1` en AWS es la única disponible en ese plan.

El `.env` nunca se sube al repositorio (está en `.gitignore`).

### 3. Crear el índice

```bash
cd pre-entrega-4
python setup_index.py
```

Crea el índice Serverless (dimensión 1536, métrica coseno, `aws/us-east-1`),
espera a que esté listo para recibir escrituras e imprime su estado. Si ya
existe, solo verifica que sea compatible.

```bash
python setup_index.py --info      # ver estado y vectores por namespace
python setup_index.py --delete    # borrarlo (pide confirmación escribiendo el nombre)
```

### 4. Ingestar el corpus

```bash
python ingest.py
```

Es **idempotente**: si el namespace ya tiene tantos vectores como chunks hay en
`data/`, no re-indexa nada. Para forzar una reindexación completa:

```bash
python ingest.py --recreate
```

Para ver cómo queda el chunking sin llamar a ninguna API ni gastar cuota:

```bash
python ingest.py --dry-run
```

### 5. Consultar

```bash
python main.py                                    # preguntas de ejemplo
python main.py "¿Cómo limito los pedidos simultáneos?"
python main.py --modo bm25 --solo-recuperar "run_in_executor"
```

### 6. Evaluar

```bash
python evaluate.py                # compara los tres modos
python evaluate.py --modo hibrido
python evaluate.py --json         # salida estructurada
```

---

## Métricas

Salida real de `python evaluate.py` contra el índice `asyncio-docs`
(16 chunks, namespace `python-asyncio`, embeddings Gemini a 1536 dim):

```
==============================================================
RESUMEN — Golden Set de 5 preguntas, k=5
==============================================================
modo         Recall@5    Precision@5    P norm.      MRR
--------------------------------------------------------------
hibrido          1.00           0.44       0.95     1.00
denso            1.00           0.44       0.95     1.00
bm25             1.00           0.32       0.65     0.85
--------------------------------------------------------------
Mejor combinación recall/precisión: hibrido
```

Y con `--k 3`, que es más exigente:

```
==============================================================
RESUMEN — Golden Set de 5 preguntas, k=3
==============================================================
modo         Recall@3    Precision@3    P norm.      MRR
--------------------------------------------------------------
hibrido          1.00           0.67       0.93     0.90
denso            1.00           0.67       0.93     1.00
bm25             0.80           0.47       0.60     0.80
--------------------------------------------------------------
```

### Cómo leer estos números

**Recall@5 = 1.00 en los tres modos no es un logro, es una limitación del
setup.** Con 16 chunks en el índice, pedir 5 ya cubre casi un tercio del corpus:
es muy difícil no acertar. La métrica recién empieza a discriminar en `--k 3`,
donde BM25 solo cae a 0.80.

**Precision@5 = 0.44 parece bajo pero está al 95% del techo.** Cuatro de las
cinco preguntas tienen un único documento relevante, y ese documento aportó
exactamente 2 chunks al índice: el máximo alcanzable es 2/5 = 0.40, porque los
otros 3 slots van a estar ocupados por material irrelevante sí o sí. La columna
`P norm.` (0.95) es la que dice cuán cerca está el recuperador de lo mejor que
este corpus permite.

**Dónde se ve el aporte de cada mitad.** BM25 es el más débil de los tres, pero
no falla parejo:

| Pregunta | BM25 | Denso |
|---|---|---|
| "¿Por qué no hay que tragarse un `CancelledError`?" | puesto 1 — el término es raro, IDF alto | puesto 1 |
| "¿Cómo limito a cinco los pedidos simultáneos?" | P@5 = 0.20 — la respuesta es `Semaphore` y la palabra no está en la pregunta | P@5 = 0.40 |
| "¿Con qué clase se mockea una corrutina?" | **puesto 4** | puesto 1 |

Es exactamente el patrón que predice la teoría: BM25 gana en identificadores
literales y pierde cuando la consulta no comparte vocabulario con el documento.

### El resultado incómodo: el híbrido no le gana al denso

Con este corpus y este Golden Set, **el recuperador híbrido empata con el denso
puro** (Recall, Precision y P norm. idénticos) y a `k=3` incluso queda por
debajo en MRR (0.90 vs 1.00). No hay una ganancia que mostrar.

La razón es que RRF es una fusión democrática: los errores de BM25 también se
propagan. En la pregunta del mock, BM25 pone el documento correcto en el puesto
4, y ese voto arrastra hacia abajo el consenso aunque el denso lo haya puesto
primero.

El barrido de pesos lo confirma (`ENSEMBLE_WEIGHTS=0.5,0.5 python evaluate.py --modo hibrido`):

| pesos BM25/denso | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| 0.5 / 0.5 | 1.00 | 0.44 | 0.87 |
| 0.4 / 0.6 | 1.00 | 0.44 | 0.90 |
| **0.3 / 0.7** (default) | 1.00 | 0.44 | **1.00** |
| 0.2 / 0.8 | 1.00 | 0.44 | 1.00 |

Cuanto menos pesa BM25, mejor anda el híbrido — que es otra forma de decir que
acá el componente léxico no está aportando.

**Advertencia metodológica:** el default de 0.3/0.7 se eligió mirando estos
mismos resultados, sobre 5 preguntas. Eso es ajustar contra el conjunto de
evaluación, y con n=5 el valor no es estadísticamente defendible. Se deja
documentado en vez de presentarlo como un hallazgo.

**Por qué esto igual no invalida el diseño.** El corpus tiene 8 documentos con
vocabulario muy distinto entre sí, que es el caso *fácil* para la búsqueda
vectorial: los embeddings alcanzan para separarlos. El híbrido paga cuando hay
muchos documentos parecidos y la diferencia está en un identificador exacto —
un corpus de miles de páginas donde `asyncio.wait` y `asyncio.wait_for` viven
en secciones casi idénticas. Para demostrarlo haría falta un corpus de otro
orden de magnitud, no otro ajuste de pesos.

---

## Limitaciones conocidas

- **El corpus es chico** (8 documentos, 16 chunks). Con k=5 sobre 16 chunks,
  Recall@5 satura en 1.00 para los tres modos: el Golden Set mide que el
  pipeline funciona, no que el sistema escale. Por eso se reporta también MRR,
  que sí distingue entre "lo encontró" y "lo encontró y lo puso primero". Es
  también la razón por la que el recuperador híbrido no logra superar al denso
  puro — ver el análisis en [Métricas](#métricas).
- **BM25 corre en local, no en Pinecone.** BM25 no es un servicio: es un cálculo
  sobre el corpus entero, que tiene que estar en memoria. `ingest.py` escribe
  `bm25_corpus.jsonl` con exactamente los mismos chunks que subió a Pinecone,
  y `rag.py` arma el índice léxico desde ahí. Para un corpus de millones de
  documentos esto no escala y habría que pasar a los **sparse vectors** nativos
  de Pinecone, que sí corren del lado del servidor.
- **El tier gratuito de Gemini limita las peticiones por minuto y por modelo.**
  Corriendo la demo aparecieron `503 UNAVAILABLE` y un `429 RESOURCE_EXHAUSTED`
  (límite: 5 requests por minuto). El SDK reintenta solo con backoff exponencial
  y el script terminó bien, pero si el error persiste se puede cambiar de modelo
  con `CHAT_MODEL` en el `.env`: cada modelo tiene su propia cuota. La
  recuperación no consume cuota de chat, solo la de embeddings, así que
  `evaluate.py` y `--solo-recuperar` son mucho más baratos que `main.py`.
- **La rama de OpenAI está implementada pero no se probó en vivo** en este
  entorno, por no contar con una API key con crédito. La validación end-to-end
  se hizo con Gemini.
