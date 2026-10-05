# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente ReAct construido con **LangGraph**: decide por su cuenta qué
herramientas usar, encadena varias consultas para llegar a una respuesta, se
corrige cuando una herramienta le devuelve un error, y **recuerda** la
conversación entre turnos — incluso entre ejecuciones distintas del programa,
porque el estado vive en SQLite.

El dominio es una tienda: clientes, pedidos, items y stock, en una base SQLite
real que el agente consulta con cuatro herramientas asíncronas.

Es la continuación de las entregas anteriores: la 3 y la 4 recuperaban
información, pero alguien tenía que decidir qué buscar. Acá esa decisión la
toma el modelo, y el aporte del código es la **topología** que le permite
equivocarse y volver a intentar.

---

## Qué cumple de la consigna

| Criterio | Dónde está |
|---|---|
| Autonomía: el agente decide cuándo llamar a una herramienta, sin `if/else` | [`agent.py`](agent.py) — `tools_condition` es la única bifurcación, y mira el mensaje del modelo |
| Ciclo de retorno ante error o info incompleta | [`traces/reintento.log`](traces/reintento.log) y [`traces/ambiguedad.log`](traces/ambiguedad.log) |
| Resiliencia de estado con `thread_id` | [`traces/memoria.log`](traces/memoria.log) — 3 turnos encadenados |
| `StateGraph` que hereda de `MessagesState` | `EstadoAgente` en [`agent.py`](agent.py) |
| Al menos 1 herramienta con `@tool` y docstring descriptivo | 4 en [`tools.py`](tools.py) |
| Persistencia con SqliteSaver | `AsyncSqliteSaver` en [`agent.py`](agent.py) — ver la nota más abajo |
| Prueba multi-paso (herramienta invocada ≥2 veces) | [`traces/multipaso.log`](traces/multipaso.log) — 3 llamadas encadenadas |
| `recursion_limit` definido | `config_thread()` en [`agent.py`](agent.py), default 10 |
| Traza de ejecución como `.json` o log | [`traces/`](traces/) — las 4 en los dos formatos |
| Tipado estático y `asyncio` | Todo el código: `from __future__ import annotations`, nodos y herramientas `async def` |
| Sin API keys en el repo | Variables de entorno, `.env` en `.gitignore` |

---

## El grafo

```
        START
          │
          ▼
      ┌────────┐   tools_condition
      │ modelo │ ─────────────────► END   (no pidió herramientas: ya respondió)
      └────────┘
          │  (pidió una o más herramientas)
          ▼
   ┌──────────────┐
   │ herramientas │
   └──────────────┘
          │
          └──────► vuelve a `modelo`
```

Esa arista de retorno es todo el razonamiento cíclico. El modelo ve el
resultado de la herramienta como un mensaje más y vuelve a decidir: puede pedir
otra herramienta, reintentar la misma con argumentos corregidos, o contestar.

**No hay una sola línea que programe la secuencia.** El código no sabe que para
responder "¿qué compró Mariana?" hay que llamar a `buscar_cliente`, después a
`buscar_pedidos` y después a `detalle_pedido`. Eso lo arma el modelo leyendo
los docstrings.

---

## Decisiones de diseño

### Los docstrings son el código que más importa

El LLM elige qué herramienta llamar leyendo **únicamente** el nombre, la firma
y el docstring. No ve el cuerpo de la función. Por eso cada docstring en
[`tools.py`](tools.py) dice tres cosas, y la tercera es la que hace que el
agente razone en varios pasos solo:

1. Qué hace y cuándo conviene usarla.
2. Qué devuelve, con un ejemplo concreto de la forma del resultado.
3. **Con qué otra herramienta se encadena.**

Por ejemplo, `buscar_pedidos` dice explícitamente *"Esta herramienta devuelve
los TOTALES de cada pedido, no qué productos tenía adentro. Para eso usá
`detalle_pedido` con el pedido_id"*. Sin esa frase el agente se queda en el
primer paso y contesta a medias.

### Los errores traen la salida puesta

Ninguna herramienta lanza excepciones hacia el grafo. Devuelven un dict con
`"error"` y, cuando se puede, **opciones concretas para reintentar**:

```python
{"error": "No existe el SKU 'TECLADO-87'.",
 "skus_validos": [{"sku": "TEC-MEC-87", "nombre": "Teclado mecánico 87 teclas"}, ...]}
```

Esa es la diferencia entre un agente que se traba y uno que se corrige. Un
`ValueError` pelado no le dice al modelo qué hacer después; un error con la
lista de valores válidos, sí. La traza de [`reintento`](traces/reintento.log)
muestra el ciclo completo: SKU inventado → error con catálogo → reintento con
el código correcto → respuesta.

### Ambigüedad: el agente pregunta en vez de adivinar

Hay dos clientes llamados Juan en la base, puestos ahí a propósito. Cuando
`buscar_cliente` encuentra varios, devuelve un error que dice *"preguntale al
usuario a cuál se refiere antes de seguir"* junto con los candidatos. El
prompt de sistema refuerza la regla. El resultado está en
[`ambiguedad.log`](traces/ambiguedad.log): el agente enumera los dos Juanes,
pregunta, y en el turno siguiente —"el de Córdoba"— resuelve solo que eso
significa `cliente_id=103`.

### Un reducer propio además de `messages`

`EstadoAgente` hereda de `MessagesState`, que ya trae `messages` con
`add_messages` como reducer. Se le suma un contador:

```python
class EstadoAgente(MessagesState):
    turnos_del_modelo: Annotated[int, operator.add]
```

`operator.add` es el mismo mecanismo aplicado a un `int`: cuando el nodo
devuelve `{"turnos_del_modelo": 1}`, LangGraph **suma** en lugar de
reemplazar. Sirve para leer de un vistazo cuántas vueltas dio el ciclo, y es la
forma más chica de mostrar que el estado se actualiza con reducers y no por
asignación.

### `AsyncSqliteSaver` y no el `SqliteSaver` de la consigna

Son la misma clase del mismo paquete (`langgraph-checkpoint-sqlite`), en su
versión asíncrona. El grafo entero es `async`, y el saver sincrónico hace E/S
bloqueante sobre el archivo: congelaría el event loop en **cada** checkpoint,
que es exactamente el error que documenta el corpus de la Pre-entrega 4
("Bloquear el event loop"). Usar el sincrónico adentro de un grafo async sería
escribir el bug a propósito.

### El checkpointer es un context manager

`abrir_agente()` es un `@asynccontextmanager` porque `AsyncSqliteSaver`
mantiene una conexión abierta que hay que cerrar. Envolverlo así evita el
clásico "la última respuesta no quedó guardada" cuando el proceso termina antes
del flush.

### Una base SQLite real, no un diccionario

Las herramientas abren conexiones con `aiosqlite` y ejecutan SQL. Importa
porque el agente tiene que lidiar con lo que una base devuelve en serio —cero
filas, varias filas, un id que no existe— y no con un `dict.get()` que siempre
sale bien. Los datos están armados para que ninguna pregunta interesante se
resuelva de un solo salto.

### `temperature=0`

Un agente que elige herramientas tiene que ser reproducible. Con temperatura
alta, la misma pregunta puede llamar a `buscar_pedidos` una vez y a
`detalle_pedido` la siguiente, y entonces la traza que se entrega deja de
representar lo que hace el sistema.

### Carga perezosa, igual que en las entregas 3 y 4

Nada se construye al importar. `import agent` funciona sin una sola API key
cargada; el error aparece recién cuando se intenta usar el modelo, y dice qué
variable falta.

---

## Estructura

```
pre-entrega-5/
├── agent.py            # el StateGraph, los nodos, el checkpointer
├── tools.py            # las 4 herramientas @tool, asíncronas
├── config.py           # configuración + LLM perezoso (@lru_cache)
├── seed_db.py          # crea y puebla data/tienda.db
├── main.py             # CLI: chat con thread_id
├── trace.py            # corre los escenarios y escribe traces/
├── traces/             # las 4 trazas, en .json y .log
│   ├── multipaso.{json,log}
│   ├── memoria.{json,log}
│   ├── reintento.{json,log}
│   └── ambiguedad.{json,log}
├── data/               # generado, no versionado (tienda.db + checkpoints.sqlite)
├── requirements.txt
└── README.md
```

---

## Cómo levantar el entorno

### 1. Dependencias

```bash
# desde la raíz del repo
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source venv/bin/activate       # Mac/Linux

pip install -r pre-entrega-5/requirements.txt
```

Probado con Python 3.14; requiere 3.12 o superior.

### 2. Variables de entorno

En el `.env` de la **raíz del repo** (ver `.env.example`):

```bash
GEMINI_API_KEY=...        # gratis en https://aistudio.google.com/apikey
PROVIDER=gemini           # o "openai" / "anthropic"
# CHAT_MODEL=gemini-flash-lite-latest   # opcional: pisa el modelo default
# RECURSION_LIMIT=10                    # opcional: techo de pasos del grafo
```

El `.env` nunca se sube al repositorio (está en `.gitignore`).

### 3. Crear la base de la tienda

```bash
cd pre-entrega-5
python seed_db.py
```

Crea `data/tienda.db` con 5 clientes, 6 productos, 6 pedidos y 9 items. Con
`--force` la recrea desde cero. (Las herramientas la crean solas si falta, así
que este paso es opcional.)

### 4. Hablar con el agente

```bash
python main.py                                      # chat interactivo
python main.py "¿Cuántos pedidos tiene Mariana Riccio?"
python main.py --verboso "¿Qué compró Mariana en su último pedido?"
```

Para probar la memoria, usá el mismo `--thread` en dos corridas distintas:

```bash
python main.py --thread soporte-42 "¿Cuántos pedidos tiene Mariana Riccio?"
python main.py --thread soporte-42 "¿y cuál fue el último?"
python main.py --thread soporte-42 --historial
```

### 5. Regenerar las trazas

```bash
python trace.py                      # los cuatro escenarios
python trace.py --escenario memoria  # uno solo
python trace.py --listar
python trace.py --solo-logs          # rehace los .log desde los .json, sin API
```

---

## Las trazas

Cuatro escenarios, cada uno en `.json` (completo, para inspeccionar) y `.log`
(legible, con el formato ReAct de la consigna).

| Escenario | Turnos del modelo | Llamadas a tools | Qué demuestra |
|---|---|---|---|
| [`multipaso`](traces/multipaso.log) | 4 | 3 | Encadena tres herramientas distintas sin que nadie le programe la secuencia |
| [`memoria`](traces/memoria.log) | 7 | 5 | Tres turnos sobre un `thread_id`; el 2º y el 3º no se entienden sin el historial |
| [`reintento`](traces/reintento.log) | 3 | 2 | La herramienta falla, el agente lee el error y se corrige |
| [`ambiguedad`](traces/ambiguedad.log) | 4 | 2 | El agente pregunta en vez de adivinar, y después retoma el hilo |

### El caso multi-paso, recortado

```
Usuario: "¿Cuánto gastó en total Mariana Riccio y qué productos tenía su último pedido?"
  -> El agente decide usar la herramienta: buscar_cliente(nombre='Mariana Riccio')
  -> La herramienta devuelve: {"cliente_id": 102, "nombre": "Mariana Riccio", ...}
  -> El agente decide usar la herramienta: buscar_pedidos(cliente_id=102)
  -> La herramienta devuelve: {"cantidad_de_pedidos": 3, "total_gastado": 14500.0, ...}
  -> El agente decide usar la herramienta: detalle_pedido(pedido_id='P-1027')
  -> La herramienta devuelve: {"items": [{"producto": "Teclado mecánico 87 teclas", ...}]}
  -> El agente razona: ya tiene los datos -> responde.

Respuesta: Mariana Riccio gastó en total $14.500 a lo largo de 3 pedidos.
Su último pedido (P-1027, del 21/09/2025) incluía 1 Teclado mecánico 87 teclas
($3.500) y 1 Mouse ergonómico inalámbrico ($1.000).
```

### El ciclo de retorno, recortado

```
Usuario: "¿Cuánto stock queda del teclado? El código creo que es TECLADO-87."
  -> El agente decide usar la herramienta: consultar_stock(sku='TECLADO-87')
  -> La herramienta devuelve [ERROR]: {"error": "No existe el SKU 'TECLADO-87'.",
                                       "skus_validos": [...]}
  -> El agente decide usar la herramienta: consultar_stock(sku='TEC-MEC-87')
  -> La herramienta devuelve: {"producto": "Teclado mecánico 87 teclas", "stock": 12}
  -> El agente razona: ya tiene los datos -> responde.

Respuesta: Quedan 12 unidades en stock del Teclado mecánico 87 teclas (TEC-MEC-87).
```

---

## Pruebas hechas a mano

Además de las trazas, se verificó:

- **Persistencia entre procesos distintos.** Un `python main.py --thread
  soporte-42 "¿Cuántos pedidos tiene Mariana Riccio?"`, y después, en una
  ejecución **nueva** del programa, `python main.py --thread soporte-42 "¿y
  cuál fue el último?"`. El segundo proceso resolvió que "el último" era el
  P-1027 de Mariana sin que nadie se lo repitiera: el historial se levantó de
  `data/checkpoints.sqlite`.

- **El `recursion_limit` corta de verdad.** Con `RECURSION_LIMIT=2` sobre una
  pregunta que necesita 3 herramientas, el grafo lanza `GraphRecursionError` y
  el CLI lo atrapa con un mensaje accionable en vez de propagar el stack trace:

  ```
  [El agente superó el límite de 2 pasos sin llegar a una respuesta. Probá
  reformulando la pregunta, o subí RECURSION_LIMIT en el .env si el caso
  realmente la necesita.]
  ```

- **Las cuatro herramientas, incluidos sus caminos de error**: SKU inexistente,
  `cliente_id` inexistente, nombre sin coincidencias y nombre ambiguo.

---

## Limitaciones conocidas

- **El modelo es Gemini, no OpenAI ni Anthropic.** La consigna pide uno de esos
  dos, pero es la única API key con cupo en este proyecto. El código soporta
  los tres (`PROVIDER=openai` o `anthropic` en el `.env`) y la única diferencia
  es qué rama de `config.get_llm()` se ejecuta; las ramas de OpenAI y Anthropic
  no se probaron en vivo por no contar con keys con crédito.

- **El tier gratuito de Gemini tiene cuota diaria por modelo.** Generando las
  trazas se agotaron los 20 requests diarios de `gemini-flash-latest` a mitad
  de camino. Las cuatro trazas finales se generaron con
  `CHAT_MODEL=gemini-flash-lite-latest`, que tiene su propia cuota. El modelo
  usado queda registrado en cada `.json` y en el encabezado de cada `.log`, así
  que la traza siempre dice con qué se produjo.

- **El estado crece sin límite.** Cada turno suma mensajes al thread y todos se
  le reenvían al modelo. Para una conversación larga habría que resumir o
  podar el historial — LangGraph lo permite con un nodo intermedio, pero esta
  entrega no lo implementa.

- **Las trazas se generan con bases efímeras.** `trace.py` usa `:memory:` en
  vez del archivo de checkpoints, para que una corrida anterior del mismo
  `thread_id` no contamine el historial y la traza sea reproducible. El CLI sí
  usa el archivo.
