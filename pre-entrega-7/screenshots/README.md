# Capturas del dashboard de observabilidad

Esta carpeta guarda la evidencia de las trazas. Son cuatro capturas, y abajo
está exactamente dónde sale cada una.

El archivo [`resultado-prueba-de-carga.json`](resultado-prueba-de-carga.json)
es la medición del lado del cliente de la misma corrida: sirve para cruzar con
lo que muestra el dashboard.

---

## Antes de capturar: generar la corrida

```bash
# 1. Levantar la API (desde pre-entrega-7/)
uvicorn app.main:app --port 8000

# 2. En otra terminal, lanzar las 5 peticiones concurrentes
python scripts/prueba_de_carga.py --aprobar --json screenshots/resultado-prueba-de-carga.json
```

Esperá a que termine (unos 2-3 minutos) y recién ahí abrí el dashboard: las
trazas tardan unos segundos en aparecer.

---

## Las cuatro capturas

### 1. `01-trazas-del-proyecto.png`

**Dónde**: https://smith.langchain.com → proyecto **`auditoria-multiagente`** →
pestaña **Runs**.

**Qué tiene que verse**: la lista de ejecuciones, con los 5 runs llamados
`auditoria_job` de la prueba de carga. Son la evidencia de que la
instrumentación está activa y exportando.

---

### 2. `02-traza-de-un-job.png`

**Dónde**: clic en cualquiera de los `auditoria_job` de la lista.

**Qué tiene que verse**: el **árbol de spans** desplegado. Es la captura más
informativa de las cuatro, porque muestra la estructura del orquestador:

```
auditoria_job
└── LangGraph
    ├── supervisor      -> ChatGoogleGenerativeAI
    ├── investigador    -> buscar_en_documentacion, listar_incidentes
    ├── supervisor      -> ChatGoogleGenerativeAI
    ├── analista        -> resumir_incidentes, detectar_atipicos, validar_registros
    ├── supervisor      -> ChatGoogleGenerativeAI
    ├── validador
    ├── compuerta_hitl
    ├── ejecutar_accion
    └── sintetizador    -> ChatGoogleGenerativeAI
```

Desplegá los nodos para que se vean las llamadas al modelo y las herramientas.

---

### 3. `03-costo-por-ejecucion.png`

**Dónde**: en la lista de runs, la columna **Total Cost**. Si no está visible,
se agrega desde el selector de columnas (el ícono de ajustes arriba a la
derecha de la tabla).

**Qué tiene que verse**: el costo de cada uno de los 5 runs. En la corrida de
referencia fue de **~$0.0021 por ejecución**, con unos 5.400 a 6.200 tokens.

> **Si la columna aparece vacía**: es porque el modelo configurado es un alias
> (`gemini-flash-lite-latest`) y LangSmith no tiene precio para los alias, solo
> para los nombres concretos. Verificá que el `.env` tenga
> `MODELO_GEMINI_API=gemini-3.1-flash-lite` y **volvé a correr la prueba de
> carga**: el costo se calcula al momento de recibir la traza, no
> retroactivamente.

---

### 4. `04-latencia-p95.png`

**Dónde**: en el proyecto, pestaña **Monitor** (o el panel de métricas). Buscá
el gráfico de **Latency** y seleccioná **P95**. Ajustá el rango de tiempo a la
última hora para que se vea el pico de la prueba de carga.

**Qué tiene que verse**: el percentil 95 de latencia de las ejecuciones.

> **Por qué este número no coincide con el del script**: el p95 de LangSmith
> mide la **ejecución del grafo**, desde que un worker toma el trabajo. El p95
> del script mide **de punta a punta**, e incluye la espera en la cola. Con 3
> workers y 5 peticiones, dos esperan turno, así que el del script es bastante
> mayor. Los dos son correctos y miden cosas distintas: compararlos es lo que
> dice si la cola está siendo el cuello de botella.

---

## Opcional: la pausa de aprobación

Si querés documentar también el HITL, corré la prueba **sin** `--aprobar`:

```bash
python scripts/prueba_de_carga.py
```

Los jobs quedan en `ESPERANDO_APROBACION`. Capturá entonces:

- `GET /tasks/{job_id}` mostrando `requiere_aprobacion: true` y la propuesta.
- `GET /escalaciones` vacío antes de aprobar, y con la escalación después.

Es la evidencia de que la acción crítica no se ejecuta sin autorización.
