"""
agents/research_agent.py

El especialista en **ir a buscar los hechos**.

Tiene dos herramientas, una por cada fuente que hace falta consultar para
responder una pregunta de auditoría técnica:

- `buscar_en_documentacion` — búsqueda semántica sobre el índice de Pinecone de
  la Pre-entrega 4 (el corpus de `asyncio`). Responde "¿qué se sabe sobre esto?".
- `listar_incidentes` — consulta la base SQLite de incidentes propios.
  Responde "¿qué nos pasó a nosotros?".

Las dos son recuperación: el agente trae material, no lo interpreta. El cómputo
es trabajo del analista, y mantener esa frontera es lo que hace que la
topología tenga sentido en lugar de ser un agente partido en dos.

Resiliencia de la búsqueda documental
-------------------------------------
Pinecone es un servicio remoto con una API key que puede rotarse o vencerse. Si
no responde, la herramienta **no falla**: cae a un índice BM25 local sobre los
mismos archivos del corpus y lo dice en la respuesta. Un orquestador que se cae
entero porque una de cuatro herramientas perdió la credencial no es un
orquestador, es una cadena.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config import CORPUS_PRE_ENTREGA_4, MAX_PASOS_ESPECIALISTA, get_llm
from seed_db import asegurar_db
from utils import extraer_hallazgos, texto_de

log = logging.getLogger("agents.investigador")

TOP_K = 4

PROMPT = """Sos el agente INVESTIGADOR de un equipo de auditoría técnica.

Tu única responsabilidad es RECUPERAR material de las fuentes. No calculás \
métricas, no saques conclusiones numéricas: de eso se encarga el analista.

Tenés dos fuentes:
- `buscar_en_documentacion`: la documentación técnica del equipo sobre \
programación asíncrona en Python. Usala para entender causas y buenas prácticas.
- `listar_incidentes`: la bitácora de incidentes propios de producción. Usala \
para traer los casos reales.

Cómo trabajar:
- Si la instrucción menciona las dos cosas (qué dice la documentación Y qué \
muestran nuestros datos), usá las dos herramientas.
- Traé el material completo, no una muestra: el analista necesita todos los \
registros para que las estadísticas sean correctas.
- Cerrá con un resumen de DOS O TRES ORACIONES de qué encontraste. No repitas \
los registros uno por uno: ya quedaron guardados en el estado compartido."""


# --------------------------------------------------------------------------
# Herramienta 1: búsqueda semántica en el vector DB de la Pre-entrega 4
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _indice_pinecone():
    """
    Handle del índice de la Pre-entrega 4, o None si no está disponible.

    Devuelve None en vez de propagar la excepción: la indisponibilidad de
    Pinecone es un caso esperado, no un error del programa.
    """
    try:
        import os

        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not api_key:
            log.info("Sin PINECONE_API_KEY: la búsqueda documental usa BM25 local.")
            return None

        nombre = os.getenv("INDEX_NAME", "asyncio-docs").strip()
        pc = Pinecone(api_key=api_key)
        if nombre not in [i["name"] for i in pc.list_indexes()]:
            log.warning("El índice '%s' no existe; se usa BM25 local.", nombre)
            return None
        return pc.Index(nombre)
    except Exception as e:
        log.warning("Pinecone no disponible (%s); se usa BM25 local.", type(e).__name__)
        return None


@lru_cache(maxsize=1)
def _embeddings():
    """Embeddings de 1536 dim, los mismos con los que se indexó en la entrega 4."""
    import os

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY", ""),
        output_dimensionality=1536,
    )


@lru_cache(maxsize=1)
def _bm25_local():
    """
    Índice BM25 sobre los archivos del corpus, como respaldo sin red.

    Parte cada documento por sus encabezados de nivel 2, igual que la ingesta de
    la Pre-entrega 4, para que los fragmentos sean comparables entre las dos
    rutas de búsqueda.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    documentos: list[Document] = []
    for archivo in sorted(CORPUS_PRE_ENTREGA_4.glob("*.md")):
        crudo = archivo.read_text(encoding="utf-8")
        cuerpo = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", crudo, flags=re.DOTALL)
        titulo_match = re.search(r"^titulo:\s*(.+)$", crudo, re.MULTILINE)
        titulo = titulo_match.group(1).strip() if titulo_match else archivo.stem

        secciones = re.split(r"\n(?=##\s)", cuerpo)
        for numero, seccion in enumerate(secciones, start=1):
            if seccion.strip():
                documentos.append(
                    Document(
                        page_content=seccion.strip(),
                        metadata={"titulo": titulo, "fuente": archivo.name,
                                  "pagina": numero},
                    )
                )

    if not documentos:
        raise RuntimeError(
            f"No hay corpus en {CORPUS_PRE_ENTREGA_4}. La búsqueda documental "
            f"necesita los .md de la Pre-entrega 4."
        )

    retriever = BM25Retriever.from_documents(documentos)
    retriever.k = TOP_K
    log.info("BM25 local listo sobre %d secciones.", len(documentos))
    return retriever


@tool
def buscar_en_documentacion(consulta: str) -> dict[str, Any]:
    """
    Busca en la documentación técnica del equipo sobre programación asíncrona.

    Usala para entender la CAUSA de un problema o la práctica recomendada:
    por qué se bloquea el event loop, cómo se manejan los timeouts, qué pasa
    con la cancelación de tareas, cómo se testea código async.

    NO sirve para consultar los incidentes propios del equipo; para eso está
    `listar_incidentes`.

    Args:
        consulta: la pregunta técnica, en lenguaje natural. Ejemplo:
            "por qué se bloquea el event loop con llamadas sincrónicas".

    Devuelve:
        {"fuente_de_busqueda": "pinecone" | "bm25-local",
         "consulta": "...",
         "fragmentos": [{"titulo": "El event loop de asyncio",
                         "fuente": "01-event-loop.md", "pagina": 4,
                         "texto": "..."}]}

    Devuelve hasta 4 fragmentos, del más relevante al menos relevante.
    """
    indice = _indice_pinecone()

    if indice is not None:
        try:
            vector = _embeddings().embed_query(consulta)
            respuesta = indice.query(
                vector=vector,
                top_k=TOP_K,
                namespace="python-asyncio",
                include_metadata=True,
            )
            fragmentos = [
                {
                    "titulo": (m.get("metadata") or {}).get("titulo", "?"),
                    "fuente": (m.get("metadata") or {}).get("fuente", "?"),
                    "pagina": (m.get("metadata") or {}).get("pagina", 0),
                    "texto": (m.get("metadata") or {}).get("text", ""),
                    "score": round(m.get("score", 0.0), 4),
                }
                for m in respuesta.get("matches", [])
            ]
            if fragmentos:
                return {
                    "fuente_de_busqueda": "pinecone",
                    "consulta": consulta,
                    "fragmentos": fragmentos,
                }
            log.warning("Pinecone no devolvió resultados; se usa BM25 local.")
        except Exception as e:
            log.warning("Falló la consulta a Pinecone (%s); se usa BM25 local.",
                        type(e).__name__)

    documentos = _bm25_local().invoke(consulta)
    return {
        "fuente_de_busqueda": "bm25-local",
        "consulta": consulta,
        "fragmentos": [
            {
                "titulo": d.metadata.get("titulo", "?"),
                "fuente": d.metadata.get("fuente", "?"),
                "pagina": d.metadata.get("pagina", 0),
                "texto": d.page_content,
            }
            for d in documentos
        ],
    }


# --------------------------------------------------------------------------
# Herramienta 2: la bitácora de incidentes propios
# --------------------------------------------------------------------------

CATEGORIAS = (
    "bloqueo-event-loop",
    "timeout-no-configurado",
    "cancelacion-mal-manejada",
    "tarea-huerfana",
    "fuga-de-tareas",
)


@tool
def listar_incidentes(meses: int = 12, categoria: str | None = None) -> dict[str, Any]:
    """
    Trae los incidentes de producción del equipo desde la bitácora interna.

    Usala para responder qué pasó REALMENTE en nuestros servicios: cuántas
    caídas hubo, de qué tipo, cuánto duraron. Es la fuente de los datos
    cuantitativos que después analiza el analista.

    NO sirve para preguntas conceptuales sobre asyncio; para eso está
    `buscar_en_documentacion`.

    Args:
        meses: cuántos meses hacia atrás mirar, contados desde el incidente más
            reciente de la base. Por defecto 12, que trae la bitácora completa.
            Pedí el período completo salvo que la instrucción diga otra cosa:
            las estadísticas sobre una muestra recortada engañan.
        categoria: filtro opcional por tipo de incidente. Valores válidos:
            "bloqueo-event-loop", "timeout-no-configurado",
            "cancelacion-mal-manejada", "tarea-huerfana", "fuga-de-tareas".
            Dejalo en None para traer todas las categorías.

    Devuelve:
        {"periodo": {"desde": "2026-03-04", "hasta": "2026-10-06"},
         "filtro_categoria": null, "cantidad": 20,
         "categorias_presentes": ["bloqueo-event-loop", ...],
         "incidentes": [{"incidente_id": "INC-0188", "fecha": "2026-03-04",
                         "servicio": "api-pedidos", "categoria": "bloqueo-event-loop",
                         "severidad": "alta", "minutos_caidos": 45,
                         "descripcion": "..."}]}

    Si la categoría no existe devuelve {"error": ..., "categorias_validas": [...]}
    para que puedas reintentar con una válida.
    """
    if categoria is not None and categoria not in CATEGORIAS:
        return {
            "error": f"La categoría '{categoria}' no existe.",
            "categorias_validas": list(CATEGORIAS),
        }

    ruta = asegurar_db()
    con = sqlite3.connect(ruta)
    try:
        con.row_factory = sqlite3.Row

        # El período se cuenta desde el incidente más reciente de la base y no
        # desde la fecha de hoy: así la herramienta devuelve lo mismo hoy que
        # dentro de seis meses, y la traza entregada sigue siendo reproducible.
        (mas_reciente,) = con.execute("SELECT MAX(fecha) FROM incidentes").fetchone()
        if mas_reciente is None:
            return {"error": "La bitácora de incidentes está vacía.", "incidentes": []}

        anio, mes, dia = (int(p) for p in mas_reciente.split("-"))
        meses_totales = anio * 12 + (mes - 1) - meses
        desde = f"{meses_totales // 12:04d}-{meses_totales % 12 + 1:02d}-{dia:02d}"

        sql = "SELECT * FROM incidentes WHERE fecha >= ?"
        parametros: list[Any] = [desde]
        if categoria is not None:
            sql += " AND categoria = ?"
            parametros.append(categoria)
        sql += " ORDER BY fecha"

        filas = [dict(f) for f in con.execute(sql, parametros).fetchall()]
    finally:
        con.close()

    return {
        "periodo": {"desde": desde, "hasta": mas_reciente},
        "filtro_categoria": categoria,
        "cantidad": len(filas),
        "categorias_presentes": sorted({f["categoria"] for f in filas}),
        "incidentes": filas,
    }


HERRAMIENTAS = [buscar_en_documentacion, listar_incidentes]


# --------------------------------------------------------------------------
# El nodo del grafo
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _agente_react():
    """El subgrafo ReAct del investigador, con sus herramientas acotadas."""
    return create_react_agent(get_llm(), HERRAMIENTAS, prompt=PROMPT)


async def nodo_investigador(estado: dict[str, Any]) -> dict[str, Any]:
    """
    Corre al investigador con la instrucción puntual del supervisor.

    **Acá se evita la contaminación de contexto.** El especialista NO recibe
    `estado["messages"]`: recibe un único mensaje con la solicitud original como
    marco y la instrucción concreta que le dio el supervisor. No ve las
    deliberaciones del supervisor, ni los razonamientos del analista, ni los
    payloads que ya están en el estado. Menos contexto, menos ruido, y un prompt
    cuyo tamaño no crece con la cantidad de vueltas del ciclo.
    """
    instruccion = estado.get("instruccion_actual", "") or estado.get("solicitud", "")
    pedido = (
        f"Solicitud original del usuario (como marco):\n{estado.get('solicitud', '')}\n\n"
        f"Tu tarea puntual ahora:\n{instruccion}"
    )

    log.info("Investigador trabajando: %s", instruccion[:110])

    resultado = await _agente_react().ainvoke(
        {"messages": [HumanMessage(pedido)]},
        config={"recursion_limit": MAX_PASOS_ESPECIALISTA * 2},
    )

    mensajes = resultado["messages"]
    hallazgos = extraer_hallazgos(mensajes, "investigador")
    informe = mensajes[-1]

    log.info("Investigador produjo %d hallazgo(s).", len(hallazgos))

    return {
        # Solo el informe final del especialista vuelve al canal de mensajes.
        # Sus pasos intermedios se quedan en su propio subgrafo: el supervisor
        # necesita el resultado, no la transcripción.
        "messages": [
            HumanMessage(
                content=f"[investigador] {texto_de(informe)}",
                name="investigador",
            )
        ],
        "hallazgos": hallazgos,
    }
