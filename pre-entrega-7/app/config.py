"""
app/config.py

Configuración de la Pre-entrega 7.

Mismo criterio que las entregas 3 a 6: **todo perezoso**. Importar este módulo
no abre conexiones ni valida credenciales, así que los tests importan la app
entera sin Redis, sin API keys y sin red.

La única diferencia con las entregas anteriores es que acá hay un recurso
compartido de verdad —el pool de conexiones a Redis— y eso impone una regla:
una sola instancia por proceso, creada en el arranque de FastAPI y cerrada en
el apagado. Un pool por request sería una fuga de sockets.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

load_dotenv(REPO_ROOT / ".env")

# El orquestador de la Pre-entrega 6 se **reutiliza**, no se copia. Sus módulos
# usan imports planos (`from state import ...`), y la carpeta tiene un guion en
# el nombre, así que no es importable como paquete: hay que agregarla al path.
# Es menos elegante que un paquete instalable, pero garantiza que la API corre
# el mismo grafo que se entregó y corrigió en el módulo 6, no una copia que
# puede divergir.
PRE_ENTREGA_6 = REPO_ROOT / "pre-entrega-6"
if str(PRE_ENTREGA_6) not in sys.path:
    sys.path.insert(0, str(PRE_ENTREGA_6))

# --- Redis ---
# Sin REDIS_URL la app no arranca, y está bien que sea así: el estado de los
# jobs y los checkpoints del grafo viven ahí. Arrancar sin Redis daría una API
# que acepta trabajos y los pierde.
REDIS_URL = os.getenv("REDIS_URL", "").strip()

# Prefijos de las claves. Tenerlos acá evita que un entorno de pruebas pise los
# datos de otro compartiendo la misma base de Redis.
PREFIJO = os.getenv("REDIS_PREFIJO", "auditoria").strip()
CLAVE_JOB = f"{PREFIJO}:job"
CLAVE_INDICE_JOBS = f"{PREFIJO}:jobs"
CLAVE_CHECKPOINTS = f"{PREFIJO}:checkpoint"

# TTL de los jobs: 7 días. Un free tier de Redis tiene 30MB, y un job con su
# informe y sus hallazgos pesa unos pocos KB; sin expiración, la base se llena
# sola en algún momento y la API empieza a fallar por algo que no tiene nada
# que ver con el pedido en curso.
TTL_JOB_SEGUNDOS = int(os.getenv("TTL_JOB_SEGUNDOS", str(7 * 24 * 3600)))

# --- Workers ---
# Cantidad de corrutinas que consumen de la cola. Con 3 se pueden ver las 5
# peticiones concurrentes de la prueba de carga haciendo cola de verdad: si
# fueran 5 o más, la cola nunca tendría profundidad y la métrica de latencia
# no mostraría nada interesante.
CANTIDAD_WORKERS = int(os.getenv("CANTIDAD_WORKERS", "3"))
# Tope de la cola. Al llenarse, la API responde 503 en lugar de aceptar trabajo
# que no va a poder atender.
TAMANIO_COLA = int(os.getenv("TAMANIO_COLA", "100"))
# Techo de duración de un job. Sin esto, un grafo que se cuelga ocupa un worker
# para siempre y la API se degrada sin dar ninguna señal.
TIMEOUT_JOB_SEGUNDOS = int(os.getenv("TIMEOUT_JOB_SEGUNDOS", "300"))

# --- Observabilidad ---
PLATAFORMA_TRAZAS = os.getenv("PLATAFORMA_TRAZAS", "langsmith").strip().lower()
PROYECTO_TRAZAS = os.getenv("LANGSMITH_PROJECT", "auditoria-multiagente").strip()


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el formato de logs. Lo llama el arranque de la app."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in (
        "httpx", "urllib3", "google_genai", "langchain_google_genai",
        "pinecone", "redisvl", "uvicorn.access",
    ):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def require_redis_url() -> str:
    """
    Devuelve la URL de Redis o falla con un mensaje accionable.

    Se llama en el arranque, no al importar: así los tests pueden sustituir el
    cliente por `fakeredis` sin tener que definir la variable.
    """
    if not REDIS_URL:
        raise RuntimeError(
            "Falta REDIS_URL. Cargala en el .env de la raíz del repo.\n"
            "  - Redis Cloud (gratis, incluye RediSearch): https://redis.io/try-free\n"
            "  - Local con Docker: docker compose up -d redis\n"
            "Formato: redis://default:<password>@<host>:<puerto>"
        )
    return REDIS_URL


@lru_cache(maxsize=1)
def hay_observabilidad() -> bool:
    """¿Están las credenciales para enviar trazas?"""
    if PLATAFORMA_TRAZAS == "langsmith":
        return bool(os.getenv("LANGSMITH_API_KEY", "").strip())
    if PLATAFORMA_TRAZAS == "phoenix":
        return bool(os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip())
    return False
