"""
app/acciones.py

Las operaciones con efectos secundarios. Es el único módulo del proyecto que
cambia algo fuera del proceso, y está separado por eso.

Tenerlas en un archivo aparte sirve para una pregunta concreta que uno quiere
poder responder rápido en una revisión: *¿qué puede hacer este sistema que no
se pueda deshacer?* La respuesta es "lo que esté acá", y hoy es una sola cosa.

Toda función de este módulo asume que **ya pasó por la aprobación humana**. No
vuelve a chequearlo, porque dos lugares que verifican lo mismo terminan
divergiendo: el control está en `app/hitl.py` y la arista condicional que llega
a `nodo_ejecutar_accion` es la única vía de entrada.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import PREFIJO

log = logging.getLogger("app.acciones")

CLAVE_ESCALACIONES = f"{PREFIJO}:escalaciones"

# El cliente de Redis lo inyecta el arranque de la app. Es un módulo global a
# propósito: `nodo_ejecutar_accion` corre dentro del grafo, que no recibe el
# cliente por parámetro, y pasarlo por el estado significaría serializar un
# socket en cada checkpoint.
_redis: Any = None


def configurar_redis(cliente: Any) -> None:
    """Inyecta el cliente de Redis. Lo llama el ciclo de vida de FastAPI."""
    global _redis
    _redis = cliente


async def escalar_a_guardia(
    incidente_id: str,
    servicio: str,
    motivo: str,
    aprobado_por: str,
) -> dict[str, Any]:
    """
    Abre una escalación y notifica a la guardia del servicio.

    Es la acción crítica del sistema: tiene un efecto externo e irreversible,
    y por eso requiere aprobación humana antes de llegar acá.

    En un sistema real este sería el lugar donde se llama a PagerDuty o se abre
    el ticket. Acá el efecto es escribir la escalación en Redis, que es un
    efecto **de verdad**: queda persistida, se puede listar por
    `GET /escalaciones` y sobrevive al reinicio del proceso. La notificación
    sí está simulada, y se declara como tal en el resultado.

    Args:
        incidente_id: el incidente que origina la escalación.
        servicio: el servicio afectado, para saber a qué guardia despertar.
        motivo: el texto que ve la persona de guardia.
        aprobado_por: quién autorizó. Queda en el registro de auditoría.

    Devuelve el registro de la escalación, con su id de seguimiento.
    """
    escalacion = {
        "escalacion_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
        "incidente_id": incidente_id,
        "servicio": servicio,
        "motivo": motivo,
        "aprobado_por": aprobado_por,
        "abierta_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "abierta",
        # Honestidad sobre el alcance: la escritura es real, el aviso no.
        "notificacion": "simulada (en producción iría a PagerDuty u Opsgenie)",
    }

    if _redis is None:
        # Sin Redis la acción no puede dejar rastro. Falla en vez de devolver
        # un éxito que no ocurrió: un sistema que dice "escalé" sin haber
        # escalado es peor que uno que avisa que no pudo.
        raise RuntimeError(
            "No hay cliente de Redis configurado: la escalación no se puede "
            "persistir. Revisá el arranque de la app (configurar_redis)."
        )

    import json

    await _redis.lpush(CLAVE_ESCALACIONES, json.dumps(escalacion, ensure_ascii=False))
    await _redis.ltrim(CLAVE_ESCALACIONES, 0, 499)

    log.warning(
        "ESCALACIÓN ABIERTA %s: %s en %s (aprobó %s).",
        escalacion["escalacion_id"], incidente_id, servicio, aprobado_por,
    )
    return escalacion


async def listar_escalaciones(limite: int = 50) -> list[dict[str, Any]]:
    """Las escalaciones abiertas, más recientes primero."""
    if _redis is None:
        return []

    import json

    crudas = await _redis.lrange(CLAVE_ESCALACIONES, 0, limite - 1)
    salida: list[dict[str, Any]] = []
    for cruda in crudas:
        if isinstance(cruda, bytes):
            cruda = cruda.decode("utf-8")
        try:
            salida.append(json.loads(cruda))
        except json.JSONDecodeError:
            continue
    return salida
