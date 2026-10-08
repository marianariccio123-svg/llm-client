"""
seed_db.py

Crea y puebla `data/incidentes.db`, la base que consulta el agente investigador.

El dominio es la bitácora de incidentes de producción de un equipo que corre
servicios asíncronos en Python. Las categorías de incidente **coinciden con los
temas del corpus de documentación** de la Pre-entrega 4 (bloqueo del event
loop, cancelación mal manejada, timeouts, tareas huérfanas). Esa coincidencia
es el punto: permite una pregunta que necesita los dos especialistas a la vez
—qué dice la documentación sobre la causa, y qué muestran nuestros propios
números— y no dos mitades pegadas con cinta.

Dos cosas plantadas a propósito en los datos:

- **Un valor atípico.** El incidente INC-0214 tiene 480 minutos de caída contra
  una mediana de ~35. Existe para que `detectar_atipicos` tenga algo real que
  encontrar en lugar de confirmar que todo está bien.
- **Un registro inválido.** INC-0231 tiene `minutos_caidos = -5`, que es
  imposible. Existe para que la validación de esquema del analista reporte un
  problema concreto de calidad de datos. Si lo sacás, el nodo de validación
  pasa a ser decorativo.

Uso:
    python seed_db.py           # crea data/incidentes.db (no pisa si ya existe)
    python seed_db.py --force   # la recrea desde cero
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "incidentes.db"

ESQUEMA = """
CREATE TABLE incidentes (
    incidente_id   TEXT PRIMARY KEY,
    fecha          TEXT NOT NULL,
    servicio       TEXT NOT NULL,
    categoria      TEXT NOT NULL,
    severidad      TEXT NOT NULL,
    minutos_caidos INTEGER NOT NULL,
    descripcion    TEXT NOT NULL
);
CREATE INDEX idx_fecha     ON incidentes(fecha);
CREATE INDEX idx_categoria ON incidentes(categoria);
"""

# (id, fecha, servicio, categoria, severidad, minutos_caidos, descripcion)
INCIDENTES = [
    ("INC-0188", "2026-03-04", "api-pedidos", "bloqueo-event-loop", "alta", 45,
     "Latencia de 12s en todos los endpoints; requests.get sincrónico en el handler de checkout."),
    ("INC-0191", "2026-03-18", "worker-emails", "tarea-huerfana", "media", 20,
     "Lote de 300 mails nunca se envió; la task creada con create_task se recolectó."),
    ("INC-0195", "2026-04-02", "api-pedidos", "bloqueo-event-loop", "critica", 95,
     "Caída total durante el pico de ventas; generación de PDF sincrónica en el event loop."),
    ("INC-0198", "2026-04-11", "api-stock", "timeout-no-configurado", "alta", 60,
     "Conexiones colgadas contra el proveedor externo, sin timeout definido."),
    ("INC-0201", "2026-04-23", "worker-reportes", "bloqueo-event-loop", "media", 30,
     "Reporte mensual congeló el worker; pandas corriendo dentro de una corrutina."),
    ("INC-0203", "2026-05-07", "api-pedidos", "cancelacion-mal-manejada", "alta", 50,
     "Pagos en estado inconsistente: un except Exception se tragó el CancelledError."),
    ("INC-0206", "2026-05-15", "api-stock", "bloqueo-event-loop", "media", 25,
     "Picos de latencia por hashing de contraseñas en el handler de login."),
    ("INC-0209", "2026-05-29", "worker-emails", "timeout-no-configurado", "baja", 15,
     "Reintentos infinitos contra el SMTP sin wait_for."),
    ("INC-0212", "2026-06-08", "api-pedidos", "bloqueo-event-loop", "alta", 55,
     "Degradación general; lectura sincrónica de un CSV de 400MB en el arranque del request."),
    # El atípico: una caída de 8 horas contra una mediana de ~35 minutos.
    ("INC-0214", "2026-06-19", "api-pagos", "cancelacion-mal-manejada", "critica", 480,
     "Ocho horas sin conciliar pagos: cancelación silenciada dejó transacciones a medio commitear."),
    ("INC-0217", "2026-07-03", "api-stock", "fuga-de-tareas", "media", 35,
     "Memoria del proceso creciendo sin techo; tasks que nunca terminaban."),
    ("INC-0219", "2026-07-14", "worker-reportes", "bloqueo-event-loop", "media", 40,
     "Worker trabado 40 minutos comprimiendo un ZIP dentro del loop."),
    ("INC-0222", "2026-07-28", "api-pedidos", "timeout-no-configurado", "alta", 65,
     "Cascada de timeouts aguas abajo por no acotar la llamada al servicio de envíos."),
    ("INC-0225", "2026-08-06", "api-pagos", "bloqueo-event-loop", "critica", 85,
     "Checkout caído; validación de tarjeta usando una librería sincrónica."),
    ("INC-0228", "2026-08-20", "worker-emails", "tarea-huerfana", "baja", 10,
     "Notificaciones push perdidas, sin referencia fuerte a las tasks."),
    # El registro inválido: minutos_caidos negativo.
    ("INC-0231", "2026-09-02", "api-stock", "fuga-de-tareas", "media", -5,
     "Registro mal cargado por el script de importación; los minutos quedaron en negativo."),
    ("INC-0234", "2026-09-11", "api-pedidos", "bloqueo-event-loop", "alta", 50,
     "Timeouts del cliente móvil; serialización de un JSON gigante bloqueando el loop."),
    ("INC-0237", "2026-09-23", "api-pagos", "cancelacion-mal-manejada", "media", 30,
     "Reembolsos duplicados tras un deploy; limpieza sin shield en el finally."),
    ("INC-0240", "2026-10-01", "api-stock", "bloqueo-event-loop", "media", 28,
     "Latencia intermitente por time.sleep en un retry hecho a mano."),
    ("INC-0243", "2026-10-06", "worker-reportes", "timeout-no-configurado", "baja", 12,
     "Job nocturno colgado contra el data warehouse."),
]


def crear(force: bool = False) -> Path:
    """Crea la base. Devuelve la ruta. Con `force`, borra la existente."""
    DATA_DIR.mkdir(exist_ok=True)

    if DB_PATH.exists():
        if not force:
            print(f"{DB_PATH} ya existe. Usá --force para recrearla.")
            return DB_PATH
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(ESQUEMA)
        con.executemany(
            "INSERT INTO incidentes VALUES (?, ?, ?, ?, ?, ?, ?)", INCIDENTES
        )
        con.commit()
    finally:
        con.close()

    categorias = sorted({i[3] for i in INCIDENTES})
    print(f"Base creada en {DB_PATH}")
    print(f"  {len(INCIDENTES)} incidentes, {len(categorias)} categorías")
    print(f"  categorías: {', '.join(categorias)}")
    return DB_PATH


def asegurar_db() -> Path:
    """Crea la base si falta. La llaman las herramientas para no fallar en seco."""
    if not DB_PATH.exists():
        crear()
    return DB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recrear desde cero")
    args = parser.parse_args()
    crear(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
