"""
app/schemas.py

Los contratos de la API y del estado de los jobs.

Separado de la lógica a propósito: un cliente de esta API necesita saber qué
mandar y qué esperar, y eso tiene que poder leerse en un solo archivo. FastAPI
además genera el OpenAPI a partir de estas clases, así que lo que se documente
acá es lo que aparece en `/docs`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EstadoJob(StrEnum):
    """
    Los estados por los que pasa un trabajo.

    El cliente hace polling sobre `GET /tasks/{id}` y decide según este campo,
    así que la lista tiene que ser exhaustiva: cualquier job que no esté en un
    estado terminal pero tampoco avanzando sería un cliente colgado para
    siempre.

    Estados terminales: COMPLETADO y FALLIDO. El resto son transitorios.
    """

    PENDIENTE = "PENDIENTE"            # encolado, ningún worker lo tomó
    EJECUTANDO = "EJECUTANDO"          # un worker lo está corriendo
    ESPERANDO_APROBACION = "ESPERANDO_APROBACION"  # pausado por el nodo HITL
    COMPLETADO = "COMPLETADO"
    FALLIDO = "FALLIDO"

    @property
    def es_terminal(self) -> bool:
        return self in (EstadoJob.COMPLETADO, EstadoJob.FALLIDO)


class SolicitudTarea(BaseModel):
    """El body de `POST /tasks`."""

    solicitud: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="La consulta de auditoría a procesar",
        examples=[
            "Nuestros servicios asíncronos se vienen cayendo seguido. ¿Qué dice "
            "la documentación sobre las causas y qué muestran nuestros incidentes?"
        ],
    )

    @field_validator("solicitud")
    @classmethod
    def _sin_espacios_sobrantes(cls, v: str) -> str:
        limpia = v.strip()
        if not limpia:
            raise ValueError("La solicitud no puede ser solo espacios.")
        return limpia


class TareaEncolada(BaseModel):
    """La respuesta de `POST /tasks`. Se devuelve con 202 Accepted."""

    job_id: str = Field(..., description="Identificador para consultar el estado")
    estado: EstadoJob = EstadoJob.PENDIENTE
    url_estado: str = Field(..., description="Dónde hacer polling")
    encolado_en: str


class PropuestaAccion(BaseModel):
    """
    Una acción con efectos secundarios que el sistema quiere ejecutar.

    Es lo que el cliente ve cuando un job queda en ESPERANDO_APROBACION: la
    descripción de qué se va a hacer, por qué se considera crítica y qué efecto
    tendría. Sin esto, aprobar sería firmar en blanco.
    """

    accion: str = Field(..., description="Nombre de la operación a ejecutar")
    motivo: str = Field(..., description="Por qué el sistema la propone")
    criticidad: Literal["alta", "critica"] = Field(
        ..., description="Por qué requiere aprobación humana"
    )
    efectos: list[str] = Field(
        default_factory=list,
        description="Qué cambia en el mundo si se aprueba",
    )
    parametros: dict[str, Any] = Field(default_factory=dict)


class DecisionAprobacion(BaseModel):
    """El body de `POST /tasks/{id}/approve`."""

    aprobado: bool = Field(..., description="True ejecuta la acción, False la descarta")
    aprobado_por: str = Field(
        ..., min_length=2, max_length=120,
        description="Quién aprueba o rechaza. Queda en el registro de auditoría.",
    )
    comentario: str = Field(
        default="", max_length=500,
        description="Opcional: justificación de la decisión",
    )


class Job(BaseModel):
    """
    El registro completo de un trabajo, tal como vive en Redis.

    Se serializa a JSON y se guarda bajo la clave `job:{job_id}`. Es la única
    fuente de verdad del estado para la API: los workers escriben acá y los
    endpoints leen de acá, así que reiniciar el proceso no pierde nada.
    """

    job_id: str
    estado: EstadoJob
    solicitud: str

    creado_en: str
    actualizado_en: str
    iniciado_en: str | None = None
    terminado_en: str | None = None

    # Resultado del camino feliz.
    informe: str | None = None
    hallazgos: list[dict[str, Any]] = Field(default_factory=list)
    pasos_del_supervisor: int = 0

    # Camino HITL.
    propuesta: PropuestaAccion | None = None
    decision: DecisionAprobacion | None = None
    accion_ejecutada: dict[str, Any] | None = None

    # Camino de error. `error` es el mensaje para el cliente; `error_tipo`
    # permite distinguir un fallo de infraestructura de uno de dominio sin
    # parsear texto.
    error: str | None = None
    error_tipo: str | None = None

    # Para la prueba de carga: cuánto tardó de punta a punta.
    duracion_segundos: float | None = None

    @staticmethod
    def ahora() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RespuestaEstado(BaseModel):
    """
    La respuesta de `GET /tasks/{id}`.

    Es un subconjunto de `Job`: se omiten los campos internos y se agrega
    `requiere_aprobacion`, que le dice al cliente si tiene que llamar a
    `/approve` en lugar de seguir esperando. Sin ese campo, un cliente que
    hace polling sobre un job pausado no tiene forma de saber que la pelota
    está de su lado.
    """

    job_id: str
    estado: EstadoJob
    solicitud: str
    creado_en: str
    actualizado_en: str
    duracion_segundos: float | None = None

    requiere_aprobacion: bool = False
    propuesta: PropuestaAccion | None = None
    decision: DecisionAprobacion | None = None
    accion_ejecutada: dict[str, Any] | None = None

    informe: str | None = None
    hallazgos: list[dict[str, Any]] = Field(default_factory=list)
    pasos_del_supervisor: int = 0

    error: str | None = None
    error_tipo: str | None = None

    @classmethod
    def desde_job(cls, job: Job) -> RespuestaEstado:
        return cls(
            **job.model_dump(
                include={
                    "job_id", "estado", "solicitud", "creado_en", "actualizado_en",
                    "duracion_segundos", "propuesta", "decision", "accion_ejecutada",
                    "informe", "hallazgos", "pasos_del_supervisor", "error",
                    "error_tipo",
                }
            ),
            requiere_aprobacion=job.estado == EstadoJob.ESPERANDO_APROBACION,
        )


class Salud(BaseModel):
    """La respuesta de `GET /health`."""

    estado: Literal["ok", "degradado"]
    redis: bool = Field(..., description="¿Responde el PING?")
    observabilidad: str = Field(..., description="Plataforma de trazas activa")
    jobs_en_cola: int
    workers: int
    detalle: str = ""
