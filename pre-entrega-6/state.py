"""
state.py

El esquema del estado compartido del orquestador.

La idea central
---------------
En un sistema multi-agente el riesgo no es que un agente falle: es que el
**contexto se pierda o se degrade** al pasar de uno a otro. Si lo único que
comparten los agentes es el historial de mensajes, cada dato cuantitativo tiene
que viajar como texto, y un número que el LLM reescribe tres veces es un número
que en algún momento va a salir mal.

Por eso el estado tiene dos canales separados:

- `messages` — la conversación. Sirve para coordinar y para que el supervisor
  entienda qué pasó. Es texto y puede degradarse.
- `hallazgos` — los datos **estructurados** que produjo cada agente, con su
  procedencia. No pasa por ningún prompt: los nodos lo escriben y las
  herramientas del analista lo leen directo del estado vía `InjectedState`.

El informe final se arma sobre `hallazgos`, no sobre `messages`. Esa es la
diferencia entre un número citado y un número recordado.

Trazabilidad
------------
Cada `Hallazgo` dice qué agente lo produjo, con qué herramienta y cuándo. El
requisito de "rastrear qué agente ha contribuido con qué información" se cumple
en el tipo de dato, no en una convención que haya que recordar respetar.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator

# Los nombres de los nodos a los que el supervisor puede derivar. Se usa el
# mismo Literal en el estado, en la salida estructurada del supervisor y en el
# tipo de retorno de la función de ruteo, así que un nombre mal escrito es un
# error de tipo y no un `KeyError` en runtime.
Destino = Literal["investigador", "analista", "validador"]

NombreAgente = Literal["investigador", "analista", "validador", "sintetizador"]


class Hallazgo(BaseModel):
    """
    Una contribución concreta de un agente, con su procedencia.

    `datos` lleva el payload estructurado tal como lo devolvió la herramienta.
    Es lo que después lee el analista: los registros no se le piden al LLM que
    los repita, se leen de acá.
    """

    agente: NombreAgente
    herramienta: str = Field(..., description="Herramienta que produjo el dato")
    resumen: str = Field(..., description="Una línea legible, para el supervisor")
    datos: dict[str, Any] = Field(
        default_factory=dict, description="Payload estructurado, sin pasar por el LLM"
    )
    momento: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class Rubrica(BaseModel):
    """
    La rúbrica con la que el supervisor evalúa si la tarea está terminada.

    Son preguntas binarias a propósito. Un supervisor al que se le pregunta
    "¿está completo?" contesta que sí demasiado pronto; uno al que se le piden
    cuatro condiciones verificables tiene que mirar el estado.
    """

    hay_evidencia_documental: bool = Field(
        ..., description="¿Se consultó la documentación técnica y trajo fragmentos?"
    )
    hay_datos_cuantitativos: bool = Field(
        ..., description="¿Se recuperaron incidentes concretos de la base?"
    )
    hay_analisis_estadistico: bool = Field(
        ..., description="¿El analista calculó métricas sobre esos datos?"
    )
    hay_validacion_de_datos: bool = Field(
        ..., description="¿Se verificó la calidad/integridad de los registros?"
    )

    @property
    def completa(self) -> bool:
        """True solo si las cuatro condiciones se cumplen."""
        return all(
            (
                self.hay_evidencia_documental,
                self.hay_datos_cuantitativos,
                self.hay_analisis_estadistico,
                self.hay_validacion_de_datos,
            )
        )

    def faltantes(self) -> list[str]:
        """Nombres de las condiciones que todavía no se cumplen."""
        return [
            nombre
            for nombre, cumplida in (
                ("evidencia documental", self.hay_evidencia_documental),
                ("datos cuantitativos", self.hay_datos_cuantitativos),
                ("análisis estadístico", self.hay_analisis_estadistico),
                ("validación de datos", self.hay_validacion_de_datos),
            )
            if not cumplida
        ]


class DecisionSupervisor(BaseModel):
    """
    La salida estructurada del supervisor.

    Se le pide salida estructurada y no texto libre justamente para que el
    ruteo no dependa de parsear una frase. El `Literal` del campo `siguiente`
    hace que el modelo no pueda inventar un nodo que no existe: si lo intenta,
    la validación de Pydantic lo rechaza antes de llegar al grafo.
    """

    razonamiento: str = Field(
        ..., description="Por qué deriva a ese agente, en una o dos oraciones"
    )
    rubrica: Rubrica
    siguiente: Destino = Field(
        ..., description="Nodo que debe intervenir ahora"
    )
    instruccion: str = Field(
        ...,
        description=(
            "La tarea puntual para ese agente. Es lo ÚNICO que el especialista "
            "recibe como pedido, así que tiene que ser autosuficiente."
        ),
    )

    @field_validator("instruccion")
    @classmethod
    def _no_vacia(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "La instrucción no puede estar vacía: el especialista no ve el "
                "historial completo, así que se quedaría sin saber qué hacer."
            )
        return v.strip()


class EstadoOrquestador(MessagesState):
    """
    El estado compartido del grafo.

    Hereda de `MessagesState`, que ya trae `messages` con `add_messages` como
    reducer. Los campos acumulativos usan `operator.add`; los que describen "la
    última decisión" se sobreescriben, que es el comportamiento por defecto.
    """

    # La pregunta original del usuario. Se guarda aparte porque los
    # especialistas no reciben el historial completo y la necesitan como marco.
    solicitud: str

    # Acumulativos: cada agente appendea lo suyo.
    hallazgos: Annotated[list[Hallazgo], operator.add]
    pasos_del_supervisor: Annotated[int, operator.add]

    # Última decisión del supervisor (se sobreescribe en cada vuelta).
    siguiente: Destino
    instruccion_actual: str
    razonamiento_supervisor: str
    rubrica_actual: Rubrica | None

    # Resultado del nodo validador y salida final.
    validacion_aprobada: bool
    motivo_validacion: str
    informe_final: str


def hallazgos_de(estado: dict[str, Any], agente: NombreAgente) -> list[Hallazgo]:
    """Filtra los hallazgos producidos por un agente puntual."""
    return [h for h in estado.get("hallazgos", []) if h.agente == agente]


def incidentes_recuperados(estado: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Junta los incidentes que el investigador dejó en el estado.

    Es el puente entre los dos especialistas, y la razón por la que las
    herramientas del analista no necesitan que el LLM les dicte los datos: los
    registros se leen de acá, exactamente como salieron de la base.
    """
    registros: list[dict[str, Any]] = []
    vistos: set[str] = set()

    for hallazgo in estado.get("hallazgos", []):
        for incidente in hallazgo.datos.get("incidentes", []):
            clave = incidente.get("incidente_id", "")
            if clave and clave not in vistos:
                vistos.add(clave)
                registros.append(incidente)

    return registros
