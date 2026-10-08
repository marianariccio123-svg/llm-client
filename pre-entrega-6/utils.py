"""
utils.py

Utilidades chicas compartidas por los nodos del grafo.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def texto_de(mensaje: Any) -> str:
    """
    Extrae el texto de un mensaje de LangChain, venga como venga.

    Hace falta porque no todos los proveedores devuelven lo mismo en
    `.content`: OpenAI manda un string pelado, mientras que Gemini y Anthropic
    mandan una lista de bloques tipados (`[{"type": "text", "text": "..."}]`).
    Interpolar el `.content` crudo en un f-string funcionaría con uno y
    pegaría el repr de una lista con los otros — y ese repr termina dentro del
    prompt del supervisor, que es el peor lugar donde puede terminar.
    """
    contenido = getattr(mensaje, "content", mensaje)

    if isinstance(contenido, str):
        return contenido

    if isinstance(contenido, list):
        partes = [
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        ]
        return "\n".join(p for p in partes if p)

    return str(contenido)


def _resumir_hallazgo(herramienta: str, payload: dict[str, Any]) -> str:
    """
    Una línea legible de qué aportó una herramienta.

    El supervisor lee estos resúmenes en vez de los payloads completos. Es la
    otra mitad de la defensa contra la contaminación de contexto: el dato
    estructurado queda en `Hallazgo.datos` para quien lo necesite, y el
    supervisor ve una síntesis de tamaño constante.
    """
    if "error" in payload:
        return f"{herramienta} falló: {payload['error'][:120]}"

    if herramienta == "buscar_en_documentacion":
        fragmentos = payload.get("fragmentos", [])
        titulos = sorted({f.get("titulo", "?") for f in fragmentos})
        return (
            f"{len(fragmentos)} fragmentos de documentación "
            f"(vía {payload.get('fuente_de_busqueda', '?')}) sobre: {', '.join(titulos)}"
        )

    if herramienta == "listar_incidentes":
        periodo = payload.get("periodo", {})
        return (
            f"{payload.get('cantidad', 0)} incidentes entre "
            f"{periodo.get('desde', '?')} y {periodo.get('hasta', '?')}, en "
            f"{len(payload.get('categorias_presentes', []))} categorías"
        )

    if herramienta == "resumir_incidentes":
        dominante = (payload.get("por_categoria") or [{}])[0]
        metricas = payload.get("metricas_minutos", {})
        return (
            f"{payload.get('total_incidentes', 0)} incidentes, "
            f"{payload.get('total_minutos_caidos', 0)} minutos caídos en total; "
            f"mediana {metricas.get('mediana', '?')} min; causa dominante "
            f"{dominante.get('categoria', '?')} ({dominante.get('porcentaje', '?')}%)"
        )

    if herramienta == "detectar_atipicos":
        cantidad = payload.get("cantidad_atipicos", 0)
        if not cantidad:
            return (
                f"Sin atípicos por IQR (límite "
                f"{payload.get('limite_superior', '?')} min)"
            )
        peor = payload["atipicos"][0]
        return (
            f"{cantidad} atípico(s); el mayor es {peor.get('incidente_id', '?')} "
            f"con {peor.get('minutos_caidos', '?')} min "
            f"({peor.get('veces_la_mediana', '?')}x la mediana)"
        )

    if herramienta == "validar_registros":
        return (
            f"{payload.get('validos', 0)}/{payload.get('registros_evaluados', 0)} "
            f"registros válidos ({payload.get('tasa_de_validez', '?')}%), "
            f"{payload.get('invalidos', 0)} con problemas"
        )

    return f"Resultado de {herramienta}"


def extraer_hallazgos(mensajes: list[Any], agente: str) -> list[Any]:
    """
    Convierte los `ToolMessage` de un especialista en `Hallazgo` estructurados.

    Es el paso que hace que los datos **no** dependan de que el LLM los repita:
    el payload que devolvió la herramienta se guarda tal cual en el estado, y de
    ahí lo leen las herramientas del analista y el sintetizador final.
    """
    from state import Hallazgo

    hallazgos = []

    for mensaje in mensajes:
        if not isinstance(mensaje, ToolMessage):
            continue
        try:
            payload = json.loads(mensaje.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue

        herramienta = mensaje.name or "?"
        hallazgos.append(
            Hallazgo(
                agente=agente,
                herramienta=herramienta,
                resumen=_resumir_hallazgo(herramienta, payload),
                datos=payload,
            )
        )

    return hallazgos
