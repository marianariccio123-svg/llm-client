"""
tests/stub_llm.py

Puente al LLM guionado de la Pre-entrega 6.

Por qué no es un `from ... import ...` normal
---------------------------------------------
El stub ya existe en `pre-entrega-6/tests/stub_llm.py` y no tiene sentido
copiarlo: una copia divergiría en cuanto alguien arregle algo de un lado.

Pero no se puede importar con un `import` común. Esta entrega también tiene un
paquete llamado `tests`, y el de acá gana: `from tests.stub_llm import ...`
resuelve a este archivo, no al de la entrega anterior. Agregar la otra carpeta
al `sys.path` no ayuda, porque el conflicto es de nombre de paquete, no de
ruta.

La salida es cargar el módulo por su ruta de archivo, que es explícito y no
depende de ningún orden de `sys.path`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ORIGEN = (
    Path(__file__).resolve().parent.parent.parent
    / "pre-entrega-6" / "tests" / "stub_llm.py"
)

if not _ORIGEN.exists():
    raise ImportError(
        f"No se encontró el stub de la Pre-entrega 6 en {_ORIGEN}. "
        f"Los tests de esta entrega lo reutilizan: hace falta que la carpeta "
        f"pre-entrega-6/ esté presente en el repo."
    )

_NOMBRE = "stub_llm_m6"

_spec = importlib.util.spec_from_file_location(_NOMBRE, _ORIGEN)
_modulo = importlib.util.module_from_spec(_spec)

# El registro en `sys.modules` va ANTES de ejecutar el módulo, y no es opcional.
# `LLMGuionado` es un modelo Pydantic con `from __future__ import annotations`,
# así que sus anotaciones son strings que Pydantic resuelve buscando el módulo
# donde se definió la clase. Si no está registrado, la clase queda "not fully
# defined" y falla recién al instanciarla, con un error que no menciona los
# imports para nada.
sys.modules[_NOMBRE] = _modulo
_spec.loader.exec_module(_modulo)

GuionAgotado = _modulo.GuionAgotado
LLMGuionado = _modulo.LLMGuionado
mensaje_con_herramientas = _modulo.mensaje_con_herramientas

__all__ = ["GuionAgotado", "LLMGuionado", "mensaje_con_herramientas"]
