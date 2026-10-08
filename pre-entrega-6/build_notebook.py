"""
build_notebook.py

Construye y ejecuta `demo.ipynb`, el notebook que demuestra el flujo de
delegación del orquestador.

El notebook se genera desde acá en vez de editarse a mano por una razón
práctica: un `.ipynb` es un JSON con outputs embebidos, y editarlo a mano
termina en diffs ilegibles y en celdas cuyo output no corresponde al código.
Generarlo garantiza que lo que se ve ejecutado es exactamente lo que dice el
código.

Uso:
    python build_notebook.py              # construye y ejecuta (consume API)
    python build_notebook.py --sin-correr  # solo construye, sin ejecutar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK = BASE_DIR / "demo.ipynb"

CELDAS: list[tuple[str, str]] = [
    ("md", """# Orquestador multi-agente — demostración del flujo de delegación

Este notebook muestra el orquestador resolviendo una consulta que **ningún
agente puede responder solo**:

> *Nuestros servicios asíncronos se vienen cayendo seguido. ¿Qué dice nuestra
> documentación técnica sobre las causas más probables, y qué muestran nuestros
> propios incidentes del último año?*

La primera mitad necesita búsqueda semántica sobre la documentación del equipo.
La segunda necesita consultar la bitácora de incidentes y calcular estadísticas
sobre ella. Son dos dominios distintos, y el informe final tiene sentido solo si
alguien los conecta.

Lo que se va a ver, en orden:

1. La topología del grafo.
2. Las herramientas acotadas de cada especialista.
3. **El flujo de delegación paso a paso**, en vivo.
4. La trazabilidad: qué agente aportó qué.
5. La rúbrica de validación y el informe final.
6. Los tests con LLM simulado, que cubren lo mismo sin gastar API."""),

    ("code", """import sys
from pathlib import Path

# El notebook corre desde pre-entrega-6/, igual que el CLI.
sys.path.insert(0, str(Path.cwd()))

from config import MAX_PASOS_SUPERVISOR, RECURSION_LIMIT, get_model_name, get_provider, setup_logging
from graph import abrir_orquestador, config_thread, diagrama_mermaid, entrada
from seed_db import asegurar_db

# Los logs del orquestador van a WARNING: en el notebook queremos ver el flujo
# de delegación, no la telemetría de cada llamada. El CLI (`python main.py
# --paso-a-paso`) los muestra en INFO.
import logging
setup_logging(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

asegurar_db()

print(f"proveedor:            {get_provider()} / {get_model_name()}")
print(f"tope del supervisor:  {MAX_PASOS_SUPERVISOR} pasos")
print(f"recursion_limit:      {RECURSION_LIMIT}")"""),

    ("md", """## 1. La topología

Jerárquica: un supervisor enruta, dos especialistas trabajan, un validador
verifica y un sintetizador cierra. Los especialistas **nunca** deciden quién
sigue — siempre devuelven el control al supervisor.

El diagrama se genera desde el grafo compilado, no se escribe a mano."""),

    ("code", """print(diagrama_mermaid())"""),

    ("md", """## 2. Las herramientas de cada especialista

Acotadas a propósito. El investigador no puede calcular y el analista no puede
buscar: esa frontera es lo que hace que el supervisor tenga algo que coordinar.

Las tres herramientas del analista reciben los datos por `InjectedState`, así
que el LLM decide *cuándo* llamarlas pero no le dicta los registros."""),

    ("code", """from agents.analyst_agent import HERRAMIENTAS as TOOLS_ANALISTA
from agents.research_agent import HERRAMIENTAS as TOOLS_INVESTIGADOR

for nombre, herramientas in [("INVESTIGADOR", TOOLS_INVESTIGADOR), ("ANALISTA", TOOLS_ANALISTA)]:
    print(f"\\n{nombre}")
    for h in herramientas:
        argumentos = [a for a in h.args.keys()]
        primera_linea = (h.description or "").strip().split("\\n")[0]
        print(f"  - {h.name}({', '.join(argumentos) or '— lee del estado —'})")
        print(f"      {primera_linea}")"""),

    ("md", """## 3. El flujo de delegación, en vivo

`astream(stream_mode="updates")` entrega lo que devolvió cada nodo a medida que
termina. Es la forma de ver la delegación mientras pasa, en lugar de esperar al
resultado final.

Prestá atención al orden: el supervisor manda primero al investigador (sin datos
no hay nada que analizar), después al analista, y recién entonces al validador."""),

    ("code", """from main import CONSULTA_DEMO, describir_paso

# Checkpointer en memoria: la demo no deja estado de corridas anteriores.
async with abrir_orquestador(":memory:") as orquestador:
    config = config_thread("demo-notebook")
    async for fragmento in orquestador.astream(
        entrada(CONSULTA_DEMO), config=config, stream_mode="updates"
    ):
        for nodo, actualizacion in fragmento.items():
            print(f"\\n> {describir_paso(nodo, actualizacion)}")

    final = (await orquestador.aget_state(config)).values"""),

    ("md", """## 4. Trazabilidad: qué agente aportó qué

El estado no guarda solo la conversación: guarda `hallazgos`, una lista de
contribuciones estructuradas con su procedencia. Cada número del informe final
se puede rastrear hasta la herramienta que lo produjo."""),

    ("code", """print(f"{'#':>2}  {'agente':<14} {'herramienta':<24} resumen")
print("-" * 100)
for i, h in enumerate(final["hallazgos"], start=1):
    print(f"{i:>2}  {h.agente:<14} {h.herramienta:<24} {h.resumen}")

print(f"\\npasos de supervisión: {final['pasos_del_supervisor']}")
print(f"hallazgos totales:    {len(final['hallazgos'])}")"""),

    ("md", """### Los datos crudos siguen ahí

Los payloads se guardan tal como los devolvieron las herramientas. El
sintetizador lee de acá —no del historial de mensajes— y por eso los números del
informe son los de la base y no una reconstrucción del modelo."""),

    ("code", """import json

resumen = next(h for h in final["hallazgos"] if h.herramienta == "resumir_incidentes")
print(json.dumps(resumen.datos["metricas_minutos"], ensure_ascii=False, indent=2))
print("\\nPor categoría:")
for fila in resumen.datos["por_categoria"]:
    print(f"  {fila['categoria']:<26} {fila['cantidad']:>2} incidentes "
          f"({fila['porcentaje']:>5}%)  {fila['minutos_totales']:>5} min")

atipicos = next(h for h in final["hallazgos"] if h.herramienta == "detectar_atipicos")
print(f"\\nAtípicos (IQR, límite {atipicos.datos['limite_superior']} min):")
for a in atipicos.datos["atipicos"]:
    print(f"  {a['incidente_id']}  {a['minutos_caidos']} min  "
          f"({a['veces_la_mediana']}x la mediana)")"""),

    ("md", """## 5. La rúbrica y la validación

El validador **no usa LLM**. Recorre los hallazgos y comprueba qué herramientas
corrieron efectivamente. Si el supervisor declarara el trabajo terminado sin
haber pasado por el analista, el validador lo detectaría y lo devolvería: no le
cree al modelo, le cree al estado."""),

    ("code", """rubrica = final["rubrica_actual"]

for criterio, cumplido in rubrica.model_dump().items():
    print(f"  [{'x' if cumplido else ' '}] {criterio}")

print(f"\\nrúbrica completa: {rubrica.completa}")
print(f"validación:       {final['motivo_validacion']}")"""),

    ("md", """## 6. El informe final

Lo redacta el sintetizador a partir de los hallazgos estructurados. Conecta las
dos mitades: lo que dice la documentación sobre la causa y lo que muestran
nuestros propios datos."""),

    ("code", """print(final["informe_final"])"""),

    ("md", """## 7. Lo mismo, sin gastar API: los tests con LLM simulado

Todo lo anterior cuesta llamadas al modelo y depende de que hoy decida lo mismo
que ayer. Sirve como demostración, no como test.

La suite reemplaza el modelo por una cola de respuestas prefabricadas
(`tests/stub_llm.py`) y ejercita el grafo entero de forma determinista: el ruteo,
los ciclos ReAct, el tope de pasos, el paso de contexto y la persistencia. Corre
sin red y sin ninguna API key."""),

    ("code", """import subprocess

resultado = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True, cwd=str(Path.cwd()),
)
print(resultado.stdout.strip())
print("exit code:", resultado.returncode)"""),

    ("md", """---

### Resumen de lo demostrado

| Requisito de la consigna | Dónde se vio |
|---|---|
| Topología jerárquica con supervisor como router | celdas 1 y 3 |
| Dos especialistas con dominios distintos | celda 2 |
| Estado compartido estructurado con trazabilidad | celda 4 |
| El supervisor decide si está completo o hay que refinar | celda 5 |
| Validación antes del `END` | celda 5 |
| Freno contra el supervisor infinito | `MAX_PASOS_SUPERVISOR`, celdas 1 y 7 |
| Contaminación de contexto evitada | `InjectedState`, celdas 2 y 4 |"""),
]


def construir() -> nbf.NotebookNode:
    """Arma el notebook a partir de la lista de celdas."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(contenido) if tipo == "md"
        else nbf.v4.new_code_cell(contenido)
        for tipo, contenido in CELDAS
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return notebook


def ejecutar(notebook: nbf.NotebookNode) -> nbf.NotebookNode:
    """Corre el notebook y le embebe los outputs reales."""
    from nbclient import NotebookClient

    cliente = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(BASE_DIR)}},
    )
    cliente.execute()
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sin-correr", action="store_true",
                        help="construir el notebook sin ejecutarlo")
    args = parser.parse_args()

    notebook = construir()
    print(f"Notebook armado: {len(notebook.cells)} celdas.")

    if not args.sin_correr:
        print("Ejecutando (consume llamadas a la API)...")
        notebook = ejecutar(notebook)
        print("Ejecutado.")

    nbf.write(notebook, NOTEBOOK)
    print(f"Escrito en {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
