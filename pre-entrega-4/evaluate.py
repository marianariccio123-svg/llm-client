"""
evaluate.py

Evaluación del recuperador contra un Golden Set de 5 preguntas de las que
conocemos de antemano el documento fuente.

Métricas
--------
Las dos se calculan a nivel **documento**, no a nivel chunk: lo que importa es
si el sistema trajo el material correcto, no si trajo el pedazo exacto.

- **Recall@k** — ¿está `documento_id_esperado` entre los k recuperados?
  Es binaria por pregunta (0 o 1) y se promedia sobre el Golden Set.
  Responde: "¿el sistema encuentra lo que hace falta?".

- **Precision@k** — de los k chunks devueltos, ¿qué proporción pertenece a un
  documento relevante? Responde: "¿cuánta basura arrastra en el camino?".

- **MRR** (extra, no pedida por la consigna) — el inverso de la posición en la
  que aparece el documento esperado. Se agrega porque con un corpus de 8
  documentos y k=5, Recall@5 satura en 1.00 para los tres modos y deja de
  distinguirlos: traerlo primero y traerlo cuarto puntúan igual. MRR sí
  separa "lo encontró" de "lo encontró y lo puso arriba", que es justamente
  donde se ve el aporte del recuperador híbrido.

Sobre el techo de Precision@k
-----------------------------
Precision@5 tiene un máximo alcanzable que casi nunca es 1.0. Si una pregunta
tiene un solo documento relevante y ese documento aportó 3 chunks al índice,
lo mejor que puede pasar es 3/5 = 0.60: los otros 2 slots van a estar ocupados
por algo irrelevante sí o sí, porque no existe más material relevante que
traer. Reportar 0.60 como si fuera un mal resultado sería un error de lectura.

Por eso se informan las dos cifras: la **cruda** (comparable con cualquier otro
sistema) y la **normalizada** (cruda ÷ techo alcanzable), que mide qué tan
cerca está el recuperador de lo mejor que este corpus permite.

Uso:
    python evaluate.py                 # compara hibrido, denso y bm25
    python evaluate.py --modo hibrido  # evalúa un solo modo
    python evaluate.py --k 3           # cambia el k de las métricas
    python evaluate.py --json          # salida en JSON, para pegar en el README
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from config import BM25_CORPUS_PATH, GOLDEN_SET_PATH, TOP_K, setup_logging
from rag import RAGSystem
from schemas import MetricasModo, PreguntaGolden

log = logging.getLogger("evaluate")

MODOS = ("hibrido", "denso", "bm25")


def cargar_golden_set() -> list[PreguntaGolden]:
    if not GOLDEN_SET_PATH.exists():
        raise RuntimeError(f"Falta el Golden Set en {GOLDEN_SET_PATH}.")
    crudo = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return [PreguntaGolden.model_validate(item) for item in crudo]


def contar_chunks_por_documento() -> Counter[str]:
    """
    Cuántos chunks aportó cada documento al índice.

    Sale del JSONL de la ingesta, que por construcción tiene exactamente los
    mismos chunks que se subieron a Pinecone. Se usa para calcular el techo de
    Precision@k.
    """
    if not BM25_CORPUS_PATH.exists():
        raise RuntimeError(
            f"Falta {BM25_CORPUS_PATH.name}. Corré primero: python ingest.py"
        )
    conteo: Counter[str] = Counter()
    with BM25_CORPUS_PATH.open(encoding="utf-8") as f:
        for linea in f:
            conteo[json.loads(linea)["doc_id"]] += 1
    return conteo


def evaluar_modo(
    modo: str,
    golden: list[PreguntaGolden],
    chunks_por_doc: Counter[str],
    k: int,
    verbose: bool = True,
) -> tuple[MetricasModo, list[dict]]:
    """Corre las k consultas del Golden Set contra un modo y calcula las métricas."""
    sistema = RAGSystem(modo=modo, k=k)
    detalle: list[dict] = []

    for caso in golden:
        documentos = sistema.retrieve(caso.pregunta)
        ids_recuperados = [d.metadata.get("doc_id", "?") for d in documentos]
        relevantes = set(caso.documentos_relevantes)

        aciertos = sum(1 for doc_id in ids_recuperados if doc_id in relevantes)
        recall = 1.0 if caso.documento_id_esperado in ids_recuperados else 0.0
        precision = aciertos / k

        # Techo: no se pueden recuperar más chunks relevantes de los que existen.
        disponibles = sum(chunks_por_doc.get(doc_id, 0) for doc_id in relevantes)
        techo = min(disponibles, k) / k
        precision_norm = precision / techo if techo else 0.0

        detalle.append({
            "pregunta": caso.pregunta,
            "esperado": caso.documento_id_esperado,
            "recuperados": ids_recuperados,
            "posicion_esperado": (
                ids_recuperados.index(caso.documento_id_esperado) + 1
                if recall else None
            ),
            "recall": recall,
            "reciprocal_rank": round(
                1.0 / (ids_recuperados.index(caso.documento_id_esperado) + 1), 3
            ) if recall else 0.0,
            "precision": round(precision, 3),
            "precision_max": round(techo, 3),
            "precision_normalizada": round(precision_norm, 3),
        })

        if verbose:
            marca = "OK " if recall else "NO "
            posicion = detalle[-1]["posicion_esperado"]
            print(f"  {marca} {caso.pregunta}")
            print(f"       esperado: {caso.documento_id_esperado}"
                  + (f" (salió en el puesto {posicion})" if posicion else " (NO apareció)"))
            print(f"       top-{k}:   {', '.join(ids_recuperados)}")
            print(f"       P@{k}={precision:.2f} (techo {techo:.2f}) "
                  f"-> normalizada {precision_norm:.2f}")

    n = len(golden)
    metricas = MetricasModo(
        modo=modo,
        k=k,
        recall_at_k=sum(d["recall"] for d in detalle) / n,
        precision_at_k=sum(d["precision"] for d in detalle) / n,
        precision_normalizada=sum(d["precision_normalizada"] for d in detalle) / n,
        mrr=sum(d["reciprocal_rank"] for d in detalle) / n,
        preguntas=n,
    )
    return metricas, detalle


def imprimir_resumen(resultados: list[MetricasModo], k: int) -> None:
    print("\n" + "=" * 62)
    print(f"RESUMEN — Golden Set de {resultados[0].preguntas} preguntas, k={k}")
    print("=" * 62)
    print(f"{'modo':<10} {'Recall@'+str(k):>10} {'Precision@'+str(k):>14} "
          f"{'P norm.':>10} {'MRR':>8}")
    print("-" * 62)
    for m in resultados:
        print(f"{m.modo:<10} {m.recall_at_k:>10.2f} {m.precision_at_k:>14.2f} "
              f"{m.precision_normalizada:>10.2f} {m.mrr:>8.2f}")
    print("-" * 62)

    if len(resultados) > 1:
        mejor = max(resultados, key=lambda m: (m.recall_at_k, m.mrr, m.precision_at_k))
        print(f"Mejor combinación recall/precisión: {mejor.modo}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=MODOS, help="evaluar un solo modo")
    parser.add_argument("--k", type=int, default=TOP_K, help=f"k de las métricas (default {TOP_K})")
    parser.add_argument("--json", action="store_true", help="salida en JSON")
    args = parser.parse_args()

    setup_logging(logging.WARNING if args.json else logging.INFO)

    golden = cargar_golden_set()
    chunks_por_doc = contar_chunks_por_documento()
    modos = [args.modo] if args.modo else list(MODOS)

    resultados: list[MetricasModo] = []
    detalles: dict[str, list[dict]] = {}

    for modo in modos:
        if not args.json:
            print(f"\n--- Modo: {modo} ---")
        metricas, detalle = evaluar_modo(
            modo, golden, chunks_por_doc, args.k, verbose=not args.json
        )
        resultados.append(metricas)
        detalles[modo] = detalle

    if args.json:
        print(json.dumps(
            {
                "k": args.k,
                "resumen": [m.model_dump() for m in resultados],
                "detalle": detalles,
            },
            ensure_ascii=False,
            indent=2,
        ))
    else:
        imprimir_resumen(resultados, args.k)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as e:
        logging.getLogger("evaluate").error("%s", e)
        sys.exit(1)
