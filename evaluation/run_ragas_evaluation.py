"""
Evaluación científica del sistema RAG de TEIA usando el framework RAGAS.

Estrategia: el script importa el retriever de src/ directamente para obtener
el texto real de los fragmentos recuperados (necesario para RAGAS), sin pasar
por la API REST (que solo devuelve nombres de archivo en el campo 'sources').

Uso:
    # Desde la raíz del proyecto:
    python evaluation/run_ragas_evaluation.py

Dependencias:
    pip install ragas datasets langchain-community langchain-ollama

Referencia: Es et al. (2023). RAGAS: Automated Evaluation of Retrieval
Augmented Generation. arXiv:2309.15217
"""

import json
import sys
import os
import asyncio
from pathlib import Path

# ---------------------------------------------------------------------------
# Añadir src/ al path para importar el retriever y el generador directamente
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)

# ---------------------------------------------------------------------------
# Configurar RAGAS para usar Ollama como LLM juez (sin costo, local)
#
# ADVERTENCIA: qwen2.5:7b-instruct es un modelo de 7B parámetros. RAGAS
# requiere que el LLM juez genere salidas JSON estructuradas (especialmente
# para Faithfulness y Answer Relevancy). Los modelos pequeños a veces no
# respetan el formato esperado, lo que produce NaN en esas métricas.
#
# Si ocurre esto, las opciones son:
#   1. Usar un modelo más grande (llama3:70b, mixtral:8x7b) si hay suficiente VRAM
#   2. Usar solo las métricas que no requieren juez LLM:
#      context_precision, answer_similarity
# ---------------------------------------------------------------------------
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

ragas_llm = LangchainLLMWrapper(
    ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,          # Determinístico para reproducibilidad
        format="json",          # Fuerza salida JSON estructurada
    )
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
)

# Inyectar LLM y embeddings en todas las métricas
METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
]

for metric in METRICS:
    if hasattr(metric, "llm"):
        metric.llm = ragas_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = ragas_embeddings


# ---------------------------------------------------------------------------
# Importar servicios de TEIA (retriever + generador)
# ---------------------------------------------------------------------------
from services.rag_retriever import get_rag_retriever
from services.ollama_service import generate_response
from config import settings

# Módulos válidos del curso
MODULE_DISPLAY_NAMES = {
    "caracterizacion":        "Caracterización de la Asignatura",
    "factores_situacionales": "Factores Situacionales",
    "resultados_aprendizaje": "Resultados de Aprendizaje del Curso",
    "actividades_aprendizaje":"Actividades de Aprendizaje",
    "evaluacion":             "Estrategias de Evaluación",
    "secuencia":              "Secuencia Didáctica",
}

GOLDEN_DATASET_PATH = ROOT / "docs" / "golden_dataset.json"
RESULTS_PATH        = ROOT / "docs" / "ragas_results.csv"
RESULTS_DETAIL_PATH = ROOT / "docs" / "ragas_results_detail.json"

TOP_K = 5  # Mismo valor que usa el sistema en producción


async def run_single_query(retriever, entry: dict) -> dict:
    """Ejecuta una consulta al sistema TEIA y devuelve los datos para RAGAS."""
    question = entry["question"]
    module   = entry["module"]

    # 1. Recuperar fragmentos (texto completo, no solo nombres de archivo)
    chunks = retriever.retrieve(query=question, module=module, top_k=TOP_K)

    # 2. Extraer el texto de cada fragmento para RAGAS
    contexts = [c["content"] for c in chunks] if chunks else ["Sin contexto disponible"]

    # 3. Construir el string de contexto para el generador (igual que en /ask)
    if chunks:
        parts = [f"[Fragmento {i+1}]\n{c['content']}" for i, c in enumerate(chunks)]
        sources_label = ", ".join({c["source"] for c in chunks})
        context_str = (
            "CONTEXTO RECUPERADO DEL CURSO:\n"
            + "\n\n".join(parts)
            + f"\n\nFUENTES: {sources_label}"
        )
    else:
        context_str = f"No se encontraron fragmentos para el módulo '{module}'."

    # 4. Generar respuesta con el LLM (sin historial — evaluación single-turn)
    module_display = MODULE_DISPLAY_NAMES.get(module, "General")
    answer = await generate_response(
        question=question,
        context=context_str,
        module=module_display,
        history=[]
    )

    return {
        "question":    question,
        "answer":      answer,
        "contexts":    contexts,           # Lista de strings — formato RAGAS
        "ground_truth": entry["ground_truth"],
    }


async def main():
    # ------------------------------------------------------------------
    # 1. Cargar el golden dataset
    # ------------------------------------------------------------------
    print(f"Cargando golden dataset desde {GOLDEN_DATASET_PATH}...")
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    # Filtrar entradas sin ground_truth (aún sin rellenar)
    golden_filled = [e for e in golden if e.get("ground_truth", "").strip()]
    total = len(golden_filled)

    if total == 0:
        print(
            "\n[ERROR] El golden dataset no tiene ninguna entrada con ground_truth.\n"
            "Rellena el campo 'ground_truth' en docs/golden_dataset.json antes de ejecutar."
        )
        return

    print(f"  → {total} entradas con ground_truth listas para evaluar.")
    print(f"  → {len(golden) - total} entradas pendientes de rellenar (se omiten).\n")

    # ------------------------------------------------------------------
    # 2. Inicializar retriever
    # ------------------------------------------------------------------
    print("Inicializando RAG retriever...")
    retriever = get_rag_retriever()
    if not retriever.is_available:
        print("[ERROR] El índice ChromaDB está vacío. Ejecuta la indexación primero.")
        return
    print(f"  → Índice disponible: {retriever.chunk_count} fragmentos indexados.\n")

    # ------------------------------------------------------------------
    # 3. Ejecutar el sistema para cada pregunta del golden dataset
    # ------------------------------------------------------------------
    print("Ejecutando consultas al sistema TEIA...")
    results = []
    for i, entry in enumerate(golden_filled, 1):
        print(f"  [{i:02d}/{total}] {entry['id']} ({entry['module']}, {entry['difficulty']}) ...", end=" ")
        try:
            result = await run_single_query(retriever, entry)
            results.append(result)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  → {len(results)}/{total} consultas completadas exitosamente.\n")

    if not results:
        print("[ERROR] Ninguna consulta fue exitosa. Verifica que Ollama esté corriendo.")
        return

    # ------------------------------------------------------------------
    # 4. Guardar respuestas generadas en el golden dataset (para revisión)
    # ------------------------------------------------------------------
    answers_map = {r["question"]: r["answer"] for r in results}
    contexts_map = {r["question"]: r["contexts"] for r in results}
    for entry in golden:
        if entry["question"] in answers_map:
            entry["generated_answer"]  = answers_map[entry["question"]]
            entry["retrieved_contexts"] = contexts_map[entry["question"]]

    with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
    print("  → Respuestas guardadas en docs/golden_dataset.json\n")

    # ------------------------------------------------------------------
    # 5. Crear Dataset de HuggingFace (formato requerido por RAGAS)
    # ------------------------------------------------------------------
    dataset = Dataset.from_list(results)

    # ------------------------------------------------------------------
    # 6. Ejecutar evaluación RAGAS
    # ------------------------------------------------------------------
    print("Ejecutando evaluación RAGAS (esto puede tomar varios minutos)...")
    print("  Métricas: faithfulness, answer_relevancy, context_precision,")
    print("            context_recall, answer_correctness, answer_similarity\n")

    # RunConfig: secuencial (max_workers=1) para evitar TimeoutErrors con LLMs locales.
    # timeout=180s por llamada — qwen2.5:7b-instruct puede tardar ~60-90s en respuestas largas.
    run_cfg = RunConfig(timeout=180, max_workers=1)
    scores = evaluate(dataset, metrics=METRICS, run_config=run_cfg)

    # ------------------------------------------------------------------
    # 7. Guardar y mostrar resultados
    # ------------------------------------------------------------------
    df = scores.to_pandas()

    # Añadir metadatos del golden dataset
    df["id"]         = [e["id"] for e in golden_filled[:len(df)]]
    df["module"]     = [e["module"] for e in golden_filled[:len(df)]]
    df["difficulty"] = [e["difficulty"] for e in golden_filled[:len(df)]]

    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nResultados guardados en {RESULTS_PATH}")

    # Resumen agregado
    numeric_cols = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "answer_correctness", "answer_similarity"]
    available_cols = [c for c in numeric_cols if c in df.columns]

    print("\n" + "="*60)
    print("RESUMEN DE MÉTRICAS RAGAS — CONFIGURACIÓN BASE")
    print("="*60)
    summary = df[available_cols].describe().loc[["mean", "std", "min", "max"]]
    print(summary.to_string())

    print("\n" + "="*60)
    print("MÉTRICAS PROMEDIO POR MÓDULO")
    print("="*60)
    print(df.groupby("module")[available_cols].mean().to_string())

    print("\n" + "="*60)
    print("UMBRALES DE REFERENCIA (Es et al., 2023)")
    print("="*60)
    thresholds = {
        "context_precision":  0.75,
        "context_recall":     0.70,
        "faithfulness":       0.80,
        "answer_relevancy":   0.75,
        "answer_correctness": 0.60,
        "answer_similarity":  0.70,
    }
    for metric, threshold in thresholds.items():
        if metric in df.columns:
            mean_val = df[metric].mean()
            status = "CUMPLE" if mean_val >= threshold else "NO CUMPLE"
            print(f"  {metric:<25} {mean_val:.3f}  (umbral: {threshold})  [{status}]")

    print(f"\nEvaluación completada. Resultados en: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
