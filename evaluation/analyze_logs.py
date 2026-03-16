"""
Análisis operacional de TEIA desde los logs de queries.jsonl.

Calcula métricas operacionales (latencia, scores de recuperación, cobertura
por módulo) y genera los gráficos para la tesis sin necesidad de un golden
dataset ni de un LLM juez.

Uso:
    # Desde la raíz del proyecto:
    python evaluation/analyze_logs.py

Salidas:
    docs/latency_distribution.png     — Histograma de latencias
    docs/scores_by_module.png         — Box plot de scores por módulo
    docs/confidence_distribution.png  — Histograma de confianza
    docs/queries_per_module.png       — Consultas por módulo (bar chart)
    docs/operational_summary.csv      — Tabla resumen de métricas operacionales
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla (compatible con servidores)

ROOT      = Path(__file__).parent.parent
LOGS_PATH = ROOT / "data" / "logs" / "queries.jsonl"
DOCS_PATH = ROOT / "docs"

MODULES_ES = {
    "caracterizacion":         "Caracterización",
    "factores_situacionales":  "Factores\nSituacionales",
    "resultados_aprendizaje":  "Resultados de\nAprendizaje",
    "actividades_aprendizaje": "Actividades de\nAprendizaje",
    "evaluacion":              "Evaluación",
    "secuencia":               "Secuencia\nDidáctica",
}


def load_logs() -> pd.DataFrame:
    if not LOGS_PATH.exists():
        print(f"[ERROR] No se encontró el archivo de logs en {LOGS_PATH}")
        print("  Asegúrate de haber ejecutado el sistema con algunas consultas primero.")
        sys.exit(1)

    records = []
    with open(LOGS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[ERROR] El archivo de logs está vacío.")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  → {len(df)} interacciones cargadas desde {LOGS_PATH}")
    return df


def plot_latency_distribution(df: pd.DataFrame):
    """Histograma de distribución de latencias de respuesta."""
    if "processing_time_ms" not in df.columns:
        print("  [OMITIDO] No hay columna 'processing_time_ms' en los logs.")
        return

    df["processing_time_s"] = df["processing_time_ms"] / 1000

    fig, ax = plt.subplots(figsize=(8, 5))
    df["processing_time_s"].hist(bins=20, ax=ax, color="#4472C4", edgecolor="white")

    mean_s = df["processing_time_s"].mean()
    ax.axvline(mean_s, color="red", linestyle="--", label=f"Media: {mean_s:.1f}s")

    ax.set_title("Distribución de latencias de respuesta — TEIA", fontsize=13)
    ax.set_xlabel("Tiempo de respuesta (segundos)", fontsize=11)
    ax.set_ylabel("Número de consultas", fontsize=11)
    ax.legend()
    plt.tight_layout()

    out = DOCS_PATH / "latency_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {out}")


def plot_scores_by_module(df: pd.DataFrame):
    """Box plot de scores de similitud coseno por módulo."""
    if "chunk_scores" not in df.columns:
        print("  [OMITIDO] No hay columna 'chunk_scores' en los logs.")
        return

    # Expandir lista de scores a filas individuales
    df_exp = df[["module", "chunk_scores"]].copy()
    df_exp = df_exp.explode("chunk_scores")
    df_exp["chunk_scores"] = pd.to_numeric(df_exp["chunk_scores"], errors="coerce")
    df_exp = df_exp.dropna(subset=["chunk_scores"])

    if df_exp.empty:
        print("  [OMITIDO] Los chunk_scores están vacíos.")
        return

    # Renombrar módulos para el gráfico
    df_exp["module_label"] = df_exp["module"].map(MODULES_ES).fillna(df_exp["module"])

    fig, ax = plt.subplots(figsize=(10, 6))
    df_exp.boxplot(
        column="chunk_scores",
        by="module_label",
        ax=ax,
        boxprops=dict(color="#4472C4"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="#4472C4"),
        capprops=dict(color="#4472C4"),
    )

    ax.set_title("Distribución de scores de recuperación por módulo", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Score de similitud coseno", fontsize=11)
    plt.suptitle("")  # Eliminar el título automático de boxplot
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = DOCS_PATH / "scores_by_module.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {out}")


def plot_confidence_distribution(df: pd.DataFrame):
    """Histograma de distribución de confianza por consulta."""
    if "confidence" not in df.columns:
        print("  [OMITIDO] No hay columna 'confidence' en los logs.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    df["confidence"].hist(bins=15, ax=ax, color="#70AD47", edgecolor="white")

    mean_conf = df["confidence"].mean()
    ax.axvline(mean_conf, color="red", linestyle="--", label=f"Media: {mean_conf:.2f}")

    ax.set_title("Distribución de confianza de recuperación — TEIA", fontsize=13)
    ax.set_xlabel("Score de confianza", fontsize=11)
    ax.set_ylabel("Número de consultas", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend()
    plt.tight_layout()

    out = DOCS_PATH / "confidence_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {out}")


def plot_queries_per_module(df: pd.DataFrame):
    """Bar chart de número de consultas por módulo."""
    if "module" not in df.columns:
        print("  [OMITIDO] No hay columna 'module' en los logs.")
        return

    counts = df["module"].value_counts()
    labels = [MODULES_ES.get(m, m) for m in counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts.values, color="#4472C4", edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom", fontsize=10
        )

    ax.set_title("Distribución de consultas por módulo del curso", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Número de consultas", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = DOCS_PATH / "queries_per_module.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → Guardado: {out}")


def print_operational_summary(df: pd.DataFrame):
    """Calcula y muestra la tabla de métricas operacionales."""
    print("\n" + "="*60)
    print("RESUMEN OPERACIONAL — TEIA")
    print("="*60)

    print(f"\n  Total de consultas registradas : {len(df)}")

    if "processing_time_ms" in df.columns:
        lat = df["processing_time_ms"] / 1000
        print(f"  Latencia media                 : {lat.mean():.2f} s")
        print(f"  Latencia mediana               : {lat.median():.2f} s")
        print(f"  Latencia máxima                : {lat.max():.2f} s")
        print(f"  Latencia mínima                : {lat.min():.2f} s")
        print(f"  Percentil 95 (p95)             : {lat.quantile(0.95):.2f} s")

    if "confidence" in df.columns:
        print(f"\n  Confianza media                : {df['confidence'].mean():.3f}")
        print(f"  Confianza mediana              : {df['confidence'].median():.3f}")

    if "chunk_scores" in df.columns:
        all_scores = df["chunk_scores"].explode()
        all_scores = pd.to_numeric(all_scores, errors="coerce").dropna()
        if not all_scores.empty:
            print(f"\n  Score de recuperación medio    : {all_scores.mean():.3f}")
            print(f"  Score de recuperación mediano  : {all_scores.median():.3f}")
            print(f"  Score de recuperación mínimo   : {all_scores.min():.3f}")

    if "module" in df.columns:
        print("\n  Consultas por módulo:")
        for module, count in df["module"].value_counts().items():
            pct = count / len(df) * 100
            print(f"    {module:<30} {count:3d} ({pct:.1f}%)")

    # Guardar CSV resumen
    summary_data = {}
    if "processing_time_ms" in df.columns:
        lat = df["processing_time_ms"] / 1000
        summary_data.update({
            "total_consultas": len(df),
            "latencia_media_s": round(lat.mean(), 3),
            "latencia_mediana_s": round(lat.median(), 3),
            "latencia_p95_s": round(lat.quantile(0.95), 3),
            "latencia_max_s": round(lat.max(), 3),
        })
    if "confidence" in df.columns:
        summary_data.update({
            "confianza_media": round(df["confidence"].mean(), 3),
            "confianza_mediana": round(df["confidence"].median(), 3),
        })

    summary_df = pd.DataFrame([summary_data])
    out = DOCS_PATH / "operational_summary.csv"
    summary_df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n  → Resumen guardado en {out}")


def main():
    print(f"Análisis operacional de TEIA")
    print(f"Cargando logs desde {LOGS_PATH}...\n")

    df = load_logs()

    print("\nGenerando gráficos...\n")
    plot_latency_distribution(df)
    plot_scores_by_module(df)
    plot_confidence_distribution(df)
    plot_queries_per_module(df)

    print_operational_summary(df)
    print("\nAnálisis completado.")


if __name__ == "__main__":
    main()
