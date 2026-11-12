"""
Pipeline de observabilidad y detección de Data Drift y Data Quality para MiBici.

Utiliza Evidently para comparar el conjunto de referencia (entrenamiento histórico)
frente al conjunto actual (producción / mes más reciente: Septiembre 2026).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
except ImportError:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

from mibici.config import PROCESSED_DATA_DIR, PROJ_ROOT
from mibici.features import (
    calcular_demanda_diaria_p1,
    preparar_demanda_diaria_p1,
    generar_caracteristicas_regresion_p1,
)

app = typer.Typer(help="Detección de Data Drift y Calidad de Datos con Evidently.")

FEATURE_COLS_P1 = [
    "estacion_id",
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "viajes_dia_anterior",
    "viajes_hace_7_dias",
    "promedio_7_dias",
    "promedio_14_dias",
    "viajes_hace_14_dias",
    "viajes_hace_28_dias",
    "promedio_28_dias",
    "promedio_mismo_dia_4_semanas",
]
TARGET_COL_P1 = "viajes_iniciados"


def preparar_conjuntos_drift(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes_limpios.parquet",
    cobertura_minima: float = 0.95,
    fecha_corte_actual: str = "2026-09-01",
    guardar_artefactos: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye las características para regresión (P1) y las divide en dos conjuntos:
    - Referencia (Entrenamiento histórico): Fechas previas a fecha_corte_actual (eliminando warmup inicial).
    - Actual (Producción / Mes reciente): Fechas >= fecha_corte_actual (Septiembre 2026).

    Retorna:
    --------
    (df_referencia, df_actual) con las variables del modelo + variable objetivo + fecha.
    """
    logger.info(f"Cargando viajes procesados desde {viajes_path}...")
    if not viajes_path.exists():
        raise FileNotFoundError(f"No existe el archivo de viajes: {viajes_path}")

    df_viajes = pd.read_parquet(
        viajes_path,
        columns=["Inicio_del_viaje", "Origen_Id"],
    )

    logger.info("Calculando demanda y generando características de regresión...")
    df_demanda = calcular_demanda_diaria_p1(df_viajes)
    df_preparado = preparar_demanda_diaria_p1(
        df_demanda,
        cobertura_minima=cobertura_minima,
    )
    df_features = generar_caracteristicas_regresion_p1(df_preparado)

    # Eliminar registros iniciales sin historial completo de 28 días
    df_features = df_features.dropna(subset=FEATURE_COLS_P1).copy()

    columnas_totales = ["fecha"] + FEATURE_COLS_P1 + [TARGET_COL_P1]
    df_features = df_features[columnas_totales].copy()

    df_referencia = df_features[df_features["fecha"] < fecha_corte_actual].copy()
    df_actual = df_features[df_features["fecha"] >= fecha_corte_actual].copy()

    logger.info(
        f"Conjuntos generados: Referencia={len(df_referencia):,} filas "
        f"({df_referencia['fecha'].min().strftime('%Y-%m-%d')} a {df_referencia['fecha'].max().strftime('%Y-%m-%d')}), "
        f"Actual={len(df_actual):,} filas "
        f"({df_actual['fecha'].min().strftime('%Y-%m-%d')} a {df_actual['fecha'].max().strftime('%Y-%m-%d')})."
    )

    if guardar_artefactos:
        ref_path = PROCESSED_DATA_DIR / "regression_reference.parquet"
        curr_path = PROCESSED_DATA_DIR / "regression_current.parquet"
        df_referencia.to_parquet(ref_path, index=False)
        df_actual.to_parquet(curr_path, index=False)
        logger.success(f"Artefactos guardados en:\n  - {ref_path}\n  - {curr_path}")

    return df_referencia, df_actual


def ejecutar_reporte_drift(
    df_referencia: pd.DataFrame,
    df_actual: pd.DataFrame,
    output_html: Path = PROJ_ROOT / "reports" / "evidently_drift_report.html",
    output_json: Path = PROJ_ROOT / "reports" / "drift_summary.json",
    output_csv: Path = PROJ_ROOT / "reports" / "drift_metrics.csv",
    output_figures_dir: Path = PROJ_ROOT / "reports" / "figures" / "drift",
) -> Dict[str, Any]:
    """
    Ejecuta el reporte de Evidently con DataDriftPreset y DataQualityPreset.
    Exporta el reporte interactivo en HTML, extrae métricas cuantitativas en JSON/CSV
    y genera gráficos estáticos de alta resolución para el reporte ejecutivo.
    """
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    cols_analisis = FEATURE_COLS_P1 + [TARGET_COL_P1]
    ref_eval = df_referencia[cols_analisis].copy()
    curr_eval = df_actual[cols_analisis].copy()

    logger.info("Ejecutando suite de Evidently (DataDriftPreset + DataQualityPreset)...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_eval, current_data=curr_eval)

    # 1. Guardar HTML interactivo
    report.save_html(str(output_html))
    logger.success(f"Reporte HTML guardado en {output_html}")

    # 2. Extraer métricas estructuradas del reporte
    report_dict = report.as_dict()
    resumen_drift: Dict[str, Any] = {
        "reference_rows": len(ref_eval),
        "current_rows": len(curr_eval),
        "total_columns_evaluated": len(cols_analisis),
        "dataset_drift_detected": False,
        "share_drifted_columns": 0.0,
        "drift_by_columns": {},
    }

    drift_rows = []
    for m in report_dict.get("metrics", []):
        res = m.get("result", {})
        if "dataset_drift" in res:
            resumen_drift["dataset_drift_detected"] = bool(res.get("dataset_drift", False))
            resumen_drift["share_drifted_columns"] = float(res.get("share_of_drifted_columns", 0.0))
        if "drift_by_columns" in res:
            for col, info in res["drift_by_columns"].items():
                drift_detected = bool(info.get("drift_detected", False))
                drift_score = float(info.get("drift_score", 0.0))
                stattest = str(info.get("stattest_name", ""))
                threshold = float(info.get("drift_threshold", 0.05))
                col_data = {
                    "column": col,
                    "drift_detected": drift_detected,
                    "drift_score": drift_score,
                    "threshold": threshold,
                    "stat_test": stattest,
                    "is_target": col == TARGET_COL_P1,
                }
                resumen_drift["drift_by_columns"][col] = col_data
                drift_rows.append(col_data)

    # Guardar JSON y CSV
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(resumen_drift, f, indent=2, ensure_ascii=False)
    logger.success(f"Resumen JSON guardado en {output_json}")

    df_drift_table = pd.DataFrame(drift_rows)
    df_drift_table.to_csv(output_csv, index=False)
    logger.success(f"Tabla de métricas guardada en {output_csv}")

    # 3. Generar visualizaciones estáticas para el PDF del entregable
    logger.info("Generando figuras estáticas para el reporte ejecutivo...")
    _generar_figuras_drift(ref_eval, curr_eval, df_drift_table, output_figures_dir)

    return resumen_drift


def _generar_figuras_drift(
    df_ref: pd.DataFrame,
    df_curr: pd.DataFrame,
    df_drift: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Genera gráficos estáticos para la documentación y PDF final."""
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Figura 1: Barras de Drift Score por columna
    plt.figure(figsize=(10, 6))
    df_plot = df_drift.sort_values(by="drift_score", ascending=True).copy()
    colores = ["#d9534f" if d else "#5cb85c" for d in df_plot["drift_detected"]]

    bars = plt.barh(df_plot["column"], df_plot["drift_score"], color=colores, edgecolor="black", alpha=0.85)
    plt.axvline(0.1, color="orange", linestyle="--", label="Umbral de Alerta Drift (0.1)")
    plt.title("Data Drift Score por Variable (Evidently - Distancia Wasserstein)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Drift Score (Mayor score = Mayor desviación)", fontsize=11)
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig1_path = output_dir / "evidently_drift_scores.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Guardada figura: {fig1_path}")

    # Figura 2: Comparativa de distribuciones de variables con mayor drift
    drifted_cols = df_drift[df_drift["drift_detected"]]["column"].tolist()
    if not drifted_cols:
        drifted_cols = ["viajes_dia_anterior", "promedio_7_dias", "viajes_hace_7_dias", "viajes_iniciados"]

    # Seleccionar top variables continuas con mayor drift score (excluyendo constantes como mes)
    cols_continuas = [c for c in df_drift["column"] if c != "mes"]
    top_cols = (
        df_drift[df_drift["column"].isin(cols_continuas)]
        .sort_values(by="drift_score", ascending=False)["column"]
        .head(4)
        .tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, col in enumerate(top_cols):
        ax = axes[idx]
        sns.kdeplot(
            df_ref[col],
            ax=ax,
            label=f"Referencia (Media={df_ref[col].mean():.1f})",
            color="#1f77b4",
            linewidth=2,
            fill=True,
            alpha=0.25,
            warn_singular=False,
        )
        sns.kdeplot(
            df_curr[col],
            ax=ax,
            label=f"Actual (Media={df_curr[col].mean():.1f})",
            color="#ff7f0e",
            linewidth=2,
            fill=True,
            alpha=0.25,
            warn_singular=False,
        )
        drift_val = df_drift.loc[df_drift["column"] == col, "drift_score"].iloc[0]
        ax.set_title(f"Distribución: {col} (Wasserstein={drift_val:.4f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Densidad", fontsize=10)
        ax.legend(loc="upper right")

    plt.suptitle("Comparativa de Distribuciones: Período de Entrenamiento vs. Mes Actual", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig2_path = output_dir / "distribucion_variables_top_drift.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Guardada figura: {fig2_path}")

    # Figura 3: Target Drift (viajes_iniciados)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_ref[TARGET_COL_P1], label=f"Referencia (Media={df_ref[TARGET_COL_P1].mean():.1f})", color="#2ca02c", linewidth=2.5, fill=True, alpha=0.3)
    sns.kdeplot(df_curr[TARGET_COL_P1], label=f"Actual (Media={df_curr[TARGET_COL_P1].mean():.1f})", color="#d62728", linewidth=2.5, fill=True, alpha=0.3)
    plt.title("Concept/Target Drift: Demanda Diaria (viajes_iniciados)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Número de Viajes Iniciados por Estación al Día", fontsize=11)
    plt.ylabel("Densidad", fontsize=11)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    fig3_path = output_dir / "target_drift_demanda.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Guardada figura: {fig3_path}")


@app.command()
def run(
    viajes_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "viajes_limpios.parquet",
        help="Ruta al parquet de viajes limpios.",
    ),
    cobertura_minima: float = typer.Option(
        0.95,
        help="Cobertura mínima requerida para estaciones.",
    ),
    fecha_corte_actual: str = typer.Option(
        "2026-09-01",
        help="Fecha inicial para el conjunto de producción / actual.",
    ),
):
    """Ejecuta el pipeline completo de preparación y generación del reporte de drift."""
    logger.info("=== Iniciando Pipeline de Drift y Calidad de Datos (Evidently) ===")
    df_ref, df_curr = preparar_conjuntos_drift(
        viajes_path=viajes_path,
        cobertura_minima=cobertura_minima,
        fecha_corte_actual=fecha_corte_actual,
        guardar_artefactos=True,
    )
    resumen = ejecutar_reporte_drift(df_ref, df_curr)
    logger.success(
        f"Pipeline finalizado. Dataset drift={resumen['dataset_drift_detected']}, "
        f"columnas con drift: {resumen['share_drifted_columns']:.1%}."
    )


if __name__ == "__main__":
    app()
