"""
Pruebas unitarias para el módulo de monitoreo y detección de drift (Evidently).
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from mibici.monitoring.drift import (
    FEATURE_COLS_P1,
    TARGET_COL_P1,
    ejecutar_reporte_drift,
    preparar_conjuntos_drift,
)


@pytest.fixture
def datos_sinteticos_drift():
    """Genera datos sintéticos pequeños para verificar la suite de Evidently."""
    n_samples = 100
    cols = FEATURE_COLS_P1 + [TARGET_COL_P1]

    # Datos de referencia
    ref_dict = {col: [10.0 + i * 0.1 for i in range(n_samples)] for col in cols}
    ref_dict["dia_semana"] = [i % 7 for i in range(n_samples)]
    ref_dict["mes"] = [5 for _ in range(n_samples)]
    ref_dict["es_fin_de_semana"] = [1 if (i % 7) in [5, 6] else 0 for i in range(n_samples)]
    ref_dict["estacion_id"] = [(i % 10) + 1 for i in range(n_samples)]
    df_ref = pd.DataFrame(ref_dict)

    # Datos actuales con drift inducido
    curr_dict = {col: [30.0 + i * 0.5 for i in range(n_samples)] for col in cols}
    curr_dict["dia_semana"] = [(i + 2) % 7 for i in range(n_samples)]
    curr_dict["mes"] = [9 for _ in range(n_samples)]
    curr_dict["es_fin_de_semana"] = [1 if ((i + 2) % 7) in [5, 6] else 0 for i in range(n_samples)]
    curr_dict["estacion_id"] = [(i % 10) + 1 for i in range(n_samples)]
    df_curr = pd.DataFrame(curr_dict)

    return df_ref, df_curr


def test_ejecutar_reporte_drift_estructura(datos_sinteticos_drift):
    """Verifica que la ejecución de Evidently genere todos los artefactos requeridos."""
    df_ref, df_curr = datos_sinteticos_drift

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_html = tmp_path / "drift_report.html"
        output_json = tmp_path / "drift_summary.json"
        output_csv = tmp_path / "drift_metrics.csv"
        figures_dir = tmp_path / "figures"

        resumen = ejecutar_reporte_drift(
            df_referencia=df_ref,
            df_actual=df_curr,
            output_html=output_html,
            output_json=output_json,
            output_csv=output_csv,
            output_figures_dir=figures_dir,
        )

        assert output_html.exists()
        assert output_html.stat().st_size > 1000

        assert output_json.exists()
        assert "share_drifted_columns" in resumen
        assert "drift_by_columns" in resumen
        assert len(resumen["drift_by_columns"]) == len(FEATURE_COLS_P1) + 1

        assert output_csv.exists()
        df_csv = pd.read_csv(output_csv)
        assert "column" in df_csv.columns
        assert "drift_score" in df_csv.columns
        assert "drift_detected" in df_csv.columns

        # Verificar figuras generadas
        assert (figures_dir / "evidently_drift_scores.png").exists()
        assert (figures_dir / "distribucion_variables_top_drift.png").exists()
        assert (figures_dir / "target_drift_demanda.png").exists()


def test_preparar_conjuntos_drift_columnas_y_fechas():
    """Verifica que la preparación de datos divida correctamente por fecha de corte."""
    parquet_path = Path("data/processed/regression_reference.parquet")
    if not parquet_path.exists():
        pytest.skip("No se han generado los datasets procesados.")

    df_ref = pd.read_parquet("data/processed/regression_reference.parquet")
    df_curr = pd.read_parquet("data/processed/regression_current.parquet")

    assert not df_ref.empty
    assert not df_curr.empty

    for col in FEATURE_COLS_P1 + [TARGET_COL_P1, "fecha"]:
        assert col in df_ref.columns
        assert col in df_curr.columns

    assert pd.to_datetime(df_ref["fecha"]).max() < pd.to_datetime("2026-09-01")
    assert pd.to_datetime(df_curr["fecha"]).min() >= pd.to_datetime("2026-09-01")
