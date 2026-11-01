"""
Pruebas automatizadas de calidad de datos, integridad y detección de Data Drift con DeepChecks.

Cubre:
1. Chequeos de integridad sobre un conjunto de datos (nulos, duplicados, columnas constantes,
   tipos mezclados, colinealidad entre variables).
2. Chequeos de validación entre particiones Train y Test (fuga de muestras, categorías nuevas,
   Feature Drift, Label Drift y Multivariate Drift).
3. Integración con las tablas de características de P1 (Regresión), P2 (Clasificación)
   y P3 (Clustering), incluyendo exportación de reportes HTML interactivos.
"""

import numpy as np
import pandas as pd
import pytest

from mibici.config import PROCESSED_DATA_DIR
from mibici.validation.deepchecks_suite import (
    DataDuplicates,
    FeatureDrift,
    FeatureFeatureCorrelation,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    NewCategoryTrainTest,
    PercentOfNulls,
    TrainTestSamplesMix,
    construir_dataset_deepchecks,
    crear_suite_integridad_datos,
    crear_suite_train_test_drift,
    ejecutar_validacion_completa_p1,
    ejecutar_validacion_completa_p2,
    ejecutar_validacion_completa_p3,
    resumir_resultado_suite,
)


@pytest.fixture
def datasets_train_test_estables():
    """
    Genera conjuntos de entrenamiento y prueba sintéticos sin anomalías ni drift,
    representando observaciones de demanda por estación.
    """
    rng = np.random.RandomState(42)
    n_train = 240
    n_test = 120

    df_train = pd.DataFrame(
        {
            "estacion_id": rng.choice([10, 20, 30, 40], size=n_train),
            "dia_semana": rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_train),
            "es_fin_de_semana": rng.choice([0, 1], size=n_train, p=[0.7, 0.3]),
            "viajes_dia_anterior": rng.normal(loc=35.0, scale=8.0, size=n_train).round(3),
            "promedio_7_dias": rng.normal(loc=34.5, scale=5.0, size=n_train).round(3),
            "capacidad": rng.choice([15, 20, 25, 30], size=n_train),
            "viajes_iniciados": rng.normal(loc=36.0, scale=7.5, size=n_train).round(2),
        }
    )

    df_test = pd.DataFrame(
        {
            "estacion_id": rng.choice([10, 20, 30, 40], size=n_test),
            "dia_semana": rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_test),
            "es_fin_de_semana": rng.choice([0, 1], size=n_test, p=[0.7, 0.3]),
            "viajes_dia_anterior": rng.normal(loc=35.2, scale=8.1, size=n_test).round(3),
            "promedio_7_dias": rng.normal(loc=34.7, scale=5.1, size=n_test).round(3),
            "capacidad": rng.choice([15, 20, 25, 30], size=n_test),
            "viajes_iniciados": rng.normal(loc=36.1, scale=7.4, size=n_test).round(2),
        }
    )

    return df_train, df_test


# =====================================================================
# 1. PRUEBAS DE INTEGRIDAD DE DATOS (CASOS NORMALES Y CASOS LÍMITE)
# =====================================================================


def test_deepchecks_integridad_datos_limpios(datasets_train_test_estables):
    """Verifica que un dataset íntegro apruebe todas las condiciones de la suite de integridad."""
    df_train, _ = datasets_train_test_estables
    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    suite = crear_suite_integridad_datos(
        max_null_ratio=0.0,
        max_duplicates_ratio=0.0,
        max_correlated_features_threshold=0.95,
    )
    resultado = suite.run(ds_train)
    resumen = resumir_resultado_suite(resultado)

    assert resultado.passed() is True
    assert resumen["passed"] is True
    assert resumen["checks_con_alertas"] == 0


def test_deepchecks_detecta_valores_faltantes(datasets_train_test_estables):
    """Caso límite: verifica que DeepChecks detecte y repruebe la presencia de valores nulos."""
    df_train, _ = datasets_train_test_estables
    df_con_nulos = df_train.copy()
    df_con_nulos.loc[0:15, "viajes_dia_anterior"] = np.nan

    ds = construir_dataset_deepchecks(
        df_con_nulos,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.0)
    res = check.run(ds)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_filas_duplicadas(datasets_train_test_estables):
    """Caso límite: verifica que DeepChecks detecte registros duplicados por encima del umbral."""
    df_train, _ = datasets_train_test_estables
    df_duplicado = pd.concat([df_train, df_train.iloc[:30]], ignore_index=True)

    ds = construir_dataset_deepchecks(
        df_duplicado,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = DataDuplicates().add_condition_ratio_less_or_equal(0.0)
    res = check.run(ds)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_columna_constante(datasets_train_test_estables):
    """Caso límite: verifica que DeepChecks detecte columnas degeneradas con un único valor."""
    df_train, _ = datasets_train_test_estables
    df_constante = df_train.copy()
    df_constante["columna_constante"] = 1

    ds = construir_dataset_deepchecks(
        df_constante,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = IsSingleValue().add_condition_not_single_value()
    res = check.run(ds)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_tipos_mezclados(datasets_train_test_estables):
    """Caso límite: verifica que DeepChecks detecte mezcla de cadenas y números en una columna."""
    df_train, _ = datasets_train_test_estables
    df_mixto = df_train.copy()
    columna_mixta = df_mixto["capacidad"].astype(object).tolist()
    for idx in range(15):
        columna_mixta[idx] = "capacidad_invalida"
    df_mixto["capacidad"] = pd.Series(columna_mixta, dtype=object)

    ds = construir_dataset_deepchecks(
        df_mixto,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana", "capacidad"],
        label_type="regression",
    )

    check = MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.5))
    res = check.run(ds)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_colinealidad_extrema(datasets_train_test_estables):
    """Caso límite: verifica que FeatureFeatureCorrelation detecte variables redundantes."""
    df_train, _ = datasets_train_test_estables
    df_colineal = df_train.copy()
    df_colineal["copia_lineal_viajes"] = df_colineal["viajes_dia_anterior"] * 2.0 + 1.0

    ds = construir_dataset_deepchecks(
        df_colineal,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=0.95, n_pairs=0
    )
    res = check.run(ds)

    assert res.passed_conditions() is False


# =====================================================================
# 2. PRUEBAS DE VALIDACIÓN TRAIN-TEST Y DATA DRIFT
# =====================================================================


def test_deepchecks_train_test_sin_drift(datasets_train_test_estables):
    """Verifica que particiones consistentes aprueben la suite completa de Train-Test y Drift."""
    df_train, df_test = datasets_train_test_estables

    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    suite = crear_suite_train_test_drift(
        max_samples_mix_ratio=0.05,
        max_feature_drift_score=0.25,
        max_label_drift_score=0.25,
        incluir_multivariate_drift=True,
    )
    resultado = suite.run(train_dataset=ds_train, test_dataset=ds_test)
    resumen = resumir_resultado_suite(resultado)

    assert resultado.passed() is True
    assert resumen["passed"] is True


def test_deepchecks_detecta_feature_drift_severo(datasets_train_test_estables):
    """Caso límite: verifica que FeatureDrift detecte un cambio drástico en las variables de entrada."""
    df_train, df_test = datasets_train_test_estables
    df_test_drifted = df_test.copy()
    df_test_drifted["viajes_dia_anterior"] = df_test_drifted["viajes_dia_anterior"] + 150.0
    df_test_drifted["promedio_7_dias"] = df_test_drifted["promedio_7_dias"] * 5.0

    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test_drifted,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = FeatureDrift().add_condition_drift_score_less_than(
        max_allowed_categorical_score=0.2,
        max_allowed_numeric_score=0.2,
        allowed_num_features_exceeding_threshold=0,
    )
    res = check.run(ds_train, ds_test)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_label_drift_severo(datasets_train_test_estables):
    """Caso límite: verifica que LabelDrift detecte un desplazamiento en la variable objetivo."""
    df_train, df_test = datasets_train_test_estables
    df_test_label_drift = df_test.copy()
    df_test_label_drift["viajes_iniciados"] = df_test_label_drift["viajes_iniciados"] + 200.0

    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test_label_drift,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = LabelDrift().add_condition_drift_score_less_than(max_allowed_drift_score=0.15)
    res = check.run(ds_train, ds_test)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_fuga_muestras_train_test(datasets_train_test_estables):
    """Caso límite: verifica que TrainTestSamplesMix detecte filtración de datos de train en test."""
    df_train, df_test = datasets_train_test_estables
    df_test_contaminado = pd.concat([df_test, df_train.iloc[:40]], ignore_index=True)

    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test_contaminado,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0.0)
    res = check.run(ds_train, ds_test)

    assert res.passed_conditions() is False


def test_deepchecks_detecta_categorias_nuevas_en_test(datasets_train_test_estables):
    """Caso límite: verifica que NewCategoryTrainTest detecte estaciones no vistas en train."""
    df_train, df_test = datasets_train_test_estables
    df_test_nuevas = df_test.copy()
    df_test_nuevas.loc[:30, "estacion_id"] = 999

    ds_train = construir_dataset_deepchecks(
        df_train,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test_nuevas,
        label="viajes_iniciados",
        cat_features=["estacion_id", "dia_semana", "es_fin_de_semana"],
        label_type="regression",
    )

    check = NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0.0)
    res = check.run(ds_train, ds_test)

    assert res.passed_conditions() is False


# =====================================================================
# 3. PRUEBAS DE PIPELINES COMPLETOS (P1, P2, P3) Y EXPORTACIÓN HTML
# =====================================================================


def test_ejecutar_validacion_completa_p1_y_exportar_html(tmp_path):
    """Valida el flujo completo de DeepChecks para P1 (Regresión) y la exportación del reporte HTML."""
    rng = np.random.RandomState(42)
    fechas = pd.date_range("2026-04-01", "2026-09-25", freq="D")
    registros = []
    for est_id in [1, 2, 3]:
        base = est_id * 12.0
        for f in fechas:
            registros.append(
                {
                    "fecha": f,
                    "estacion_id": est_id,
                    "dia_semana": f.dayofweek,
                    "mes": f.month,
                    "es_fin_de_semana": int(f.dayofweek in [5, 6]),
                    "viajes_dia_anterior": round(base + rng.normal(0, 3), 3),
                    "viajes_hace_7_dias": round(base + rng.normal(0, 3), 3),
                    "promedio_7_dias": round(base + rng.normal(0, 2), 3),
                    "promedio_14_dias": round(base + rng.normal(0, 1.8), 3),
                    "viajes_hace_14_dias": round(base + rng.normal(0, 3), 3),
                    "viajes_hace_28_dias": round(base + rng.normal(0, 3), 3),
                    "promedio_28_dias": round(base + rng.normal(0, 1.5), 3),
                    "promedio_mismo_dia_4_semanas": round(base + rng.normal(0, 2.2), 3),
                    "viajes_iniciados": max(1, round(base + rng.normal(0, 3))),
                }
            )
    df_p1 = pd.DataFrame(registros)
    html_out = tmp_path / "reporte_p1.html"

    resultado = ejecutar_validacion_completa_p1(
        df_p1,
        fecha_corte_train="2026-07-01",
        fecha_corte_test="2026-09-01",
        html_output_path=html_out,
    )

    assert html_out.exists()
    assert html_out.stat().st_size > 1000
    assert resultado["integridad"]["passed"] is True
    assert resultado["drift"]["passed"] is True


def test_ejecutar_validacion_completa_p2_sintetico():
    """Valida el flujo completo de DeepChecks para P2 (Clasificación de riesgo)."""
    rng = np.random.RandomState(42)
    n = 300
    df_p2 = pd.DataFrame(
        {
            "ventana": pd.date_range("2026-01-01", periods=n, freq="2h"),
            "capacidad": rng.choice([15, 20, 25], size=n),
            "hora": rng.choice([6, 8, 10, 12, 14, 16, 18, 20], size=n),
            "dia_semana": rng.choice([0, 1, 2, 3, 4, 5, 6], size=n),
            "es_fin_de_semana": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
            "lag_salidas": rng.uniform(0, 15, size=n).round(3),
            "lag_llegadas": rng.uniform(0, 15, size=n).round(3),
            "lag_flujo_neto": rng.normal(0, 5, size=n).round(3),
            "en_riesgo": rng.choice([0, 1], size=n, p=[0.75, 0.25]),
        }
    )

    resultado = ejecutar_validacion_completa_p2(df_p2)

    assert resultado["integridad"]["passed"] is True
    assert resultado["drift"]["passed"] is True


def test_ejecutar_validacion_completa_p3_sintetico():
    """Valida el flujo completo de DeepChecks para P3 (Clustering de estaciones)."""
    rng = np.random.RandomState(42)
    n = 40
    df_p3 = pd.DataFrame(
        {
            "estacion_id": np.arange(1, n + 1),
            "nombre": [f"Estacion_{i}" for i in range(1, n + 1)],
            "latitud": rng.uniform(20.60, 20.75, size=n),
            "longitud": rng.uniform(-103.42, -103.30, size=n),
            "capacidad": rng.choice([12, 15, 18, 20, 24, 30], size=n),
            "ratio_flujo_neto": rng.uniform(-0.4, 0.4, size=n),
            "pct_salidas_pico_matutino": rng.uniform(0.1, 0.4, size=n),
            "pct_salidas_pico_vespertino": rng.uniform(0.1, 0.4, size=n),
            "pct_salidas_fin_de_semana": rng.uniform(0.05, 0.3, size=n),
            "duracion_promedio_salidas_min": rng.uniform(8.0, 18.0, size=n),
            "viajes_diarios_por_anclaje": rng.uniform(0.5, 6.0, size=n),
        }
    )

    resultado = ejecutar_validacion_completa_p3(df_p3)

    assert resultado["integridad"]["passed"] is True
    assert resultado["integridad"]["checks_con_alertas"] == 0


def test_deepchecks_datos_reales_p3_si_existen():
    """Ejecuta la suite de integridad de DeepChecks sobre los perfiles reales de P3 si existen."""
    p3_path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet"
    if not p3_path.exists():
        pytest.skip("Archivo features_p3_estaciones.parquet no generado localmente.")

    df_p3 = pd.read_parquet(p3_path)
    resultado = ejecutar_validacion_completa_p3(df_p3, usar_solo_features_modelo=True)

    assert resultado["integridad"]["passed"] is True
    assert resultado["integridad"]["checks_aprobados"] >= 7
