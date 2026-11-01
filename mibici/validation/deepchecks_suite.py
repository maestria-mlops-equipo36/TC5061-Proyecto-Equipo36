"""
Validación automatizada de integridad de datos y detección de Data Drift con DeepChecks.

Este módulo implementa suites de validación para los conjuntos de datos del proyecto
MiBici (P1 Regresión, P2 Clasificación y P3 Clustering), cubriendo:
1. Integridad del dataset: valores nulos, registros duplicados, columnas constantes,
   tipos de datos mezclados, variantes de cadenas y correlaciones entre variables.
2. Validación Train-Test y Drift: mezcla de muestras entre particiones, proporción de
   tamaños, discrepancia de categorías, deriva de características (Feature Drift),
   deriva de variable objetivo (Label Drift) y deriva multivariada (Multivariate Drift).
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
import types
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    FIGURES_DIR,
    P3_FEATURE_COLS_ESPACIAL,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


def aplicar_parches_compatibilidad_deepchecks() -> None:
    """
    Aplica parches de compatibilidad hacia adelante para ejecutar DeepChecks 0.19.x
    en entornos modernos (Python 3.12-3.14, NumPy >= 2.0, Pandas >= 2.1/3.0 y
    scikit-learn >= 1.6).
    """
    # 1. Alias eliminados en NumPy 2.0
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]
    if not hasattr(np, "infty"):
        np.infty = np.inf  # type: ignore[attr-defined]
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int_"):
        np.int_ = np.int64  # type: ignore[attr-defined]

    # 2. Alias DataFrame.applymap eliminado en Pandas >= 2.1 / 3.0
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map  # type: ignore[attr-defined]

    # 3. Shim de pkg_resources para entornos con setuptools >= 81
    if "pkg_resources" not in sys.modules:
        from packaging.version import parse as parse_version

        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.parse_version = parse_version  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = pkg_resources

    # 4. Compatibilidad de scorers y __sklearn_tags__ en scikit-learn >= 1.6
    import sklearn.base as _sk_base
    from sklearn.base import BaseEstimator
    from sklearn.metrics import _scorer, get_scorer
    import sklearn.utils._tags as _sk_tags

    if "max_error" not in _scorer._SCORERS and "neg_max_error" in _scorer._SCORERS:
        _scorer._SCORERS["max_error"] = get_scorer("neg_max_error")

    orig_get_tags = getattr(_sk_tags, "_orig_get_tags_mibici", _sk_tags.get_tags)

    def _compat_get_tags(estimator: Any) -> Any:
        try:
            return orig_get_tags(estimator)
        except AttributeError:
            tags = BaseEstimator().__sklearn_tags__()
            est_type = getattr(estimator, "_estimator_type", None)
            if est_type:
                tags.estimator_type = est_type
                if (
                    est_type == "classifier"
                    and getattr(tags, "classifier_tags", None) is None
                ):
                    from sklearn.utils._tags import ClassifierTags

                    tags.classifier_tags = ClassifierTags()
            return tags

    _sk_tags._orig_get_tags_mibici = orig_get_tags  # type: ignore[attr-defined]
    _sk_tags.get_tags = _compat_get_tags
    _sk_base.get_tags = _compat_get_tags


# Aplicar parches antes de importar deepchecks
aplicar_parches_compatibilidad_deepchecks()

from deepchecks.core import ConditionCategory, ConditionResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    DataDuplicates,
    DatasetsSizeComparison,
    FeatureDrift,
    FeatureFeatureCorrelation,
    FeatureLabelCorrelation,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    MultivariateDrift,
    NewCategoryTrainTest,
    PercentOfNulls,
    StringMismatch,
    TrainTestSamplesMix,
)


def _parchar_feature_feature_correlation_pandas3() -> None:
    """
    Corrige la condición de FeatureFeatureCorrelation para Pandas >= 3.0, donde
    `DataFrame.stack()` conserva valores NaN por defecto a menos que se invoque `.dropna()`.
    """

    def add_condition_max_number_of_pairs_above_threshold(
        self: Any, threshold: float = 0.9, n_pairs: int = 0
    ) -> Any:
        def condition(result: pd.DataFrame) -> ConditionResult:
            stacked = result[result.abs() > threshold].stack().dropna()
            results_ge = stacked.index.to_list()
            high_corr_pairs = [(i, j) for (i, j) in results_ge if str(i) < str(j)]

            if len(high_corr_pairs) > n_pairs:
                return ConditionResult(
                    ConditionCategory.FAIL,
                    f"Correlation is greater than {threshold} for pairs {high_corr_pairs}",
                )
            return ConditionResult(
                ConditionCategory.PASS,
                f"All correlations are less than {threshold} except pairs {high_corr_pairs}",
            )

        return self.add_condition(
            f"Not more than {n_pairs} pairs are correlated above {threshold}",
            condition,
        )

    FeatureFeatureCorrelation.add_condition_max_number_of_pairs_above_threshold = (
        add_condition_max_number_of_pairs_above_threshold
    )


_parchar_feature_feature_correlation_pandas3()

app = typer.Typer(
    help="Ejecución de suites de calidad de datos y Data Drift con DeepChecks."
)


def construir_dataset_deepchecks(
    df: pd.DataFrame,
    label: str | None = None,
    cat_features: Sequence[str] | None = None,
    datetime_name: str | None = None,
    index_name: str | None = None,
    features: Sequence[str] | None = None,
    label_type: str | None = None,
) -> Dataset:
    """
    Construye un objeto Dataset de DeepChecks a partir de un DataFrame de pandas.

    Parámetros
    ----------
    df : pd.DataFrame
        Tabla de datos a validar.
    label : Optional[str]
        Nombre de la columna objetivo (si aplica).
    cat_features : Optional[Sequence[str]]
        Lista de nombres de columnas categóricas.
    datetime_name : Optional[str]
        Nombre de la columna de fecha/tiempo (si existe y está en el DataFrame).
    index_name : Optional[str]
        Nombre de la columna índice (si aplica).
    features : Optional[Sequence[str]]
        Subconjunto explícito de columnas de características.
    label_type : Optional[str]
        Tipo de etiqueta ('regression', 'classification_label', etc.).

    Retorna
    -------
    Dataset
        Instancia configurada de `deepchecks.tabular.Dataset`.
    """
    df_clean = df.copy()
    cat_cols = [c for c in (cat_features or []) if c in df_clean.columns]
    dt_col = (
        datetime_name
        if (datetime_name and datetime_name in df_clean.columns)
        else None
    )

    kwargs: dict[str, Any] = {
        "df": df_clean,
        "cat_features": cat_cols,
    }
    if label and label in df_clean.columns:
        kwargs["label"] = label
    if dt_col:
        kwargs["datetime_name"] = dt_col
    if index_name and index_name in df_clean.columns:
        kwargs["index_name"] = index_name
    if features is not None:
        kwargs["features"] = [f for f in features if f in df_clean.columns]
    if label_type is not None:
        kwargs["label_type"] = label_type

    return Dataset(**kwargs)


def crear_suite_integridad_datos(
    nombre: str = "Suite de Integridad de Datos - MiBici",
    max_null_ratio: float = 0.0,
    max_duplicates_ratio: float = 0.0,
    max_correlated_features_threshold: float = 0.98,
    max_correlated_pairs: int = 2,
    incluir_correlacion_label: bool = True,
) -> Suite:
    """
    Crea una suite de DeepChecks para validar la calidad e integridad de un dataset.

    Chequeos incluidos:
    - PercentOfNulls: verifica ausencia (o umbral máximo) de valores faltantes.
    - DataDuplicates: verifica ausencia (o umbral máximo) de filas duplicadas.
    - IsSingleValue: detecta columnas constantes sin variabilidad informativa.
    - MixedNulls: detecta representaciones heterogéneas de nulos.
    - MixedDataTypes: detecta mezcla de tipos de datos en una misma columna.
    - StringMismatch: detecta variaciones de formato/mayúsculas en columnas de texto.
    - FeatureFeatureCorrelation: evalúa colinealidad extrema entre variables.
    - FeatureLabelCorrelation: evalúa posible fuga de datos entre features y etiqueta.
    """
    checks = [
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(max_null_ratio),
        DataDuplicates().add_condition_ratio_less_or_equal(max_duplicates_ratio),
        IsSingleValue().add_condition_not_single_value(),
        MixedNulls().add_condition_different_nulls_less_equal_to(0),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range((0.01, 0.5)),
        StringMismatch().add_condition_no_variants(),
        FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
            threshold=max_correlated_features_threshold,
            n_pairs=max_correlated_pairs,
        ),
    ]
    if incluir_correlacion_label:
        checks.append(
            FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.95)
        )

    return Suite(nombre, *checks)


def crear_suite_train_test_drift(
    nombre: str = "Suite de Validación Train-Test y Data Drift - MiBici",
    max_samples_mix_ratio: float = 0.05,
    max_feature_drift_score: float = 0.35,
    allowed_num_features_exceeding_drift: int = 0,
    max_label_drift_score: float = 0.35,
    max_multivariate_drift_score: float = 0.40,
    incluir_multivariate_drift: bool = True,
) -> Suite:
    """
    Crea una suite de DeepChecks para validar la partición Train vs Test y detectar Data Drift.

    Chequeos incluidos:
    - TrainTestSamplesMix: detecta filtración de registros idénticos entre train y test.
    - DatasetsSizeComparison: verifica que el conjunto de prueba no sea mayor al de entrenamiento.
    - NewCategoryTrainTest: detecta categorías nuevas o ausentes entre train y test.
    - FeatureDrift: cuantifica la deriva de distribución univariada por característica.
    - LabelDrift: cuantifica la deriva de distribución en la variable objetivo.
    - MultivariateDrift: entrena un clasificador de dominio para medir drift multivariado.
    """
    checks = [
        TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(
            max_samples_mix_ratio
        ),
        DatasetsSizeComparison().add_condition_train_dataset_greater_or_equal_test(),
        NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0.05),
        FeatureDrift().add_condition_drift_score_less_than(
            max_allowed_categorical_score=max_feature_drift_score,
            max_allowed_numeric_score=max_feature_drift_score,
            allowed_num_features_exceeding_threshold=allowed_num_features_exceeding_drift,
        ),
        LabelDrift().add_condition_drift_score_less_than(
            max_allowed_drift_score=max_label_drift_score,
        ),
    ]
    if incluir_multivariate_drift:
        checks.append(
            MultivariateDrift().add_condition_overall_drift_value_less_than(
                max_multivariate_drift_score
            )
        )

    return Suite(nombre, *checks)


def _obtener_nombre_check(item: Any) -> str:
    """Obtiene un nombre legible para un resultado de chequeo de DeepChecks."""
    header = getattr(item, "header", None)
    if header:
        return str(header)
    chk = getattr(item, "check", None)
    if chk is not None and hasattr(chk, "name"):
        try:
            return str(chk.name())
        except (AttributeError, TypeError, ValueError):
            return type(chk).__name__
    return type(item).__name__


def resumir_resultado_suite(suite_result: Any) -> dict[str, Any]:
    """
    Extrae un diccionario serializable a JSON con el resumen de resultados de una Suite.
    """
    detalles_checks: list[dict[str, Any]] = []
    total_condiciones_pasadas = 0
    total_condiciones_falladas = 0

    for item in suite_result.results:
        nombre_chk = _obtener_nombre_check(item)
        if hasattr(item, "passed_conditions"):
            paso = bool(item.passed_conditions())
            condiciones = []
            for cond_res in getattr(item, "conditions_results", []):
                condiciones.append(
                    {
                        "nombre": getattr(cond_res, "name", ""),
                        "categoria": str(getattr(cond_res, "category", "")),
                        "detalles": str(getattr(cond_res, "details", "")),
                        "paso": bool(cond_res.is_pass()),
                    }
                )
            if paso:
                total_condiciones_pasadas += 1
            else:
                total_condiciones_falladas += 1

            detalles_checks.append(
                {
                    "check": nombre_chk,
                    "ejecutado": True,
                    "paso_condiciones": paso,
                    "condiciones": condiciones,
                }
            )
        else:
            detalles_checks.append(
                {
                    "check": nombre_chk,
                    "ejecutado": False,
                    "paso_condiciones": True,
                    "nota": str(item),
                }
            )

    return {
        "suite_nombre": getattr(suite_result, "name", "DeepChecks Suite"),
        "passed": bool(suite_result.passed()),
        "checks_evaluados": len(detalles_checks),
        "checks_aprobados": total_condiciones_pasadas,
        "checks_con_alertas": total_condiciones_falladas,
        "resultados": detalles_checks,
    }


def ejecutar_validacion_completa_p1(
    df_features_p1: pd.DataFrame,
    fecha_corte_train: str = "2026-07-01",
    fecha_corte_test: str = "2026-09-01",
    max_muestras_por_particion: int = 5000,
    html_output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Ejecuta las suites de Integridad y de Train-Test Drift sobre la tabla de características
    de la Pregunta 1 (Demanda diaria por estación - Regresión).
    """
    feature_cols = [
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
    label_col = "viajes_iniciados"
    # Nota: 'mes' es ordinal/temporal (y en el corte temporal Oct 2025-Jun 2026 vs Sep 2026,
    # el mes 9 solo existe en test). Las categóricas invariantes entre particiones son:
    cat_cols = ["estacion_id", "dia_semana", "es_fin_de_semana"]

    df_clean = df_features_p1.dropna(subset=feature_cols + [label_col]).copy()
    df_clean["fecha"] = pd.to_datetime(df_clean["fecha"])

    df_train = df_clean[df_clean["fecha"] < pd.Timestamp(fecha_corte_train)].copy()
    df_test = df_clean[df_clean["fecha"] >= pd.Timestamp(fecha_corte_test)].copy()

    if len(df_train) > max_muestras_por_particion:
        df_train = df_train.sample(n=max_muestras_por_particion, random_state=42)
    if len(df_test) > max_muestras_por_particion:
        df_test = df_test.sample(n=max_muestras_por_particion, random_state=42)

    cols_eval = feature_cols + [label_col]
    ds_train = construir_dataset_deepchecks(
        df_train[cols_eval],
        label=label_col,
        cat_features=cat_cols,
        label_type="regression",
    )
    ds_test = construir_dataset_deepchecks(
        df_test[cols_eval],
        label=label_col,
        cat_features=cat_cols,
        label_type="regression",
    )

    # En P1, promedios móviles de 7, 14 y 28 días tienen alta colinealidad esperada (>0.98)
    suite_integridad = crear_suite_integridad_datos(
        nombre="P1 Regresión - Integridad de Datos",
        max_null_ratio=0.0,
        max_duplicates_ratio=0.02,
        max_correlated_features_threshold=0.98,
        max_correlated_pairs=6,
    )
    res_integridad = suite_integridad.run(ds_train)

    # 'mes' presenta deriva temporal esperada por el corte cronológico (Sep=9 frente a Oct-Jun)
    suite_drift = crear_suite_train_test_drift(
        nombre="P1 Regresión - Validación Train vs Test y Data Drift",
        max_samples_mix_ratio=0.05,
        max_feature_drift_score=0.35,
        allowed_num_features_exceeding_drift=1,
        max_label_drift_score=0.35,
        incluir_multivariate_drift=False,
    )
    res_drift = suite_drift.run(train_dataset=ds_train, test_dataset=ds_test)

    if html_output_path is not None:
        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        suite_combinada = Suite(
            "Reporte DeepChecks P1: Integridad y Drift (Train vs Test)",
            suite_integridad,
            suite_drift,
        )
        res_combinado = suite_combinada.run(
            train_dataset=ds_train, test_dataset=ds_test
        )
        if html_output_path.exists():
            html_output_path.unlink()
        res_combinado.save_as_html(str(html_output_path), as_widget=False)
        logger.success(f"Reporte HTML P1 exportado en: {html_output_path}")

    return {
        "integridad": resumir_resultado_suite(res_integridad),
        "drift": resumir_resultado_suite(res_drift),
        "res_integridad": res_integridad,
        "res_drift": res_drift,
    }


def ejecutar_validacion_completa_p2(
    df_features_p2: pd.DataFrame,
    max_muestras_por_particion: int = 5000,
    html_output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Ejecuta las suites de Integridad y de Train-Test Drift sobre la tabla de características
    de la Pregunta 2 (Clasificación de riesgo operativo en ventanas de 2h).
    """
    feature_cols = [
        "capacidad",
        "hora",
        "dia_semana",
        "es_fin_de_semana",
        "lag_salidas",
        "lag_llegadas",
        "lag_flujo_neto",
    ]
    label_col = "en_riesgo"
    cat_cols = ["hora", "dia_semana", "es_fin_de_semana"]

    df_clean = df_features_p2.dropna(subset=feature_cols + [label_col]).copy()
    if "ventana" in df_clean.columns:
        df_clean = df_clean.sort_values("ventana").reset_index(drop=True)

    n = len(df_clean)
    corte_train = int(n * 0.70)
    corte_val = int(n * 0.85)

    df_train = df_clean.iloc[:corte_train].copy()
    df_test = df_clean.iloc[corte_val:].copy()

    # Eliminar duplicados exactos en el subconjunto de columnas predictoras para evaluar integridad única
    cols_eval = feature_cols + [label_col]
    df_train_unique = df_train[cols_eval].drop_duplicates().copy()
    df_test_unique = df_test[cols_eval].drop_duplicates().copy()

    if len(df_train_unique) > max_muestras_por_particion:
        df_train_unique = df_train_unique.sample(
            n=max_muestras_por_particion, random_state=42
        )
    if len(df_test_unique) > max_muestras_por_particion:
        df_test_unique = df_test_unique.sample(
            n=max_muestras_por_particion, random_state=42
        )

    ds_train = construir_dataset_deepchecks(
        df_train_unique,
        label=label_col,
        cat_features=cat_cols,
        label_type="binary",
    )
    ds_test = construir_dataset_deepchecks(
        df_test_unique,
        label=label_col,
        cat_features=cat_cols,
        label_type="binary",
    )

    suite_integridad = crear_suite_integridad_datos(
        nombre="P2 Clasificación - Integridad de Datos",
        max_null_ratio=0.0,
        max_duplicates_ratio=0.0,
        max_correlated_features_threshold=0.98,
        max_correlated_pairs=2,
    )
    res_integridad = suite_integridad.run(ds_train)

    suite_drift = crear_suite_train_test_drift(
        nombre="P2 Clasificación - Validación Train vs Test y Data Drift",
        max_samples_mix_ratio=0.15,
        max_feature_drift_score=0.35,
        allowed_num_features_exceeding_drift=0,
        max_label_drift_score=0.35,
        incluir_multivariate_drift=False,
    )
    res_drift = suite_drift.run(train_dataset=ds_train, test_dataset=ds_test)

    if html_output_path is not None:
        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        suite_combinada = Suite(
            "Reporte DeepChecks P2: Integridad y Drift (Train vs Test)",
            suite_integridad,
            suite_drift,
        )
        res_combinado = suite_combinada.run(
            train_dataset=ds_train, test_dataset=ds_test
        )
        if html_output_path.exists():
            html_output_path.unlink()
        res_combinado.save_as_html(str(html_output_path), as_widget=False)
        logger.success(f"Reporte HTML P2 exportado en: {html_output_path}")

    return {
        "integridad": resumir_resultado_suite(res_integridad),
        "drift": resumir_resultado_suite(res_drift),
        "res_integridad": res_integridad,
        "res_drift": res_drift,
    }


def ejecutar_validacion_completa_p3(
    df_features_p3: pd.DataFrame,
    usar_solo_features_modelo: bool = True,
    html_output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Ejecuta la suite de Integridad de Datos sobre la tabla de perfiles de estaciones
    de la Pregunta 3 (Clustering).
    """
    if usar_solo_features_modelo:
        feature_cols = list(P3_FEATURE_COLS_ESPACIAL) + [
            "capacidad",
            "duracion_promedio_salidas_min",
        ]
    else:
        feature_cols = [
            "latitud",
            "longitud",
            "capacidad",
            "salidas_totales",
            "llegadas_totales",
            "total_viajes",
            "ratio_flujo_neto",
            "pct_salidas_pico_matutino",
            "pct_salidas_pico_vespertino",
            "pct_salidas_fin_de_semana",
            "duracion_promedio_salidas_min",
            "viajes_diarios_por_anclaje",
        ]
    cols_presentes = [c for c in feature_cols if c in df_features_p3.columns]

    ds_p3 = construir_dataset_deepchecks(
        df_features_p3[cols_presentes],
        cat_features=[],
    )

    suite_p3 = crear_suite_integridad_datos(
        nombre="P3 Clustering - Integridad de Perfiles de Estaciones",
        max_null_ratio=0.0,
        max_duplicates_ratio=0.0,
        max_correlated_features_threshold=0.98,
        max_correlated_pairs=0,
        incluir_correlacion_label=False,
    )
    res_p3 = suite_p3.run(ds_p3)

    if html_output_path is not None:
        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        if html_output_path.exists():
            html_output_path.unlink()
        res_p3.save_as_html(str(html_output_path), as_widget=False)
        logger.success(f"Reporte HTML P3 exportado en: {html_output_path}")

    return {
        "integridad": resumir_resultado_suite(res_p3),
        "res_integridad": res_p3,
    }


def graficar_resumen_deepchecks(
    resumen_global: dict[str, Any],
    res_drift_p1: Any | None,
    output_path: Path,
) -> Path:
    """
    Genera una figura resumen de la ejecución de DeepChecks (estado de chequeos de
    integridad/drift por pregunta de negocio y puntajes de deriva de características en P1).
    """
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Panel A: Resumen de checks aprobados vs evaluados por suite
    suites_labels: list[str] = []
    aprobados: list[int] = []
    alertas: list[int] = []

    if "P1_regresion" in resumen_global:
        s_int = resumen_global["P1_regresion"]["integridad"]
        s_drf = resumen_global["P1_regresion"]["drift"]
        suites_labels.extend(["P1 Integridad\n(Regresión)", "P1 Train-Test Drift\n(Regresión)"])
        aprobados.extend([s_int["checks_aprobados"], s_drf["checks_aprobados"]])
        alertas.extend([s_int["checks_con_alertas"], s_drf["checks_con_alertas"]])

    if "P2_clasificacion" in resumen_global:
        s_int2 = resumen_global["P2_clasificacion"]["integridad"]
        s_drf2 = resumen_global["P2_clasificacion"]["drift"]
        suites_labels.extend(["P2 Integridad\n(Clasificación)", "P2 Train-Test Drift\n(Clasificación)"])
        aprobados.extend([s_int2["checks_aprobados"], s_drf2["checks_aprobados"]])
        alertas.extend([s_int2["checks_con_alertas"], s_drf2["checks_con_alertas"]])

    if "P3_clustering" in resumen_global:
        s_int3 = resumen_global["P3_clustering"]
        suites_labels.append("P3 Integridad\n(Clustering)")
        aprobados.append(s_int3["checks_aprobados"])
        alertas.append(s_int3["checks_con_alertas"])

    ax1 = axes[0]
    x = np.arange(len(suites_labels))
    bars_ok = ax1.bar(x, aprobados, width=0.55, color="#2a9d8f", label="Chequeos Aprobados (PASS)")
    ax1.bar(x, alertas, width=0.55, bottom=aprobados, color="#e76f51", label="Alertas / Fallos")
    ax1.set_xticks(x)
    ax1.set_xticklabels(suites_labels, fontsize=9)
    ax1.set_ylabel("Número de Chequeos Evaluados")
    ax1.set_title("Resumen de Suites DeepChecks (Integridad y Drift)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(aprobados or [8]) + 2)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)
    for bar, val in zip(bars_ok, aprobados):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val / 2.0,
            f"{val}/{val} PASS",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8.5,
        )

    # Panel B: Puntajes de Feature Drift por variable en P1
    ax2 = axes[1]
    drift_scores: dict[str, float] = {}
    if res_drift_p1 is not None:
        for item in res_drift_p1.results:
            if _obtener_nombre_check(item) == "Feature Drift" and hasattr(item, "value"):
                val_dict = item.value
                if isinstance(val_dict, dict):
                    for feat, info in val_dict.items():
                        if isinstance(info, dict) and "Drift score" in info:
                            drift_scores[str(feat)] = float(info["Drift score"])

    if drift_scores:
        sorted_items = sorted(drift_scores.items(), key=lambda kv: kv[1])
        feats = [k for k, _ in sorted_items]
        scores = [v for _, v in sorted_items]
        colors = ["#e9c46a" if s > 0.35 else "#264653" for s in scores]
        y_pos = np.arange(len(feats))
        ax2.barh(y_pos, scores, color=colors, height=0.65)
        ax2.axvline(0.35, color="#d62828", linestyle="--", linewidth=1.3, label="Umbral Drift (0.35)")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feats, fontsize=8.5)
        ax2.set_xlabel("Puntaje de Drift (PSI / Cramer's V / KS)")
        ax2.set_title("P1: Deriva por Característica (Train vs Test)", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.4)
        ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Gráfica resumen de DeepChecks guardada en: {output_path}")
    return output_path


@app.command()
def main(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes_limpios.parquet",
    estaciones_path: Path = PROCESSED_DATA_DIR / "estaciones.parquet",
    p3_features_path: Path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet",
    output_dir: Path = REPORTS_DIR,
) -> None:
    """
    Ejecuta las validaciones de DeepChecks sobre los datos procesados del proyecto y
    exporta los reportes HTML, figura de evidencia y resumen en JSON dentro de `reports/`.
    """
    from mibici.features import (
        calcular_demanda_diaria_p1,
        calcular_features_riesgo_p2,
        generar_caracteristicas_regresion_p1,
        preparar_demanda_diaria_p1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    resumen_global: dict[str, Any] = {}
    res_drift_p1_obj = None

    if p3_features_path.exists():
        logger.info(f"Cargando perfiles de estaciones P3 desde: {p3_features_path}")
        df_p3 = pd.read_parquet(p3_features_path)
        html_p3 = output_dir / "deepchecks_p3_integridad.html"
        out_p3 = ejecutar_validacion_completa_p3(df_p3, html_output_path=html_p3)
        resumen_global["P3_clustering"] = out_p3["integridad"]
        logger.info(
            f"Resultado P3 Integridad: passed={out_p3['integridad']['passed']}"
        )
    else:
        logger.warning(f"No se encontró {p3_features_path}; omitiendo reporte P3.")

    if viajes_path.exists():
        logger.info(f"Cargando viajes limpios para validación P1 desde: {viajes_path}")
        df_viajes_p1 = pd.read_parquet(
            viajes_path, columns=["Inicio_del_viaje", "Origen_Id"]
        )
        df_demanda = calcular_demanda_diaria_p1(df_viajes_p1)
        df_prep = preparar_demanda_diaria_p1(df_demanda)
        df_feat_p1 = generar_caracteristicas_regresion_p1(df_prep)

        html_p1 = output_dir / "deepchecks_p1_integridad_drift.html"
        out_p1 = ejecutar_validacion_completa_p1(df_feat_p1, html_output_path=html_p1)
        res_drift_p1_obj = out_p1["res_drift"]
        resumen_global["P1_regresion"] = {
            "integridad": out_p1["integridad"],
            "drift": out_p1["drift"],
        }
        logger.info(
            f"Resultado P1 Integridad: passed={out_p1['integridad']['passed']}, "
            f"Drift: passed={out_p1['drift']['passed']}"
        )

        if estaciones_path.exists():
            logger.info("Ejecutando validación DeepChecks para P2 (Clasificación)...")
            df_viajes_p2 = pd.read_parquet(
                viajes_path,
                columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"],
            )
            df_estaciones = pd.read_parquet(estaciones_path)
            df_feat_p2 = calcular_features_riesgo_p2(df_viajes_p2, df_estaciones, freq="2h")
            html_p2 = output_dir / "deepchecks_p2_integridad_drift.html"
            out_p2 = ejecutar_validacion_completa_p2(df_feat_p2, html_output_path=html_p2)
            resumen_global["P2_clasificacion"] = {
                "integridad": out_p2["integridad"],
                "drift": out_p2["drift"],
            }
            logger.info(
                f"Resultado P2 Integridad: passed={out_p2['integridad']['passed']}, "
                f"Drift: passed={out_p2['drift']['passed']}"
            )
    else:
        logger.warning(f"No se encontró {viajes_path}; omitiendo reportes P1 y P2.")

    summary_json_path = output_dir / "deepchecks_summary.json"
    summary_json_path.write_text(
        json.dumps(resumen_global, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success(f"Resumen JSON de DeepChecks guardado en: {summary_json_path}")

    if resumen_global:
        fig_path = FIGURES_DIR / "deepchecks_resumen_validacion.png"
        graficar_resumen_deepchecks(resumen_global, res_drift_p1_obj, fig_path)


if __name__ == "__main__":
    app()
