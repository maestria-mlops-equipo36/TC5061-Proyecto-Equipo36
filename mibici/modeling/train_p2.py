"""
Módulo de Entrenamiento y Evaluación para la Pregunta 2 (P2).

Problema de Negocio:
    Clasificación del riesgo operativo de desabasto o saturación en cicloestaciones 
    de MiBici Guadalajara para la planeación del equipo de rebalanceo.

Atiende observaciones de revisión:
1. Partición estrictamente temporal: Train (70%), Validación (15%), Prueba (15%).
2. Métricas de la clase positiva (en_riesgo = 1): Precision, Recall, F1, además de F1-macro y ROC-AUC.
3. Figuras en reports/figures/: Matriz de confusión y Curva ROC comparativa.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from mibici.config import (
    DATA_MANIFEST_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from mibici.features import calcular_features_riesgo_p2


def obtener_hash_datos() -> str:
    """Recupera el hash SHA-256 del dataset desde el manifiesto."""
    if DATA_MANIFEST_PATH.exists():
        with open(DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            return manifest.get("hash_consolidado_parquet", "sin_hash")[:8]
    return "sin_hash"


def ejecutar_experimento_p2():
    viajes_path = PROCESSED_DATA_DIR / "viajes_limpios.parquet"
    estaciones_path = PROCESSED_DATA_DIR / "estaciones.parquet"

    df_viajes = pd.read_parquet(viajes_path)
    df_estaciones = pd.read_parquet(estaciones_path)
    df_p2 = calcular_features_riesgo_p2(df_viajes, df_estaciones, freq="2h")

    # 1. Partición temporal en 3 bloques (Train: 70%, Val: 15%, Test: 15%)
    df_p2 = df_p2.sort_values("ventana").reset_index(drop=True)
    n = len(df_p2)
    corte_train = int(n * 0.70)
    corte_val = int(n * 0.85)

    train = df_p2.iloc[:corte_train]
    val = df_p2.iloc[corte_train:corte_val]
    test = df_p2.iloc[corte_val:]

    pred_columns = [
        "capacidad",
        "hora",
        "dia_semana",
        "es_fin_de_semana",
        "lag_salidas",
        "lag_llegadas",
        "lag_flujo_neto",
    ]
    X_train, y_train = train[pred_columns], train["en_riesgo"]
    X_val, y_val = val[pred_columns], val["en_riesgo"]
    X_test, y_test = test[pred_columns], test["en_riesgo"]

    # MLflow Setup dinámico con fallback a 5000
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    version_datos = obtener_hash_datos()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Corrida 1: Baseline (DummyClassifier)
    # -------------------------------------------------------------------------
    with mlflow.start_run(run_name="P2_Baseline_DummyClassifier"):
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("tipo_modelo", "baseline")
        mlflow.log_param("version_datos", version_datos)
        mlflow.log_param("strategy", "most_frequent")

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        y_prob_dummy = dummy.predict_proba(X_test)[:, 1]

        # Métricas generales y de la clase positiva solicitadas
        f1_macro = f1_score(y_test, y_pred_dummy, average="macro", zero_division=0)
        prec_pos = precision_score(y_test, y_pred_dummy, pos_label=1, zero_division=0)
        rec_pos = recall_score(y_test, y_pred_dummy, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred_dummy, pos_label=1, zero_division=0)

        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_precision_pos", prec_pos)
        mlflow.log_metric("test_recall_pos", rec_pos)
        mlflow.log_metric("test_f1_pos", f1_pos)

    # -------------------------------------------------------------------------
    # Corrida 2: Modelo Candidato (LogisticRegression con validación)
    # -------------------------------------------------------------------------
    with mlflow.start_run(run_name="P2_Candidato_LogisticRegression"):
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("tipo_modelo", "candidato")
        mlflow.log_param("version_datos", version_datos)
        mlflow.log_param("C", 1.0)
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("random_state", RANDOM_SEED)
        mlflow.log_param("class_weight", "balanced")

        modelo = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_SEED,
        )
        modelo.fit(X_train, y_train)

        # Evaluación en Test
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        f1_macro_lr = f1_score(y_test, y_pred, average="macro")
        auc_lr = roc_auc_score(y_test, y_prob)
        prec_pos_lr = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_pos_lr = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos_lr = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        mlflow.log_metric("test_f1_macro", f1_macro_lr)
        mlflow.log_metric("test_roc_auc", auc_lr)
        mlflow.log_metric("test_precision_pos", prec_pos_lr)
        mlflow.log_metric("test_recall_pos", rec_pos_lr)
        mlflow.log_metric("test_f1_pos", f1_pos_lr)

        # 1. Matriz de confusión
        cm_path = FIGURES_DIR / "p2_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=True)
        ax.set_title("P2: Matriz de Confusión (Riesgo en Estación)")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(str(cm_path))

        # 2. Curva ROC Comparativa (Baseline vs Logistic Regression)
        roc_path = FIGURES_DIR / "p2_roc_curve.png"
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob)
        fpr_dm, tpr_dm, _ = roc_curve(y_test, y_prob_dummy)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="blue")
        plt.plot(fpr_dm, tpr_dm, label="Baseline Dummy (AUC = 0.500)", color="gray", linestyle="--")
        plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
        plt.title("P2: Curva ROC Comparativa")
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(str(roc_path))

        mlflow.sklearn.log_model(modelo, "modelo_clasificacion_p2")
        print(f"[✓] Candidato P2 registrado con éxito:")
        print(f"    test_f1_macro: {f1_macro_lr:.4f} | test_roc_auc: {auc_lr:.4f}")
        print(f"    Clase Positiva -> Precision: {prec_pos_lr:.4f} | Recall: {rec_pos_lr:.4f} | F1: {f1_pos_lr:.4f}")


if __name__ == "__main__":
    ejecutar_experimento_p2()