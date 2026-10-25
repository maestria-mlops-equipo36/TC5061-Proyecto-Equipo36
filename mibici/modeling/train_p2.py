"""
Módulo de Entrenamiento y Evaluación para la Pregunta 2 (P2).

Problema de Negocio:
    Clasificación del riesgo operativo de desabasto o saturación en cicloestaciones 
    de MiBici Guadalajara para la planeación del equipo de rebalanceo.

Atiende especificaciones de MLflow Projects (Segunda Entrega):
1. Partición estrictamente temporal: Train (70%), Validación (15%), Prueba (15%).
2. Métricas de la clase positiva (en_riesgo = 1): Precision, Recall, F1, además de F1-macro y ROC-AUC.
3. Figuras en reports/figures/: Matriz de confusión, Curva ROC y Curva Precisión-Recall.
4. Registro en Model Registry con alias 'champion' y firma de modelo.
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
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


def preparar_datos_p2():
    viajes_path = PROCESSED_DATA_DIR / "viajes_limpios.parquet"
    estaciones_path = PROCESSED_DATA_DIR / "estaciones.parquet"

    df_viajes = pd.read_parquet(viajes_path)
    df_estaciones = pd.read_parquet(estaciones_path)
    df_p2 = calcular_features_riesgo_p2(df_viajes, df_estaciones, freq="2h")

    # Partición temporal en 3 bloques (Train: 70%, Val: 15%, Test: 15%)
    df_p2 = df_p2.sort_values("ventana").reset_index(drop=True)
    n = len(df_p2)
    corte_train = int(n * 0.70)
    corte_val = int(n * 0.85)

    train = df_p2.iloc[:corte_train]
    val = df_p2.iloc[corte_train:corte_val]
    test = df_p2.iloc[corte_val:]

    pred_columns = ["capacidad","hora","dia_semana","es_fin_de_semana","lag_salidas","lag_llegadas","lag_flujo_neto",]
    return (
        train[pred_columns],
        train["en_riesgo"],
        val[pred_columns],
        val["en_riesgo"],
        test[pred_columns],
        test["en_riesgo"],
    )


def p2_clasificacion(
    modelo_tipo="logistic_regression",
    c_param=1.0,
    solver="lbfgs",
    penalty="l2",
    stage_of_work="candidate",
    evaluar_test=True,
):
    """Función unificada compatible con CLI de run_project y MLproject."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Solo cambiar de experimento si no estamos dentro de un run previo de MLflow Projects
    if not mlflow.active_run():
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    version_datos = obtener_hash_datos()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos_p2()

    # Log dataset formal de entrenamiento
    dataset_mlflow = mlflow.data.from_pandas(
        pd.concat([X_train, y_train], axis=1),
        targets="en_riesgo",
        name="p2_train_dataset",
    )

    # Reutilizar la corrida activa de MLflow Projects si existe
    active_run = mlflow.active_run()
    run_context = active_run if active_run else mlflow.start_run(run_name=f"P2_{modelo_tipo}")

    with run_context:
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("question", "p2")
        mlflow.set_tag("model_family", "baseline" if modelo_tipo == "baseline" else "linear_model")
        mlflow.set_tag("stage_of_work", stage_of_work)

        mlflow.log_param("version_datos", version_datos)
        mlflow.log_param("modelo_tipo", modelo_tipo)
        mlflow.log_input(dataset_mlflow, context="training")

        # ---------------------------------------------------------------------
        # 1. BASELINE (DummyClassifier)
        # ---------------------------------------------------------------------
        if modelo_tipo == "baseline":
            mlflow.log_param("strategy", "most_frequent")
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)

            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

            mlflow.log_metric("test_f1_macro", f1_macro)
            mlflow.log_metric("test_precision_pos", prec_pos)
            mlflow.log_metric("test_recall_pos", rec_pos)
            mlflow.log_metric("test_f1_pos", f1_pos)

            print(f" Baseline P2 registrado exitosamente:")
            print(f"    test_f1_macro: {f1_macro:.4f} | test_recall_pos: {rec_pos:.4f}")
            return f1_macro

        # ---------------------------------------------------------------------
        # 2. MODELO CANDIDATO (LogisticRegression Parametrizado)
        # ---------------------------------------------------------------------
        mlflow.log_param("C", c_param)
        mlflow.log_param("solver", solver)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", RANDOM_SEED)

        modelo = LogisticRegression(
            C=c_param,
            solver=solver,
            penalty=penalty,
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_SEED,
        )
        modelo.fit(X_train, y_train)

        # Validación
        y_pred_val = modelo.predict(X_val)
        val_f1 = f1_score(y_val, y_pred_val, average="macro")
        mlflow.log_metric("val_f1_macro", val_f1)
        # Durante la búsqueda no utilizar el conjunto Test
        if not evaluar_test:
            print(
                f"Candidato P2 evaluado en validación: "
                f"C={c_param}, solver={solver}, "
                f"val_f1_macro={val_f1:.4f}"
            )
            return val_f1

        # Prueba
        y_pred_test = modelo.predict(X_test)
        y_prob_test = modelo.predict_proba(X_test)[:, 1]

        f1_macro_lr = f1_score(y_test, y_pred_test, average="macro")
        auc_lr = roc_auc_score(y_test, y_prob_test)
        prec_pos_lr = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        rec_pos_lr = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        f1_pos_lr = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)

        mlflow.log_metric("test_f1_macro", f1_macro_lr)
        mlflow.log_metric("test_roc_auc", auc_lr)
        mlflow.log_metric("test_precision_pos", prec_pos_lr)
        mlflow.log_metric("test_recall_pos", rec_pos_lr)
        mlflow.log_metric("test_f1_pos", f1_pos_lr)

        # Artefacto 1: Matriz de confusión
        cm_path = FIGURES_DIR / "p2_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax, cmap="Blues", colorbar=True)
        ax.set_title("P2: Matriz de Confusión (Riesgo en Estación)")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        # Artefacto 2: Curva Precisión-Recall
        pr_path = FIGURES_DIR / "p2_precision_recall_curve.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_prob_test, ax=ax, color="purple")
        ax.set_title("P2: Curva Precisión-Recall")
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()
        mlflow.log_artifact(str(pr_path), artifact_path="figures")

        # Artefacto 3: Curva ROC
        roc_path = FIGURES_DIR / "p2_roc_curve.png"
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="blue")
        plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
        plt.title("P2: Curva ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="figures")

        # Registro nativo del modelo con firma y ejemplo
        signature = infer_signature(X_train, modelo.predict(X_train))
        input_example = X_train.iloc[:5]

        model_name = "mibici-equipo36-p2-riesgo"
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo_clasificacion_p2",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )

        print(f" Candidato P2 registrado exitosamente (C={c_param}, solver={solver}):")
        print(f"    Val F1: {val_f1:.4f} | Test F1-macro: {f1_macro_lr:.4f} | ROC-AUC: {auc_lr:.4f}")
        print(f"    Clase Positiva -> Precision: {prec_pos_lr:.4f} | Recall: {rec_pos_lr:.4f} | F1: {f1_pos_lr:.4f}")

        return val_f1


if __name__ == "__main__":
    p2_clasificacion()
