import json
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score

from mibici.config import (
    DATA_MANIFEST_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from mibici.features import calcular_features_riesgo_p2


def obtener_hash_datos() -> str:
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

    # 80% train & 20% test (partición estrictamente temporal)
    df_p2 = df_p2.sort_values("ventana").reset_index(drop=True)
    corte = int(len(df_p2) * 0.8)
    train, test = df_p2.iloc[:corte], df_p2.iloc[corte:]

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
    X_test, y_test = test[pred_columns], test["en_riesgo"]

    # MLflow Tracking
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    version_datos = obtener_hash_datos()

    # 1. Baseline run: DummyClassifier
    with mlflow.start_run(run_name="P2_Baseline_DummyClassifier"):
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("tipo_modelo", "baseline")
        mlflow.log_param("version_datos", version_datos)
        mlflow.log_param("strategy", "most_frequent")

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        f1_dummy = f1_score(y_test, y_pred_dummy, average="macro", zero_division=0)
        mlflow.log_metric("test_f1_macro", f1_dummy)
        print(f"[✓] Baseline P2 registrado -> test_f1_macro: {f1_dummy:.4f}")

    # 2. Candidate run: Regresión Logística
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
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("test_f1_macro", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # Matriz de confusión
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = FIGURES_DIR / "p2_confusion_matrix.png"

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues",
            colorbar=True,
        )
        ax.set_title("P2: Matriz de Confusión (Riesgo en Estación)")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(modelo, "modelo_clasificacion_p2")
        print(
            f"[✓] Candidato P2 registrado -> test_f1_macro: {f1:.4f} | test_roc_auc: {auc:.4f}"
        )


if __name__ == "__main__":
    ejecutar_experimento_p2()