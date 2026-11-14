
"""Entrenamiento alternativo P1 para comparación A/B."""

import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import MLFLOW_EXPERIMENT_NAME, RANDOM_SEED
from mibici.modeling.train import entrenar_random_forest_p1

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

DATA_PATH = Path("data/processed/regression_reference.parquet")

FEATURES = [
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

PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 3,
}


def main():
    df = pd.read_parquet(DATA_PATH)

    train = df[df["fecha"] < "2026-07-01"].copy()
    val = df[df["fecha"] >= "2026-07-01"].copy()

    if train.empty or val.empty:
        raise ValueError("Los conjuntos de entrenamiento o validación están vacíos.")

    X_train, y_train = train[FEATURES], train["viajes_iniciados"]
    X_val, y_val = val[FEATURES], val["viajes_iniciados"]

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="P1_random_forest_alternativo") as run:
        mlflow.set_tags({
            "pregunta": "P1",
            "stage_of_work": "candidate",
            "modelo_familia": "random_forest",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        })

        mlflow.log_params({
            **PARAMS,
            "random_seed": RANDOM_SEED,
            "n_train": len(train),
            "n_val": len(val),
            "fecha_fin_train": "2026-06-30",
            "fecha_inicio_val": "2026-07-01",
            "fecha_fin_val": "2026-08-31",
            "features_list": ",".join(FEATURES),
        })

        # Entrenar con el conjunto histórico inicial.
        model = entrenar_random_forest_p1(
            X_train, y_train, **PARAMS
        )

        pred = np.maximum(model.predict(X_val), 0)

        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))

        mlflow.log_metrics({
            "val_mae": float(mae),
            "val_rmse": float(rmse),
        })

        print(f"MAE validación:  {mae:.4f}")
        print(f"RMSE validación: {rmse:.4f}")

        # Reentrenar con todo el historial hasta agosto.
        X_final = df[FEATURES]
        y_final = df["viajes_iniciados"]
        model.fit(X_final, y_final)

        # Definir la firma del modelo con tipos explícitos.

        input_schema = Schema([
            ColSpec("long", "estacion_id"),
            ColSpec("integer", "dia_semana"),
            ColSpec("integer", "mes"),
            ColSpec("long", "es_fin_de_semana"),
            ColSpec("double", "viajes_dia_anterior"),
            ColSpec("double", "viajes_hace_7_dias"),
            ColSpec("double", "promedio_7_dias"),
            ColSpec("double", "promedio_14_dias"),
            ColSpec("double", "viajes_hace_14_dias"),
            ColSpec("double", "viajes_hace_28_dias"),
            ColSpec("double", "promedio_28_dias"),
            ColSpec("double", "promedio_mismo_dia_4_semanas"),
        ])

        signature = ModelSignature(
            inputs=input_schema,
            outputs=Schema([ColSpec("double")]),
        )

        example = X_final.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo_p1_alternativo",
            signature=signature,
            input_example=example,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"\nRun ID: {run.info.run_id}")
        print("Modelo alternativo guardado en MLflow Tracking.")
        print("El alias champion no ha sido modificado.")


if __name__ == "__main__":
    typer.run(main)
