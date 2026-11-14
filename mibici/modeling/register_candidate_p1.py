
"""Registro reproducible del modelo alternativo P1 en MLflow."""

import os

import mlflow
import typer
from mlflow import MlflowClient

MODEL_NAME = "mibici-equipo36-p1-demanda"


def main(
    run_id: str = typer.Option(..., help="Run ID del modelo alternativo"),
):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Confirmar que existe el modelo champion.
    champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )
    champion_version = champion.version

    # Buscar el modelo generado durante el entrenamiento.
    models = client.search_logged_models(
        experiment_ids=["5"],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if len(models) != 1:
        raise RuntimeError(
            f"Se esperaba un modelo para el Run ID {run_id}, "
            f"pero se encontraron {len(models)}."
        )

    model_uri = f"models:/{models[0].model_id}"

    # Buscar si ya existe una versión registrada.
    versions = client.search_model_versions(
        f"name = '{MODEL_NAME}'"
    )

    existing = next(
        (
            version for version in versions
            if version.run_id == run_id
            and version.source == model_uri
        ),
        None,
    )

    if existing:
        version_number = existing.version
        print(f"Versión existente encontrada: {version_number}")
    else:
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )
        version_number = registered.version
        print(f"Nueva versión registrada: {version_number}")

    # Asignar alias candidate sin modificar champion.
    client.set_registered_model_alias(
        MODEL_NAME,
        "candidate",
        version_number,
    )

    champion_final = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    if champion_final.version != champion_version:
        raise RuntimeError(
            "El alias champion cambió inesperadamente."
        )

    print("\nREGISTRO P1 COMPLETADO")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Champion: versión {champion_final.version}")
    print(f"Candidate: versión {version_number}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    typer.run(main)
