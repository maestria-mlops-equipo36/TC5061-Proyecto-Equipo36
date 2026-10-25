"""
Registro de modelos ganadores de MiBici en MLflow Model Registry.

Permite registrar un modelo a partir de una corrida existente
y asignarle el alias 'champion'.

Ejemplo:
    python -m mibici.modeling.register_models \
        --question p1 \
        --run-id RUN_ID \
        --model-uri runs:/RUN_ID/modelo_p1_random_forest
"""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient


MODEL_CONFIG = {
    "p1": {
        "name": "mibici-equipo36-p1-demanda",
        "description": (
            "Modelo de regresión para predecir la demanda diaria "
            "de viajes por estación de MiBici."
        ),
    },
    "p2": {
        "name": "mibici-equipo36-p2-riesgo",
        "description": (
            "Modelo de clasificación para identificar estaciones "
            "con riesgo operativo de desabasto o saturación."
        ),
    },
    "p3": {
        "name": "mibici-equipo36-p3-grupos",
        "description": (
            "Modelo de clustering para agrupar estaciones "
            "según sus patrones de uso y características."
        ),
    },
}


def registrar_modelo(question, run_id, model_uri, replace_champion=False):
    """Registra un modelo ganador y le asigna el alias champion."""

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    config = MODEL_CONFIG[question]
    model_name = config["name"]

    print(f"\nPregunta: {question.upper()}")
    print(f"Modelo: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_uri}")

    # 1. Verificar que la corrida existe y terminó correctamente
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"La corrida {run_id} no está finalizada correctamente."
        )

    # 2. Verificar que la URI corresponde a la corrida indicada
    if model_uri.startswith("runs:/"):
        if not model_uri.startswith(f"runs:/{run_id}/"):
            raise ValueError(
                "La URI del modelo no corresponde al run_id proporcionado."
            )

    elif model_uri.startswith("models:/"):
        model_id = model_uri.removeprefix("models:/")

        # Se aceptan únicamente URI de Logged Models, no aliases.
        if not model_id or "/" in model_id or "@" in model_id:
            raise ValueError(
                "Se requiere una URI de Logged Model: models:/<model_id>."
            )

        logged_model = client.get_logged_model(model_id)

        if logged_model.source_run_id != run_id:
            raise ValueError(
                "El Logged Model no corresponde al run_id proporcionado."
            )

    else:
        raise ValueError(
            "URI no soportada. Utiliza runs:/ o models:/<model_id>."
        )

    # 3. Comprobar que el modelo puede cargarse antes de registrarlo
    print("Verificando que el modelo pueda cargarse...")
    mlflow.pyfunc.load_model(model_uri)

    # 4. Comprobar si ya existe un champion
    try:
        champion_actual = client.get_model_version_by_alias(
            model_name,
            "champion",
        )
    except mlflow.exceptions.MlflowException as exc:
        if exc.error_code not in (
            "RESOURCE_DOES_NOT_EXIST",
            "ENDPOINT_NOT_FOUND",
        ):
            raise
        champion_actual = None

    if champion_actual and not replace_champion:
        raise RuntimeError(
            f"Ya existe champion v{champion_actual.version}. "
            "Utiliza --replace-champion para reemplazarlo."
        )

    # 5. Registrar el modelo
    print("Registrando modelo...")

    version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    # 6. Agregar descripción
    client.update_registered_model(
        name=model_name,
        description=config["description"],
    )

    # 7. Conservar información de linaje
    client.set_model_version_tag(
        model_name,
        version.version,
        "source_run_id",
        run_id,
    )

    client.set_model_version_tag(
        model_name,
        version.version,
        "question",
        question,
    )

    client.set_model_version_tag(
        model_name,
        version.version,
        "git_commit",
        run.data.tags.get(
            "git_commit",
            run.data.tags.get("commit", "no_disponible"),
        ),
    )

    client.set_model_version_tag(
        model_name,
        version.version,
        "data_version",
        run.data.tags.get(
            "data_version",
            run.data.params.get("data_version", "no_disponible"),
        ),
    )

    # 8. Asignar alias de producción
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version.version,
    )

    print("\nRegistro completado:")
    print(f"  Modelo: {model_name}")
    print(f"  Versión: {version.version}")
    print("  Alias: champion")
    print(f"  Origen: {run_id}")

    return version


def main():
    parser = argparse.ArgumentParser(
        description="Registro de modelos ganadores de MiBici"
    )

    parser.add_argument(
        "--question",
        choices=MODEL_CONFIG.keys(),
        required=True,
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Identificador de la corrida ganadora.",
    )

    parser.add_argument(
        "--model-uri",
        required=True,
        help="URI del modelo guardado en MLflow.",
    )

    parser.add_argument(
        "--replace-champion",
        action="store_true",
        help="Permite reemplazar el alias champion existente.",
    )

    args = parser.parse_args()

    registrar_modelo(
        question=args.question,
        run_id=args.run_id,
        model_uri=args.model_uri,
        replace_champion=args.replace_champion,
    )


if __name__ == "__main__":
    main()