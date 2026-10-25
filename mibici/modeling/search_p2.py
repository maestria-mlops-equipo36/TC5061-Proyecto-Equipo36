"""
Módulo de Búsqueda Sistemática de Hiperparámetros para P2.
Implementa:
- Ejecución de Baseline y >= 3 versiones adicionales.
- Búsqueda organizada como Nested Runs (run padre de búsqueda y corridas hijas).
- Promoción del modelo con mejor F1-macro de validación al Model Registry con alias 'champion'.
"""

import os
import mlflow
from mibici.config import MLFLOW_EXPERIMENT_NAME
from mibici.modeling.train_p2 import p2_clasificacion


def ejecutar_busqueda_p2():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Baseline de P2 (Clasificación de Riesgo de Accidentes)")
    p2_clasificacion(modelo_tipo="baseline", stage_of_work="baseline")
    configuraciones = [
        {"C": 0.01, "solver": "lbfgs", "penalty": "l2"},
        {"C": 0.1, "solver": "lbfgs", "penalty": "l2"},
        {"C": 1.0, "solver": "lbfgs", "penalty": "l2"},
        {"C": 10.0, "solver": "lbfgs", "penalty": "l2"},
    ]
    print("Búsqueda de Hiperparámetros (p2)")

    # Reutilizar la corrida de MLflow Projects si ya existe.
    active_run = mlflow.active_run()
    run_context = (
        mlflow.start_run(run_name="P2_Busqueda_Optimizacion_Parent")
        if active_run is None
        else mlflow.start_run(
            run_name="P2_Busqueda_Optimizacion_Parent",
            nested=True,
        )
    )

    with run_context as parent_run:
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("question", "p2")
        mlflow.set_tag("stage_of_work", "search")

        mejor_val_f1 = -1.0
        mejor_cfg = None
        mejor_run_id = None

        for cfg in configuraciones:
            run_name_hijo = f"P2_LR_C{cfg['C']}_{cfg['solver']}"
            with mlflow.start_run(run_name=run_name_hijo, nested=True) as child_run:
                val_f1 = p2_clasificacion(
                    modelo_tipo="logistic_regression",
                    c_param=cfg["C"],
                    solver=cfg["solver"],
                    penalty=cfg["penalty"],
                    stage_of_work="candidate",
                )
                if val_f1 is not None and val_f1 > mejor_val_f1:
                    mejor_val_f1 = val_f1
                    mejor_cfg = cfg
                    mejor_run_id = child_run.info.run_id

        mlflow.log_metric("best_val_f1_macro", mejor_val_f1)
        if mejor_run_id is not None:
            mlflow.set_tag("best_child_run_id", mejor_run_id)
        if mejor_cfg:
            mlflow.log_param("best_C", mejor_cfg["C"])
            mlflow.log_param("best_solver", mejor_cfg["solver"])
            print(f"Best config found: C={mejor_cfg['C']} (Val F1: {mejor_val_f1:.4f})")

        # Promover al mejor candidato según F1-macro de validación.
        if mejor_run_id is None:
            raise RuntimeError("No se encontró un candidato válido para P2.")

        client = mlflow.tracking.MlflowClient()
        model_name = "mibici-equipo36-p2-riesgo"

        versiones = client.search_model_versions(
            f"name='{model_name}'"
        )

        versiones_ganadoras = [
            version
            for version in versiones
            if version.run_id == mejor_run_id
        ]

        if not versiones_ganadoras:
            raise RuntimeError(
                f"No se encontró una versión registrada para el run {mejor_run_id}"
            )

        version_ganadora = max(
            versiones_ganadoras,
            key=lambda version: int(version.version),
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version_ganadora.version,
        )

        # Documentar automáticamente el modelo champion.
        run_ganador = client.get_run(mejor_run_id)

        client.update_registered_model(
            name=model_name,
            description=(
                "Modelo de clasificación para predecir el riesgo operativo "
                "de desabasto o saturación de estaciones MiBici. "
                "Utiliza regresión logística con balanceo de clases. "
                "El modelo champion se selecciona mediante F1-macro "
                "sobre el conjunto de validación."
            ),
        )

        tags = {
            "source_run_id": mejor_run_id,
            "question": "p2",
        }

        git_commit = run_ganador.data.tags.get("mlflow.source.git.commit")
        data_version = run_ganador.data.params.get("version_datos")

        if git_commit:
            tags["git_commit"] = git_commit

        if data_version:
            tags["data_version"] = data_version

        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=version_ganadora.version,
                key=key,
                value=value,
            )

        print(
            f"Champion P2: versión {version_ganadora.version} "
            f"(Run ID: {mejor_run_id}, Val F1: {mejor_val_f1:.4f})"
        )


if __name__ == "__main__":
    ejecutar_busqueda_p2()