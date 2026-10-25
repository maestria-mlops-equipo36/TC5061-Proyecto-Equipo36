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
    with mlflow.start_run(run_name="P2_Busqueda_Optimizacion_Parent") as parent_run:
        mlflow.set_tag("equipo", "equipo36")
        mlflow.set_tag("question", "p2")
        mlflow.set_tag("stage_of_work", "search")

        mejor_val_f1 = -1.0
        mejor_cfg = None
        mejor_run_id = None

        for cfg in configuraciones:
            run_name_hijo = f"P2_LR_C{cfg['C']}_{cfg['solver']}"
            with mlflow.start_run(run_name=run_name_hijo, nested=True):
                val_f1 = p2_clasificacion(
                    modelo_tipo="logistic_regression",
                    c_param=cfg["C"],
                    solver=cfg["solver"],
                    penalty=cfg["penalty"],
                    stage_of_work="candidate",
                    evaluar_test=False,
                )
                if val_f1 is not None and val_f1 > mejor_val_f1:
                    mejor_val_f1 = val_f1
                    mejor_cfg = cfg
                    mejor_run_id = mlflow.active_run().info.run_id

        mlflow.log_metric("best_val_f1_macro", mejor_val_f1)
        if mejor_cfg:
            mlflow.log_param("best_C", mejor_cfg["C"])
            mlflow.log_param("best_solver", mejor_cfg["solver"])
            print(f"Best config found: C={mejor_cfg['C']} (Val F1: {mejor_val_f1:.4f})")

    # ---------------------------------------------------------
    # Evaluación final y registro del modelo ganador
    # ---------------------------------------------------------

    if mejor_cfg is None:
        raise RuntimeError(
            "No se encontró una configuración válida para P2."
        )

    print("\nEvaluación final del mejor modelo P2")
    print(
        f"Mejor configuración: C={mejor_cfg['C']}, "
        f"solver={mejor_cfg['solver']}, "
        f"penalty={mejor_cfg['penalty']}"
    )

    # Ejecutar únicamente el modelo ganador sobre Test
    with mlflow.start_run(run_name="P2_Champion_Final") as final_run:

        p2_clasificacion(
            modelo_tipo="logistic_regression",
            c_param=mejor_cfg["C"],
            solver=mejor_cfg["solver"],
            penalty=mejor_cfg["penalty"],
            stage_of_work="final",
            evaluar_test=True,
        )

        final_run_id = final_run.info.run_id

    # ---------------------------------------------------------
    # Promoción del modelo ganador al Model Registry
    # ---------------------------------------------------------

    client = mlflow.tracking.MlflowClient()
    model_name = "mibici-equipo36-p2-riesgo"

    try:
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        best_version = next(
            (
                version
                for version in versions
                if version.run_id == final_run_id
            ),
            None,
        )

        if best_version is None:
            raise RuntimeError(
                "No se encontró la versión registrada "
                f"para el run_id {final_run_id}."
            )

        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=best_version.version,
        )

        print(
            f"\nChampion P2 registrado correctamente:\n"
            f"Modelo: {model_name}\n"
            f"Versión: {best_version.version}\n"
            f"Run ID: {final_run_id}\n"
            f"F1-macro validación: {mejor_val_f1:.4f}"
        )

    except Exception as e:
        print(f"ERROR al promover el Champion P2: {e}")
        raise

if __name__ == "__main__":
    ejecutar_busqueda_p2()
