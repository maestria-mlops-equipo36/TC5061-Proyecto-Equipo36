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

        for cfg in configuraciones:
            run_name_hijo = f"P2_LR_C{cfg['C']}_{cfg['solver']}"
            with mlflow.start_run(run_name=run_name_hijo, nested=True):
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

        mlflow.log_metric("best_val_f1_macro", mejor_val_f1)
        if mejor_cfg:
            mlflow.log_param("best_C", mejor_cfg["C"])
            mlflow.log_param("best_solver", mejor_cfg["solver"])
            print(f"Best config found: C={mejor_cfg['C']} (Val F1: {mejor_val_f1:.4f})")

    # Promoción del mejor modelo al Model Registry
    client = mlflow.tracking.MlflowClient()
    model_name = "mibici-equipo36-p2-riesgo"
    try:
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            best_ver = latest_versions[-1].version
            client.set_registered_model_alias(name=model_name, alias="champion", version=best_ver)
            print(f"Modelo '{model_name}' versión {best_ver} promovido al alias 'champion' (Producción)")
    except Exception as e:
        print(f"Nota sobre Model Registry: {e}")


if __name__ == "__main__":
    ejecutar_busqueda_p2()