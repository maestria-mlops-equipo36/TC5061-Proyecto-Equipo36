"""Búsqueda de hiperparámetros para P1 con MLflow.

Permite entrenar nuevas corridas o seleccionar el ganador de corridas existentes.
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from mibici.modeling.train import p1_regresion


CONFIGURACIONES = [
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 100, "max_depth": 12},
    {"n_estimators": 300, "max_depth": 12},
]


def seleccionar_ganador(corridas):
    candidatas = [
        corrida for corrida in corridas
        if corrida.info.status == "FINISHED"
        and "val_mae" in corrida.data.metrics
        and "n_estimators" in corrida.data.params
        and "max_depth" in corrida.data.params
    ]
    if not candidatas:
        raise RuntimeError("No hay corridas hijas finalizadas con métricas y parámetros P1.")
    return min(candidatas, key=lambda corrida: corrida.data.metrics["val_mae"])


def registrar_ganador(ganador):
    """Registrar selección en la corrida activa, sin modificar parámetros existentes."""
    mlflow.log_metric("best_val_mae", ganador.data.metrics["val_mae"])
    if "val_rmse" in ganador.data.metrics:
        mlflow.log_metric("best_val_rmse", ganador.data.metrics["val_rmse"])
    for nombre in ("n_estimators", "max_depth", "min_samples_leaf"):
        if nombre in ganador.data.params:
            mlflow.set_tag(f"best_{nombre}", ganador.data.params[nombre])
    mlflow.set_tag("best_child_run_id", ganador.info.run_id)
    print("\n=== Mejor configuración P1 ===")
    print(f"Parámetros: {ganador.data.params}")
    print(f"MAE validación: {ganador.data.metrics['val_mae']:.4f}")
    print(f"Run ID ganador: {ganador.info.run_id}")


def main():
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros P1")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Número de configuraciones a entrenar (1 a 3)")
    parser.add_argument("--reuse-parent-run-id", default=None,
                        help="Seleccionar el mejor hijo de una corrida padre existente, sin entrenar")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000",
                        help="Servidor MLflow")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experimento = mlflow.set_experiment("mibici-equipo36")
    print("Proyecto: MiBici Equipo 36")
    print("Pregunta: P1 - Regresión")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    if args.reuse_parent_run_id:
        cliente = MlflowClient()
        padre = cliente.get_run(args.reuse_parent_run_id)
        if padre.info.experiment_id != experimento.experiment_id:
            raise RuntimeError("La corrida padre no pertenece a mibici-equipo36.")
        hijos = cliente.search_runs(
            [experimento.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{padre.info.run_id}'",
            max_results=1000,
        )
        ganador = seleccionar_ganador(hijos)
        with mlflow.start_run(run_name="P1_MODELO_GANADOR_FINAL"):
            mlflow.set_tag("pregunta", "P1")
            mlflow.set_tag("tarea", "seleccion_modelo_existente")
            mlflow.set_tag("source_parent_run_id", padre.info.run_id)
            registrar_ganador(ganador)
        print("P1: selección registrada sin volver a entrenar.")
        return

    if not 1 <= args.n_trials <= len(CONFIGURACIONES):
        parser.error("--n-trials debe estar entre 1 y 3.")

    with mlflow.start_run(run_name="P1_hyperparameter_search") as padre:
        print(f"Corrida padre: {padre.info.run_id}")
        mlflow.set_tag("pregunta", "P1")
        mlflow.set_tag("tarea", "hyperparameter_search")
        mlflow.log_param("n_trials", args.n_trials)
        ids_hijos = []

        for i, config in enumerate(CONFIGURACIONES[:args.n_trials], start=1):
            with mlflow.start_run(run_name=f"P1_trial_{i}", nested=True) as hijo:
                mlflow.log_params(config)
                mlflow.log_param("min_samples_leaf", 5)
                print(f"Entrenando trial {i}: {config}")
                p1_regresion(
                    modelo_tipo="random_forest",
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_leaf=5,
                )
                ids_hijos.append(hijo.info.run_id)
                resultado = mlflow.get_run(hijo.info.run_id)
                print(f"Trial {i}: val_mae={resultado.data.metrics['val_mae']:.4f}")

        cliente = MlflowClient()
        ganador = seleccionar_ganador([cliente.get_run(run_id) for run_id in ids_hijos])
        registrar_ganador(ganador)
        print("P1: búsqueda finalizada correctamente.")


if __name__ == "__main__":
    main()
