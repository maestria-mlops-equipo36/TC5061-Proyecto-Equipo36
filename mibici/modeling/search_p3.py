"""Búsqueda de hiperparámetros para P3 (Clustering) con MLflow Nested Runs.

Permite entrenar una búsqueda organizada como corrida padre y corridas hijas
anidadas, o recuperar el ganador de una corrida padre existente sin reentrenar.
"""

import argparse
from contextlib import nullcontext
import os
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    TEAM_NAME,
)
from mibici.modeling.train import (
    cargar_manifest_version,
    obtener_git_commit,
    p3_clustering,
)

CONFIGURACIONES = [
    {
        "modelo_tipo": "kmeans",
        "n_clusters": 3,
        "covariance_type": "full",
        "incluir_espacial": True,
    },
    {
        "modelo_tipo": "kmeans",
        "n_clusters": 4,
        "covariance_type": "full",
        "incluir_espacial": True,
    },
    {
        "modelo_tipo": "kmeans",
        "n_clusters": 5,
        "covariance_type": "full",
        "incluir_espacial": True,
    },
    {
        "modelo_tipo": "gmm",
        "n_clusters": 4,
        "covariance_type": "full",
        "incluir_espacial": True,
    },
    {
        "modelo_tipo": "gmm",
        "n_clusters": 4,
        "covariance_type": "diag",
        "incluir_espacial": True,
    },
]


def seleccionar_ganador(corridas):
    """Selecciona la corrida hija con mayor silhouette_score y menor davies_bouldin_score."""
    candidatas = [
        corrida
        for corrida in corridas
        if corrida.info.status == "FINISHED"
        and "silhouette_score" in corrida.data.metrics
        and "davies_bouldin_score" in corrida.data.metrics
        and "modelo_tipo" in corrida.data.params
        and "n_clusters" in corrida.data.params
    ]
    if not candidatas:
        raise RuntimeError(
            "No hay corridas hijas finalizadas con métricas y parámetros P3."
        )
    return max(
        candidatas,
        key=lambda corrida: (
            corrida.data.metrics["silhouette_score"],
            -corrida.data.metrics["davies_bouldin_score"],
        ),
    )


def registrar_ganador(ganador):
    """Registra la selección ganadora en la corrida activa sin modificar parámetros existentes."""
    mlflow.log_metric(
        "best_silhouette_score",
        ganador.data.metrics["silhouette_score"],
    )
    mlflow.log_metric(
        "best_davies_bouldin_score",
        ganador.data.metrics["davies_bouldin_score"],
    )
    if "calinski_harabasz_score" in ganador.data.metrics:
        mlflow.log_metric(
            "best_calinski_harabasz_score",
            ganador.data.metrics["calinski_harabasz_score"],
        )

    for nombre in (
        "modelo_tipo",
        "n_clusters",
        "covariance_type",
        "incluir_espacial",
    ):
        if nombre in ganador.data.params:
            mlflow.set_tag(f"best_{nombre}", ganador.data.params[nombre])

    mlflow.set_tag("best_child_run_id", ganador.info.run_id)
    mlflow.set_tag("best_child_run_name", ganador.data.tags.get("mlflow.runName", ""))

    print("\n=== Mejor configuración P3 (Clustering) ===")
    print(f"Parámetros: {ganador.data.params}")
    print(f"Silhouette Score: {ganador.data.metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Score: {ganador.data.metrics['davies_bouldin_score']:.4f}")
    if "calinski_harabasz_score" in ganador.data.metrics:
        print(
            f"Calinski-Harabasz Score: {ganador.data.metrics['calinski_harabasz_score']:.2f}"
        )
    print(f"Run ID ganador: {ganador.info.run_id}")


def ejecutar_busqueda_p3(
    n_trials: int = len(CONFIGURACIONES),
    reuse_parent_run_id: Optional[str] = None,
    tracking_uri: Optional[str] = None,
):
    """Ejecuta la búsqueda con nested runs para P3 o recupera el ganador."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    experimento = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_commit = obtener_git_commit()
    data_version = cargar_manifest_version()

    print("Proyecto: MiBici Equipo 36")
    print("Pregunta: P3 - Clustering de Estaciones")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    if reuse_parent_run_id:
        cliente = MlflowClient()
        padre = cliente.get_run(reuse_parent_run_id)
        if padre.info.experiment_id != experimento.experiment_id:
            raise RuntimeError(
                f"La corrida padre no pertenece a {MLFLOW_EXPERIMENT_NAME}."
            )
        hijos = cliente.search_runs(
            [experimento.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{padre.info.run_id}'",
            max_results=1000,
        )
        ganador = seleccionar_ganador(hijos)
        with mlflow.start_run(run_name="P3_MODELO_GANADOR_FINAL"):
            mlflow.set_tags({
                "equipo": TEAM_NAME,
                "pregunta": "P3",
                "question": "p3",
                "tarea": "seleccion_modelo_existente",
                "stage_of_work": "candidate",
                "git_commit": git_commit,
                "commit": git_commit,
                "data_version": data_version,
                "source_parent_run_id": padre.info.run_id,
            })
            registrar_ganador(ganador)
        print("P3: selección registrada sin volver a entrenar.")
        return ganador

    if not 1 <= n_trials <= len(CONFIGURACIONES):
        raise ValueError(
            f"--n-trials debe estar entre 1 y {len(CONFIGURACIONES)}."
        )

    run_activo = mlflow.active_run()
    if run_activo is not None:
        contexto_padre = nullcontext(run_activo)
    else:
        env_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        contexto_padre = mlflow.start_run(
            run_id=env_run_id,
            run_name="P3_hyperparameter_search",
        )

    with contexto_padre as padre:
        print(f"Corrida padre: {padre.info.run_id}")
        mlflow.set_tags({
            "mlflow.runName": "P3_hyperparameter_search",
            "equipo": TEAM_NAME,
            "pregunta": "P3",
            "question": "p3",
            "tarea": "hyperparameter_search",
            "model_family": "clustering_search",
            "modelo_familia": "clustering_search",
            "stage_of_work": "search",
            "git_commit": git_commit,
            "commit": git_commit,
            "data_version": data_version,
        })
        try:
            mlflow.log_param("n_trials", n_trials)
        except Exception:
            pass

        ids_hijos = []

        for i, config in enumerate(CONFIGURACIONES[:n_trials], start=1):
            trial_name = (
                f"P3_trial_{i}_{config['modelo_tipo']}_k{config['n_clusters']}"
                f"_{config['covariance_type']}"
            )
            with mlflow.start_run(run_name=trial_name, nested=True) as hijo:
                print(f"Entrenando trial {i}: {config}")
                p3_clustering(
                    modelo_tipo=config["modelo_tipo"],
                    n_clusters=config["n_clusters"],
                    covariance_type=config["covariance_type"],
                    incluir_espacial=config["incluir_espacial"],
                    stage_of_work="search",
                )
                ids_hijos.append(hijo.info.run_id)
                resultado = mlflow.get_run(hijo.info.run_id)
                sil = resultado.data.metrics["silhouette_score"]
                db = resultado.data.metrics["davies_bouldin_score"]
                print(
                    f"Trial {i}: silhouette={sil:.4f}, davies_bouldin={db:.4f}"
                )

        cliente = MlflowClient()
        ganador = seleccionar_ganador(
            [cliente.get_run(run_id) for run_id in ids_hijos]
        )
        registrar_ganador(ganador)
        print("P3: búsqueda de hiperparámetros finalizada correctamente.")
        return ganador


def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda de hiperparámetros P3 (Clustering)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=len(CONFIGURACIONES),
        help=f"Número de configuraciones a entrenar (1 a {len(CONFIGURACIONES)})",
    )
    parser.add_argument(
        "--reuse-parent-run-id",
        default=None,
        help="Seleccionar el mejor hijo de una corrida padre existente, sin entrenar",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="Servidor MLflow",
    )
    args = parser.parse_args()

    ejecutar_busqueda_p3(
        n_trials=args.n_trials,
        reuse_parent_run_id=args.reuse_parent_run_id,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
