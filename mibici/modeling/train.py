import json
import subprocess
from pathlib import Path
from typing import List, Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    DATA_MANIFEST_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEAM_NAME,
)
from mibici.plots import graficar_clusters_mapa, graficar_perfil_clusters

app = typer.Typer(help="Entrenamiento y experimentación con MLFlow para modelos de MiBici.")


def obtener_git_commit() -> str:
    """Obtiene el hash del commit actual de Git para trazabilidad en MLFlow."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "desconocido"


def cargar_manifest_version() -> str:
    """Carga la versión y hash del dataset desde data_manifest.json."""
    if DATA_MANIFEST_PATH.exists():
        try:
            with open(DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("dataset_version", "v1")
        except Exception:
            pass
    return "v1"


@app.command()
def p3_clustering(
    features_path: Path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet",
    modelo_tipo: str = "kmeans",
    n_clusters: int = 4,
    random_seed: int = RANDOM_SEED,
    incluir_espacial: bool = True,
):
    """
    Ejecuta el experimento de clustering para la Pregunta 3 (P3) y lo registra en MLFlow Tracking.
    """
    logger.info(f"Iniciando experimento P3: modelo={modelo_tipo}, clusters={n_clusters}, semilla={random_seed}")
    
    if not features_path.exists():
        logger.error(f"No existe la matriz de características en {features_path}. Ejecuta features.py primero.")
        raise typer.Exit(code=1)

    df_features = pd.read_parquet(features_path)
    logger.info(f"Cargadas {len(df_features)} estaciones.")

    # Definir variables utilizadas para el perfilamiento
    feature_cols = [
        "ratio_flujo_neto",
        "pct_salidas_pico_matutino",
        "pct_salidas_pico_vespertino",
        "pct_salidas_fin_de_semana",
        "viajes_diarios_por_anclaje",
    ]
    if incluir_espacial:
        feature_cols.extend(["latitud", "longitud"])

    X = df_features[feature_cols].copy()

    # Preprocesamiento: Escalado estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configuración de MLFlow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_commit = obtener_git_commit()
    data_version = cargar_manifest_version()

    run_name = f"P3_{modelo_tipo}_k{n_clusters}_{'espacial' if incluir_espacial else 'temporal'}"

    with mlflow.start_run(run_name=run_name):
        # 1. Registrar Etiquetas (Tags)
        mlflow.set_tag("equipo", TEAM_NAME)
        mlflow.set_tag("pregunta", "P3")
        mlflow.set_tag("tarea", "clustering")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("modelo_familia", modelo_tipo)

        # 2. Registrar Parámetros (Params)
        mlflow.log_param("modelo_tipo", modelo_tipo)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("incluir_espacial", incluir_espacial)
        mlflow.log_param("features_list", ",".join(feature_cols))
        mlflow.log_param("n_estaciones", len(df_features))

        # 3. Ajuste del Modelo
        if modelo_tipo == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=10,
                max_iter=300,
            )
            labels = model.fit_predict(X_scaled)
            mlflow.log_metric("inertia", float(model.inertia_))
        elif modelo_tipo == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_seed,
                covariance_type="full",
            )
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            mlflow.log_metric("bic", float(model.bic(X_scaled)))
            mlflow.log_metric("aic", float(model.aic(X_scaled)))
        else:
            raise ValueError(f"Tipo de modelo no soportado: {modelo_tipo}")

        # 4. Cálculo y Registro de Métricas Intrínsecas de Clustering
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
        ch_score = float(calinski_harabasz_score(X_scaled, labels))

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        logger.info(f"Resultados de Métricas: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}, Calinski-Harabasz={ch_score:.2f}")

        # 5. Generar y Registrar Artefactos
        df_eval = df_features.copy()
        df_eval["cluster"] = labels

        # a) Gráfica geográfica
        mapa_path = FIGURES_DIR / f"{run_name}_mapa.png"
        graficar_clusters_mapa(
            df_eval,
            columna_cluster="cluster",
            output_path=mapa_path,
            titulo=f"Distribución de Clusters ({modelo_tipo.upper()}, k={n_clusters})",
        )
        mlflow.log_artifact(str(mapa_path), artifact_path="figures")

        # b) Gráfica de perfil
        perfil_path = FIGURES_DIR / f"{run_name}_perfil.png"
        graficar_perfil_clusters(
            df_eval,
            columnas_features=feature_cols,
            columna_cluster="cluster",
            output_path=perfil_path,
            titulo=f"Perfil de Comportamiento ({modelo_tipo.upper()}, k={n_clusters})",
        )
        mlflow.log_artifact(str(perfil_path), artifact_path="figures")

        # c) Tabla de asignación de estaciones
        clusters_csv_path = PROCESSED_DATA_DIR / f"{run_name}_estaciones_clusters.csv"
        df_eval.to_csv(clusters_csv_path, index=False)
        mlflow.log_artifact(str(clusters_csv_path), artifact_path="data")

        # d) Guardar modelo serializado localmente (ignorado por Git)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file = MODELS_DIR / f"{run_name}.joblib"
        joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, model_file)

        logger.success(f"Run de MLFlow finalizado exitosamente: {run_name}")


if __name__ == "__main__":
    app()
