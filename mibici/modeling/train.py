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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
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
from mibici.features import (
    calcular_demanda_diaria_p1,
    preparar_demanda_diaria_p1,
    generar_caracteristicas_regresion_p1,
)
from mibici.plots import (
    graficar_clusters_mapa,
    graficar_perfil_clusters,
    graficar_reales_vs_predichos_p1,
    graficar_distribucion_errores_p1,
    graficar_importancia_caracteristicas_p1,
)

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


@app.command()
def p1_regresion(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes_limpios.parquet",
    cobertura_minima: float = 0.95,
    modelo_tipo: str = "baseline",
):
    """
    Ejecuta los modelos de regresión para la Pregunta 1 (P1).

    Modelos disponibles:
    - baseline: demanda observada siete días antes.
    - random_forest: bosque aleatorio con 12 características temporales
      e históricas y una configuración fija.

    Registra los resultados de validación y prueba final en MLflow.
    """

    logger.info(f"Iniciando experimento P1: modelo={modelo_tipo}.")

    # 1. Verificar y cargar datos
    if not viajes_path.exists():
        logger.error(
            f"No existe el archivo de viajes: {viajes_path}. "
            "Ejecuta primero dataset.py."
        )
        raise typer.Exit(code=1)

    df_viajes = pd.read_parquet(
        viajes_path,
        columns=["Inicio_del_viaje", "Origen_Id"],
    )

    # 2. Generar características para regresión
    df_demanda = calcular_demanda_diaria_p1(df_viajes)

    df_preparado = preparar_demanda_diaria_p1(
        df_demanda,
        cobertura_minima=cobertura_minima,
    )

    df_features = generar_caracteristicas_regresion_p1(df_preparado)

    # 3. Definir las 12 características del modelo de regresión
    feature_cols = [
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

    # Eliminar los primeros 28 días sin historial suficiente
    df_features = df_features.dropna(subset=feature_cols)

    # 4. División cronológica
    df_train = df_features[
        df_features["fecha"] < "2026-07-01"
    ].copy()

    df_val = df_features[
        (df_features["fecha"] >= "2026-07-01")
        & (df_features["fecha"] < "2026-09-01")
    ].copy()

    df_test = df_features[
        df_features["fecha"] >= "2026-09-01"
    ].copy()

    logger.info(
        f"División temporal: train={len(df_train):,}, "
        f"val={len(df_val):,}, test={len(df_test):,}."
    )

    # 5. Definir características y variable objetivo
    X_train = df_train[feature_cols]
    y_train = df_train["viajes_iniciados"]

    X_val = df_val[feature_cols]
    y_val = df_val["viajes_iniciados"]

    # 6. Entrenar o ejecutar el modelo seleccionado
    model = None

    if modelo_tipo == "baseline":
        # Predicción: demanda observada siete días antes
        y_pred = df_val["viajes_hace_7_dias"].to_numpy()

    elif modelo_tipo == "random_forest":
        logger.info(
            f"Entrenando Random Forest: árboles=100, "
            f"profundidad=12, min_samples_leaf=5."
        )

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # La demanda no puede ser negativa
        y_pred = np.maximum(y_pred, 0)

    else:
        raise ValueError(
            f"Tipo de modelo no soportado: {modelo_tipo}. "
            "Utiliza 'baseline' o 'random_forest'."
        )

    # 7. Evaluación sobre validación
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    logger.info(
        f"Resultados P1 {modelo_tipo} (validación): "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}"
    )

    # 8. Configuración de MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    git_commit = obtener_git_commit()
    data_version = cargar_manifest_version()

    run_name = f"P1_{modelo_tipo}"

    with mlflow.start_run(run_name=run_name):

        # a) Etiquetas
        mlflow.set_tag("equipo", TEAM_NAME)
        mlflow.set_tag("pregunta", "P1")
        mlflow.set_tag("tarea", "regresion")
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("modelo_familia", modelo_tipo)

        # b) Parámetros
        mlflow.log_param("modelo_tipo", modelo_tipo)
        mlflow.log_param("features_list", ",".join(feature_cols))

        if modelo_tipo == "baseline":
            mlflow.log_param("baseline_lag_dias", 7)

        elif modelo_tipo == "random_forest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 12)
            mlflow.log_param("min_samples_leaf", 5)
            mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("cobertura_minima", cobertura_minima)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("fecha_inicio_train", "2025-10-29")
        mlflow.log_param("fecha_fin_train", "2026-06-30")
        mlflow.log_param("fecha_inicio_val", "2026-07-01")
        mlflow.log_param("fecha_fin_val", "2026-08-31")
        mlflow.log_param("fecha_fin_train_final", "2026-08-31")
        mlflow.log_param("fecha_inicio_test", "2026-09-01")
        mlflow.log_param("fecha_fin_test", "2026-09-30")
        mlflow.log_param("n_estaciones", df_features["estacion_id"].nunique())
        mlflow.log_param("n_train", len(df_train))
        mlflow.log_param("n_val", len(df_val))
        mlflow.log_param("n_test_reservado", len(df_test))

        mlflow.log_metric("val_mae", float(mae))
        mlflow.log_metric("val_rmse", float(rmse))

        # Generar y registrar visualizaciones de Random Forest
        if modelo_tipo == "random_forest":

            # 1. Valores reales vs. predichos
            reales_predichos_path = (
                FIGURES_DIR / f"{run_name}_reales_vs_predichos.png"
            )

            graficar_reales_vs_predichos_p1(
                y_real=y_val.to_numpy(),
                y_pred=y_pred,
                output_path=reales_predichos_path,
            )

            mlflow.log_artifact(
                str(reales_predichos_path),
                artifact_path="figures",
            )

            # 2. Distribución de errores
            errores_path = (
                FIGURES_DIR / f"{run_name}_distribucion_errores.png"
            )

            graficar_distribucion_errores_p1(
                y_real=y_val.to_numpy(),
                y_pred=y_pred,
                output_path=errores_path,
            )

            mlflow.log_artifact(
                str(errores_path),
                artifact_path="figures",
            )

            # 3. Importancia de características
            importancia_path = (
                FIGURES_DIR / f"{run_name}_importancia_caracteristicas.png"
            )

            graficar_importancia_caracteristicas_p1(
                model=model,
                feature_cols=feature_cols,
                output_path=importancia_path,
            )

            mlflow.log_artifact(
                str(importancia_path),
                artifact_path="figures",
            )

        # Evaluación final sobre septiembre
        df_train_final = pd.concat([df_train, df_val])

        if modelo_tipo == "baseline":
            y_pred_test = df_test["viajes_hace_7_dias"].to_numpy()

        else:
            model.fit(
                df_train_final[feature_cols],
                df_train_final["viajes_iniciados"],
            )

            y_pred_test = np.maximum(
                model.predict(df_test[feature_cols]),
                0,
            )

        test_mae = mean_absolute_error(
            df_test["viajes_iniciados"],
            y_pred_test,
        )

        test_rmse = np.sqrt(
            mean_squared_error(
                df_test["viajes_iniciados"],
                y_pred_test,
            )
        )

        logger.info(
            f"Resultados P1 {modelo_tipo} (prueba septiembre): "
            f"MAE={test_mae:.4f}, RMSE={test_rmse:.4f}"
        )

        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("test_rmse", float(test_rmse))

        # Guardar modelo entrenado cuando corresponda
        if model is not None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            model_file = MODELS_DIR / f"{run_name}.joblib"

            joblib.dump(
                {
                    "model": model,
                    "feature_cols": feature_cols,
                },
                model_file,
            )

            mlflow.log_artifact(
                str(model_file),
                artifact_path="models",
            )

        logger.success(
            f"Run de MLflow finalizado exitosamente: {run_name}"
        )

if __name__ == "__main__":
    app()
