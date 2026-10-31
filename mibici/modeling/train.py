import os
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from contextlib import nullcontext
from mlflow.models import infer_signature
from tempfile import TemporaryDirectory

import typer

from mibici.config import (
    DATA_MANIFEST_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P3_DEFAULT_COVARIANCE_TYPE,
    P3_DEFAULT_INCLUIR_ESPACIAL,
    P3_DEFAULT_MODEL,
    P3_DEFAULT_N_CLUSTERS,
    P3_FEATURE_COLS_BASE,
    P3_FEATURES_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEAM_NAME,
)
from mibici.features import (
    calcular_demanda_diaria_p1,
    preparar_demanda_diaria_p1,
    generar_caracteristicas_regresion_p1,
)
from mibici.modeling.baseline_reglas import (
    asegurar_features_p3,
    ejecutar_baseline_reglas_p3,
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
    features_path: Path = P3_FEATURES_PATH,
    modelo_tipo: str = P3_DEFAULT_MODEL,
    n_clusters: int = P3_DEFAULT_N_CLUSTERS,
    covariance_type: str = P3_DEFAULT_COVARIANCE_TYPE,
    random_seed: int = RANDOM_SEED,
    incluir_espacial: bool = P3_DEFAULT_INCLUIR_ESPACIAL,
    stage_of_work: Optional[str] = None,
):
    """
    Ejecuta el experimento de clustering para la Pregunta 3 (P3) y lo registra en MLFlow Tracking.
    """
    if modelo_tipo in ("baseline", "reglas", "reglas_intensidad"):
        return ejecutar_baseline_reglas_p3(
            features_path=features_path,
            stage_of_work=stage_of_work or "baseline",
        )

    logger.info(
        f"Iniciando experimento P3: modelo={modelo_tipo}, clusters={n_clusters}, "
        f"covariance_type={covariance_type}, semilla={random_seed}"
    )

    try:
        features_path = asegurar_features_p3(features_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1)

    df_features = pd.read_parquet(features_path)
    logger.info(f"Cargadas {len(df_features)} estaciones.")

    # Definir variables utilizadas para el perfilamiento
    feature_cols = list(P3_FEATURE_COLS_BASE)
    if incluir_espacial:
        feature_cols.extend(["latitud", "longitud"])

    X = df_features[feature_cols].astype(float).copy()

    # Preprocesamiento: Escalado estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configuración de MLFlow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    git_commit = obtener_git_commit()
    data_version = cargar_manifest_version()

    sufijo_espacial = "espacial" if incluir_espacial else "temporal"
    if modelo_tipo == "gmm" and covariance_type != "full":
        run_name = f"P3_{modelo_tipo}_{covariance_type}_k{n_clusters}_{sufijo_espacial}"
    else:
        run_name = f"P3_{modelo_tipo}_k{n_clusters}_{sufijo_espacial}"

    if stage_of_work is None:
        if modelo_tipo == "kmeans" and n_clusters == 3 and incluir_espacial:
            etapa = "baseline"
        else:
            etapa = "candidate"
    else:
        etapa = stage_of_work

    run_activo = mlflow.active_run()
    if run_activo is not None:
        contexto_run = nullcontext(run_activo)
    else:
        env_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        contexto_run = mlflow.start_run(
            run_id=env_run_id,
            run_name=run_name,
        )

    with contexto_run:
        # 1. Registrar Etiquetas (Tags)
        mlflow.set_tags({
            "mlflow.runName": run_name,
            "equipo": TEAM_NAME,
            "pregunta": "P3",
            "question": "p3",
            "tarea": "clustering",
            "git_commit": git_commit,
            "commit": git_commit,
            "modelo_familia": modelo_tipo,
            "model_family": modelo_tipo,
            "stage_of_work": etapa,
            "data_version": data_version,
        })

        # 2. Registrar Parámetros (Params)
        params_corrida = {
            "modelo_tipo": modelo_tipo,
            "n_clusters": n_clusters,
            "random_seed": random_seed,
            "data_version": data_version,
            "incluir_espacial": incluir_espacial,
            "features_list": ",".join(feature_cols),
            "n_estaciones": len(df_features),
        }
        if modelo_tipo == "gmm":
            params_corrida["covariance_type"] = covariance_type

        for k_param, v_param in params_corrida.items():
            try:
                mlflow.log_param(k_param, v_param)
            except Exception:
                pass

        # 3. Ajuste del Modelo y Pipeline (Scaler + Modelo)
        if modelo_tipo == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_seed,
                n_init=10,
                max_iter=300,
            )
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            labels = pipeline.fit_predict(X)
            mlflow.log_metric("inertia", float(pipeline.named_steps["model"].inertia_))
        elif modelo_tipo == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_seed,
                covariance_type=covariance_type,
            )
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipeline.fit(X)
            labels = pipeline.predict(X)
            gmm_fitted = pipeline.named_steps["model"]
            X_scaled_pipe = pipeline.named_steps["scaler"].transform(X)
            mlflow.log_metric("bic", float(gmm_fitted.bic(X_scaled_pipe)))
            mlflow.log_metric("aic", float(gmm_fitted.aic(X_scaled_pipe)))
        else:
            raise ValueError(f"Tipo de modelo no soportado: {modelo_tipo}")

        # 4. Cálculo y Registro de Métricas Intrínsecas de Clustering
        sil_score = float(silhouette_score(X_scaled, labels))
        db_score = float(davies_bouldin_score(X_scaled, labels))
        ch_score = float(calinski_harabasz_score(X_scaled, labels))

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        tamanos_grupos = {}
        for grupo in range(n_clusters):
            conteo = int((labels == grupo).sum())
            tamanos_grupos[f"n_estaciones_grupo_{grupo}"] = conteo
            mlflow.log_metric(f"n_estaciones_grupo_{grupo}", conteo)

        mlflow.log_metric("min_cluster_size", int(min(tamanos_grupos.values())))
        mlflow.log_metric("max_cluster_size", int(max(tamanos_grupos.values())))

        logger.info(
            f"Resultados de Métricas: Silhouette={sil_score:.4f}, "
            f"Davies-Bouldin={db_score:.4f}, Calinski-Harabasz={ch_score:.2f}"
        )

        # 5. Generar y Registrar Artefactos (Punto 5 de la entrega)
        # a) Registrar dataset de entrenamiento con mlflow.log_input
        try:
            dataset_input = mlflow.data.from_pandas(
                df_features,
                source=str(features_path),
                name="features_p3_estaciones",
            )
            mlflow.log_input(dataset_input, context="training")
        except Exception as exc:
            logger.warning(f"No se pudo registrar mlflow.log_input: {exc}")

        df_eval = df_features.copy()
        df_eval["cluster"] = labels

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        # b) Gráfica geográfica
        mapa_path = FIGURES_DIR / f"{run_name}_mapa.png"
        graficar_clusters_mapa(
            df_eval,
            columna_cluster="cluster",
            output_path=mapa_path,
            titulo=f"Distribución de Clusters ({modelo_tipo.upper()}, k={n_clusters})",
        )
        mlflow.log_artifact(str(mapa_path), artifact_path="plots")
        mlflow.log_artifact(str(mapa_path), artifact_path="figures")

        # c) Gráfica de perfil
        perfil_path = FIGURES_DIR / f"{run_name}_perfil.png"
        graficar_perfil_clusters(
            df_eval,
            columnas_features=feature_cols,
            columna_cluster="cluster",
            output_path=perfil_path,
            titulo=f"Perfil de Comportamiento ({modelo_tipo.upper()}, k={n_clusters})",
        )
        mlflow.log_artifact(str(perfil_path), artifact_path="plots")
        mlflow.log_artifact(str(perfil_path), artifact_path="figures")

        # d) Tabla de asignación de estaciones y manifiesto de versión de datos
        clusters_csv_path = PROCESSED_DATA_DIR / f"{run_name}_estaciones_clusters.csv"
        df_eval.to_csv(clusters_csv_path, index=False)
        mlflow.log_artifact(str(clusters_csv_path), artifact_path="data")
        if DATA_MANIFEST_PATH.exists():
            mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")

        # e) Archivo de configuración de la corrida en subcarpeta config/
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / f"{run_name}_config.json"
            config_payload = {
                "run_name": run_name,
                "stage_of_work": etapa,
                "git_commit": git_commit,
                "params": params_corrida,
                "metrics": {
                    "silhouette_score": sil_score,
                    "davies_bouldin_score": db_score,
                    "calinski_harabasz_score": ch_score,
                    **tamanos_grupos,
                },
            }
            config_path.write_text(
                json.dumps(config_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(config_path), artifact_path="config")

        # f) Registrar el modelo nativo en MLflow con firma (signature) e input_example
        input_example = X.iloc[:5].copy()
        signature = infer_signature(X, labels)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # g) Guardar modelo serializado localmente (ignorado por Git)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file = MODELS_DIR / f"{run_name}.joblib"
        joblib.dump(
            {
                "pipeline": pipeline,
                "model": pipeline.named_steps["model"],
                "scaler": pipeline.named_steps["scaler"],
                "feature_cols": feature_cols,
            },
            model_file,
        )

        logger.success(f"Run de MLFlow finalizado exitosamente: {run_name}")

def entrenar_random_forest_p1(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_seed=RANDOM_SEED,
):
    """
    Entrena un Random Forest para P1 con una semilla controlada.

    Permite ejecutar pruebas de reproducibilidad sin depender
    de MLflow ni de los archivos históricos de MiBici.
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


@app.command()
def p1_regresion(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes_limpios.parquet",
    cobertura_minima: float = 0.95,
    modelo_tipo: str = "baseline",
    n_estimators: int = 100,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
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
            f"Entrenando Random Forest: árboles={n_estimators}, "
            f"profundidad={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}."
        )

        model = entrenar_random_forest_p1(
            X_train=X_train,
            y_train=y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

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

    run_activo = mlflow.active_run()

    if run_activo is not None:
        contexto_run = nullcontext(run_activo)
    else:
        contexto_run = mlflow.start_run(
            run_id=os.getenv("MLFLOW_RUN_ID"),
            run_name=run_name,
        )

    with contexto_run:
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
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
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

            # Registrar modelo P1 en formato nativo de MLflow
            input_example = df_train_final[feature_cols].head(5).copy()

            signature = infer_signature(
                input_example,
                model.predict(input_example),
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="modelo_p1_random_forest",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            logger.success(
                "Modelo P1 registrado en MLflow con signature e input_example."
            )

        logger.success(
            f"Run de MLflow finalizado exitosamente: {run_name}"
        )

if __name__ == "__main__":
    app()
