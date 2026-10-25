"""Referencia adicional para P3: agrupación por intensidad de uso."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    DATA_MANIFEST_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    P3_FEATURE_COLS_ESPACIAL,
    P3_FEATURES_PATH,
    PROCESSED_DATA_DIR,
    TEAM_NAME,
)
from mibici.features import calcular_perfil_estaciones_p3
from mibici.plots import graficar_clusters_mapa, graficar_perfil_clusters

FEATURE_COLS = P3_FEATURE_COLS_ESPACIAL


def asegurar_features_p3(features_path: Path = P3_FEATURES_PATH) -> Path:
    """Genera el archivo de características de P3 si aún no existe tras la ingesta."""
    if features_path.exists():
        return features_path

    viajes_path = PROCESSED_DATA_DIR / "viajes_limpios.parquet"
    estaciones_path = PROCESSED_DATA_DIR / "estaciones.parquet"
    if not viajes_path.exists() or not estaciones_path.exists():
        raise FileNotFoundError(
            f"No existe {features_path} ni los insumos de ingesta ({viajes_path}). "
            "Ejecutar primero el punto de entrada 'ingest' (mibici.dataset)."
        )

    logger.info(
        f"Generando automáticamente {features_path} a partir de viajes_limpios.parquet..."
    )
    df_viajes = pd.read_parquet(viajes_path)
    df_estaciones = pd.read_parquet(estaciones_path)
    df_features = calcular_perfil_estaciones_p3(df_viajes, df_estaciones)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(features_path, index=False)
    df_features.to_csv(features_path.with_suffix(".csv"), index=False)
    return features_path


def ejecutar_baseline_reglas_p3(
    features_path: Path = P3_FEATURES_PATH,
    stage_of_work: str = "baseline",
) -> dict:
    """Ejecuta el baseline basado en reglas de intensidad para P3 y lo registra en MLflow."""
    from mibici.modeling.train import cargar_manifest_version, obtener_git_commit

    features_path = asegurar_features_p3(features_path)
    df = pd.read_parquet(features_path)
    X = df[FEATURE_COLS].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Las características contienen valores faltantes o infinitos.")

    intensidad = X["viajes_diarios_por_anclaje"]
    p33, p67 = intensidad.quantile([0.33, 0.67]).to_numpy()

    if p33 >= p67:
        raise ValueError("Los percentiles coinciden; no permiten definir tres niveles.")

    # Los valores iguales a un umbral permanecen en el nivel inferior.
    labels = np.select(
        [intensidad <= p33, intensidad <= p67],
        [0, 1],
        default=2,
    )

    n_grupos = len(np.unique(labels))
    if n_grupos != 3 or len(df) <= n_grupos:
        raise ValueError("Se requieren tres grupos no vacíos y más de tres estaciones.")

    X_scaled = StandardScaler().fit_transform(X)
    metricas = {
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(
            calinski_harabasz_score(X_scaled, labels)
        ),
    }

    df_resultado = df.copy()
    df_resultado["cluster"] = labels
    df_resultado["nivel_uso"] = pd.Series(labels, index=df.index).map(
        {0: "bajo", 1: "medio", 2: "alto"}
    )

    run_name = "P3_reglas_intensidad_k3_espacial"
    git_commit = obtener_git_commit()
    data_version = cargar_manifest_version()

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
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
        mlflow.set_tags({
            "mlflow.runName": run_name,
            "equipo": TEAM_NAME,
            "pregunta": "P3",
            "question": "p3",
            "tarea": "clustering",
            "git_commit": git_commit,
            "commit": git_commit,
            "modelo_familia": "reglas_intensidad",
            "model_family": "reglas_intensidad",
            "stage_of_work": stage_of_work,
            "rol": "referencia_adicional",
            "data_version": data_version,
        })
        params = {
            "modelo_tipo": "reglas_intensidad",
            "n_clusters": n_grupos,
            "n_estaciones": len(df),
            "incluir_espacial": True,
            "features_list": ",".join(FEATURE_COLS),
            "data_version": data_version,
            "percentil_inferior": 0.33,
            "percentil_superior": 0.67,
            "umbral_inferior": round(float(p33), 6),
            "umbral_superior": round(float(p67), 6),
        }
        for k_param, v_param in params.items():
            try:
                mlflow.log_param(k_param, v_param)
            except Exception:
                pass

        mlflow.log_metrics(metricas)

        for grupo in range(3):
            mlflow.log_metric(
                f"n_estaciones_grupo_{grupo}",
                int((labels == grupo).sum()),
            )

        # Registrar dataset de entrada con mlflow.log_input
        try:
            dataset_input = mlflow.data.from_pandas(
                df,
                source=str(features_path),
                name="features_p3_estaciones",
            )
            mlflow.log_input(dataset_input, context="training")
        except Exception as exc:
            logger.warning(f"No se pudo registrar mlflow.log_input: {exc}")

        # Gráficas de mapa y perfiles para comparación visual
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        mapa_path = FIGURES_DIR / f"{run_name}_mapa.png"
        graficar_clusters_mapa(
            df_resultado,
            columna_cluster="cluster",
            output_path=mapa_path,
            titulo="Distribución de Clusters (Reglas de Intensidad, k=3)",
        )
        mlflow.log_artifact(str(mapa_path), artifact_path="plots")
        mlflow.log_artifact(str(mapa_path), artifact_path="figures")

        perfil_path = FIGURES_DIR / f"{run_name}_perfil.png"
        graficar_perfil_clusters(
            df_resultado,
            columnas_features=FEATURE_COLS,
            columna_cluster="cluster",
            output_path=perfil_path,
            titulo="Perfil de Comportamiento (Reglas de Intensidad, k=3)",
        )
        mlflow.log_artifact(str(perfil_path), artifact_path="plots")
        mlflow.log_artifact(str(perfil_path), artifact_path="figures")

        # Artefactos en data/, config/ y referencia_reglas/
        if DATA_MANIFEST_PATH.exists():
            mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")

        with TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            estaciones_csv = carpeta / "estaciones_niveles.csv"
            metricas_json = carpeta / "metricas.json"
            config_json = carpeta / "config_corrida_p3.json"

            df_resultado.to_csv(estaciones_csv, index=False)
            metricas_json.write_text(
                json.dumps(metricas, indent=2),
                encoding="utf-8",
            )
            config_payload = {
                "run_name": run_name,
                "stage_of_work": stage_of_work,
                "git_commit": git_commit,
                "params": params,
                "metrics": metricas,
            }
            config_json.write_text(
                json.dumps(config_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(estaciones_csv), artifact_path="data")
            mlflow.log_artifact(str(metricas_json), artifact_path="data")
            mlflow.log_artifact(str(config_json), artifact_path="config")
            mlflow.log_artifacts(str(carpeta), artifact_path="referencia_reglas")

    logger.info(f"Umbrales de intensidad: {p33:.4f} y {p67:.4f}")
    logger.info(
        "\n" + df_resultado["nivel_uso"].value_counts().to_string()
    )
    logger.success(f"Métricas de la referencia por reglas: {metricas}")
    return metricas


def main():
    ejecutar_baseline_reglas_p3()


if __name__ == "__main__":
    main()
