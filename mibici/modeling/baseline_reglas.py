"""Referencia adicional para P3: agrupación por intensidad de uso."""

import json
from tempfile import TemporaryDirectory
from pathlib import Path

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
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    TEAM_NAME,
)
from mibici.modeling.train import cargar_manifest_version, obtener_git_commit


FEATURE_COLS = [
    "ratio_flujo_neto",
    "pct_salidas_pico_matutino",
    "pct_salidas_pico_vespertino",
    "pct_salidas_fin_de_semana",
    "viajes_diarios_por_anclaje",
    "latitud",
    "longitud",
]


def main():
    features_path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"No existe {features_path}. Ejecutar primero mibici.features."
        )

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

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="P3_reglas_intensidad_k3_espacial"):
        mlflow.set_tags({
            "equipo": TEAM_NAME,
            "pregunta": "P3",
            "tarea": "clustering",
            "git_commit": obtener_git_commit(),
            "modelo_familia": "reglas_intensidad",
            "rol": "referencia_adicional",
        })
        mlflow.log_params({
            "modelo_tipo": "reglas_intensidad",
            "n_clusters": n_grupos,
            "n_estaciones": len(df),
            "incluir_espacial": True,
            "features_list": ",".join(FEATURE_COLS),
            "data_version": cargar_manifest_version(),
            "percentil_inferior": 0.33,
            "percentil_superior": 0.67,
            "umbral_inferior": float(p33),
            "umbral_superior": float(p67),
        })
        mlflow.log_metrics(metricas)

        for grupo in range(3):
            mlflow.log_metric(
                f"n_estaciones_grupo_{grupo}",
                int((labels == grupo).sum()),
            )

        with TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            df_resultado.to_csv(carpeta / "estaciones_niveles.csv", index=False)
            (carpeta / "metricas.json").write_text(
                json.dumps(metricas, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifacts(str(carpeta), artifact_path="referencia_reglas")

    logger.info(f"Umbrales de intensidad: {p33:.4f} y {p67:.4f}")
    logger.info(
        "\n" + df_resultado["nivel_uso"].value_counts().to_string()
    )
    logger.success(f"Métricas de la referencia por reglas: {metricas}")


if __name__ == "__main__":
    main()
