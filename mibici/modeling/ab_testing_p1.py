from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mibici.config import PROCESSED_DATA_DIR


MODEL_NAME = "mibici-equipo36-p1-demanda"

CURRENT_DATA_PATH = PROCESSED_DATA_DIR / "regression_current.parquet"

FEATURES = [
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

TARGET = "viajes_iniciados"

def cargar_datos_evaluacion(data_path: Path = CURRENT_DATA_PATH):

    """Carga y valida los datos del periodo actual para el A/B testing."""

    df = pd.read_parquet(data_path)

    columnas_requeridas = ["fecha", *FEATURES, TARGET]

    # Verificar que existan todas las columnas necesarias
    faltantes = set(columnas_requeridas) - set(df.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    # Verificar que el conjunto de evaluación no esté vacío
    if df.empty:
        raise ValueError("El conjunto de evaluación está vacío.")

    # Verificar que no existan valores faltantes
    if df[columnas_requeridas].isna().any().any():
        raise ValueError("El conjunto de evaluación contiene valores faltantes.")

    # Verificar que no existan registros duplicados
    if df.duplicated(subset=["fecha", "estacion_id"]).any():
        raise ValueError("Existen registros duplicados por estación y fecha.")



    # Separar las características y la variable objetivo
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return df, X, y



def cargar_modelos():
    """Carga modelos y conserva las versiones exactas utilizadas."""
    modelos = {}
    versiones = {}
    client = mlflow.tracking.MlflowClient()

    for alias in ["champion", "candidate"]:
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
        versiones[alias] = version.version

        model_uri = f"models:/{MODEL_NAME}/{version.version}"
        modelos[alias] = mlflow.pyfunc.load_model(model_uri)

        print(f"Modelo {alias} cargado (versión {version.version}).")

    return modelos, versiones


def evaluar_modelo(modelo, X, y):
    """Calcula las métricas de regresión sobre el conjunto de evaluación."""

    # Generar predicciones y evitar valores negativos
    predicciones = np.maximum(modelo.predict(X), 0)

    # Calcular métricas
    mae = mean_absolute_error(y, predicciones)
    rmse = np.sqrt(mean_squared_error(y, predicciones))
    r2 = r2_score(y, predicciones)

    metricas = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return metricas, predicciones



def comparar_modelos():
    """Compara champion y candidate sobre los mismos registros."""
    df, X, y = cargar_datos_evaluacion()
    modelos, versiones = cargar_modelos()

    resultados = {}
    predicciones = {}

    for alias, modelo in modelos.items():
        metricas, pred = evaluar_modelo(modelo, X, y)
        resultados[alias] = metricas
        predicciones[alias] = pred

    comparacion = pd.DataFrame(resultados).T

    comparacion.loc["diferencia_absoluta"] = (
        comparacion.loc["candidate"] - comparacion.loc["champion"]
    )

    comparacion.loc["diferencia_porcentual"] = (
        comparacion.loc["diferencia_absoluta"]
        / comparacion.loc["champion"].replace(0, np.nan)
        * 100
    )

    return comparacion, predicciones, df, versiones


def recomendar_modelo(comparacion):
    """Recomienda qué modelo conservar según las métricas del A/B testing."""

    champion = comparacion.loc["champion"]
    candidate = comparacion.loc["candidate"]

    # Comparar las métricas principales
    mejora_mae = candidate["mae"] < champion["mae"]
    mejora_rmse = candidate["rmse"] < champion["rmse"]

    if mejora_mae and mejora_rmse:
        recomendado = "candidate"
        justificacion = (
            "El modelo candidate presenta menores valores de MAE y RMSE "
            "sobre el conjunto de evaluación actual. Por lo tanto, "
            "se recomienda considerarlo como nueva versión vigente."
        )
    else:
        recomendado = "champion"
        justificacion = (
            "El modelo candidate no demuestra una mejora consistente "
            "en MAE y RMSE respecto al modelo champion. Por lo tanto, "
            "se recomienda conservar la versión vigente."
        )

    return recomendado, justificacion


def graficar_comparacion(comparacion, output_dir):
    """Genera una gráfica comparativa de las métricas del A/B testing."""

    modelos = ["champion", "candidate"]
    metricas = ["mae", "rmse", "r2"]
    titulos = ["MAE (menor es mejor)", "RMSE (menor es mejor)", "R² (mayor es mejor)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metrica, titulo in zip(axes, metricas, titulos):
        valores = comparacion.loc[modelos, metrica]

        barras = ax.bar(modelos, valores, color=["#4C78A8", "#F58518"])

        ax.set_title(titulo)
        ax.set_ylabel("Valor")
        ax.bar_label(barras, fmt="%.4f", padding=3)

        # Mantener la escala desde cero para evitar exagerar diferencias pequeñas
        ax.set_ylim(0, max(valores) * 1.15)

    fig.suptitle("A/B Testing P1 - Septiembre 2026", fontsize=14)
    fig.tight_layout()

    output_path = output_dir / "comparacion_metricas_p1.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfica guardada en: {output_path}")


def registrar_experimento(comparacion, recomendado, output_dir, df, versiones):
    """Registra los resultados del A/B testing en MLflow."""

    mlflow.set_experiment("mibici-equipo36-ab-testing-p1")

    with mlflow.start_run(run_name="ab-testing-champion-vs-candidate"):

        # Registrar información del experimento
        mlflow.log_param("modelo", MODEL_NAME)
        for alias, version in versiones.items():
            mlflow.log_param(f"{alias}_version", version)

        mlflow.log_param("periodo_inicio", str(df["fecha"].min().date()))
        mlflow.log_param("periodo_fin", str(df["fecha"].max().date()))
        mlflow.log_param("registros_evaluados", len(df))

        # Registrar métricas de ambos modelos
        for alias in ["champion", "candidate"]:
            for metrica in ["mae", "rmse", "r2"]:
                mlflow.log_metric(
                    f"{alias}_{metrica}",
                    comparacion.loc[alias, metrica],
                )

        # Registrar diferencias porcentuales
        mlflow.log_metric(
            "mejora_mae_pct",
            -comparacion.loc["diferencia_porcentual", "mae"],
        )

        mlflow.log_metric(
            "mejora_rmse_pct",
            -comparacion.loc["diferencia_porcentual", "rmse"],
        )

        # Registrar recomendación
        mlflow.set_tag("modelo_recomendado", recomendado)

        # Guardar evidencias
        mlflow.log_artifacts(str(output_dir), artifact_path="ab_testing")

    print("Experimento registrado correctamente en MLflow.")


def main():
    """Ejecuta el A/B testing de los modelos de regresión P1."""

    print("\nIniciando A/B testing P1...")

    # Comparar ambos modelos
    comparacion, predicciones, df, versiones = comparar_modelos()

    # Guardar resultados de la comparación A/B
    output_dir = Path("reports/ab_testing")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparacion.to_csv(output_dir / "comparacion_modelos_p1.csv")
    graficar_comparacion(comparacion, output_dir)

    # Mostrar resultados
    print("\nResultados del A/B testing:")
    print(comparacion.round(4).to_string())

    # Obtener recomendación
    recomendado, justificacion = recomendar_modelo(comparacion)

    print(f"\nModelo recomendado: {recomendado}")
    print(f"Justificación: {justificacion}")

    # Registrar resultados y evidencias en MLflow
    registrar_experimento(comparacion, recomendado, output_dir, df, versiones)

    print("\nA/B testing finalizado correctamente.")


if __name__ == "__main__":
    main()
