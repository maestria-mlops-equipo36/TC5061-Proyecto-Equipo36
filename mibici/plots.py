from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from mibici.config import FIGURES_DIR


def graficar_clusters_mapa(
    df: pd.DataFrame,
    columna_cluster: str = "cluster",
    output_path: Optional[Path] = None,
    titulo: str = "Distribución Espacial de Clusters de Estaciones MiBici",
) -> Path:
    """
    Genera un mapa de dispersión (scatter plot geográfico) de las estaciones
    coloreadas según su asignación de cluster.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "p3_clusters_mapa.png"

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("tab10", n_colors=df[columna_cluster].nunique())
    
    scatter = sns.scatterplot(
        data=df,
        x="longitud",
        y="latitud",
        hue=columna_cluster,
        palette=palette,
        s=60,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.title(titulo, fontsize=14, weight="bold", pad=12)
    plt.xlabel("Longitud", fontsize=11)
    plt.ylabel("Latitud", fontsize=11)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Gráfica de mapa de clusters guardada en: {output_path}")
    return output_path


def graficar_perfil_clusters(
    df: pd.DataFrame,
    columnas_features: list,
    columna_cluster: str = "cluster",
    output_path: Optional[Path] = None,
    titulo: str = "Perfil Promedio por Cluster de Estaciones",
) -> Path:
    """
    Genera un mapa de calor o gráfico de barras comparativo de las características
    promedio de cada cluster.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "p3_perfil_clusters.png"

    # Calcular promedios por cluster
    perfil = df.groupby(columna_cluster)[columnas_features].mean()

    # Normalizar entre 0 y 1 por columna para visualización clara en heatmap
    perfil_norm = (perfil - perfil.min()) / (perfil.max() - perfil.min() + 1e-6)

    plt.figure(figsize=(11, max(4, len(perfil) * 0.9 + 1)))
    sns.heatmap(
        perfil_norm,
        annot=perfil.round(2),
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Escala relativa (min-max)"},
    )
    plt.title(titulo, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Características de Comportamiento", fontsize=11)
    plt.ylabel("Cluster", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Gráfica de perfil de clusters guardada en: {output_path}")
    return output_path


def graficar_reales_vs_predichos_p1(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Genera una gráfica de dispersión para comparar la demanda real
    contra la demanda predicha por el modelo de regresión P1.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = FIGURES_DIR / "p1_reales_vs_predichos.png"

    plt.figure(figsize=(9, 7))

    sns.scatterplot(
        x=y_real,
        y=y_pred,
        alpha=0.25,
        s=20,
        color="steelblue",
        edgecolor=None,
    )

    # Línea de predicción perfecta
    limite = max(np.max(y_real), np.max(y_pred))

    plt.plot(
        [0, limite],
        [0, limite],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Predicción perfecta",
    )

    plt.title(
        "P1 - Demanda real vs. predicha (Random Forest)",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    plt.xlabel("Viajes reales por estación y día")
    plt.ylabel("Viajes predichos por estación y día")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Gráfica de valores reales vs. predichos guardada en: {output_path}")

    return output_path

def graficar_distribucion_errores_p1(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Genera un histograma de los errores de predicción del modelo P1.

    Error = demanda real - demanda predicha.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = FIGURES_DIR / "p1_distribucion_errores.png"

    errores = np.asarray(y_real) - np.asarray(y_pred)

    plt.figure(figsize=(10, 6))

    sns.histplot(
        errores,
        bins=60,
        kde=True,
        color="steelblue",
    )

    plt.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Error cero",
    )

    plt.title(
        "P1 - Distribución de errores (Random Forest)",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    plt.xlabel("Error (viajes reales - viajes predichos)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Gráfica de distribución de errores guardada en: {output_path}")

    return output_path

def graficar_importancia_caracteristicas_p1(
    model,
    feature_cols: list,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Genera una gráfica de barras con la importancia de las
    características utilizadas por Random Forest en P1.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = FIGURES_DIR / "p1_importancia_caracteristicas.png"

    # Obtener importancia de cada característica
    importancias = pd.DataFrame({
        "caracteristica": feature_cols,
        "importancia": model.feature_importances_,
    })

    # Ordenar de mayor a menor importancia
    importancias = importancias.sort_values(
        "importancia",
        ascending=False,
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=importancias,
        x="importancia",
        y="caracteristica",
        color="steelblue",
    )

    plt.title(
        "P1 - Importancia de características (Random Forest)",
        fontsize=14,
        weight="bold",
        pad=12,
    )

    plt.xlabel("Importancia relativa")
    plt.ylabel("Característica")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(
        f"Gráfica de importancia de características guardada en: {output_path}"
    )

    return output_path
