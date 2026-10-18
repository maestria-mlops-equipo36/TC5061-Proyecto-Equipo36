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
