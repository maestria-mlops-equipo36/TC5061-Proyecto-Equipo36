from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer(help="Generación de características (Feature Engineering) para MiBici.")


def extraer_caracteristicas_temporales(df_viajes: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega atributos temporales esenciales al registro de viajes.

    Parámetros:
    -----------
    df_viajes : pd.DataFrame
        DataFrame con registros de viajes que contiene 'Inicio_del_viaje'.

    Retorna:
    --------
    pd.DataFrame con columnas: 'hora', 'dia_semana', 'es_fin_de_semana',
    'pico_matutino', 'pico_vespertino'.
    """
    df = df_viajes.copy()
    hora = df["Inicio_del_viaje"].dt.hour
    dia_semana = df["Inicio_del_viaje"].dt.dayofweek  # 0=Lunes, 6=Domingo

    df["hora"] = hora
    df["es_fin_de_semana"] = dia_semana.isin([5, 6]).astype(int)
    # Picos típicos de movilidad laboral/escolar en GDL
    df["pico_matutino"] = hora.between(7, 10).astype(int)
    df["pico_vespertino"] = hora.between(17, 20).astype(int)
    return df


def calcular_perfil_estaciones_p3(
    df_viajes: pd.DataFrame, df_estaciones: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye las características consolidadas por estación para la Pregunta 3 (Clustering).

    Cada fila representa una estación y describe cuantitativamente sus patrones de uso,
    sin fuga temporal de información posterior a la ventana observada.

    Características calculadas:
    ---------------------------
    - salidas_totales: Número total de viajes originados en la estación.
    - llegadas_totales: Número total de viajes concluidos en la estación.
    - total_viajes: Suma de viajes (salidas + llegadas).
    - ratio_flujo_neto: (salidas - llegadas) / total_viajes (-1 indica atractor, +1 emisor).
    - pct_salidas_pico_matutino: Proporción de salidas en horario pico de la mañana (07:00-10:00).
    - pct_salidas_pico_vespertino: Proporción de salidas en horario pico de la tarde (17:00-20:00).
    - pct_salidas_fin_de_semana: Proporción de salidas en sábado o domingo.
    - duracion_promedio_salidas_min: Duración media de los viajes originados en la estación.
    - viajes_diarios_por_anclaje: Intensidad promedio de uso normalizada por capacidad de la estación.
    - latitud, longitud, capacidad: Atributos físicos y de ubicación.

    Parámetros:
    -----------
    df_viajes : pd.DataFrame
        Datos limpios de viajes consolidados.
    df_estaciones : pd.DataFrame
        Catálogo de estaciones con 'estacion_id', 'nombre', 'latitud', 'longitud', 'capacidad'.

    Retorna:
    --------
    pd.DataFrame con el vector de características por estación.
    """
    logger.info("Enriqueciendo viajes con componentes temporales...")
    df = extraer_caracteristicas_temporales(df_viajes)

    dias_totales = max(1, (df["Inicio_del_viaje"].max() - df["Inicio_del_viaje"].min()).days)
    logger.info(f"Ventana temporal observada: {dias_totales} días.")

    # 1. Agregaciones de Salidas (Origen)
    logger.info("Calculando agregaciones de salidas (origen)...")
    salidas_grp = df.groupby("Origen_Id")
    df_salidas = salidas_grp.agg(
        salidas_totales=("Viaje_Id", "count"),
        salidas_pico_matutino=("pico_matutino", "sum"),
        salidas_pico_vespertino=("pico_vespertino", "sum"),
        salidas_fin_semana=("es_fin_de_semana", "sum"),
        duracion_promedio_salidas_min=("duracion_min", "mean"),
    ).reset_index().rename(columns={"Origen_Id": "estacion_id"})

    # Proporciones relativas
    df_salidas["pct_salidas_pico_matutino"] = (
        df_salidas["salidas_pico_matutino"] / df_salidas["salidas_totales"]
    ).fillna(0.0)
    df_salidas["pct_salidas_pico_vespertino"] = (
        df_salidas["salidas_pico_vespertino"] / df_salidas["salidas_totales"]
    ).fillna(0.0)
    df_salidas["pct_salidas_fin_de_semana"] = (
        df_salidas["salidas_fin_semana"] / df_salidas["salidas_totales"]
    ).fillna(0.0)

    # 2. Agregaciones de Llegadas (Destino)
    logger.info("Calculando agregaciones de llegadas (destino)...")
    llegadas_grp = df.groupby("Destino_Id")
    df_llegadas = llegadas_grp.agg(
        llegadas_totales=("Viaje_Id", "count"),
        llegadas_pico_matutino=("pico_matutino", "sum"),
        llegadas_pico_vespertino=("pico_vespertino", "sum"),
    ).reset_index().rename(columns={"Destino_Id": "estacion_id"})

    # 3. Cruzar con el catálogo oficial de estaciones
    logger.info("Consolidando características con el catálogo de estaciones...")
    df_features = df_estaciones.merge(df_salidas, on="estacion_id", how="left")
    df_features = df_features.merge(df_llegadas, on="estacion_id", how="left")

    # Llenar nulos para estaciones recién instaladas o con 0 viajes registrados
    df_features["salidas_totales"] = df_features["salidas_totales"].fillna(0).astype(int)
    df_features["llegadas_totales"] = df_features["llegadas_totales"].fillna(0).astype(int)
    df_features["pct_salidas_pico_matutino"] = df_features["pct_salidas_pico_matutino"].fillna(0.0)
    df_features["pct_salidas_pico_vespertino"] = df_features["pct_salidas_pico_vespertino"].fillna(0.0)
    df_features["pct_salidas_fin_de_semana"] = df_features["pct_salidas_fin_de_semana"].fillna(0.0)
    df_features["duracion_promedio_salidas_min"] = df_features["duracion_promedio_salidas_min"].fillna(
        df_features["duracion_promedio_salidas_min"].median()
    )

    # Métrica de flujo neto y total
    df_features["total_viajes"] = df_features["salidas_totales"] + df_features["llegadas_totales"]
    # Ratio flujo neto: >0 estación emisora (neta de origen), <0 estación receptora (neta de destino)
    df_features["ratio_flujo_neto"] = np.where(
        df_features["total_viajes"] > 0,
        (df_features["salidas_totales"] - df_features["llegadas_totales"]) / df_features["total_viajes"],
        0.0,
    )

    # Viajes diarios por anclaje (intensidad / rotación)
    capacidad = df_features["capacidad"].replace(0, np.nan)
    df_features["viajes_diarios_por_anclaje"] = (
        (df_features["total_viajes"] / dias_totales) / capacidad
    ).fillna(0.0)

    # Seleccionar y ordenar columnas finales
    columnas_finales = [
        "estacion_id",
        "nombre",
        "latitud",
        "longitud",
        "capacidad",
        "salidas_totales",
        "llegadas_totales",
        "total_viajes",
        "ratio_flujo_neto",
        "pct_salidas_pico_matutino",
        "pct_salidas_pico_vespertino",
        "pct_salidas_fin_de_semana",
        "duracion_promedio_salidas_min",
        "viajes_diarios_por_anclaje",
    ]

    return df_features[columnas_finales]


@app.command()
def main(
    viajes_path: Path = PROCESSED_DATA_DIR / "viajes_limpios.parquet",
    estaciones_path: Path = PROCESSED_DATA_DIR / "estaciones.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet",
):
    """
    Punto de entrada de CLI para generar la tabla de características de estaciones (P3).
    """
    logger.info(f"Cargando viajes desde: {viajes_path}")
    if not viajes_path.exists():
        logger.error(f"No existe el archivo de viajes: {viajes_path}. Ejecuta primero dataset.py")
        raise typer.Exit(code=1)

    df_viajes = pd.read_parquet(viajes_path)
    df_estaciones = pd.read_parquet(estaciones_path)

    logger.info(f"Generando características para {len(df_estaciones)} estaciones...")
    df_features = calcular_perfil_estaciones_p3(df_viajes, df_estaciones)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    # También guardar una copia en CSV para inspección rápida o reporte
    df_features.to_csv(output_path.with_suffix(".csv"), index=False)

    logger.success(f"Características guardadas exitosamente en: {output_path}")
    logger.info(f"Dimensiones de la matriz de características: {df_features.shape}")
    logger.info("\nResumen estadístico de las características clave:\n" + 
                df_features[["ratio_flujo_neto", "pct_salidas_pico_matutino", "pct_salidas_pico_vespertino", "pct_salidas_fin_de_semana", "viajes_diarios_por_anclaje"]].describe().to_string())


if __name__ == "__main__":
    app()
