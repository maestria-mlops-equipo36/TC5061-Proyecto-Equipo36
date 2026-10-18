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

def calcular_features_riesgo_p2(
    df_viajes: pd.DataFrame, df_estaciones: pd.DataFrame, freq: str = "2h"
) -> pd.DataFrame:
  """Construye el dataset supervisado para la Pregunta 2 (Clasificación de Riesgo).

  Aproxima el riesgo operativo de desabasto o saturación por estación en
  ventanas temporales, cruzando los flujos netos con la capacidad instalada y
  calculando retardos históricos estrictamente anteriores al horizonte
  predictivo.
  """
  df = df_viajes.copy()
  df["ventana"] = df["Inicio_del_viaje"].dt.floor(freq)
  #Bicicletas OUT
  outBikes=(df.groupby(["ventana", "Origen_Id"]).size().reset_index(name="salidas").rename(columns={"Origen_Id": "estacion_id"}))
  #Bicicletas IN
  df["ventana_fin"]=df["Fin_del_viaje"].dt.floor(freq)
  inBikes=(df.groupby(["ventana_fin","Destino_Id"]).size().reset_index(name="llegadas").rename(columns={"ventana_fin": "ventana", "Destino_Id": "estacion_id"}))
  #Cross de IN & OUT para el balance de capacidad de bikes
  balance=pd.merge(outBikes, inBikes, on=["ventana", "estacion_id"], how="outer").fillna(0)
  balance=balance.merge(df_estaciones[["estacion_id", "capacidad", "latitud", "longitud"]],on="estacion_id",how="inner",)
  #Flujo neto y formulación de la variable objetivo
  balance["flujo_neto"] = balance["llegadas"] - balance["salidas"]
  umbral = 0.5 * balance["capacidad"]
  #case: desbalance frente a su tamaño
  balance["en_riesgo"] = ((balance["flujo_neto"].abs() >= umbral) | (balance["salidas"] >= umbral)).astype(int)
  #para el tiempo
  balance["hora"] = balance["ventana"].dt.hour
  balance["dia_semana"] = balance["ventana"].dt.dayofweek
  balance["es_fin_de_semana"] = (balance["dia_semana"] >= 5).astype(int)
  #para el tardado del intervalo previo, evitando datadrifts
  balance = balance.sort_values(["estacion_id", "ventana"]).reset_index(
      drop=True
  )
  balance["lag_salidas"] = balance.groupby("estacion_id")["salidas"].shift(1)
  balance["lag_llegadas"] = balance.groupby("estacion_id")["llegadas"].shift(1)
  balance["lag_flujo_neto"] = balance.groupby("estacion_id")["flujo_neto"].shift(
      1
  )
  return balance.dropna().reset_index(drop=True)

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


def calcular_demanda_diaria_p1(df_viajes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la demanda diaria observada por estación para la Pregunta 1 (Regresión).

    Cada fila representa una estación en una fecha y contiene el número de
    viajes iniciados. No se generan registros para días sin salidas.

    Parámetros:
    -----------
    df_viajes : pd.DataFrame
        Datos limpios de viajes con 'Inicio_del_viaje' y 'Origen_Id'.

    Retorna:
    --------
    pd.DataFrame con columnas: 'fecha', 'estacion_id', 'viajes_iniciados'.
    """
    logger.info("Calculando demanda diaria por estación para P1...")

    # 1. Seleccionar únicamente las columnas necesarias
    df = df_viajes[["Inicio_del_viaje", "Origen_Id"]].copy()

    # 2. Obtener la fecha de inicio sin la hora
    df["fecha"] = df["Inicio_del_viaje"].dt.normalize()

    # 3. Contar los viajes iniciados por estación y fecha
    df_demanda = (
        df.groupby(["fecha", "Origen_Id"])
        .size()
        .reset_index(name="viajes_iniciados")
        .rename(columns={"Origen_Id": "estacion_id"})
    )

    # 4. Ordenar los registros cronológicamente por estación
    df_demanda = (
        df_demanda.sort_values(["estacion_id", "fecha"])
        .reset_index(drop=True)
    )

    logger.success(
        f"Demanda diaria calculada: {len(df_demanda):,} registros "
        f"para {df_demanda['estacion_id'].nunique()} estaciones."
    )

    return df_demanda


def preparar_demanda_diaria_p1(
    df_demanda: pd.DataFrame,
    fecha_inicio: str = "2025-10-01",
    fecha_fin_train: str = "2026-06-30",
    fecha_fin: str = "2026-09-30",
    cobertura_minima: float = 0.95,
) -> pd.DataFrame:
    """
    Prepara una serie diaria continua por estación para P1.

    Selecciona estaciones utilizando exclusivamente la cobertura observada
    durante entrenamiento y completa los días sin salidas registradas con cero.

    Parámetros:
    -----------
    df_demanda : pd.DataFrame
        Tabla generada por calcular_demanda_diaria_p1().
    fecha_inicio : str
        Primera fecha del periodo de análisis.
    fecha_fin_train : str
        Última fecha utilizada para seleccionar estaciones.
    fecha_fin : str
        Última fecha del periodo completo.
    cobertura_minima : float
        Proporción mínima de días con salidas durante entrenamiento.

    Retorna:
    --------
    pd.DataFrame con columnas: 'fecha', 'estacion_id', 'viajes_iniciados'.
    """
    logger.info("Preparando calendario diario de estaciones para P1...")

    inicio = pd.Timestamp(fecha_inicio)
    fin_train = pd.Timestamp(fecha_fin_train)
    fin = pd.Timestamp(fecha_fin)

    if not (inicio <= fin_train <= fin):
        raise ValueError("Las fechas deben cumplir: inicio <= fin_train <= fin.")

    if not 0 < cobertura_minima <= 1:
        raise ValueError("cobertura_minima debe estar entre 0 y 1.")

    # 1. Seleccionar únicamente el periodo de entrenamiento
    df_train = df_demanda[
        (df_demanda["fecha"] >= inicio)
        & (df_demanda["fecha"] <= fin_train)
    ]

    # 2. Calcular cobertura histórica por estación
    dias_train = len(pd.date_range(inicio, fin_train))

    actividad = (
        df_train.groupby("estacion_id")["fecha"]
        .nunique()
    )

    estaciones_seleccionadas = actividad[
        actividad / dias_train >= cobertura_minima
    ].index

    logger.info(
        f"Estaciones seleccionadas: {len(estaciones_seleccionadas)} "
        f"(cobertura mínima: {cobertura_minima:.0%})."
    )

    # 3. Crear todas las combinaciones estación-fecha
    fechas = pd.date_range(inicio, fin)

    calendario = pd.MultiIndex.from_product(
        [estaciones_seleccionadas, fechas],
        names=["estacion_id", "fecha"],
    ).to_frame(index=False)

    # 4. Incorporar la demanda observada
    df_preparado = calendario.merge(
        df_demanda,
        on=["estacion_id", "fecha"],
        how="left",
    )

    # 5. Completar días sin salidas registradas
    dias_sin_registros = df_preparado["viajes_iniciados"].isna().sum()

    df_preparado["viajes_iniciados"] = (
        df_preparado["viajes_iniciados"]
        .fillna(0)
        .astype(int)
    )

    logger.info(
        f"Combinaciones estación-día sin registros completadas con cero: "
        f"{dias_sin_registros:,}."
    )

    logger.success(
        f"Calendario P1 preparado: {len(df_preparado):,} registros "
        f"para {len(estaciones_seleccionadas)} estaciones."
    )

    return df_preparado


def generar_caracteristicas_regresion_p1(
    df_demanda: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera características temporales e históricas para P1 (Regresión).

    Cada fila representa una estación en una fecha. Las características
    históricas utilizan exclusivamente viajes de días anteriores.

    Requiere una serie diaria continua por estación, generada previamente
    mediante preparar_demanda_diaria_p1().

    Parámetros:
    -----------
    df_demanda : pd.DataFrame
        DataFrame con 'fecha', 'estacion_id' y 'viajes_iniciados'.

    Retorna:
    --------
    pd.DataFrame con la demanda diaria y las características de regresión.
    """
    logger.info("Generando características históricas para P1...")

    # 1. Copiar y ordenar los datos
    df = df_demanda.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    df = df.sort_values(
        ["estacion_id", "fecha"]
    ).reset_index(drop=True)

    # 2. Verificar que no existan registros duplicados
    if df.duplicated(["estacion_id", "fecha"]).any():
        raise ValueError(
            "Existen registros duplicados para una misma estación y fecha."
        )

    # 3. Verificar que el calendario sea continuo por estación
    diferencia_dias = df.groupby("estacion_id")["fecha"].diff()

    if (
        diferencia_dias.dropna() != pd.Timedelta(days=1)
    ).any():
        raise ValueError(
            "El calendario debe tener una observación por día para cada estación."
        )

    # 4. Características de calendario
    df["dia_semana"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.month

    df["es_fin_de_semana"] = (
        df["dia_semana"].isin([5, 6]).astype(int)
    )

    # 5. Características históricas por estación
    grupos = df.groupby("estacion_id")["viajes_iniciados"]

    df["viajes_dia_anterior"] = grupos.shift(1)
    df["viajes_hace_7_dias"] = grupos.shift(7)

    # 6. Promedios móviles: excluir el día que queremos predecir
    df["promedio_7_dias"] = (
        grupos.transform(
            lambda serie: serie.shift(1).rolling(7).mean()
        )
    )

    df["promedio_14_dias"] = (
        grupos.transform(
            lambda serie: serie.shift(1).rolling(14).mean()
        )
    )

    # 7. Rezagos históricos de dos y cuatro semanas
    df["viajes_hace_14_dias"] = grupos.shift(14)
    df["viajes_hace_28_dias"] = grupos.shift(28)

    df["promedio_28_dias"] = grupos.transform(
        lambda serie: serie.shift(1).rolling(28).mean()
    )

    df["promedio_mismo_dia_4_semanas"] = (
        df["viajes_hace_7_dias"]
        + df["viajes_hace_14_dias"]
        + grupos.shift(21)
        + df["viajes_hace_28_dias"]
    ) / 4

    logger.success(
        f"Características P1 generadas: {len(df):,} registros "
        f"para {df['estacion_id'].nunique()} estaciones."
    )

    return df


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
