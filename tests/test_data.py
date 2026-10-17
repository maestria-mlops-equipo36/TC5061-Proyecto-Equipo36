from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from mibici.config import (
    PROCESSED_DATA_DIR,
    DATA_MANIFEST_PATH,
    NOMENCLATURA_PATH,
)
from mibici.features import extraer_caracteristicas_temporales, calcular_perfil_estaciones_p3


def test_manifiesto_datos_existe_y_valido():
    """Valida que el archivo de versionado de datos exista y tenga los campos requeridos."""
    assert DATA_MANIFEST_PATH.exists(), f"No se encontró el manifiesto en {DATA_MANIFEST_PATH}"
    
    with open(DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_version" in data
    assert "rango_fechas" in data
    assert "total_viajes" in data
    assert data["total_viajes"] > 0
    assert "total_estaciones_nomenclatura" in data
    assert data["total_estaciones_nomenclatura"] > 0
    assert "archivos_origen" in data
    assert len(data["archivos_origen"]) == 12  # 12 meses


def test_archivos_parquet_procesados_existen():
    """Valida que los datasets Parquet limpios existan y no estén vacíos."""
    viajes_path = PROCESSED_DATA_DIR / "viajes_limpios.parquet"
    estaciones_path = PROCESSED_DATA_DIR / "estaciones.parquet"

    assert viajes_path.exists(), f"Falta {viajes_path}"
    assert estaciones_path.exists(), f"Falta {estaciones_path}"

    df_estaciones = pd.read_parquet(estaciones_path)
    assert len(df_estaciones) >= 400
    assert "estacion_id" in df_estaciones.columns
    assert "latitud" in df_estaciones.columns
    assert "longitud" in df_estaciones.columns
    assert "capacidad" in df_estaciones.columns


def test_extraccion_caracteristicas_temporales():
    """Valida la asignación correcta de horas pico y fines de semana."""
    sample_data = pd.DataFrame({
        "Inicio_del_viaje": [
            pd.Timestamp("2026-03-02 08:30:00"),  # Lunes pico matutino
            pd.Timestamp("2026-03-02 18:15:00"),  # Lunes pico vespertino
            pd.Timestamp("2026-03-07 14:00:00"),  # Sábado fin de semana
        ]
    })
    
    df_res = extraer_caracteristicas_temporales(sample_data)
    
    assert df_res.loc[0, "pico_matutino"] == 1
    assert df_res.loc[0, "pico_vespertino"] == 0
    assert df_res.loc[0, "es_fin_de_semana"] == 0

    assert df_res.loc[1, "pico_matutino"] == 0
    assert df_res.loc[1, "pico_vespertino"] == 1
    assert df_res.loc[1, "es_fin_de_semana"] == 0

    assert df_res.loc[2, "pico_matutino"] == 0
    assert df_res.loc[2, "es_fin_de_semana"] == 1


def test_caracteristicas_p3_estaciones():
    """Valida que la tabla de features por estación cumpla con las restricciones matemáticas."""
    features_path = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet"
    assert features_path.exists(), f"Falta {features_path}"

    df_features = pd.read_parquet(features_path)

    # Validar que no haya valores NaN o infinitos en las métricas clave
    cols_clave = [
        "ratio_flujo_neto",
        "pct_salidas_pico_matutino",
        "pct_salidas_pico_vespertino",
        "pct_salidas_fin_de_semana",
        "viajes_diarios_por_anclaje",
    ]
    for col in cols_clave:
        assert not df_features[col].isna().any(), f"Columna {col} tiene valores NaN"
        assert not np.isinf(df_features[col]).any(), f"Columna {col} tiene valores Inf"

    # Restricciones matemáticas
    assert (df_features["ratio_flujo_neto"] >= -1.0).all()
    assert (df_features["ratio_flujo_neto"] <= 1.0).all()
    assert (df_features["pct_salidas_pico_matutino"] >= 0.0).all()
    assert (df_features["pct_salidas_pico_matutino"] <= 1.0).all()
    assert (df_features["pct_salidas_fin_de_semana"] >= 0.0).all()
    assert (df_features["pct_salidas_fin_de_semana"] <= 1.0).all()
