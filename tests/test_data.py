from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

from mibici.config import (
    PROCESSED_DATA_DIR,
    DATA_MANIFEST_PATH,
)
from mibici.features import (
    extraer_caracteristicas_temporales,
    calcular_perfil_estaciones_p3,
)


@pytest.fixture
def datos_sinteticos():
    """Genera datos sintéticos mínimos de viajes y estaciones para pruebas aisladas."""
    df_estaciones = pd.DataFrame({
        "estacion_id": [1, 2, 3],
        "nombre": ["Estación 1", "Estación 2", "Estación 3"],
        "latitud": [20.67, 20.68, 20.69],
        "longitud": [-103.35, -103.36, -103.37],
        "capacidad": [15, 20, 10],
    })

    df_viajes = pd.DataFrame({
        "Viaje_Id": [101, 102, 103, 104, 105],
        "Origen_Id": [1, 1, 2, 2, 3],
        "Destino_Id": [2, 3, 1, 3, 1],
        "Inicio_del_viaje": pd.to_datetime([
            "2026-03-02 08:15:00",  # Lunes pico matutino
            "2026-03-02 18:30:00",  # Lunes pico vespertino
            "2026-03-07 14:00:00",  # Sábado fin de semana
            "2026-03-03 09:00:00",  # Martes pico matutino
            "2026-03-04 12:00:00",  # Miércoles fuera de pico
        ]),
        "Fin_del_viaje": pd.to_datetime([
            "2026-03-02 08:30:00",
            "2026-03-02 18:50:00",
            "2026-03-07 14:25:00",
            "2026-03-03 09:15:00",
            "2026-03-04 12:20:00",
        ]),
        "duracion_min": [15.0, 20.0, 25.0, 15.0, 20.0],
    })

    return df_viajes, df_estaciones


def test_manifiesto_datos_si_existe():
    """Valida la estructura del manifiesto de versionado si ya fue generado."""
    if not DATA_MANIFEST_PATH.exists():
        pytest.skip("data_manifest.json no generado aún. Requiere ejecutar python -m mibici.dataset.")

    with open(DATA_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_version" in data
    assert "rango_fechas" in data
    assert "total_viajes" in data
    assert data["total_viajes"] > 0
    assert "archivos_origen" in data


def test_extraccion_caracteristicas_temporales():
    """Valida la clasificación de picos horarios y fines de semana."""
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


def test_calcular_perfil_estaciones_con_fixture(datos_sinteticos):
    """Valida la función de perfilamiento de estaciones con datos sintéticos independientes."""
    df_viajes, df_estaciones = datos_sinteticos
    df_features = calcular_perfil_estaciones_p3(df_viajes, df_estaciones)

    assert len(df_features) == len(df_estaciones)
    assert "ratio_flujo_neto" in df_features.columns
    assert "viajes_diarios_por_anclaje" in df_features.columns

    # Validar rangos matemáticos
    assert (df_features["ratio_flujo_neto"] >= -1.0).all()
    assert (df_features["ratio_flujo_neto"] <= 1.0).all()
    assert (df_features["pct_salidas_pico_matutino"] >= 0.0).all()
    assert (df_features["pct_salidas_pico_matutino"] <= 1.0).all()
    assert not df_features["ratio_flujo_neto"].isna().any()


def test_archivos_parquet_reales_si_existen():
    """Verifica la consistencia de los datos reales procesados cuando ya fueron generados."""
    viajes_path = PROCESSED_DATA_DIR / "viajes_limpios.parquet"
    estaciones_path = PROCESSED_DATA_DIR / "estaciones.parquet"

    if not viajes_path.exists() or not estaciones_path.exists():
        pytest.skip("Archivos Parquet no encontrados. Requiere ejecutar python -m mibici.dataset.")

    df_estaciones = pd.read_parquet(estaciones_path)
    assert len(df_estaciones) >= 400
    assert "estacion_id" in df_estaciones.columns


def test_parse_bool_run_project():
    """Valida el parseo de argumentos booleanos en el coordinador de MLflow Projects."""
    from mibici.modeling.run_project import _parse_bool

    assert _parse_bool("true") is True
    assert _parse_bool("True") is True
    assert _parse_bool("1") is True
    assert _parse_bool("false") is False
    assert _parse_bool("False") is False
    assert _parse_bool("0") is False


def test_configuraciones_search_p3():
    """Verifica que la búsqueda de hiperparámetros P3 incluya al menos 3 configuraciones válidas."""
    from mibici.modeling.search_p3 import CONFIGURACIONES

    assert len(CONFIGURACIONES) >= 3
    for cfg in CONFIGURACIONES:
        assert cfg["modelo_tipo"] in ("kmeans", "gmm")
        assert cfg["n_clusters"] >= 2
        assert "incluir_espacial" in cfg
