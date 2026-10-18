import numpy as np
import pandas as pd

from mibici.features import (
    calcular_demanda_diaria_p1,
    preparar_demanda_diaria_p1,
    generar_caracteristicas_regresion_p1,
)


def test_calcular_demanda_diaria_p1():
    """Verifica el conteo de viajes por estación y día."""
    viajes = pd.DataFrame({
        "Inicio_del_viaje": pd.to_datetime([
            "2026-01-01 08:00",
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-02 08:00",
        ]),
        "Origen_Id": [1, 1, 2, 1],
    })

    demanda = calcular_demanda_diaria_p1(viajes)

    assert len(demanda) == 3
    assert demanda.loc[
        (demanda["estacion_id"] == 1)
        & (demanda["fecha"] == "2026-01-01"),
        "viajes_iniciados",
    ].iloc[0] == 2

    assert demanda["viajes_iniciados"].sum() == 4


def test_preparar_demanda_diaria_p1():
    """Verifica continuidad, días sin viajes y selección sin datos futuros."""
    demanda = pd.DataFrame({
        "fecha": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-01",
            "2026-01-04",
        ]),
        "estacion_id": [1, 1, 1, 2, 2],
        "viajes_iniciados": [5, 6, 7, 3, 9],
    })

    preparado = preparar_demanda_diaria_p1(
        demanda,
        fecha_inicio="2026-01-01",
        fecha_fin_train="2026-01-03",
        fecha_fin="2026-01-04",
        cobertura_minima=0.95,
    )

    # Solo la estación 1 tiene suficiente cobertura en entrenamiento
    assert preparado["estacion_id"].unique().tolist() == [1]

    # Debe existir un registro para cada día
    assert len(preparado) == 4

    # El día sin viajes debe completarse con cero
    assert preparado.loc[
        preparado["fecha"] == "2026-01-04",
        "viajes_iniciados",
    ].iloc[0] == 0


def test_caracteristicas_historicas_p1():
    """Verifica que los lags y promedios utilicen días anteriores."""
    demanda = pd.DataFrame({
        "fecha": pd.date_range("2026-01-01", periods=35),
        "estacion_id": [1] * 35,
        "viajes_iniciados": np.arange(1, 36),
    })

    features = generar_caracteristicas_regresion_p1(demanda)

    # Día 29: demanda real = 29
    fila = features.iloc[28]

    assert fila["viajes_dia_anterior"] == 28
    assert fila["viajes_hace_7_dias"] == 22
    assert fila["viajes_hace_14_dias"] == 15
    assert fila["viajes_hace_28_dias"] == 1

    assert fila["promedio_7_dias"] == 25
    assert fila["promedio_14_dias"] == 21.5
    assert fila["promedio_28_dias"] == 14.5

    assert fila["promedio_mismo_dia_4_semanas"] == 11.5


def test_sin_fuga_temporal_p1():
    """Verifica que cambiar la demanda futura no altere features pasadas."""
    demanda = pd.DataFrame({
        "fecha": pd.date_range("2026-01-01", periods=40),
        "estacion_id": [1] * 40,
        "viajes_iniciados": np.arange(1, 41),
    })

    features_originales = generar_caracteristicas_regresion_p1(demanda)

    # Modificar únicamente la demanda del día 40
    demanda_modificada = demanda.copy()
    demanda_modificada.loc[39, "viajes_iniciados"] = 10000

    features_modificadas = generar_caracteristicas_regresion_p1(
        demanda_modificada
    )

    columnas_features = [
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

    # Las características de los primeros 39 días deben permanecer iguales
    pd.testing.assert_frame_equal(
        features_originales.loc[:38, columnas_features],
        features_modificadas.loc[:38, columnas_features],
    )