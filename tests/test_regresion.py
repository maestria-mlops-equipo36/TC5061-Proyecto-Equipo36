import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from mibici.config import RANDOM_SEED

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


def test_calcular_demanda_diaria_p1_vacio():
    """Verifica que un dataset vacío conserve la estructura esperada."""

    viajes = pd.DataFrame({
        "Inicio_del_viaje": pd.Series(dtype="datetime64[ns]"),
        "Origen_Id": pd.Series(dtype="int64"),
    })

    demanda = calcular_demanda_diaria_p1(viajes)

    # No deben generarse registros si no existen viajes
    assert demanda.empty

    # La estructura de salida debe conservarse
    assert demanda.columns.tolist() == [
        "fecha",
        "estacion_id",
        "viajes_iniciados",
    ]


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


def test_preparar_demanda_diaria_p1_fechas_invalidas():
    """Verifica que se rechacen periodos de entrenamiento inconsistentes."""

    demanda = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-01"]),
        "estacion_id": [1],
        "viajes_iniciados": [5],
    })

    with pytest.raises(
        ValueError,
        match="Las fechas deben cumplir",
    ):
        preparar_demanda_diaria_p1(
            demanda,
            fecha_inicio="2026-01-10",
            fecha_fin_train="2026-01-05",
            fecha_fin="2026-01-20",
        )

@pytest.mark.parametrize("cobertura_invalida", [0, -0.5, 1.5])
def test_preparar_demanda_diaria_p1_cobertura_invalida(cobertura_invalida):
    """Verifica que se rechacen valores de cobertura fuera del rango permitido."""

    demanda = pd.DataFrame({
        "fecha": pd.to_datetime(["2026-01-01"]),
        "estacion_id": [1],
        "viajes_iniciados": [5],
    })

    with pytest.raises(
        ValueError,
        match="cobertura_minima debe estar entre 0 y 1",
    ):
        preparar_demanda_diaria_p1(
            demanda,
            fecha_inicio="2026-01-01",
            fecha_fin_train="2026-01-02",
            fecha_fin="2026-01-03",
            cobertura_minima=cobertura_invalida,
        )


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


def test_caracteristicas_regresion_p1_duplicados():
    """Verifica que se rechacen registros duplicados por estación y fecha."""

    demanda = pd.DataFrame({
        "fecha": pd.to_datetime([
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
        ]),
        "estacion_id": [1, 1, 1],
        "viajes_iniciados": [5, 8, 6],
    })

    with pytest.raises(
        ValueError,
        match="Existen registros duplicados",
    ):
        generar_caracteristicas_regresion_p1(demanda)


def test_caracteristicas_regresion_p1_calendario_discontinuo():
    """Verifica que se rechacen series temporales con días faltantes."""

    demanda = pd.DataFrame({
        "fecha": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-04",  # Falta el 3 de enero
            "2026-01-05",
        ]),
        "estacion_id": [1, 1, 1, 1],
        "viajes_iniciados": [5, 8, 6, 10],
    })

    with pytest.raises(
        ValueError,
        match="El calendario debe tener una observación por día",
    ):
        generar_caracteristicas_regresion_p1(demanda)

    
def test_caracteristicas_regresion_p1_historial_insuficiente():
    """Verifica que los rezagos sin historial suficiente permanezcan como NaN."""

    demanda = pd.DataFrame({
        "fecha": pd.date_range("2026-01-01", periods=10),
        "estacion_id": [1] * 10,
        "viajes_iniciados": np.arange(1, 11),
    })

    features = generar_caracteristicas_regresion_p1(demanda)

    # Deben conservarse los 10 registros originales
    assert len(features) == 10

    # El primer día no tiene información histórica
    assert pd.isna(features.loc[0, "viajes_dia_anterior"])
    assert pd.isna(features.loc[0, "promedio_7_dias"])

    # El segundo día ya puede utilizar la demanda del día anterior
    assert features.loc[1, "viajes_dia_anterior"] == 1

    # El octavo día ya cuenta con siete días de historial
    assert features.loc[7, "viajes_hace_7_dias"] == 1
    assert features.loc[7, "promedio_7_dias"] == 4

    # Ningún registro tiene suficiente historial para rezagos de 14 o 28 días
    assert features["viajes_hace_14_dias"].isna().all()
    assert features["viajes_hace_28_dias"].isna().all()
    assert features["promedio_28_dias"].isna().all()


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


def test_reproducibilidad_entrenamiento_p1():
    """Verifica que Random Forest produzca resultados idénticos con la misma semilla."""

    # Generar datos sintéticos reproducibles
    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.DataFrame({
        "estacion_id": rng.integers(1, 20, size=100),
        "dia_semana": rng.integers(0, 7, size=100),
        "mes": rng.integers(1, 13, size=100),
        "viajes_dia_anterior": rng.integers(0, 50, size=100),
        "viajes_hace_7_dias": rng.integers(0, 50, size=100),
    })

    y = (
        0.6 * X["viajes_dia_anterior"]
        + 0.4 * X["viajes_hace_7_dias"]
        + rng.normal(0, 2, size=100)
    )

    # División fija de entrenamiento y evaluación
    X_train, X_test = X.iloc[:80], X.iloc[80:]
    y_train, y_test = y.iloc[:80], y.iloc[80:]

    # Configuración del modelo ganador de P1
    parametros = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    # Primera ejecución
    modelo_1 = RandomForestRegressor(**parametros)
    modelo_1.fit(X_train, y_train)
    predicciones_1 = np.maximum(modelo_1.predict(X_test), 0)

    # Segunda ejecución independiente
    modelo_2 = RandomForestRegressor(**parametros)
    modelo_2.fit(X_train, y_train)
    predicciones_2 = np.maximum(modelo_2.predict(X_test), 0)

    # Verificar reproducibilidad de las predicciones
    np.testing.assert_allclose(
        predicciones_1,
        predicciones_2,
        rtol=1e-12,
        atol=1e-12,
    )

    # Verificar reproducibilidad de la métrica MAE
    mae_1 = mean_absolute_error(y_test, predicciones_1)
    mae_2 = mean_absolute_error(y_test, predicciones_2)

    assert mae_1 == pytest.approx(mae_2, rel=1e-12, abs=1e-12)