import numpy as np
import pytest
import pandas as pd

from mibici.modeling.ab_testing_p1 import (
    cargar_datos_evaluacion,
    evaluar_modelo,
    recomendar_modelo,
)

class ModeloSimulado:
    """Simula un modelo para probar el cálculo de métricas."""

    def predict(self, X):
        return np.array([10, 20, 30, 40])


def test_evaluar_modelo():
    """Verifica que las métricas se calculen correctamente."""

    modelo = ModeloSimulado()

    X = np.zeros((4, 2))
    y = np.array([12, 18, 33, 37])

    metricas, predicciones = evaluar_modelo(modelo, X, y)

    assert metricas["mae"] == pytest.approx(2.5)
    assert metricas["rmse"] == pytest.approx(np.sqrt(6.5))
    assert metricas["r2"] == pytest.approx(1 - 26 / 426)

    np.testing.assert_array_equal(
        predicciones,
        np.array([10, 20, 30, 40]),
    )


def test_recomendar_candidate():
    """Verifica que se recomiende candidate cuando mejora ambas métricas."""

    comparacion = pd.DataFrame(
        {
            "mae": [7.23, 7.21],
            "rmse": [12.29, 12.11],
            "r2": [0.883, 0.886],
        },
        index=["champion", "candidate"],
    )

    recomendado, justificacion = recomendar_modelo(comparacion)

    assert recomendado == "candidate"
    assert "menores valores de MAE y RMSE" in justificacion


def test_recomendar_champion():
    """Verifica que se conserve champion si candidate no mejora ambas métricas."""

    comparacion = pd.DataFrame(
        {
            "mae": [7.23, 7.20],
            "rmse": [12.29, 12.50],
            "r2": [0.883, 0.879],
        },
        index=["champion", "candidate"],
    )

    recomendado, justificacion = recomendar_modelo(comparacion)

    assert recomendado == "champion"
    assert "no demuestra una mejora consistente" in justificacion


def test_datos_duplicados(tmp_path):
    """Verifica que se detecten registros duplicados por estación y fecha."""

    from mibici.modeling.ab_testing_p1 import FEATURES

    # Crear datos simulados con una estación y fecha duplicadas
    df = pd.DataFrame(0, index=range(2), columns=FEATURES)

    df["estacion_id"] = [1, 1]
    df["fecha"] = pd.to_datetime(["2026-09-01", "2026-09-01"])
    df["viajes_iniciados"] = [20, 25]

    # Guardar datos de prueba
    data_path = tmp_path / "regression_current.parquet"
    df.to_parquet(data_path)

    # Verificar que la función detecte los duplicados
    with pytest.raises(ValueError, match="registros duplicados"):
        cargar_datos_evaluacion(data_path)


def test_datos_faltantes(tmp_path):
    """Verifica que se detecten valores faltantes en los datos."""

    from mibici.modeling.ab_testing_p1 import FEATURES

    # Crear datos simulados
    df = pd.DataFrame(0, index=range(2), columns=FEATURES)

    df["estacion_id"] = [1, 2]
    df["fecha"] = pd.to_datetime(["2026-09-01", "2026-09-01"])
    df["viajes_iniciados"] = [20, 25]

    # Introducir un valor faltante
    df.loc[0, "promedio_7_dias"] = np.nan

    # Guardar datos de prueba
    data_path = tmp_path / "regression_current.parquet"
    df.to_parquet(data_path)

    # Verificar que la función detecte los valores faltantes
    with pytest.raises(ValueError, match="valores faltantes"):
        cargar_datos_evaluacion(data_path)
