import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from mibici.modeling.train import entrenar_random_forest_p1


def test_reproducibilidad_random_forest_p1():
    """
    Verifica que dos entrenamientos con la misma semilla
    produzcan predicciones y métricas idénticas.
    """

    # Dataset sintético reproducible
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "estacion_id": rng.integers(1, 10, 100),
        "dia_semana": rng.integers(0, 7, 100),
        "viajes_dia_anterior": rng.integers(0, 50, 100),
        "promedio_7_dias": rng.uniform(0, 50, 100),
    })

    y = (
        0.5 * X["viajes_dia_anterior"]
        + 0.3 * X["promedio_7_dias"]
        + rng.normal(0, 2, 100)
    )

    # División fija de datos
    X_train = X.iloc[:80]
    X_test = X.iloc[80:]

    y_train = y.iloc[:80]
    y_test = y.iloc[80:]

    # Primer entrenamiento
    modelo_1 = entrenar_random_forest_p1(
        X_train,
        y_train,
        n_estimators=20,
        random_seed=42,
    )

    # Segundo entrenamiento
    modelo_2 = entrenar_random_forest_p1(
        X_train,
        y_train,
        n_estimators=20,
        random_seed=42,
    )

    # Predicciones
    pred_1 = modelo_1.predict(X_test)
    pred_2 = modelo_2.predict(X_test)

    # Verificar reproducibilidad de predicciones
    np.testing.assert_allclose(
        pred_1,
        pred_2,
        rtol=1e-10,
        atol=1e-12,
    )   

    # Verificar reproducibilidad de métricas
    mae_1 = mean_absolute_error(y_test, pred_1)
    mae_2 = mean_absolute_error(y_test, pred_2)

    np.testing.assert_allclose(
        mae_1,
        mae_2,
        rtol=1e-10,
        atol=1e-12,
    )
