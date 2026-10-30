import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from mibici.config import RANDOM_SEED
from mibici.features import calcular_features_riesgo_p2

from mibici.modeling.train_p2 import preparar_datos_p2

def test_calcular_riesgo_p2():
    """Verifica la clasificación de estaciones con y sin riesgo operativo."""

    # Estación con capacidad de 10 bicicletas
    df_estaciones = pd.DataFrame({
        "estacion_id": [1],
        "capacidad": [10],
        "latitud": [20.67],
        "longitud": [-103.35],
    })

    # Tres ventanas temporales con distintos niveles de movimiento
    df_viajes = pd.DataFrame({
        "Origen_Id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Destino_Id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Inicio_del_viaje": pd.to_datetime([
            "2026-03-02 08:10:00",
            "2026-03-02 10:10:00",
            "2026-03-02 10:20:00",
            "2026-03-02 12:05:00",
            "2026-03-02 12:10:00",
            "2026-03-02 12:15:00",
            "2026-03-02 12:20:00",
            "2026-03-02 12:25:00",
            "2026-03-02 12:30:00",
        ]),
        "Fin_del_viaje": pd.to_datetime([
            "2026-03-02 08:20:00",
            "2026-03-02 10:30:00",
            "2026-03-02 12:40:00",
            "2026-03-02 12:45:00",
            "2026-03-02 14:10:00",
            "2026-03-02 14:15:00",
            "2026-03-02 14:20:00",
            "2026-03-02 14:25:00",
            "2026-03-02 14:30:00",
        ]),
    })

    # Ejecutar la función real de P2
    resultado = calcular_features_riesgo_p2(
        df_viajes,
        df_estaciones,
        freq="2h",
    )

    # Consultar las ventanas que queremos evaluar
    ventana_10 = resultado[
        resultado["ventana"] == pd.Timestamp("2026-03-02 10:00:00")
    ]

    ventana_12 = resultado[
        resultado["ventana"] == pd.Timestamp("2026-03-02 12:00:00")
    ]

    # Verificar que ambas ventanas estén presentes
    assert len(ventana_10) == 1
    assert len(ventana_12) == 1

    # Ventana de las 10:00: movimientos por debajo del umbral
    assert ventana_10.iloc[0]["en_riesgo"] == 0

    # Ventana de las 12:00: salidas superiores al umbral
    assert ventana_12.iloc[0]["en_riesgo"] == 1

def test_umbral_exacto_riesgo_p2():
    """Verifica que el riesgo se active al alcanzar exactamente el 50 % de capacidad."""

    df_estaciones = pd.DataFrame({
        "estacion_id": [1],
        "capacidad": [10],
        "latitud": [20.67],
        "longitud": [-103.35],
    })

    # Crear viajes en tres ventanas de dos horas
    registros = []

    for hora, salidas in [(8, 1), (10, 4), (12, 5)]:
        for minuto in range(salidas):
            inicio = pd.Timestamp("2026-03-02") + pd.Timedelta(
                hours=hora, minutes=minuto
            )

            registros.append({
                "Origen_Id": 1,
                "Destino_Id": 1,
                "Inicio_del_viaje": inicio,
                "Fin_del_viaje": inicio + pd.Timedelta(minutes=15),
            })

    df_viajes = pd.DataFrame(registros)

    # Calcular características y etiquetas de riesgo
    resultado = calcular_features_riesgo_p2(
        df_viajes,
        df_estaciones,
        freq="2h",
    )

    ventana_10 = resultado.loc[
        resultado["ventana"] == pd.Timestamp("2026-03-02 10:00:00")
    ].iloc[0]

    ventana_12 = resultado.loc[
        resultado["ventana"] == pd.Timestamp("2026-03-02 12:00:00")
    ].iloc[0]

    # Debajo del umbral: 4 salidas < 5
    assert ventana_10["salidas"] == 4
    assert ventana_10["en_riesgo"] == 0

    # Umbral exacto: 5 salidas >= 5
    assert ventana_12["salidas"] == 5
    assert ventana_12["en_riesgo"] == 1


def test_independencia_estaciones_p2():
    """Verifica que los retardos históricos se calculen por estación."""

    df_estaciones = pd.DataFrame({
        "estacion_id": [1, 2],
        "capacidad": [20, 20],
        "latitud": [20.67, 20.68],
        "longitud": [-103.35, -103.36],
    })

    registros = []

    # Cada estación tiene un historial diferente
    movimientos = [
        (1, 8, 3),
        (2, 8, 7),
        (1, 10, 5),
        (2, 10, 2),
    ]

    for estacion_id, hora, cantidad in movimientos:
        for minuto in range(cantidad):
            inicio = pd.Timestamp("2026-03-02") + pd.Timedelta(
                hours=hora, minutes=minuto
            )

            registros.append({
                "Origen_Id": estacion_id,
                "Destino_Id": estacion_id,
                "Inicio_del_viaje": inicio,
                "Fin_del_viaje": inicio + pd.Timedelta(minutes=15),
            })

    df_viajes = pd.DataFrame(registros)

    resultado = calcular_features_riesgo_p2(
        df_viajes,
        df_estaciones,
        freq="2h",
    )

    # Seleccionar la ventana de las 10:00 para ambas estaciones
    ventana_10 = resultado[
        resultado["ventana"] == pd.Timestamp("2026-03-02 10:00:00")
    ].set_index("estacion_id")

    # Verificar que ambas estaciones estén presentes
    assert set(ventana_10.index) == {1, 2}

    # Estación 1: historial de tres movimientos
    assert ventana_10.loc[1, "lag_salidas"] == 3
    assert ventana_10.loc[1, "lag_llegadas"] == 3
    assert ventana_10.loc[1, "lag_flujo_neto"] == 0

    # Estación 2: historial de siete movimientos
    assert ventana_10.loc[2, "lag_salidas"] == 7
    assert ventana_10.loc[2, "lag_llegadas"] == 7
    assert ventana_10.loc[2, "lag_flujo_neto"] == 0

    # Verificar que los valores históricos sean diferentes
    assert ventana_10.loc[1, "lag_salidas"] != ventana_10.loc[2, "lag_salidas"]


def test_sin_fuga_temporal_p2():
    """Verifica que los viajes futuros no alteren las características históricas."""

    df_estaciones = pd.DataFrame({
        "estacion_id": [1],
        "capacidad": [20],
        "latitud": [20.67],
        "longitud": [-103.35],
    })

    registros = []

    # Construir cuatro ventanas temporales
    for hora, cantidad in [(8, 2), (10, 3), (12, 4), (14, 2)]:
        for minuto in range(cantidad):
            inicio = pd.Timestamp("2026-03-02") + pd.Timedelta(
                hours=hora, minutes=minuto
            )

            registros.append({
                "Origen_Id": 1,
                "Destino_Id": 1,
                "Inicio_del_viaje": inicio,
                "Fin_del_viaje": inicio + pd.Timedelta(minutes=15),
            })

    df_original = pd.DataFrame(registros)

    # Calcular características con los datos originales
    resultado_original = calcular_features_riesgo_p2(
        df_original,
        df_estaciones,
        freq="2h",
    )

    # Agregar diez viajes adicionales en una ventana futura (14:00)
    viajes_futuros = []

    for minuto in range(10):
        inicio = pd.Timestamp("2026-03-02 14:30:00") + pd.Timedelta(
            minutes=minuto
        )

        viajes_futuros.append({
            "Origen_Id": 1,
            "Destino_Id": 1,
            "Inicio_del_viaje": inicio,
            "Fin_del_viaje": inicio + pd.Timedelta(minutes=15),
        })

    df_modificado = pd.concat(
        [df_original, pd.DataFrame(viajes_futuros)],
        ignore_index=True,
    )

    # Recalcular características después de modificar el futuro
    resultado_modificado = calcular_features_riesgo_p2(
        df_modificado,
        df_estaciones,
        freq="2h",
    )

    # Comparar únicamente las ventanas anteriores a las 14:00
    limite_temporal = pd.Timestamp("2026-03-02 14:00:00")

    columnas_historicas = [
        "ventana",
        "estacion_id",
        "lag_salidas",
        "lag_llegadas",
        "lag_flujo_neto",
    ]

    historico_original = resultado_original.loc[
        resultado_original["ventana"] < limite_temporal,
        columnas_historicas,
    ].reset_index(drop=True)

    historico_modificado = resultado_modificado.loc[
        resultado_modificado["ventana"] < limite_temporal,
        columnas_historicas,
    ].reset_index(drop=True)

    # Los cambios futuros no deben afectar el historial anterior
    pd.testing.assert_frame_equal(
        historico_original,
        historico_modificado,
    )

    # Comprobar que efectivamente modificamos los datos futuros
    salidas_originales = resultado_original.loc[
        resultado_original["ventana"] == limite_temporal,
        "salidas",
    ].iloc[0]

    salidas_modificadas = resultado_modificado.loc[
        resultado_modificado["ventana"] == limite_temporal,
        "salidas",
    ].iloc[0]

    assert salidas_originales == 2
    assert salidas_modificadas == 12


def test_preparar_datos_p2(monkeypatch):
    """Verifica las proporciones, características y etiquetas de los conjuntos P2."""

    df_estaciones = pd.DataFrame({
        "estacion_id": [1, 2],
        "capacidad": [10, 10],
        "latitud": [20.67, 20.68],
        "longitud": [-103.35, -103.36],
    })

    registros = []

    # Crear 40 ventanas de dos horas para ambas estaciones
    inicio_base = pd.Timestamp("2026-03-02 00:00:00")

    for ventana_idx in range(40):
        inicio_ventana = inicio_base + pd.Timedelta(hours=2 * ventana_idx)

        for estacion_id in [1, 2]:
            # Alternar entre ventanas con pocos y muchos movimientos
            cantidad = 2 if ventana_idx % 2 == 0 else 6

            for minuto in range(cantidad):
                inicio = inicio_ventana + pd.Timedelta(minutes=minuto)

                registros.append({
                    "Origen_Id": estacion_id,
                    "Destino_Id": estacion_id,
                    "Inicio_del_viaje": inicio,
                    "Fin_del_viaje": inicio + pd.Timedelta(minutes=15),
                })

    df_viajes = pd.DataFrame(registros)

    # Sustituir temporalmente la lectura de archivos Parquet
    def leer_parquet_simulado(path, *args, **kwargs):
        if path.name == "viajes_limpios.parquet":
            return df_viajes.copy()

        if path.name == "estaciones.parquet":
            return df_estaciones.copy()

        raise ValueError(f"Archivo inesperado: {path}")

    monkeypatch.setattr(pd, "read_parquet", leer_parquet_simulado)

    # Ejecutar la función real de preparación de P2
    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos_p2()

    # Verificar que las características y etiquetas tengan igual longitud
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Verificar que se conserven todas las observaciones
    total = len(X_train) + len(X_val) + len(X_test)

    # La primera observación de cada estación se elimina por los lags
    assert total == (40 - 1) * 2

    # Verificar proporciones de partición
    assert len(X_train) == int(total * 0.70)
    assert len(X_val) == int(total * 0.85) - int(total * 0.70)
    assert len(X_test) == total - int(total * 0.85)

    # Verificar las características utilizadas por el modelo
    columnas_esperadas = [
        "capacidad",
        "hora",
        "dia_semana",
        "es_fin_de_semana",
        "lag_salidas",
        "lag_llegadas",
        "lag_flujo_neto",
    ]

    assert list(X_train.columns) == columnas_esperadas
    assert list(X_val.columns) == columnas_esperadas
    assert list(X_test.columns) == columnas_esperadas

    # Verificar que las etiquetas sean binarias
    for etiquetas in [y_train, y_val, y_test]:
        assert set(etiquetas.unique()).issubset({0, 1})

    # Verificar que no existan valores faltantes
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()
    assert not X_test.isna().any().any()


#REVISAR ESTA PRUEBA
def test_separacion_temporal_estricta_p2(monkeypatch):
    """Verifica que entrenamiento, validación y prueba no compartan ventanas."""

    import mibici.modeling.train_p2 as modulo_p2

    # Crear 10 ventanas temporales con 3 estaciones por ventana
    ventanas = pd.date_range(
        start="2026-03-02 08:00:00",
        periods=10,
        freq="2h",
    )

    registros = []

    for ventana in ventanas:
        for estacion_id in [1, 2, 3]:
            registros.append({
                "ventana": ventana,
                "estacion_id": estacion_id,
                "capacidad": 20,
                "hora": ventana.hour,
                "dia_semana": ventana.dayofweek,
                "es_fin_de_semana": int(ventana.dayofweek >= 5),
                "lag_salidas": 2,
                "lag_llegadas": 3,
                "lag_flujo_neto": 1,
                "en_riesgo": 0,
            })

    df_sintetico = pd.DataFrame(registros)

    # Evitar la lectura de archivos reales
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    # Sustituir únicamente la generación de características
    monkeypatch.setattr(
        modulo_p2,
        "calcular_features_riesgo_p2",
        lambda *args, **kwargs: df_sintetico.copy(),
    )

    # Ejecutar la preparación real
    X_train, y_train, X_val, y_val, X_test, y_test = (
        modulo_p2.preparar_datos_p2()
    )

    # Recuperar las ventanas correspondientes a cada partición.
    # La función ordena por ventana y conserva ese orden al dividir.
    df_ordenado = df_sintetico.sort_values("ventana").reset_index(drop=True)

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    ventanas_train = df_ordenado.iloc[:n_train]["ventana"]
    ventanas_val = df_ordenado.iloc[n_train:n_train + n_val]["ventana"]
    ventanas_test = df_ordenado.iloc[n_train + n_val:]["ventana"]

    # Verificar que todas las observaciones estén distribuidas
    assert n_train + n_val + n_test == len(df_sintetico)

    # Ninguna ventana debe aparecer en dos conjuntos diferentes
    assert ventanas_train.max() < ventanas_val.min(), (
        "Fuga temporal: entrenamiento y validación comparten "
        "al menos una ventana temporal."
    )

    assert ventanas_val.max() < ventanas_test.min(), (
        "Fuga temporal: validación y prueba comparten "
        "al menos una ventana temporal."
    )