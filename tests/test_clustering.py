import pandas as pd
import pytest
import numpy as np

from mibici.features import extraer_caracteristicas_temporales
from mibici.features import calcular_perfil_estaciones_p3

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    P3_DEFAULT_N_CLUSTERS,
    P3_FEATURE_COLS_ESPACIAL,
    RANDOM_SEED,
)

from sklearn.mixture import GaussianMixture

from mibici.config import P3_DEFAULT_COVARIANCE_TYPE

from sklearn.metrics import adjusted_rand_score

def test_caracteristicas_temporales_p3():
    """Verifica horarios pico, fines de semana y conservación de datos."""

    # Fechas con horarios representativos y casos límite
    fechas = pd.to_datetime([
        "2026-03-02 06:59:00",  # Lunes, antes del pico matutino
        "2026-03-02 07:00:00",  # Lunes, inicio del pico matutino
        "2026-03-02 10:59:00",  # Lunes, final del pico matutino
        "2026-03-02 11:00:00",  # Lunes, fuera del pico matutino
        "2026-03-02 17:00:00",  # Lunes, inicio del pico vespertino
        "2026-03-02 20:59:00",  # Lunes, final del pico vespertino
        "2026-03-02 21:00:00",  # Lunes, fuera del pico vespertino
        "2026-03-07 12:00:00",  # Sábado
        "2026-03-08 12:00:00",  # Domingo
    ])

    df_viajes = pd.DataFrame({
        "Inicio_del_viaje": fechas,
        "Viaje_Id": range(1, len(fechas) + 1),
    })

    # Conservar una copia para comprobar que no se modifica el original
    df_original = df_viajes.copy(deep=True)

    # Ejecutar la función real del proyecto
    resultado = extraer_caracteristicas_temporales(df_viajes)

    # Verificar los horarios pico matutinos
    assert resultado["pico_matutino"].tolist() == [
        0, 1, 1, 0, 0, 0, 0, 0, 0
    ]

    # Verificar los horarios pico vespertinos
    assert resultado["pico_vespertino"].tolist() == [
        0, 0, 0, 0, 1, 1, 0, 0, 0
    ]

    # Verificar que solamente sábado y domingo sean fin de semana
    assert resultado["es_fin_de_semana"].tolist() == [
        0, 0, 0, 0, 0, 0, 0, 1, 1
    ]

    # Verificar que las horas se extraigan correctamente
    assert resultado["hora"].tolist() == [
        6, 7, 10, 11, 17, 20, 21, 12, 12
    ]

    # Verificar que se conserven todos los registros
    assert len(resultado) == len(df_viajes)

    # Verificar que la función no modifique el DataFrame original
    pd.testing.assert_frame_equal(df_viajes, df_original)


def test_calcular_perfil_estaciones_p3():
    """Verifica los cálculos del perfil de estaciones con valores conocidos."""

    # Catálogo sintético de estaciones
    df_estaciones = pd.DataFrame({
        "estacion_id": [101, 202],
        "nombre": ["Estación A", "Estación B"],
        "latitud": [20.67, 20.68],
        "longitud": [-103.35, -103.36],
        "capacidad": [10, 20],
    })

    # Viajes sintéticos con horarios, duraciones y flujos conocidos
    df_viajes = pd.DataFrame({
        "Viaje_Id": [1, 2, 3, 4],
        "Origen_Id": [101, 101, 101, 202],
        "Destino_Id": [202, 202, 202, 101],
        "Inicio_del_viaje": pd.to_datetime([
            "2026-03-02 08:00:00",  # Lunes, pico matutino
            "2026-03-02 18:00:00",  # Lunes, pico vespertino
            "2026-03-07 12:00:00",  # Sábado, fin de semana
            "2026-03-02 12:00:00",  # Lunes, fuera de pico
        ]),
        "duracion_min": [10.0, 20.0, 30.0, 40.0],
    })

    # Ejecutar la función real del proyecto
    resultado = calcular_perfil_estaciones_p3(
        df_viajes,
        df_estaciones,
    )

    # Facilitar la consulta por identificador de estación
    perfiles = resultado.set_index("estacion_id")

    estacion_101 = perfiles.loc[101]
    estacion_202 = perfiles.loc[202]


    # Estación 101: principalmente emisora
    assert estacion_101["salidas_totales"] == 3
    assert estacion_101["llegadas_totales"] == 1
    assert estacion_101["total_viajes"] == 4

    assert estacion_101["ratio_flujo_neto"] == pytest.approx(0.5)

    assert estacion_101["pct_salidas_pico_matutino"] == pytest.approx(1 / 3)
    assert estacion_101["pct_salidas_pico_vespertino"] == pytest.approx(1 / 3)
    assert estacion_101["pct_salidas_fin_de_semana"] == pytest.approx(1 / 3)

    assert estacion_101["duracion_promedio_salidas_min"] == pytest.approx(20.0)

    # Estación 202: principalmente receptora
    assert estacion_202["salidas_totales"] == 1
    assert estacion_202["llegadas_totales"] == 3
    assert estacion_202["total_viajes"] == 4

    assert estacion_202["ratio_flujo_neto"] == pytest.approx(-0.5)

    assert estacion_202["pct_salidas_pico_matutino"] == pytest.approx(0.0)
    assert estacion_202["pct_salidas_pico_vespertino"] == pytest.approx(0.0)
    assert estacion_202["pct_salidas_fin_de_semana"] == pytest.approx(0.0)

    assert estacion_202["duracion_promedio_salidas_min"] == pytest.approx(40.0)


    # Intensidad de uso normalizada por capacidad
    # El código calcula la ventana observada como la diferencia
    # entre las fechas máxima y mínima, expresada en días.
    dias_totales = max(
        1,
        (
            df_viajes["Inicio_del_viaje"].max()
            - df_viajes["Inicio_del_viaje"].min()
        ).days,
    )

    assert estacion_101["viajes_diarios_por_anclaje"] == pytest.approx(
        (4 / dias_totales) / 10
    )

    assert estacion_202["viajes_diarios_por_anclaje"] == pytest.approx(
        (4 / dias_totales) / 20
    )


    # Integridad de los resultados
    assert len(resultado) == 2
    assert resultado["estacion_id"].is_unique

    columnas_numericas = resultado.select_dtypes(include="number")

    assert np.isfinite(columnas_numericas.to_numpy()).all()


def test_estaciones_sin_actividad_p3():
    """Verifica que las estaciones sin viajes se conserven y no generen NaN."""

    # Catálogo con dos estaciones activas y una sin actividad
    df_estaciones = pd.DataFrame({
        "estacion_id": [101, 202, 303],
        "nombre": ["Estación A", "Estación B", "Estación sin actividad"],
        "latitud": [20.67, 20.68, 20.69],
        "longitud": [-103.35, -103.36, -103.37],
        "capacidad": [10, 20, 15],
    })

    # La estación 303 no participa en ningún viaje
    df_viajes = pd.DataFrame({
        "Viaje_Id": [1, 2, 3],
        "Origen_Id": [101, 101, 202],
        "Destino_Id": [202, 202, 101],
        "Inicio_del_viaje": pd.to_datetime([
            "2026-03-02 08:00:00",
            "2026-03-02 18:00:00",
            "2026-03-03 12:00:00",
        ]),
        "duracion_min": [10.0, 30.0, 40.0],
    })

    # Ejecutar la función real del proyecto
    resultado = calcular_perfil_estaciones_p3(
        df_viajes,
        df_estaciones,
    )

    perfiles = resultado.set_index("estacion_id")


    # 1. Verificar que se conserven todas las estaciones
    assert len(resultado) == 3
    assert set(resultado["estacion_id"]) == {101, 202, 303}
    assert resultado["estacion_id"].is_unique


    # 2. Verificar los contadores de la estación sin actividad
    estacion_303 = perfiles.loc[303]

    assert estacion_303["salidas_totales"] == 0
    assert estacion_303["llegadas_totales"] == 0
    assert estacion_303["total_viajes"] == 0


    # 3. Verificar proporciones y flujo neto
    assert estacion_303["ratio_flujo_neto"] == pytest.approx(0.0)

    assert estacion_303["pct_salidas_pico_matutino"] == pytest.approx(0.0)
    assert estacion_303["pct_salidas_pico_vespertino"] == pytest.approx(0.0)
    assert estacion_303["pct_salidas_fin_de_semana"] == pytest.approx(0.0)


    # 4. Verificar intensidad de uso
    assert estacion_303["viajes_diarios_por_anclaje"] == pytest.approx(0.0)


    # 5. Verificar imputación de duración promedio
    # Estación 101: (10 + 30) / 2 = 20 minutos
    # Estación 202: 40 minutos
    # Mediana de ambas estaciones = 30 minutos
    assert estacion_303["duracion_promedio_salidas_min"] == pytest.approx(30.0)


    # 6. Verificar ausencia de valores faltantes o infinitos
    columnas_numericas = resultado.select_dtypes(include="number")

    assert not columnas_numericas.isna().any().any()
    assert np.isfinite(columnas_numericas.to_numpy()).all()


def test_capacidad_cero_p3():
    """Verifica que una estación con capacidad cero no genere valores infinitos."""

    # 1. Crear estaciones con capacidades diferentes
    df_estaciones = pd.DataFrame({
        "estacion_id": [101, 202],
        "nombre": ["Estación sin capacidad", "Estación normal"],
        "latitud": [20.67, 20.68],
        "longitud": [-103.35, -103.36],
        "capacidad": [0, 10],
    })

    # 2. Crear viajes donde ambas estaciones tengan actividad
    df_viajes = pd.DataFrame({
        "Viaje_Id": [1, 2, 3],
        "Origen_Id": [101, 101, 202],
        "Destino_Id": [202, 202, 101],
        "Inicio_del_viaje": pd.to_datetime([
            "2026-03-02 08:00:00",
            "2026-03-03 12:00:00",
            "2026-03-04 18:00:00",
        ]),
        "duracion_min": [10.0, 20.0, 30.0],
    })

    # 3. Ejecutar la función real del proyecto
    resultado = calcular_perfil_estaciones_p3(
        df_viajes,
        df_estaciones,
    )

    perfiles = resultado.set_index("estacion_id")

    estacion_101 = perfiles.loc[101]
    estacion_202 = perfiles.loc[202]

    # 4. Verificar que la estación con capacidad cero conserve sus datos
    assert estacion_101["capacidad"] == 0
    assert estacion_101["salidas_totales"] == 2
    assert estacion_101["llegadas_totales"] == 1
    assert estacion_101["total_viajes"] == 3

    # 5. Verificar que la intensidad sea cero y no infinita
    assert estacion_101["viajes_diarios_por_anclaje"] == pytest.approx(0.0)

    # 6. Verificar el cálculo de la estación con capacidad normal
    dias_totales = max(
        1,
        (
            df_viajes["Inicio_del_viaje"].max()
            - df_viajes["Inicio_del_viaje"].min()
        ).days,
    )

    intensidad_esperada = (3 / dias_totales) / 10

    assert estacion_202["viajes_diarios_por_anclaje"] == pytest.approx(
        intensidad_esperada
    )

    # 7. Verificar que no existan valores numéricos inválidos
    columnas_numericas = resultado.select_dtypes(include="number")

    assert not columnas_numericas.isna().any().any()
    assert np.isfinite(columnas_numericas.to_numpy()).all()

    # 8. Verificar que ambas estaciones permanezcan en el resultado
    assert len(resultado) == 2
    assert set(resultado["estacion_id"]) == {101, 202}


def test_entrenamiento_kmeans_p3():
    """Verifica el entrenamiento y las asignaciones de K-Means en P3."""

    # 1. Generar cuatro grupos sintéticos de estaciones
    rng = np.random.default_rng(RANDOM_SEED)

    n_clusters = P3_DEFAULT_N_CLUSTERS
    estaciones_por_grupo = 15

    # Cada centro representa un perfil de estación diferente.
    # Orden de columnas: flujo, pico AM, pico PM, fin de semana,
    # intensidad, latitud y longitud.
    centros = np.array([
        [0.70, 0.80, 0.10, 0.10, 2.50, 20.60, -103.30],
        [-0.70, 0.10, 0.80, 0.10, 1.50, 20.70, -103.40],
        [0.00, 0.20, 0.20, 0.80, 0.50, 20.80, -103.50],
        [0.10, 0.40, 0.40, 0.30, 4.00, 20.90, -103.60],
    ])

    # Introducir pequeñas variaciones para simular estaciones reales
    desviaciones = np.array([
        0.03, 0.03, 0.03, 0.03, 0.10, 0.005, 0.005
    ])

    grupos = []
    grupos_esperados = []

    for grupo_id, centro in enumerate(centros):
        observaciones = rng.normal(
            loc=centro,
            scale=desviaciones,
            size=(estaciones_por_grupo, len(centro)),
        )

        grupos.append(observaciones)
        grupos_esperados.extend([grupo_id] * estaciones_por_grupo)

    # 2. Construir el DataFrame con las características reales de P3
    X = pd.DataFrame(
        np.vstack(grupos),
        columns=P3_FEATURE_COLS_ESPACIAL,
    )

    grupos_esperados = np.array(grupos_esperados)

    assert list(X.columns) == P3_FEATURE_COLS_ESPACIAL
    assert len(X) == n_clusters * estaciones_por_grupo

    # 3. Crear el mismo pipeline utilizado en p3_clustering()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_SEED,
                n_init=10,
                max_iter=300,
            ),
        ),
    ])

    # 4. Entrenar el modelo y obtener las etiquetas
    labels = pipeline.fit_predict(X)

    # Debe existir una etiqueta por estación
    assert len(labels) == len(X)

    # Deben generarse exactamente cuatro clusters
    assert len(np.unique(labels)) == n_clusters

    # Las etiquetas deben estar dentro del rango esperado
    assert set(labels).issubset(set(range(n_clusters)))

    # 5. Verificar que el escalamiento se haya realizado correctamente
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    np.testing.assert_allclose(
        X_scaled.mean(axis=0),
        np.zeros(X.shape[1]),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        X_scaled.std(axis=0),
        np.ones(X.shape[1]),
        atol=1e-12,
    )

    # 6. Verificar que las estaciones de cada grupo permanezcan juntas
    # No comparamos números de cluster porque sus etiquetas son arbitrarias.
    for grupo_id in range(n_clusters):
        indices = grupos_esperados == grupo_id
        etiquetas_grupo = labels[indices]

        assert len(np.unique(etiquetas_grupo)) == 1

    # Verificar que grupos distintos reciban etiquetas diferentes
    etiquetas_representativas = [
        labels[np.where(grupos_esperados == grupo_id)[0][0]]
        for grupo_id in range(n_clusters)
    ]

    assert len(set(etiquetas_representativas)) == n_clusters

    # 7. Verificar la calidad y consistencia del modelo
    silhouette = silhouette_score(X_scaled, labels)

    assert np.isfinite(silhouette)
    assert silhouette > 0.5

    # La inercia debe ser finita y no negativa
    inercia = pipeline.named_steps["model"].inertia_

    assert np.isfinite(inercia)
    assert inercia >= 0


def test_entrenamiento_gmm_p3():
    """Verifica el entrenamiento, las etiquetas y las probabilidades de GMM."""

    # 1. Generar cuatro grupos sintéticos de estaciones
    rng = np.random.default_rng(RANDOM_SEED)

    n_clusters = P3_DEFAULT_N_CLUSTERS
    estaciones_por_grupo = 20

    # Centros con diferentes perfiles de comportamiento y ubicación
    centros = np.array([
        [0.70, 0.80, 0.10, 0.10, 2.50, 20.60, -103.30],
        [-0.70, 0.10, 0.80, 0.10, 1.50, 20.70, -103.40],
        [0.00, 0.20, 0.20, 0.80, 0.50, 20.80, -103.50],
        [0.10, 0.40, 0.40, 0.30, 4.00, 20.90, -103.60],
    ])

    desviaciones = np.array([
        0.03, 0.03, 0.03, 0.03, 0.10, 0.005, 0.005
    ])

    grupos = []
    grupos_esperados = []

    for grupo_id, centro in enumerate(centros):
        observaciones = rng.normal(
            loc=centro,
            scale=desviaciones,
            size=(estaciones_por_grupo, len(centro)),
        )

        grupos.append(observaciones)
        grupos_esperados.extend([grupo_id] * estaciones_por_grupo)

    # 2. Construir el dataset con las características de P3
    X = pd.DataFrame(
        np.vstack(grupos),
        columns=P3_FEATURE_COLS_ESPACIAL,
    )

    grupos_esperados = np.array(grupos_esperados)

    assert len(X) == n_clusters * estaciones_por_grupo
    assert list(X.columns) == P3_FEATURE_COLS_ESPACIAL

    # 3. Crear el mismo pipeline de GMM utilizado en P3
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            GaussianMixture(
                n_components=n_clusters,
                random_state=RANDOM_SEED,
                covariance_type=P3_DEFAULT_COVARIANCE_TYPE,
            ),
        ),
    ])

    # 4. Entrenar el modelo y obtener las etiquetas
    pipeline.fit(X)
    labels = pipeline.predict(X)

    assert len(labels) == len(X)
    assert len(np.unique(labels)) == n_clusters
    assert set(labels).issubset(set(range(n_clusters)))

    # 5. Verificar las probabilidades de pertenencia
    probabilidades = pipeline.predict_proba(X)

    assert probabilidades.shape == (len(X), n_clusters)

    # Cada probabilidad debe estar entre 0 y 1
    assert np.isfinite(probabilidades).all()
    assert (probabilidades >= 0).all()
    assert (probabilidades <= 1).all()

    # Las probabilidades de cada estación deben sumar 1
    np.testing.assert_allclose(
        probabilidades.sum(axis=1),
        np.ones(len(X)),
        rtol=1e-12,
        atol=1e-12,
    )

    # La etiqueta asignada debe corresponder a la mayor probabilidad
    np.testing.assert_array_equal(
        labels,
        probabilidades.argmax(axis=1),
    )

    # 6. Verificar que los cuatro grupos sintéticos se distingan
    etiquetas_representativas = []

    for grupo_id in range(n_clusters):
        indices = grupos_esperados == grupo_id
        etiquetas_grupo = labels[indices]

        # Las estaciones del mismo grupo deben permanecer juntas
        assert len(np.unique(etiquetas_grupo)) == 1

        etiquetas_representativas.append(etiquetas_grupo[0])

    # Cada grupo sintético debe recibir una etiqueta diferente
    assert len(set(etiquetas_representativas)) == n_clusters

    # 7. Verificar las métricas AIC y BIC utilizadas en P3
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    modelo_ajustado = pipeline.named_steps["model"]

    aic = modelo_ajustado.aic(X_scaled)
    bic = modelo_ajustado.bic(X_scaled)

    assert np.isfinite(aic)
    assert np.isfinite(bic)

    # Verificar que el modelo haya convergido
    assert modelo_ajustado.converged_


@pytest.mark.parametrize("modelo_tipo", ["kmeans", "gmm"])
def test_reproducibilidad_entrenamiento_p3(modelo_tipo):
    """Verifica que P3 produzca agrupaciones reproducibles con la misma semilla."""

    # 1. Generar datos sintéticos reproducibles
    rng = np.random.default_rng(RANDOM_SEED)

    centros = np.array([
        [0.70, 0.80, 0.10, 0.10, 2.50, 20.60, -103.30],
        [-0.70, 0.10, 0.80, 0.10, 1.50, 20.70, -103.40],
        [0.00, 0.20, 0.20, 0.80, 0.50, 20.80, -103.50],
        [0.10, 0.40, 0.40, 0.30, 4.00, 20.90, -103.60],
    ])

    desviaciones = np.array([
        0.03, 0.03, 0.03, 0.03, 0.10, 0.005, 0.005
    ])

    grupos = [
        rng.normal(
            loc=centro,
            scale=desviaciones,
            size=(20, len(centro)),
        )
        for centro in centros
    ]

    X = pd.DataFrame(
        np.vstack(grupos),
        columns=P3_FEATURE_COLS_ESPACIAL,
    )

    # 2. Crear una función para construir modelos independientes
    def crear_pipeline():
        if modelo_tipo == "kmeans":
            modelo = KMeans(
                n_clusters=P3_DEFAULT_N_CLUSTERS,
                random_state=RANDOM_SEED,
                n_init=10,
                max_iter=300,
            )
        else:
            modelo = GaussianMixture(
                n_components=P3_DEFAULT_N_CLUSTERS,
                random_state=RANDOM_SEED,
                covariance_type=P3_DEFAULT_COVARIANCE_TYPE,
            )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", modelo),
        ])

    # 3. Crear y entrenar dos pipelines independientes
    pipeline_1 = crear_pipeline()
    pipeline_2 = crear_pipeline()

    pipeline_1.fit(X)
    pipeline_2.fit(X)

    predicciones_1 = pipeline_1.predict(X)
    predicciones_2 = pipeline_2.predict(X)

    # 4. Verificar que ambas ejecuciones produzcan las mismas agrupaciones
    ari = adjusted_rand_score(
        predicciones_1,
        predicciones_2,
    )

    assert ari == pytest.approx(1.0)

    # 5. Verificar la reproducibilidad de la métrica Silhouette
    X_scaled_1 = pipeline_1.named_steps["scaler"].transform(X)
    X_scaled_2 = pipeline_2.named_steps["scaler"].transform(X)

    silhouette_1 = silhouette_score(X_scaled_1, predicciones_1)
    silhouette_2 = silhouette_score(X_scaled_2, predicciones_2)

    assert silhouette_1 == pytest.approx(
        silhouette_2,
        rel=1e-10,
        abs=1e-10,
    )

    # 6. Verificar métricas específicas de cada algoritmo
    modelo_1 = pipeline_1.named_steps["model"]
    modelo_2 = pipeline_2.named_steps["model"]

    if modelo_tipo == "kmeans":
        assert modelo_1.inertia_ == pytest.approx(
            modelo_2.inertia_,
            rel=1e-10,
            abs=1e-10,
        )

    else:
        # Comparar probabilidades de pertenencia de GMM
        probabilidades_1 = pipeline_1.predict_proba(X)
        probabilidades_2 = pipeline_2.predict_proba(X)

        np.testing.assert_allclose(
            probabilidades_1,
            probabilidades_2,
            rtol=1e-10,
            atol=1e-10,
        )

        # Comparar métricas AIC y BIC
        aic_1 = modelo_1.aic(X_scaled_1)
        aic_2 = modelo_2.aic(X_scaled_2)

        bic_1 = modelo_1.bic(X_scaled_1)
        bic_2 = modelo_2.bic(X_scaled_2)

        assert aic_1 == pytest.approx(
            aic_2,
            rel=1e-10,
            abs=1e-10,
        )

        assert bic_1 == pytest.approx(
            bic_2,
            rel=1e-10,
            abs=1e-10,
        )