from fastapi.testclient import TestClient
from mibici.serving.app import app
import pytest
import mibici.serving.model as serving_model

client = TestClient(app)

from mibici.serving.model import EJEMPLO_ENTRADA_P1

DATOS_VALIDOS = EJEMPLO_ENTRADA_P1.copy()


def test_health_status_code():
    """Verifica que el endpoint de salud responde con HTTP 200."""
    response = client.get("/health")

    assert response.status_code == 200


def test_health_response():
    """Verifica que el endpoint devuelve la información esperada."""
    response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "service": "mibici-api",
    }


def test_predict_entrada_valida():
    """Verifica que una solicitud válida devuelve HTTP 200."""

    datos = {
        "estacion_id": 10,
        "dia_semana": 2,
        "mes": 10,
        "es_fin_de_semana": 0,
        "viajes_dia_anterior": 25.0,
        "viajes_hace_7_dias": 28.0,
        "promedio_7_dias": 26.5,
        "promedio_14_dias": 27.3,
        "viajes_hace_14_dias": 30.0,
        "viajes_hace_28_dias": 24.0,
        "promedio_28_dias": 26.8,
        "promedio_mismo_dia_4_semanas": 27.0,
    }

    response = client.post("/predict", json=datos)

    assert response.status_code == 200

    resultado = response.json()

    assert resultado["estacion_id"] == 10
    assert isinstance(resultado["demanda_estimada"], (int, float))
    assert resultado["demanda_estimada"] >= 0
    assert isinstance(resultado["modelo"], str)


@pytest.mark.parametrize("campo_faltante", list(DATOS_VALIDOS.keys()))
def test_predict_campo_obligatorio_faltante(campo_faltante):
    """Verifica que la API rechaza solicitudes con campos faltantes."""

    datos = DATOS_VALIDOS.copy()
    datos.pop(campo_faltante)

    response = client.post("/predict", json=datos)

    assert response.status_code == 422

    errores = response.json()["detail"]

    assert any(
        error["loc"] == ["body", campo_faltante]
        and error["type"] == "missing"
        for error in errores
    )


@pytest.mark.parametrize(
    "campo, valor_invalido",
    [
        ("estacion_id", -1),
        ("dia_semana", -1),
        ("dia_semana", 7),
        ("mes", 0),
        ("mes", 13),
        ("es_fin_de_semana", -1),
        ("es_fin_de_semana", 2),
        ("viajes_dia_anterior", -5.0),
        ("viajes_hace_7_dias", -10.0),
        ("promedio_7_dias", -1.5),
        ("promedio_14_dias", -2.0),
        ("viajes_hace_14_dias", -3.0),
        ("viajes_hace_28_dias", -4.0),
        ("promedio_28_dias", -1.0),
        ("promedio_mismo_dia_4_semanas", -2.5),
    ],
)
def test_predict_valores_fuera_de_rango(campo, valor_invalido):
    """Verifica que la API rechaza características fuera de rango."""

    datos = DATOS_VALIDOS.copy()
    datos[campo] = valor_invalido

    response = client.post("/predict", json=datos)

    assert response.status_code == 422

    errores = response.json()["detail"]

    assert any(
        error["loc"] == ["body", campo]
        for error in errores
    )


def test_predict_campo_adicional():
    """Verifica que la API rechaza características no definidas."""

    datos = DATOS_VALIDOS.copy()
    datos["temperatura"] = 30.0

    response = client.post("/predict", json=datos)

    assert response.status_code == 422

    errores = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "temperatura"]
        and error["type"] == "extra_forbidden"
        for error in errores
    )


@pytest.mark.parametrize(
    "campo, valor_invalido",
    [
        ("estacion_id", "estacion_desconocida"),
        ("estacion_id", 10.5),
        ("dia_semana", "lunes"),
        ("mes", "octubre"),
        ("es_fin_de_semana", "si"),
        ("viajes_dia_anterior", "muchos"),
        ("viajes_hace_7_dias", "sin_datos"),
        ("promedio_7_dias", "desconocido"),
        ("promedio_14_dias", [10, 20]),
        ("viajes_hace_14_dias", {"valor": 30}),
        ("viajes_hace_28_dias", "N/A"),
        ("promedio_28_dias", "invalido"),
        ("promedio_mismo_dia_4_semanas", "sin_promedio"),
    ],
)
def test_predict_tipos_incompatibles(campo, valor_invalido):
    """Verifica que la API rechaza tipos de datos incompatibles."""

    datos = DATOS_VALIDOS.copy()
    datos[campo] = valor_invalido

    response = client.post("/predict", json=datos)

    assert response.status_code == 422

    errores = response.json()["detail"]

    assert any(
        error["loc"] == ["body", campo]
        for error in errores
    )


@pytest.mark.parametrize(
    "campo, valor_limite",
    [
        ("estacion_id", 0),
        ("dia_semana", 0),
        ("dia_semana", 6),
        ("mes", 1),
        ("mes", 12),
        ("es_fin_de_semana", 0),
        ("es_fin_de_semana", 1),
        ("viajes_dia_anterior", 0.0),
        ("viajes_hace_7_dias", 0.0),
        ("promedio_7_dias", 0.0),
        ("promedio_14_dias", 0.0),
        ("viajes_hace_14_dias", 0.0),
        ("viajes_hace_28_dias", 0.0),
        ("promedio_28_dias", 0.0),
        ("promedio_mismo_dia_4_semanas", 0.0),
    ],
)
def test_predict_valores_limite(campo, valor_limite):
    """Verifica que la API acepta los límites permitidos."""

    datos = DATOS_VALIDOS.copy()
    datos[campo] = valor_limite

    response = client.post("/predict", json=datos)

    assert response.status_code == 200

    resultado = response.json()

    assert resultado["estacion_id"] == datos["estacion_id"]
    assert resultado["demanda_estimada"] >= 0


@pytest.mark.parametrize(
    "entrada_invalida",
    [
        {},
        [],
        None,
        "datos_invalidos",
        [DATOS_VALIDOS],
    ],
)
def test_predict_estructura_invalida(entrada_invalida):
    """Verifica que la API rechaza estructuras JSON incorrectas."""

    response = client.post("/predict", json=entrada_invalida)

    assert response.status_code == 422

    errores = response.json()["detail"]

    assert isinstance(errores, list)
    assert len(errores) > 0


@pytest.mark.xfail(
    reason="Pendiente integrar /predict con la función de inferencia P1",
    strict=True,
)
def test_predict_utiliza_funcion_inferencia(monkeypatch):
    """Verifica que FastAPI utiliza la función de inferencia del modelo P1."""

    llamadas = []

    def mock_predecir_demanda(datos, modelo=None):
        llamadas.append(datos)

        return {
            "estacion_id": datos["estacion_id"],
            "demanda_estimada": 42.5,
            "modelo": "modelo-prueba",
        }

    monkeypatch.setattr(
        serving_model,
        "predecir_demanda",
        mock_predecir_demanda,
    )

    response = client.post("/predict", json=DATOS_VALIDOS)

    assert response.status_code == 200
    assert len(llamadas) == 1
    assert llamadas[0] == DATOS_VALIDOS

    resultado = response.json()

    assert resultado["estacion_id"] == DATOS_VALIDOS["estacion_id"]
    assert resultado["demanda_estimada"] == 42.5
    assert resultado["modelo"] == "modelo-prueba"


def test_inferencia_p1_con_modelo_mock():
    """Verifica la inferencia P1 utilizando un modelo simulado."""

    class ModeloMock:
        _mibici_model_uri = "models:/mibici-equipo36-p1-demanda/2"

        def __init__(self):
            self.entradas_recibidas = []

        def predict(self, datos):
            self.entradas_recibidas.append(datos.copy())
            return [42.5]

    modelo_mock = ModeloMock()

    resultado = serving_model.predecir_demanda(
        datos=DATOS_VALIDOS,
        modelo=modelo_mock,
    )

    assert len(modelo_mock.entradas_recibidas) == 1

    df_recibido = modelo_mock.entradas_recibidas[0]

    assert len(df_recibido) == 1
    assert len(df_recibido.columns) == 12
    assert set(df_recibido.columns) == set(DATOS_VALIDOS.keys())

    for campo, valor in DATOS_VALIDOS.items():
        assert df_recibido.iloc[0][campo] == valor

    assert resultado["estacion_id"] == DATOS_VALIDOS["estacion_id"]
    assert resultado["demanda_estimada"] == 42.5
    assert isinstance(resultado["modelo"], str)
    assert resultado["modelo"]


def test_inferencia_p1_prediccion_negativa():
    """Verifica que una predicción negativa se ajuste a cero."""

    class ModeloMock:
        _mibici_model_uri = "models:/mibici-equipo36-p1-demanda/2"

        def predict(self, datos):
            return [-15.0]

    resultado = serving_model.predecir_demanda(
        datos=DATOS_VALIDOS,
        modelo=ModeloMock(),
    )

    assert resultado["demanda_estimada"] == 0.0
    assert resultado["estacion_id"] == DATOS_VALIDOS["estacion_id"]


def test_inferencia_p1_error_modelo():
    """Verifica que se manejen los errores durante la inferencia."""

    class ModeloMock:
        def predict(self, datos):
            raise RuntimeError("Error simulado del modelo")

    with pytest.raises(
        serving_model.ErrorPrediccionModelo,
        match="Error al ejecutar la inferencia",
    ):
        serving_model.predecir_demanda(
            datos=DATOS_VALIDOS,
            modelo=ModeloMock(),
        )


@pytest.mark.parametrize(
    "prediccion_invalida",
    [
        [],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
    ],
)
def test_inferencia_p1_prediccion_invalida(prediccion_invalida):
    """Verifica que se rechacen predicciones vacías o no finitas."""

    class ModeloMock:
        def predict(self, datos):
            return prediccion_invalida

    with pytest.raises(
        serving_model.ErrorPrediccionModelo,
        match="predicción vacía o no numérica",
    ):
        serving_model.predecir_demanda(
            datos=DATOS_VALIDOS,
            modelo=ModeloMock(),
        )


@pytest.mark.integration
def test_inferencia_p1_modelo_real():
    """Verifica una predicción real con el modelo P1 registrado en MLflow."""

    resultado = serving_model.predecir_demanda(datos=DATOS_VALIDOS)

    assert resultado["estacion_id"] == DATOS_VALIDOS["estacion_id"]
    assert isinstance(resultado["demanda_estimada"], (int, float))
    assert resultado["demanda_estimada"] >= 0
    assert resultado["modelo"] == "mibici-equipo36-p1-demanda@champion"