
from fastapi import FastAPI

from mibici.serving.schemas import PrediccionEntrada, PrediccionSalida


app = FastAPI(
    title="MiBici - API de Predicción",
    description="Servicio REST para consultar modelos de Machine Learning de MiBici.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Verifica que el servicio HTTP está disponible."""
    return {
        "status": "ok",
        "service": "mibici-api",
    }


@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: PrediccionEntrada):
    """Valida las características y devuelve una predicción simulada."""

    # Valor temporal para probar el funcionamiento de la API.
    # Se sustituirá al integrar el modelo registrado en MLflow.
    demanda_simulada = datos.promedio_7_dias

    return PrediccionSalida(
        estacion_id=datos.estacion_id,
        demanda_estimada=demanda_simulada,
        modelo="simulado",
    )
