
from fastapi import FastAPI, HTTPException

from mibici.serving import model as serving_model
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
    """Ejecuta una predicción real utilizando el modelo P1."""

    try:
        resultado = serving_model.predecir_demanda(
            datos.model_dump()
        )

        return PrediccionSalida(**resultado)

    except serving_model.ErrorCargaModelo:
        raise HTTPException(
            status_code=503,
            detail="El modelo de predicción no está disponible.",
        )

    except serving_model.ErrorPrediccionModelo:
        raise HTTPException(
            status_code=500,
            detail="No fue posible ejecutar la predicción.",
        )
