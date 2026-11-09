
from pydantic import BaseModel, Field, ConfigDict


class PrediccionEntrada(BaseModel):
    """Características requeridas por el modelo P1 de MiBici."""

    model_config = ConfigDict(extra="forbid")

    estacion_id: int = Field(ge=0)
    dia_semana: int = Field(ge=0, le=6)
    mes: int = Field(ge=1, le=12)
    es_fin_de_semana: int = Field(ge=0, le=1)

    viajes_dia_anterior: float = Field(ge=0)
    viajes_hace_7_dias: float = Field(ge=0)
    promedio_7_dias: float = Field(ge=0)
    promedio_14_dias: float = Field(ge=0)
    viajes_hace_14_dias: float = Field(ge=0)
    viajes_hace_28_dias: float = Field(ge=0)
    promedio_28_dias: float = Field(ge=0)
    promedio_mismo_dia_4_semanas: float = Field(ge=0)


class PrediccionSalida(BaseModel):
    """Respuesta del servicio de predicción."""

    estacion_id: int
    demanda_estimada: float = Field(ge=0)
    modelo: str
