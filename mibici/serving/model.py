"""Módulo de recuperación de modelo desde MLflow e inferencia para P1 (Demanda Diaria).

Componente 1:
- Recupera el modelo registrado `mibici-equipo36-p1-demanda` desde MLflow Model Registry
  utilizando su nombre y alias (`champion`) o versión (`2`), o bien desde una URI/ruta
  local configurada por entorno para despliegue en contenedor.
- Prepara las 12 características de entrada en el orden y tipos de datos exactos
  requeridos por la firma del modelo (`int64`, `int32` y `float64`).
- Ejecuta predicciones y maneja errores durante la carga e inferencia.
"""

import argparse
from collections.abc import Mapping
import json
import os
from typing import Any

import joblib
from loguru import logger
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
P1_MODEL_NAME = "mibici-equipo36-p1-demanda"
P1_MODEL_ALIAS = "champion"
P1_MODEL_VERSION = "2"

# Las 12 características exactas utilizadas en el entrenamiento de P1 (Random Forest)
P1_FEATURE_COLS: list[str] = [
    "estacion_id",
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

# Tipos de datos esperados por la firma (signature) registrada en MLflow para P1
P1_FEATURE_DTYPES: dict[str, str] = {
    "estacion_id": "int64",
    "dia_semana": "int32",
    "mes": "int32",
    "es_fin_de_semana": "int64",
    "viajes_dia_anterior": "float64",
    "viajes_hace_7_dias": "float64",
    "promedio_7_dias": "float64",
    "promedio_14_dias": "float64",
    "viajes_hace_14_dias": "float64",
    "viajes_hace_28_dias": "float64",
    "promedio_28_dias": "float64",
    "promedio_mismo_dia_4_semanas": "float64",
}

EJEMPLO_ENTRADA_P1: dict[str, int | float] = {
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

# Caché en memoria del modelo cargado para evitar recargas en cada petición HTTP
_MODELO_CACHE: dict[str, Any] = {
    "modelo": None,
    "model_uri": None,
    "identificador": None,
}


class ErrorCargaModelo(RuntimeError):
    """Excepción lanzada cuando falla la recuperación o carga del modelo desde MLflow."""


class ErrorPrediccionModelo(ValueError):
    """Excepción lanzada cuando los datos de entrada son inválidos o falla la inferencia."""


def construir_model_uri(
    nombre_modelo: str | None = None,
    alias: str | None = None,
    version: str | int | None = None,
    model_uri: str | None = None,
) -> str:
    """Construye la URI de MLflow para recuperar el modelo P1.

    Prioridad de resolución:
    1. Parámetro `model_uri` explícito o variable de entorno `MIBICI_MODEL_URI`
       (permite cargar un artefacto local empaquetado en el contenedor o una URI directa).
    2. Si se proporciona `version` explícitamente y no se especifica `alias`, utiliza
       `models:/<nombre_modelo>/<version>`.
    3. Por defecto utiliza el alias configurado (`champion`):
       `models:/<nombre_modelo>@<alias>`.
    """
    uri_explicita = model_uri or os.getenv("MIBICI_MODEL_URI")
    if uri_explicita:
        return uri_explicita.strip()

    nombre = (nombre_modelo or os.getenv("MIBICI_MODEL_NAME", P1_MODEL_NAME)).strip()
    if not nombre:
        raise ErrorCargaModelo("El nombre del modelo no puede estar vacío.")

    if version is not None and alias is None:
        return f"models:/{nombre}/{version}"

    alias_efectivo = (
        alias if alias is not None else os.getenv("MIBICI_MODEL_ALIAS", P1_MODEL_ALIAS)
    ).strip()

    if alias_efectivo:
        return f"models:/{nombre}@{alias_efectivo}"

    version_efectiva = str(
        version if version is not None else os.getenv("MIBICI_MODEL_VERSION", P1_MODEL_VERSION)
    ).strip()
    return f"models:/{nombre}/{version_efectiva}"


def obtener_identificador_modelo(model_uri: str | None = None) -> str:
    """Devuelve una etiqueta legible del modelo activo para incluir en la respuesta JSON."""
    uri = model_uri or _MODELO_CACHE.get("model_uri") or construir_model_uri()
    if uri.startswith("models:/"):
        return uri.removeprefix("models:/")
    return uri


class _JoblibModelWrapper:
    """Adaptador ligero cuando se carga un artefacto .joblib empaquetado localmente."""

    def __init__(self, objeto_cargado: Any, model_uri: str) -> None:
        if isinstance(objeto_cargado, Mapping) and "model" in objeto_cargado:
            self._model = objeto_cargado["model"]
        else:
            self._model = objeto_cargado
        self._mibici_model_uri = model_uri

    def predict(self, df: pd.DataFrame) -> Any:
        return self._model.predict(df)


def _adjuntar_uri_modelo(modelo: Any, uri: str) -> Any:
    """Registra en la instancia cargada la URI efectiva desde la cual se recuperó."""
    try:
        modelo._mibici_model_uri = uri
    except (AttributeError, TypeError) as exc:
        logger.debug(f"No fue posible adjuntar _mibici_model_uri al modelo: {exc}")
    return modelo


def cargar_modelo(
    nombre_modelo: str | None = None,
    alias: str | None = None,
    version: str | int | None = None,
    tracking_uri: str | None = None,
    model_uri: str | None = None,
) -> Any:
    """Recupera y carga el modelo P1 desde MLflow Model Registry (o ruta configurada).

    Si la carga mediante alias (por ejemplo `@champion`) falla por no encontrarse el alias
    en el servidor MLflow, intenta recuperar automáticamente la versión indicada (`2` por
    defecto) antes de propagar `ErrorCargaModelo`.
    """
    uri_tracking = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    ).strip()
    mlflow.set_tracking_uri(uri_tracking)

    uri_objetivo = construir_model_uri(
        nombre_modelo=nombre_modelo,
        alias=alias,
        version=version,
        model_uri=model_uri,
    )

    logger.info(
        f"Cargando modelo de MiBici desde '{uri_objetivo}' "
        f"(MLFLOW_TRACKING_URI='{uri_tracking}')..."
    )

    if uri_objetivo.endswith((".joblib", ".pkl")):
        try:
            objeto = joblib.load(uri_objetivo)
            logger.success(f"Modelo joblib cargado exitosamente: {uri_objetivo}")
            return _JoblibModelWrapper(objeto, uri_objetivo)
        except Exception as exc_joblib:
            raise ErrorCargaModelo(
                f"No se pudo cargar el archivo de modelo local '{uri_objetivo}': {exc_joblib}"
            ) from exc_joblib

    try:
        modelo = mlflow.pyfunc.load_model(uri_objetivo)
        logger.success(f"Modelo cargado exitosamente: {uri_objetivo}")
        return _adjuntar_uri_modelo(modelo, uri_objetivo)
    except Exception as exc_primaria:
        # Si se intentó por alias (@champion) y no se forzó model_uri, intentar por versión
        nombre = (nombre_modelo or os.getenv("MIBICI_MODEL_NAME", P1_MODEL_NAME)).strip()
        version_fallback = str(
            version if version is not None else os.getenv("MIBICI_MODEL_VERSION", P1_MODEL_VERSION)
        ).strip()
        uri_fallback = f"models:/{nombre}/{version_fallback}"

        if (
            model_uri is None
            and not os.getenv("MIBICI_MODEL_URI")
            and "@" in uri_objetivo
            and version_fallback
        ):
            logger.warning(
                f"No fue posible cargar '{uri_objetivo}' ({exc_primaria}). "
                f"Intentando respaldo por versión '{uri_fallback}'..."
            )
            try:
                modelo = mlflow.pyfunc.load_model(uri_fallback)
                logger.success(f"Modelo cargado exitosamente por versión: {uri_fallback}")
                return _adjuntar_uri_modelo(modelo, uri_fallback)
            except Exception as exc_secundaria:
                raise ErrorCargaModelo(
                    f"No se pudo recuperar el modelo '{nombre}' desde MLflow "
                    f"(tracking_uri='{uri_tracking}'). Falló por alias ('{uri_objetivo}': "
                    f"{exc_primaria}) y por versión ('{uri_fallback}': {exc_secundaria})."
                ) from exc_secundaria

        raise ErrorCargaModelo(
            f"No se pudo recuperar el modelo desde '{uri_objetivo}' "
            f"(tracking_uri='{uri_tracking}'): {exc_primaria}"
        ) from exc_primaria


def obtener_modelo(forzar_recarga: bool = False, **kwargs: Any) -> Any:
    """Obtiene la instancia en caché del modelo P1 o la carga si aún no está en memoria."""
    if forzar_recarga or _MODELO_CACHE["modelo"] is None:
        modelo = cargar_modelo(**kwargs)
        uri_efectiva = getattr(modelo, "_mibici_model_uri", None) or construir_model_uri(
            nombre_modelo=kwargs.get("nombre_modelo"),
            alias=kwargs.get("alias"),
            version=kwargs.get("version"),
            model_uri=kwargs.get("model_uri"),
        )
        _MODELO_CACHE["modelo"] = modelo
        _MODELO_CACHE["model_uri"] = uri_efectiva
        _MODELO_CACHE["identificador"] = obtener_identificador_modelo(uri_efectiva)
    return _MODELO_CACHE["modelo"]


def limpiar_cache_modelo() -> None:
    """Limpia la instancia en caché del modelo (útil para pruebas automatizadas)."""
    _MODELO_CACHE["modelo"] = None
    _MODELO_CACHE["model_uri"] = None
    _MODELO_CACHE["identificador"] = None


def preparar_caracteristicas_p1(
    datos: Mapping[str, Any] | list[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Valida, ordena y convierte las 12 características de entrada para el modelo P1.

    Asegura que los tipos coincidan con la firma registrada en MLflow:
    - `estacion_id`, `es_fin_de_semana`: `int64`
    - `dia_semana`, `mes`: `int32`
    - variables históricas de viajes y promedios: `float64`
    """
    if isinstance(datos, pd.DataFrame):
        df = datos.copy()
    elif isinstance(datos, Mapping):
        df = pd.DataFrame([dict(datos)])
    elif isinstance(datos, list) and len(datos) > 0 and isinstance(datos[0], Mapping):
        df = pd.DataFrame([dict(fila) for fila in datos])
    else:
        raise ErrorPrediccionModelo(
            "Los datos de entrada deben ser un diccionario, una lista de diccionarios "
            "o un pandas.DataFrame con las 12 características de P1."
        )

    if df.empty:
        raise ErrorPrediccionModelo("El conjunto de datos de entrada está vacío.")

    faltantes = [col for col in P1_FEATURE_COLS if col not in df.columns]
    if faltantes:
        raise ErrorPrediccionModelo(
            f"Faltan características requeridas para el modelo P1: {faltantes}."
        )

    extras = [col for col in df.columns if col not in P1_FEATURE_COLS]
    if extras:
        raise ErrorPrediccionModelo(
            f"Se recibieron campos no permitidos para el modelo P1: {extras}."
        )

    # Seleccionar en el orden exacto del entrenamiento
    df = df[P1_FEATURE_COLS].copy()

    # Verificar que no existan nulos
    if df.isna().any().any():
        cols_nulas = df.columns[df.isna().any()].tolist()
        raise ErrorPrediccionModelo(
            f"Existen valores nulos (NaN/None) en las características: {cols_nulas}."
        )

    # Convertir tipos y verificar que sean numéricos y finitos
    for col, dtype_str in P1_FEATURE_DTYPES.items():
        try:
            valores_num = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ErrorPrediccionModelo(
                f"La característica '{col}' contiene un valor no numérico."
            ) from exc

        if not np.isfinite(valores_num.to_numpy(dtype="float64")).all():
            raise ErrorPrediccionModelo(
                f"La característica '{col}' contiene valores infinitos o inválidos."
            )

        # Para columnas enteras, comprobar que no tengan parte fraccionaria
        if dtype_str.startswith("int") and not np.all(
            np.equal(np.mod(valores_num.to_numpy(dtype="float64"), 1), 0)
        ):
            raise ErrorPrediccionModelo(
                f"La característica '{col}' debe ser un número entero."
            )

        df[col] = valores_num.astype(dtype_str)

    # Validaciones de dominio consistentes con las reglas de negocio de P1
    if (df["estacion_id"] < 0).any():
        raise ErrorPrediccionModelo("El campo 'estacion_id' debe ser mayor o igual a 0.")
    if ((df["dia_semana"] < 0) | (df["dia_semana"] > 6)).any():
        raise ErrorPrediccionModelo(
            "El campo 'dia_semana' debe estar entre 0 (lunes) y 6 (domingo)."
        )
    if ((df["mes"] < 1) | (df["mes"] > 12)).any():
        raise ErrorPrediccionModelo("El campo 'mes' debe estar entre 1 y 12.")
    if (~df["es_fin_de_semana"].isin([0, 1])).any():
        raise ErrorPrediccionModelo("El campo 'es_fin_de_semana' debe ser 0 o 1.")

    cols_historicas = P1_FEATURE_COLS[4:]
    for col in cols_historicas:
        if (df[col] < 0).any():
            raise ErrorPrediccionModelo(
                f"La característica histórica '{col}' no puede ser negativa."
            )

    return df


def predecir_demanda(
    datos: Mapping[str, Any] | pd.DataFrame,
    modelo: Any | None = None,
) -> dict[str, Any]:
    """Ejecuta la predicción de demanda diaria para P1 y devuelve el resultado estructurado.

    Parámetros:
    -----------
    datos : Mapping[str, Any] | pd.DataFrame
        Diccionario o DataFrame de una fila con las 12 características de entrada.
    modelo : Any, opcional
        Instancia de modelo ya cargada (con método `.predict(df)`). Si no se especifica,
        se recupera mediante `obtener_modelo()`.

    Retorna:
    --------
    dict con claves:
        - `estacion_id` (int)
        - `demanda_estimada` (float, >= 0.0)
        - `modelo` (str)
    """
    df_entrada = preparar_caracteristicas_p1(datos)
    modelo_activo = modelo if modelo is not None else obtener_modelo()

    try:
        predicciones = modelo_activo.predict(df_entrada)
    except Exception as exc:
        raise ErrorPrediccionModelo(
            f"Error al ejecutar la inferencia con el modelo P1: {exc}"
        ) from exc

    valores_pred = np.asarray(predicciones, dtype="float64").reshape(-1)
    if valores_pred.size == 0 or not np.isfinite(valores_pred[0]):
        raise ErrorPrediccionModelo(
            "El modelo devolvió una predicción vacía o no numérica."
        )

    # La demanda diaria de viajes no puede ser negativa (consistente con train.py)
    demanda_estimada = float(max(0.0, valores_pred[0]))
    estacion_id = int(df_entrada["estacion_id"].iloc[0])
    uri_modelo = getattr(modelo_activo, "_mibici_model_uri", None)
    etiqueta_modelo = (
        obtener_identificador_modelo(uri_modelo)
        if uri_modelo
        else (_MODELO_CACHE.get("identificador") or obtener_identificador_modelo())
    )

    return {
        "estacion_id": estacion_id,
        "demanda_estimada": demanda_estimada,
        "modelo": etiqueta_modelo,
    }


def main() -> None:
    """Punto de entrada CLI para verificar la carga del modelo P1 y ejecutar una predicción."""
    parser = argparse.ArgumentParser(
        description="Recuperación del modelo P1 desde MLflow e inferencia de prueba."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="URI del servidor MLflow (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MIBICI_MODEL_NAME", P1_MODEL_NAME),
        help="Nombre del modelo registrado en MLflow.",
    )
    parser.add_argument(
        "--alias",
        default=os.getenv("MIBICI_MODEL_ALIAS", P1_MODEL_ALIAS),
        help="Alias del modelo en MLflow Model Registry (por defecto: champion).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Versión específica del modelo si se desea consultar sin alias (ej. 2).",
    )
    parser.add_argument(
        "--model-uri",
        default=os.getenv("MIBICI_MODEL_URI"),
        help="URI explícita o ruta local del modelo (opcional).",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Cadena JSON con las 12 características de entrada (opcional).",
    )
    args = parser.parse_args()

    datos_entrada = (
        json.loads(args.input_json) if args.input_json else dict(EJEMPLO_ENTRADA_P1)
    )

    alias_arg = None if args.version is not None else args.alias
    modelo = obtener_modelo(
        forzar_recarga=True,
        nombre_modelo=args.model_name,
        alias=alias_arg,
        version=args.version,
        tracking_uri=args.tracking_uri,
        model_uri=args.model_uri,
    )

    resultado = predecir_demanda(datos_entrada, modelo=modelo)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
