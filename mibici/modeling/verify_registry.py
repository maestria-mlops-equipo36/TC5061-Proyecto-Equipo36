"""
Verificación de modelos registrados en MLflow Model Registry.

Comprueba:
1. Existencia del modelo registrado.
2. Alias champion.
3. Versión y corrida de origen.
4. Firma del modelo.
5. Carga mediante models:/<nombre>@champion.

Uso:
    python -m mibici.modeling.verify_registry
"""

import os

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


MODELOS = {
    "p1": "mibici-equipo36-p1-demanda",
    "p2": "mibici-equipo36-p2-riesgo",
    "p3": "mibici-equipo36-p3-grupos",
}


def verificar_modelo(client, pregunta, nombre):
    """Verifica un modelo registrado y su alias champion."""

    print(f"\n{'=' * 60}")
    print(f"Verificando {pregunta.upper()}: {nombre}")
    print("=" * 60)

    # 1. Comprobar que existe el modelo registrado
    modelo_registrado = client.get_registered_model(nombre)

    print(f"Modelo encontrado: {modelo_registrado.name}")
    print(f"Descripción: {modelo_registrado.description}")

    # 2. Recuperar la versión champion
    version = client.get_model_version_by_alias(
        name=nombre,
        alias="champion",
    )

    print(f"Alias: champion")
    print(f"Versión: {version.version}")
    print(f"Run ID: {version.run_id}")
    print(f"Source: {version.source}")

    # 3. Comprobar linaje
    print("\nLinaje:")

    for tag in ["source_run_id", "git_commit", "data_version"]:
        valor = version.tags.get(tag, "No disponible")
        print(f"  {tag}: {valor}")

    # 4. Cargar el modelo mediante el alias
    model_uri = f"models:/{nombre}@champion"

    print(f"\nCargando modelo: {model_uri}")

    modelo = mlflow.pyfunc.load_model(model_uri)

    print("Modelo cargado correctamente.")

    # 5. Verificar firma
    signature = modelo.metadata.signature

    if signature is None:
        raise RuntimeError(
            f"El modelo {nombre} no tiene firma registrada."
        )

    print(f"Firma de entrada: {signature.inputs}")
    print(f"Firma de salida: {signature.outputs}")

    return True


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow Tracking URI: {tracking_uri}")

    client = MlflowClient()

    resultados = {}

    for pregunta, nombre in MODELOS.items():
        try:
            resultados[pregunta] = verificar_modelo(
                client,
                pregunta,
                nombre,
            )
        except Exception as exc:
            resultados[pregunta] = False
            print(f"\nERROR en {pregunta.upper()}: {exc}")

    print(f"\n{'=' * 60}")
    print("RESUMEN DE VERIFICACIÓN")
    print("=" * 60)

    for pregunta, resultado in resultados.items():
        estado = "OK" if resultado else "ERROR"
        print(f"{pregunta.upper()}: {estado}")

    if not all(resultados.values()):
        raise SystemExit(1)

    print("\nLos tres modelos están disponibles en producción.")


if __name__ == "__main__":
    main()