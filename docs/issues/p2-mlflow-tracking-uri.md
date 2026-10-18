# Corrección de conexión MLflow – P2

## Problema identificado

Durante las pruebas del modelo de clasificación P2 se identificó que el archivo `mibici/modeling/train_p2.py` configura MLflow utilizando el puerto `5001`.

```python
mlflow.set_tracking_uri("http://localhost:5001")
```

Sin embargo, el servidor central de MLflow utilizado para las pruebas del equipo se encuentra en el puerto `5000`.

## Corrección solicitada

- Actualizar la configuración para utilizar el puerto `5000`.
- Preferentemente, permitir que la dirección del servidor se configure mediante la variable de entorno `MLFLOW_TRACKING_URI`, evitando valores fijos en el código.
- Verificar que se registren correctamente las dos corridas del modelo P2.

## Evidencia de validación

Al utilizar temporalmente `http://127.0.0.1:5000`, ambas corridas se registraron correctamente en MLflow 3.17.1.

| Modelo | F1 macro | ROC AUC |
|---|---:|---:|
| DummyClassifier | 0.4689 | — |
| LogisticRegression | 0.6910 | 0.8652 |

**Resultado:** Entrenamiento y registro exitosos.

## Criterios de aceptación

- El entrenamiento se conecta al servidor MLflow configurado.
- Se registran ambas corridas en el experimento `mibici-equipo36`.
- Se conservan las métricas y los artefactos generados.
- No se requiere modificar manualmente el código para cambiar el puerto.
