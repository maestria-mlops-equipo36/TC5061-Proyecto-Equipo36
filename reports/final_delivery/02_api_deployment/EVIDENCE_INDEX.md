# Índice de evidencias — Despliegue y operación de la API

| ID | Archivo | Descripción | Reporte / fuente original | Uso en la presentación |
|---|---|---|---|---|
| E05 | [Servicios y contenedores activos](E05_docker_compose.png) | Orquestación mediante `docker compose` mostrando los servicios `api` (puerto `8000`) y `mlflow` (puerto `5000` / `5001` local) en estado `Up (healthy)` | `docker-compose.yml` y `Containerfile` | Demostración de contenerización y orquestación reproducible |
| E06 | [Documentación de la API (Swagger UI)](E06_swagger.png) | Interfaz OpenAPI (`/docs`) de FastAPI con el contrato de entrada (12 variables de demanda) y respuestas (`200` y `422`) de `POST /predict` | `mibici/api/main.py` y `http://localhost:8000/docs` | Explicación del contrato y esquema de consumo del servicio |
| E07 | [Petición real y resultado de predicción](E07_predict.png) | Ejecución de petición HTTP `POST /predict` con respuesta `200 OK` y predicción generada por el modelo `mibici-equipo36-p1-demanda@champion` | `POST http://localhost:8000/predict` | Demo funcional en vivo de inferencia sobre el modelo Champion |
| E08 | [Validación del estado del servicio](E08_health.png) | Verificación de disponibilidad del servicio mediante `GET /health` con respuesta `200 OK` (`{"status":"ok","service":"mibici-api"}`) | `GET http://localhost:8000/health` | Respaldo de operabilidad y monitoreo básico de salud |

> Todas las evidencias reflejan el estado consolidado del repositorio en `integration/final-delivery` y operan con el modelo Champion P1 (`mibici-equipo36-p1-demanda`, alias `champion`, versión `2`).
