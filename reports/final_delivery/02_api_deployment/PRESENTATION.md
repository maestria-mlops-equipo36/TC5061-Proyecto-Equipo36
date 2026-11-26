# Propuesta de presentación final — Despliegue y operación de la API

Bloque de diapositivas diseñado para integrarse en la presentación del Equipo 36 (~75 segundos de narración dentro del video de 2 a 5 minutos, incluyendo la demo funcional en vivo).

## Slide 1. Arquitectura de despliegue y contenerización
**Mensaje:** Todo el stack de inferencia y registro de modelos se levanta de forma reproducible con un solo comando.

- **Imagen unificada (`Containerfile`):** Basada en `python:3.12-slim`, instala dependencias controladas (`requirements.txt`), empaqueta `mibici/` y ejecuta bajo usuario sin privilegios (`appuser`).
- **Orquestación (`docker-compose.yml`):**
  - Servicio **`mlflow`** (puerto interno `5000`) con *backend store* SQLite (`mlflow.db`), artefactos locales (`mlartifacts/`) y `healthcheck` activo.
  - Servicio **`api`** (puerto `8000`) conectado a `http://mlflow:5000` mediante `depends_on` (`condition: service_healthy`).
- **Configuración del modelo:** Carga dinámica desde Model Registry mediante variables de entorno (`MIBICI_MODEL_NAME=mibici-equipo36-p1-demanda`, `MIBICI_MODEL_ALIAS=champion`, `MIBICI_MODEL_VERSION=2`).

**Visual:** [Servicios y contenedores activos](E05_docker_compose.png).

**Notas de exposición (~20 s):** Explicar cómo Docker Compose desacopla el servidor de tracking/registry (`mlflow`) del servicio de inferencia (`api`), asegurando que la API arranque únicamente cuando MLflow está saludable.

## Slide 2. Contrato de la API y validación de salud (FastAPI)
**Mensaje:** La API expone endpoints tipados, autodocumentados y listos para monitoreo operativo.

- **Documentación interactiva (`/docs`):** Swagger UI generado automáticamente por FastAPI con esquemas Pydantic y códigos de estado explícitos (`200 OK`, `422 Validation Error`).
- **Endpoint de salud (`GET /health`):** Responde `HTTP 200 OK` con `{"status":"ok","service":"mibici-api"}`, utilizado tanto por Docker `healthcheck` como por las pruebas de humo en CI/CD.
- **Contrato de predicción (`POST /predict`):** Recibe 12 atributos temporales y de rezagos históricos (`estacion_id`, `dia_semana`, `mes`, `es_fin_de_semana`, `viajes_dia_anterior`, `viajes_hace_7_dias`, `promedio_7_dias`, `promedio_14_dias`, `viajes_hace_14_dias`, `viajes_hace_28_dias`, `promedio_28_dias`, `promedio_mismo_dia_4_semanas`).

**Visual:** [Documentación Swagger UI](E06_swagger.png) y [Validación `/health`](E08_health.png).

**Notas de exposición (~20 s):** Mostrar rápidamente la interfaz Swagger UI y confirmar que `/health` reporta disponibilidad inmediata del microservicio.

## Slide 3. Demostración funcional en vivo (`POST /predict`)
**Mensaje:** El sistema responde en tiempo real utilizando el modelo Champion registrado en MLflow.

- **Petición real ejecutada (`estacion_id: 15`, día laboral de noviembre):**
  - Respuesta `HTTP/1.1 200 OK` (`content-type: application/json`).
  - Salida: `{"estacion_id":15,"demanda_estimada":0.0,"modelo":"mibici-equipo36-p1-demanda@champion"}`.
- **Trazabilidad:** La respuesta incluye explícitamente el identificador y alias del modelo servido (`@champion`), garantizando auditabilidad en producción.

**Visual:** [Petición real y resultado de predicción](E07_predict.png).

**Notas de exposición (~35 s):** Ejecutar el comando `curl` (o botón *Execute* en Swagger) frente a pantalla mostrando el `200 OK` y el JSON de respuesta con `@champion`.

## Transición al siguiente bloque
Con la API contenerizada y respondiendo peticiones reales en producción, el siguiente bloque muestra cómo validamos automáticamente su calidad, seguridad y rendimiento bajo carga mediante CI/CD y Locust.
