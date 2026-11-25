# Propuesta de presentación final — CI/CD, pruebas y rendimiento

Bloque de tres diapositivas diseñado para integrarse en la presentación del Equipo 36 (~60 segundos de narración dentro del video de 2 a 5 minutos).

## Slide 1. Pipeline de CI/CD con GitHub Actions
**Mensaje:** Cada cambio al repositorio se valida automáticamente antes de integrarse.

- Workflow `CI/CD - MiBici MLOps` activado en `push` y `pull_request` hacia `main` e `integration/**`.
- **Job 1 (`test`):** Python 3.12 con caché de `pip`, ejecución de PyTest y publicación de reporte `JUnit XML`.
- **Job 2 (`build-docker`):** Construcción de `Containerfile` con caché GHA, smoke test en `/health` y validación completa del stack `Docker Compose` (`MLflow + FastAPI`).
- Ejecuciones exitosas en verde sobre `main`, `integration/final-delivery` y PRs del equipo.

**Visual:** [Pipeline CI/CD](E09_github_actions.png).

**Notas de exposición (~20 s):** Explicar que ningún cambio se integra sin pasar automáticamente pruebas, build de contenedor y verificación de salud de ambos servicios en Docker Compose.

## Slide 2. Aseguramiento de calidad y seguridad
**Mensaje:** La suite automatizada protege tanto la lógica ML como la API y la seguridad del código.

- **135 / 135 pruebas aprobadas** en 13.56 s distribuidas en 10 módulos (`test_api`: 72, `test_deepchecks`: 15, `test_regresion`: 13, `test_clustering`: 8, `test_clasificacion`: 7, `test_data`: 6, `test_ab_testing_p1`: 5, `test_security`: 4, `test_reproducibilidad`: 3, `test_monitoring`: 2).
- **Seguridad estática (Bandit):** 4,579 líneas analizadas, **0 vulnerabilidades críticas o medias**.
- **Auditoría (`pip-audit`) y protección de secretos:** Validación automática de dependencias y reglas en `.gitignore`.

**Visual:** [PyTest y seguridad](E10_pytest.png).

**Notas de exposición (~18 s):** Destacar la amplitud de las 135 pruebas (contrato de la API, calidad de datos con DeepChecks, no-regresión de los tres modelos y chequeos automáticos de seguridad).

## Slide 3. Pruebas de carga y rendimiento (Locust)
**Mensaje:** La API escala linealmente hasta 50 usuarios concurrentes con 0 % de errores.

| Concurrencia | Peticiones | Fallos | Throughput | P50 `/predict` | P95 `/predict` |
|---|---:|---:|---:|---:|---:|
| 10 usuarios | 340 | 0.00 % | 12.13 req/s | 15 ms | 28 ms |
| 25 usuarios | 1,133 | 0.00 % | 38.98 req/s | 10 ms | 25 ms |
| 50 usuarios | 2,024 | 0.00 % | 69.71 req/s | 8 ms | 21 ms |

- **Hallazgo clave:** *Cold start* puntual de ~6.48 s en la primera petición por carga perezosa del modelo; recomendación: pre-carga en el `startup` de FastAPI.
- Bajo cuotas estrictas de CPU (perfil FinOps), la latencia sube por saturación de CPU, lo que motiva el escalado horizontal.

**Visual:** [Métricas de rendimiento](E12_performance.png) (respaldo: [Dashboard Locust](E11_locust.png)).

**Notas de exposición (~22 s):** Mostrar la gráfica de throughput y latencia, enfatizar el 0 % de errores en 3,497 peticiones y conectar la saturación de CPU con el siguiente bloque de Monitoreo y FinOps.

## Transición al siguiente bloque
Con la calidad automatizada y el comportamiento bajo carga validados, el siguiente bloque presenta el monitoreo en producción (Evidently), el A/B Testing, el análisis FinOps y el Model Card.
