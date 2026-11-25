# Índice de evidencias — CI/CD, pruebas y rendimiento

| ID | Archivo | Descripción | Reporte / fuente original | Uso en la presentación |
|---|---|---|---|---|
| E09 | [GitHub Actions CI/CD](E09_github_actions.png) | Pipeline `CI/CD - MiBici MLOps` en verde (`test` + `build-docker` + smoke tests de Docker Compose) e historial de corridas exitosas | `.github/workflows/ci.yml` y API de GitHub Actions (Runs #16, #18, #19) | Demostración de automatización e integración continua |
| E10 | [PyTest y seguridad](E10_pytest.png) | Ejecución de 135/135 pruebas automatizadas en 13.56 s + escaneo Bandit (`0 High / 0 Medium` en 4,579 LOC) y `pip-audit` | `tests/`, `reports/bandit_security_report.txt`, `reports/pip_audit_report.txt` | Respaldo de calidad, pruebas unitarias y seguridad |
| E11 | [Dashboard Locust](E11_locust.png) | Evolución temporal de throughput (req/s), latencia P50/P95 y tabla completa por endpoint para 10, 25 y 50 usuarios | `locustfile.py` y `reports/load_testing/locust_*_stats*.csv` | Evidencia de comportamiento del sistema bajo carga |
| E12 | [Métricas de rendimiento](E12_performance.png) | Comparativa de escalabilidad, percentiles P50/P95/P99 en `POST /predict`, detección de *cold start* y contraste estándar vs. FinOps | `reports/load_testing/` y `reports/finops/resumen_consolidado.csv` | Cierre del bloque: hallazgos, cuellos de botella y mejoras |

> Todas las evidencias reflejan el estado consolidado del repositorio en `integration/final-delivery` (commit `e36ba9f`) y son consistentes con el modelo Champion P1 (`Random Forest`) servido mediante FastAPI y MLflow.
