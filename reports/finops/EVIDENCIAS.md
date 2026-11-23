# Evidencias de monitoreo y análisis FinOps — MiBici

Equipo 36 · 21 de noviembre de 2026

Las capturas documentan el despliegue, las pruebas de carga y el análisis de recursos. Los datos tabulares oficiales se encuentran en `reports/finops/` y las gráficas comparativas en `reports/figures/finops/`.

## Evidencias de ejecución

### 01. Inicio MLflow y respuesta de salud
![Inicio MLflow y respuesta de salud](../figures/finops/evidencias/01_mlflow_inicio_y_health.png)

### 02. Contenedores API y MLflow saludables
![Contenedores API y MLflow saludables](../figures/finops/evidencias/02_docker_contenedores_saludables.png)

### 03. Predicción real HTTP 200
![Predicción real HTTP 200](../figures/finops/evidencias/03_api_prediccion_real_http_200.png)

### 04. Ejecución del monitoreo de línea base
![Ejecución del monitoreo de línea base](../figures/finops/evidencias/04_baseline_ejecucion_monitoreo.png)

### 05. Resumen de CPU y memoria en línea base
![Resumen de CPU y memoria en línea base](../figures/finops/evidencias/05_baseline_resumen_cpu_memoria.png)

### 06. Resultados de la prueba preliminar de 10 usuarios
![Resultados de la prueba preliminar de 10 usuarios](../figures/finops/evidencias/06_locust_10u_prueba_preliminar_resultados.png)

### 07. Inicio de la prueba preliminar de 10 usuarios
![Inicio de la prueba preliminar de 10 usuarios](../figures/finops/evidencias/07_locust_10u_prueba_preliminar_inicio.png)

### 08. Inicio de la prueba definitiva de 10 usuarios con monitoreo
![Inicio de la prueba definitiva de 10 usuarios con monitoreo](../figures/finops/evidencias/08_locust_10u_monitoreo_inicio.png)

### 09. Resultados de Locust con 10 usuarios
![Resultados de Locust con 10 usuarios](../figures/finops/evidencias/09_locust_10u_resultados_finales.png)

### 10. CPU y memoria con 10 usuarios
![CPU y memoria con 10 usuarios](../figures/finops/evidencias/10_recursos_10u_resumen.png)

### 11. Inicio de la prueba de 25 usuarios con monitoreo
![Inicio de la prueba de 25 usuarios con monitoreo](../figures/finops/evidencias/11_locust_25u_monitoreo_inicio.png)

### 12. Resultados de Locust con 25 usuarios
![Resultados de Locust con 25 usuarios](../figures/finops/evidencias/12_locust_25u_resultados_finales.png)

### 13. CPU y memoria con 25 usuarios
![CPU y memoria con 25 usuarios](../figures/finops/evidencias/13_recursos_25u_resumen.png)

### 14. Inicio de la prueba de 50 usuarios con monitoreo
![Inicio de la prueba de 50 usuarios con monitoreo](../figures/finops/evidencias/14_locust_50u_monitoreo_inicio.png)

### 15. Resultados de Locust con 50 usuarios
![Resultados de Locust con 50 usuarios](../figures/finops/evidencias/15_locust_50u_resultados_finales.png)

### 16. CPU y memoria con 50 usuarios
![CPU y memoria con 50 usuarios](../figures/finops/evidencias/16_recursos_50u_resumen.png)

### 17. Resumen consolidado de los tres escenarios
![Resumen consolidado de los tres escenarios](../figures/finops/evidencias/17_finops_resumen_consolidado.png)

### 18. Estimación de costos EC2 AWS
![Estimación de costos EC2 AWS](../figures/finops/evidencias/18_finops_estimacion_aws_ec2.png)

## Consideraciones metodológicas

- Las capturas 06 y 07 corresponden a una ejecución preliminar y no deben utilizarse para calcular las métricas finales.
- Los escenarios definitivos se ejecutaron con 10, 25 y 50 usuarios concurrentes durante 90 segundos, con monitoreo simultáneo de recursos.
- Para resultados oficiales, utilizar `reports/finops/resumen_consolidado.csv`, los archivos `locust_*_finops_stats.csv` y `recursos_*u.csv`.
- La tarifa AWS EC2 empleada es una hipótesis de estimación, no un costo facturado. El cálculo mostrado considera únicamente cómputo EC2; no incluye EBS, red, impuestos ni otros servicios.
- La API utiliza un modelo cargado desde un montaje local de solo lectura para estas pruebas; el servicio MLflow se ejecuta en paralelo.
