# Monitoreo de Data Drift y Calidad de Datos con Evidently (Problema 1 - Regresión)

## 1. Definición y Justificación de los Conjuntos de Datos

Para evaluar la estabilidad del modelo de regresión (estimación de demanda diaria de viajes por estación en el sistema MiBici), se delimitaron cronológicamente dos conjuntos a partir de los datos limpios:

* **Conjunto de Referencia (Entrenamiento Histórico):**
  * **Período:** 29 de octubre de 2025 al 31 de agosto de 2026 (106,836 registros correspondientes a 348 estaciones continuas).
  * **Justificación:** Representa el período base utilizado para el ajuste inicial de los modelos de regresión y captura el comportamiento estacional estándar (otoño, invierno, primavera y parte del verano). Los primeros 28 días fueron descartados debido al calentamiento (*warmup*) requerido por las características de rezago de 28 días.
* **Conjunto Actual (Producción / Operación Reciente):**
  * **Período:** 1 de septiembre de 2026 al 30 de septiembre de 2026 (10,440 registros para las mismas 348 estaciones).
  * **Justificación:** Corresponde al mes más reciente disponible en los datos abiertos de MiBici. Funciona como el lote de producción no visto donde el modelo opera en tiempo real. Este mes es especialmente crítico en el Área Metropolitana de Guadalajara por la reactivación total del ciclo escolar universitario/laboral y la culminación de la temporada de lluvias.

Ambos conjuntos están persistidos para el equipo en:
* `data/processed/regression_reference.parquet`
* `data/processed/regression_current.parquet`

---

## 2. Metodología de Monitoreo con Evidently

El pipeline implementado en `mibici/monitoring/drift.py` utiliza la suite estándar de Evidently integrando:
1. **`DataDriftPreset`:** Análisis de variación distribucional columna por columna y métrica global de drift de conjunto.
   * **Variables numéricas:** Distancia Wasserstein normalizada (*Wasserstein distance*) con umbral de 0.10, aplicada a 12 de las 13 variables evaluadas.
   * **Variable binaria (`es_fin_de_semana`):** Distancia Jensen-Shannon (*Jensen-Shannon distance*) con umbral de 0.10.
   Los métodos estadísticos y sus umbrales son determinados automáticamente por Evidently (versión 0.7.23). El pipeline recupera los umbrales efectivos mediante el atributo `stattest_threshold`, garantizando la consistencia entre los resultados estadísticos, los reportes exportados y las visualizaciones.
2. **`DataQualityPreset`:** Verificación de integridad de datos, conteo de nulos, correlaciones, valores constantes y estadísticas descriptivas (mínimo, máximo, media, percentiles).

### Artefactos Generados
* **Reporte HTML Interactivo:** `reports/evidently_drift_report.html` (visualización completa navegable con widgets interactivos).
* **Resumen estructurado JSON:** `reports/drift_summary.json` (consumible por servicios de observabilidad o pipelines de CI/CD).
* **Tabla de métricas CSV:** `reports/drift_metrics.csv` (para integración directa en el reporte final).
* **Gráficas ejecutivas:**
  * `reports/figures/drift/evidently_drift_scores.png`: Resumen de scores de drift por variable.
  * `reports/figures/drift/distribucion_variables_top_drift.png`: Comparativa KDE de distribuciones entre referencia y actual.
  * `reports/figures/drift/target_drift_demanda.png`: Análisis de desviación de la variable objetivo (`viajes_iniciados`).

---

## 3. Resultados Cuantitativos del Drift

A continuación se resume el estado de drift detectado para cada variable, utilizando los umbrales efectivos recuperados de Evidently:

| Variable | Tipo | Prueba Estadística | Drift Score | Umbral | Drift Detectado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `viajes_iniciados` *(Target)* | Continua | Wasserstein (normed) | **0.2664** | 0.10 | **SÍ** |
| `viajes_dia_anterior` | Continua | Wasserstein (normed) | **0.2424** | 0.10 | **SÍ** |
| `promedio_7_dias` | Continua | Wasserstein (normed) | **0.2090** | 0.10 | **SÍ** |
| `viajes_hace_7_dias` | Continua | Wasserstein (normed) | **0.1625** | 0.10 | **SÍ** |
| `promedio_14_dias` | Continua | Wasserstein (normed) | **0.1592** | 0.10 | **SÍ** |
| `mes` | Discreta (calendario) | Wasserstein (normed) | **1.1629** | 0.10 | **SÍ** |
| `promedio_28_dias` | Continua | Wasserstein (normed) | 0.0955 | 0.10 | No |
| `viajes_hace_14_dias` | Continua | Wasserstein (normed) | 0.0711 | 0.10 | No |
| `promedio_mismo_dia_4_semanas` | Continua | Wasserstein (normed) | 0.0704 | 0.10 | No |
| `dia_semana` | Discreta | Wasserstein (normed) | 0.0632 | 0.10 | No |
| `viajes_hace_28_dias` | Continua | Wasserstein (normed) | 0.0228 | 0.10 | No |
| `es_fin_de_semana` | Binaria | Jensen-Shannon | 0.0158 | 0.10 | No |
| `estacion_id` | Identificador | Wasserstein (normed) | 0.0000 | 0.10 | No |

**Resumen Global:**

* Proporción total de columnas con drift: **46.15%** (6 de 13 variables evaluadas).
* Feature Drift: **41.67%** (5 de 12 predictores).
* Target Drift: Detectado en `viajes_iniciados`, con un score de **0.2664**.
* Drift a nivel Dataset: **No detectado** (`dataset_drift_detected = False`), debido a que la proporción de columnas afectadas no alcanza el umbral global del 50%.

Aunque no se activa una alerta global de Dataset Drift, los cambios observados en las variables de demanda reciente y en la variable objetivo justifican mantener el monitoreo y evaluar el desempeño predictivo del modelo sobre datos recientes.

Los resultados fueron reproducidos durante la integración final del proyecto, utilizando Evidently 0.7.23 y los mismos conjuntos de referencia y actual. Se verificó la consistencia entre los scores, los umbrales efectivos y las clasificaciones de drift.

---

## 4. Interpretación Técnica y de Negocio del Drift

### 4.1 Variables Afectadas y Naturaleza del Drift

1. **Target Drift en `viajes_iniciados` (Wasserstein = 0.2664):**
   * La distribución de la demanda real de viajes presenta cambios entre el período histórico y septiembre de 2026. El retorno a clases y actividades laborales, así como los cambios estacionales, podrían contribuir a esta variación, aunque el análisis no permite establecer una relación causal.

2. **Feature Drift en rezagos y promedios de corto plazo:**
   * Las variables `viajes_dia_anterior`, `promedio_7_dias`, `viajes_hace_7_dias` y `promedio_14_dias` presentan cambios en sus distribuciones. Estos indicadores reflejan variaciones recientes en los patrones de demanda utilizados por el modelo para generar predicciones.

3. **Drift en la variable de calendario `mes`:**
   * El score de 1.1629 es consistente con la comparación de un período histórico que abarca múltiples meses frente a un conjunto actual correspondiente exclusivamente a septiembre. Por lo tanto, este resultado debe interpretarse considerando la naturaleza temporal de los conjuntos de datos.

4. **Estabilidad en ventanas largas y otras variables:**
   * Las características `promedio_28_dias` y `promedio_mismo_dia_4_semanas` presentan menor drift, con scores inferiores al umbral de 0.10.
   * `promedio_28_dias`, con un score de 0.0955, se encuentra cerca del umbral y representa una variable de interés para futuros monitoreos.
   * `es_fin_de_semana` y `estacion_id` no presentan drift detectado, lo cual indica estabilidad en sus distribuciones dentro de los conjuntos evaluados.

### 4.2 Implicaciones para el Desempeño del Modelo

* **Riesgo de degradación predictiva:** Los cambios observados en las variables de entrada y en la variable objetivo pueden afectar la capacidad de generalización del modelo entrenado con datos históricos.

* **Evaluación del desempeño:** La detección de drift justifica evaluar las métricas MAE, RMSE y R² sobre datos recientes para determinar si existe una degradación en la calidad de las predicciones.

* **Relación con el A/B Testing:** Como parte del ciclo de mejora del modelo, se realizó una comparación offline entre las versiones v2 y v3 del modelo de regresión utilizando los datos de septiembre de 2026. Los resultados y criterios de selección se encuentran documentados en `docs/ab_testing_p1.md`.

* **Monitoreo continuo:** Aunque no se activa una alerta global de Dataset Drift, se recomienda mantener el seguimiento periódico de los predictores afectados, la variable objetivo y las métricas de desempeño.

* **Alcance del análisis:** Los resultados demuestran cambios distribucionales en las variables de entrada y en la variable objetivo (Feature Drift y Target Drift). Sin embargo, no constituyen por sí solos evidencia de Concept Drift ni justifican automáticamente el reentrenamiento o la promoción de una nueva versión del modelo.

---

## 5. Instrucciones de Reproducción

Para ejecutar el pipeline de monitoreo y regenerar los reportes en cualquier momento:

```bash
# 1. Asegurar dependencias
pip install -r requirements.txt

# 2. Ejecutar pipeline de drift con CLI
python -m mibici.monitoring.drift run

# 3. Ejecutar pruebas unitarias
pytest tests/test_monitoring.py -v
```
