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
   * **Variables Continuas:** Prueba de Distancia Wasserstein normalizada (*Wasserstein distance*) con umbral de 0.05 / 0.10.
   * **Variables Categóricas / Binarias:** Distancia Jensen-Shannon (*Jensen-Shannon distance*) con umbral de 0.05.
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

A continuación se resume el estado de drift detectado para cada variable:

| Variable | Tipo | Prueba Estadística | Drift Score | Umbral | Drift Detectado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `viajes_iniciados` *(Target)* | Continua | Wasserstein (normed) | **0.2664** | 0.05 | **SÍ** |
| `viajes_dia_anterior` | Continua | Wasserstein (normed) | **0.2424** | 0.05 | **SÍ** |
| `promedio_7_dias` | Continua | Wasserstein (normed) | **0.2090** | 0.05 | **SÍ** |
| `viajes_hace_7_dias` | Continua | Wasserstein (normed) | **0.1625** | 0.05 | **SÍ** |
| `promedio_14_dias` | Continua | Wasserstein (normed) | **0.1592** | 0.05 | **SÍ** |
| `mes` | Categórica | Wasserstein (normed) | **1.1629** | 0.05 | **SÍ** |
| `promedio_28_dias` | Continua | Wasserstein (normed) | 0.0955 | 0.05 | No (Score < 0.1) |
| `viajes_hace_14_dias` | Continua | Wasserstein (normed) | 0.0711 | 0.05 | No |
| `promedio_mismo_dia_4_semanas` | Continua | Wasserstein (normed) | 0.0704 | 0.05 | No |
| `dia_semana` | Discreta | Wasserstein (normed) | 0.0632 | 0.05 | No |
| `viajes_hace_28_dias` | Continua | Wasserstein (normed) | 0.0228 | 0.05 | No |
| `es_fin_de_semana` | Binaria | Jensen-Shannon | 0.0158 | 0.05 | No |
| `estacion_id` | Identificador | Wasserstein (normed) | 0.0000 | 0.05 | No |

**Resumen Global:**
* Proporción de columnas con drift: **46.2%** (6 de 13 variables evaluadas).
* Drift a nivel Dataset: Por defecto Evidently requiere >50% de columnas afectadas para activar la alerta global de dataset drift; sin embargo, al estar afectadas las **variables de mayor importancia predictiva** y la **variable objetivo**, la alerta técnica de negocio es **ALTA**.

---

## 4. Interpretación Técnica y de Negocio del Drift

### 4.1 Variables Afectadas y Naturaleza del Drift
1. **Target Drift en `viajes_iniciados` (Wasserstein = 0.2664):**
   * La demanda real de viajes experimentó un incremento sostenido en septiembre en comparación con el promedio anual. En el conjunto de referencia, la media de viajes diarios por estación rondaba los 35.8 viajes; en septiembre 2026, la media sube notablemente debido al retorno masivo a clases y trabajo presencial tras el periodo vacacional de julio-agosto.
2. **Feature Drift en rezagos de corto plazo (`viajes_dia_anterior`, `promedio_7_dias`, `viajes_hace_7_dias`):**
   * Los indicadores de ventana corta (1 a 7 días) reflejan inmediatamente el aumento de volumen y la mayor varianza de uso diario en las estaciones con alta afluencia (zonas universitarias y corredores de oficinas como Chapultepec y Providencia).
3. **Estabilidad en ventanas largas y variables de calendario:**
   * Las características `promedio_28_dias` y `promedio_mismo_dia_4_semanas` presentan menor drift (scores < 0.10) debido a la inercia temporal de las 4 semanas previas.
   * `es_fin_de_semana` y `estacion_id` presentan drift nulo o despreciable, confirmando que la infraestructura física del sistema no cambió y que la proporción entre días hábiles y fines de semana se mantiene matemáticamente estable.

### 4.2 Implicaciones para el Desempeño del Modelo
* **Riesgo de subestimación sistemática:** El modelo base original (v1) fue entrenado predominantemente con meses de volumen moderado o vacaciones. Al enfrentarse a características de entrada infladas en septiembre, es altamente probable que las predicciones del modelo v1 se queden por debajo de la demanda real pico.
* **Degradación de métricas (MAE / RMSE):** Se anticipa un incremento en el error absoluto medio durante los días laborables del mes actual.
* **Justificación para reentrenamiento (v2) y A/B Testing:** La presencia simultánea de *Data Drift* en predictores clave y *Target Drift* proporciona evidencia cuantitativa sólida para reentrenar una versión 2 (v2) del modelo e inaugurar la prueba de hipótesis del A/B testing para justificar su pase a producción.

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
