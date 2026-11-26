# Model Card — MiBici P1: Predicción de demanda diaria

**Equipo 36 | TC5061 — MLOps**  
**Fecha de actualización:** 26 de noviembre de 2026

## 1. Información general del modelo

El modelo P1 del proyecto MiBici tiene como objetivo predecir
la demanda diaria de viajes iniciados por estación del sistema
de bicicletas públicas de Guadalajara, México.

### Identificación y trazabilidad

| Campo | Descripción |
|---|---|
| Nombre del modelo | mibici-equipo36-p1-demanda |
| Tipo de tarea | Regresión supervisada |
| Algoritmo | Random Forest Regressor |
| Versión registrada | 2 |
| Alias | champion |
| Desarrolladores | Equipo 36 |
| Framework | Scikit-learn |
| Gestión del ciclo de vida | MLflow |
| Versión de datos | mibici-2025-10_to_2026-09-v1 |
| Training Run ID | `4dd9959935814b5985dd64b41c1cd2cf` |
| Packaging Run ID | `a5c4f6e42f1c415fab2ba1f6afedec76` |
| Commit de entrenamiento | `e7fc7b580aa5f95c169a75b4a4f22786c8f65580` |

La identificación corresponde al Champion documentado durante
la validación final del Model Registry. Los artefactos y
registros de MLflow se administran en instancias locales,
por lo que la versión y el alias deben verificarse junto
con el identificador de ejecución antes de cada despliegue.

## 2. Usos previstos y fuera de alcance

### Usos previstos

- Estimar la demanda diaria de viajes por estación.
- Apoyar la planeación operativa del sistema MiBici.
- Identificar estaciones con mayor demanda esperada.
- Complementar decisiones de redistribución de bicicletas.
- Facilitar el análisis de patrones históricos de movilidad.

### Usuarios objetivo

- Operadores de sistemas de bicicletas públicas.
- Analistas de movilidad urbana.
- Equipos de ciencia de datos y MLOps.
- Personal responsable de planeación operativa.

### Usos fuera de alcance

- Predicción de demanda individual de usuarios.
- Predicciones en tiempo real a nivel de minutos u horas.
- Predicción en estaciones sin suficiente información histórica.
- Aplicación directa a otros sistemas de movilidad sin
  validación previa.
- Toma de decisiones operativas críticas sin supervisión humana.

El modelo debe utilizarse como herramienta de apoyo y no
como único criterio para la asignación de recursos.

## 3. Datos y preprocesamiento

### Origen y cobertura

Los datos provienen de los registros históricos de viajes
del sistema MiBici, correspondientes al periodo comprendido
entre octubre de 2025 y septiembre de 2026.

| Característica | Valor |
|---|---:|
| Archivos mensuales | 12 |
| Registros de viajes | 4,340,052 |
| Estaciones seleccionadas | 348 |
| Cobertura mínima requerida por estación | 95% |
| Observaciones estación-día antes del warm-up | 127,020 |
| Observaciones después del warm-up | 117,276 |

El procesamiento incluye la agregación diaria de viajes
por estación, la selección de estaciones con suficiente
cobertura histórica, la construcción de un calendario
completo y la generación de variables temporales e históricas.

Se aplicó un periodo inicial de 28 días para permitir
la construcción de las variables rezagadas.

### Particiones temporales

| Partición | Periodo | Observaciones |
|---|---|---:|
| Entrenamiento | 29/10/2025 - 30/06/2026 | 85,260 |
| Validación | 01/07/2026 - 31/08/2026 | 21,576 |
| Prueba | 01/09/2026 - 30/09/2026 | 10,440 |
| Total | | 117,276 |

La separación cronológica permite evaluar el modelo sobre
periodos posteriores a los utilizados para su entrenamiento
y reducir el riesgo de fuga de información temporal.

### Variables de entrada

| Variable | Descripción |
|---|---|
| estacion_id | Identificador de la estación |
| dia_semana | Día de la semana |
| mes | Mes del año |
| es_fin_de_semana | Indicador de fin de semana |
| viajes_dia_anterior | Demanda del día anterior |
| viajes_hace_7_dias | Demanda de hace 7 días |
| viajes_hace_14_dias | Demanda de hace 14 días |
| viajes_hace_28_dias | Demanda de hace 28 días |
| promedio_7_dias | Promedio histórico de 7 días |
| promedio_14_dias | Promedio histórico de 14 días |
| promedio_28_dias | Promedio histórico de 28 días |
| promedio_mismo_dia_4_semanas | Promedio del mismo día de la semana durante las últimas cuatro semanas |

### Variable objetivo

`viajes_iniciados`: número de viajes iniciados por estación
durante un día.

### Limitaciones de los datos

- La cobertura temporal se limita a un año de observaciones.
- La selección de estaciones puede reducir la representatividad
  de estaciones nuevas o con registros incompletos.
- El modelo no incorpora explícitamente condiciones
  meteorológicas, eventos especiales ni interrupciones
  del servicio.
- Los viajes registrados representan demanda observada,
  no necesariamente toda la demanda potencial del sistema.

## 4. Entrenamiento y configuración

El modelo Champion utiliza Random Forest Regressor,
un algoritmo de aprendizaje supervisado basado en un
ensamble de árboles de decisión.

### Hiperparámetros del Champion

| Parámetro | Valor |
|---|---:|
| n_estimators | 300 |
| max_depth | 12 |
| min_samples_leaf | 5 |
| random_state | 42 |

La configuración corresponde al modelo identificado durante
la validación final del Model Registry.

El proyecto utiliza MLflow para registrar parámetros,
métricas, artefactos y versiones de los modelos. El proceso
contempla el ajuste final con los datos de entrenamiento
y validación, manteniendo el conjunto de prueba temporal
separado.

## 5. Evaluación y rendimiento

### Métricas de evaluación

- MAE: error absoluto promedio expresado en viajes.
- RMSE: raíz del error cuadrático medio, que penaliza
  en mayor medida los errores grandes.
- R²: proporción de variabilidad explicada por el modelo.

### Resultados del Champion

Los siguientes resultados corresponden al Champion v2
identificado durante la validación final del Model Registry.

| Métrica | Validación | Prueba |
|---|---:|---:|
| MAE | 5.0110 | 7.2303 |
| RMSE | 7.8904 | 12.2722 |

El MAE de prueba indica un error absoluto promedio
de aproximadamente 7.23 viajes por estación-día.

Como referencia, el baseline basado en el promedio
histórico de siete días obtuvo un MAE de validación
de aproximadamente 6.55 viajes.

El Champion presenta un MAE de validación aproximadamente
23.5% menor que el baseline.

### Factores relevantes de evaluación

El rendimiento del modelo puede variar según:

- Nivel de demanda de cada estación.
- Día de la semana y estacionalidad.
- Disponibilidad de información histórica.
- Cambios en los patrones de movilidad.
- Eventos externos no representados en las variables.

## 6. Despliegue y uso del modelo

El proyecto proporciona un servicio de inferencia mediante
FastAPI, contenerizado y orquestado con Docker Compose.

| Componente | Tecnología |
|---|---|
| Servicio de inferencia | FastAPI |
| Contenerización | Docker |
| Orquestación | Docker Compose |
| Registro de modelos | MLflow |
| Integración continua | GitHub Actions |
| Pruebas de carga | Locust |

### Endpoints

- `GET /health`: verifica la disponibilidad de la API.
- `POST /predict`: recibe las variables de entrada y
  devuelve la demanda estimada.

La API valida los datos de entrada mediante esquemas
definidos para el servicio.

La identidad exacta del artefacto utilizado debe verificarse
en cada entorno de despliegue, debido a que las instancias
locales de MLflow pueden contener registros diferentes.

## 7. Monitoreo y mantenimiento

### Data Drift

Se utilizó Evidently para comparar las distribuciones
de las variables entre un conjunto de referencia y un
conjunto de datos recientes.

| Indicador | Resultado |
|---|---:|
| Observaciones de referencia | 106,836 |
| Observaciones actuales | 10,440 |
| Variables evaluadas, incluyendo el target | 13 |
| Variables con drift | 6 |
| Proporción de variables con drift | 46.15% |
| Umbral global de dataset drift | 50% |
| Dataset drift detectado | No |

Se identificó drift en cinco variables predictoras
y en la variable objetivo.

Aunque no se superó el umbral global, los cambios
individuales justifican mantener el seguimiento de las
distribuciones y del rendimiento del modelo.

El cambio observado en la distribución de la variable
objetivo constituye evidencia de Target Drift, pero
no demuestra por sí mismo la existencia de Concept Drift.

Reportes:

- [Documentación de monitoreo](monitoreo_drift.md)
- [Reporte Evidently](../reports/evidently_drift_report.html)
- [Métricas de drift](../reports/drift_metrics.csv)
- [Resumen de drift](../reports/drift_summary.json)

### A/B Testing

Se realizó una comparación offline entre dos modelos
registrados en una instancia local de MLflow, utilizando
el mismo conjunto de evaluación de septiembre de 2026.

| Métrica | Champion local v2 | Candidate local v3 |
|---|---:|---:|
| MAE | 7.2366 | 7.2114 |
| RMSE | 12.2953 | 12.1144 |
| R² | 0.8830 | 0.8864 |
| Número de árboles | 100 | 200 |

El Candidate local presentó una reducción aproximada
de 0.35% en MAE y 1.47% en RMSE.

Este experimento corresponde a una instancia local
distinta de la utilizada para documentar el Champion
de 300 árboles en la validación final del Model Registry.

Por lo tanto, los resultados no constituyen una
comparación directa contra dicho Champion ni justifican
por sí solos una promoción del Candidate.

Reportes:

- [Documentación de A/B Testing](ab_testing_p1.md)
- [Comparación de métricas](../reports/ab_testing/comparacion_modelos_p1.csv)
- [Gráfica comparativa](../reports/ab_testing/comparacion_metricas_p1.png)

## 8. Rendimiento operativo y FinOps

Se realizaron pruebas de carga locales mediante Locust
y monitoreo de recursos de los contenedores Docker.

### Resultados operativos

| Indicador | 10 usuarios | 25 usuarios | 50 usuarios |
|---|---:|---:|---:|
| Predicciones exitosas | 491 | 454 | 402 |
| Errores | 0 | 0 | 0 |
| Throughput (req/s) | 5.51 | 5.17 | 4.52 |
| Latencia promedio (s) | 1.15 | 4.05 | 9.58 |
| Latencia P95 (s) | 1.50 | 5.10 | 12.00 |
| CPU promedio API (%) | 132.60 | 201.20 | 231.43 |
| Memoria promedio API (MiB) | 423.12 | 438.85 | 448.39 |

El incremento de concurrencia produjo una reducción
del throughput y un aumento considerable de la latencia,
a pesar de no registrarse errores.

### Estimación de costos

Se utilizó como referencia una instancia AWS EC2
t3.xlarge en la región us-east-1, con una tarifa
histórica de USD 0.1664 por hora y 730 horas mensuales.

| Usuarios | Costo por inferencia (USD) | Costo por 1,000 predicciones (USD) |
|---|---:|---:|
| 10 | 0.000008389 | 0.008389 |
| 25 | 0.000008949 | 0.008949 |
| 50 | 0.000010227 | 0.010227 |

El costo mensual estimado de cómputo es de USD 121.47.

Estas cifras corresponden a una proyección ilustrativa
basada en el throughput observado localmente. No representan
costos reales ni rendimiento validado en AWS.

La tarifa original se conserva por trazabilidad.
Las exclusiones y diferencias respecto a la tarifa
consultada posteriormente se documentan en el reporte
original de FinOps.

La identidad exacta del artefacto utilizado durante
estas pruebas no se ha reconciliado con el Champion
documentado en la validación final del Model Registry.

Reportes:

- [Documentación FinOps](../reports/finops/README.md)
- [Resultados consolidados](../reports/finops/resumen_consolidado.csv)
- [Estimación de costos](../reports/finops/costos_aws.csv)

## 9. Consideraciones éticas, riesgos y limitaciones

### Representatividad y equidad

El modelo aprende a partir de patrones históricos
de movilidad. Estos patrones pueden reflejar diferencias
en la disponibilidad de infraestructura y en el acceso
al servicio entre distintas zonas de la ciudad.

Utilizar únicamente la demanda estimada para distribuir
bicicletas podría desfavorecer estaciones con menor
actividad histórica o con disponibilidad limitada.

Se recomienda complementar las predicciones con criterios
de cobertura territorial, accesibilidad y necesidades
operativas.

### Privacidad

Las variables utilizadas por el modelo corresponden
a información agregada por estación y día.

El modelo no requiere identificadores personales de
usuarios para generar predicciones. Se recomienda
mantener controles de acceso y minimización de datos
durante todo el ciclo de vida.

### Riesgos operativos

- Degradación del rendimiento ante cambios de demanda.
- Predicciones poco confiables para estaciones nuevas.
- Dependencia de la disponibilidad de datos históricos.
- Incremento de latencia bajo alta concurrencia.
- Inconsistencias entre registros locales de modelos.
- Uso de predicciones fuera del contexto evaluado.

## 10. Gobernanza y recomendaciones

### Mecanismos implementados

- Control de versiones mediante Git y GitHub.
- Versionado de datos y artefactos mediante DVC.
- Registro de experimentos y modelos mediante MLflow.
- Identificación de modelos mediante versiones y aliases.
- Validación automatizada mediante pruebas y CI/CD.
- Monitoreo experimental de drift.
- Comparación offline de modelos.
- Evaluación de rendimiento y costos operativos.

### Política de mantenimiento propuesta

Se recomienda realizar revisiones periódicas del modelo
cuando se incorporen nuevos datos mensuales.

Las revisiones deben considerar:

1. Calidad y disponibilidad de los datos.
2. Cambios en las distribuciones de las variables.
3. Rendimiento predictivo sobre datos recientes.
4. Latencia, throughput y consumo de recursos.
5. Costo estimado por inferencia.
6. Identidad y trazabilidad del modelo registrado.

### Criterios propuestos para promoción de modelos

Antes de promover un Candidate a Champion se recomienda:

1. Verificar las versiones y los identificadores de
   ejecución de ambos modelos.
2. Evaluar ambos modelos sobre el mismo conjunto de datos.
3. Comparar MAE, RMSE y métricas complementarias.
4. Validar la compatibilidad con el servicio de inferencia.
5. Ejecutar las pruebas automatizadas correspondientes.
6. Evaluar el impacto operativo y económico.
7. Obtener aprobación del equipo responsable.
8. Conservar la versión anterior para facilitar un rollback.

La detección de drift no debe provocar automáticamente
el reemplazo del modelo sin una evaluación adicional.

El reentrenamiento automático, la promoción automática
y el monitoreo continuo en producción se consideran
extensiones futuras, no funcionalidades implementadas.

### Recomendaciones de mejora

- Incorporar variables meteorológicas y eventos externos.
- Evaluar el rendimiento por estación y nivel de demanda.
- Establecer objetivos de latencia y disponibilidad.
- Optimizar la configuración de workers de la API.
- Evaluar la separación de MLflow y el servicio de inferencia.
- Validar las estimaciones FinOps en infraestructura real.
- Estandarizar la identidad de los modelos entre entornos.

## 11. Referencias

La estructura de esta Model Card se fundamenta en
las recomendaciones de Mitchell et al. (2019),
Gebru et al. (2021) y la documentación de Hugging Face,
adaptadas al contexto del proyecto MiBici.

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P.,
  Vasserman, L., Hutchinson, B., Spitzer, E.,
  Raji, I. D., & Gebru, T. (2019).
  [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).
  Proceedings of the Conference on Fairness,
  Accountability, and Transparency, 220–229.

- Gebru, T., Morgenstern, J., Vecchione, B.,
  Vaughan, J. W., Wallach, H., Daumé III, H.,
  & Crawford, K. (2021).
  [Datasheets for Datasets](https://arxiv.org/abs/1803.09010).
  Communications of the ACM, 64(12), 86–92.

- Hugging Face. (s. f.).
  [Model Cards](https://huggingface.co/docs/hub/model-cards).