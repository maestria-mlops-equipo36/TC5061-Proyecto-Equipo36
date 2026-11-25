# Propuesta de presentación final — Repositorio y Model Registry

Bloque de cinco diapositivas para integrar en la presentación del Equipo 36. Se propone utilizar una o dos evidencias principales por slide y conservar las restantes como respaldo técnico.

## Slide 1. Repositorio y reproducibilidad de datos
**Mensaje:** La preparación de datos se encuentra organizada y versionada.

- Repositorio estructurado y ambiente Python 3.12.
- 12 archivos de origen y 4,340,052 viajes procesados.
- Dataset `mibici-2025-10_to_2026-09-v1`.
- Manifiesto regenerado idéntico al original.

**Visual:** [Repositorio](E01_repository.png) y [verificación de integridad](E01b_dataset_integrity.png).

**Notas de exposición:** Se estableció una estructura de trabajo común y se verificó la igualdad del manifiesto. Esta comprobación no implica por sí sola identidad binaria de todos los datos.

## Slide 2. Pipeline reproducible de Machine Learning
**Mensaje:** El entrenamiento puede ejecutarse de forma parametrizada.

- 348 estaciones con cobertura mínima de 95 %.
- 127,020 registros diarios y 12 predictores.
- División cronológica: 85,260 entrenamiento, 21,576 validación y 10,440 prueba.
- MLflow Projects: ingesta y entrenamiento; run final exitoso.

**Visual:** [Feature engineering](E01b_feature_engineering.png) y [MLflow Projects](E02_mlflow_project.png).

**Notas de exposición:** Se utilizaron variables históricas y una división temporal. La prueba de septiembre se mantuvo separada del ajuste final con entrenamiento y validación.

## Slide 3. Experimentación y seguimiento
**Mensaje:** MLflow Tracking permite comparar resultados y conservar artefactos.

| Configuración | MAE validación | MAE prueba |
|---|---:|---:|
| Baseline histórico | 6.5527 | 8.5673 |
| RF 100, validación final | 5.0119 | 7.2366 |
| RF 300, Champion histórico | 5.0110 | 7.2303 |

**Visual:** [Comparación histórica](E03d_mlflow_experiment_comparison.png) y [detalle del run](E03b_mlflow_run_details.png).

**Notas de exposición:** Menor MAE indica menor error absoluto promedio en viajes por estación y día. La ejecución final de 100 árboles confirma resultados consistentes con esa configuración histórica, no una reproducción exacta del Champion. Las gráficas y el modelo se conservaron como artefactos.

## Slide 4. Gobernanza y trazabilidad
**Mensaje:** El Model Registry permite identificar versiones y recuperar su procedencia.

- P1: Champion v2 y Candidate v3.
- Champion: Random Forest de 300 árboles.
- Linaje: versión registrada → packaging run → entrenamiento original → commit y versión de datos.
- Sin cambios de alias ni promoción del modelo de validación final.

**Visual:** [Versiones y alias](E04b_mlflow_model_versions_aliases.png) y [linaje](E04c_02_mlflow_champion_metrics_lineage.png).

**Notas de exposición:** Los metadatos faltantes en el registro histórico se recuperaron siguiendo el linaje. Se distingue el almacenamiento de un modelo en Tracking de su registro como versión administrada.

## Slide 5. Validación operativa del Champion
**Mensaje:** El modelo registrado puede recuperarse y ejecutar inferencia local.

- Carga mediante alias `champion`.
- Verificación de versión, hiperparámetros y 12 entradas.
- Diez predicciones sobre observaciones reales de septiembre de 2026.
- Diez validaciones exitosas, incluidas predicciones finitas y no negativas.

**Visual:** [Inferencia del Champion](E04d_mlflow_champion_inference.png).

**Notas de exposición:** La prueba confirma carga e inferencia local. No equivale a una prueba de disponibilidad productiva. Como mejoras pendientes se identifican el almacenamiento compartido de artefactos y el control del esquema de entrada ante valores faltantes.

## Transición al siguiente bloque
Una vez comprobada la reproducibilidad y la recuperación del modelo, corresponde presentar los mecanismos de despliegue, calidad, monitoreo y operación desarrollados por el equipo, según el orden acordado para la exposición.
