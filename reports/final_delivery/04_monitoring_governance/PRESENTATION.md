# Propuesta de presentación final — Monitoreo, FinOps y gobernanza

Bloque de cinco diapositivas para integrar en la presentación del Equipo 36. Se propone utilizar una o dos evidencias principales por slide y conservar los reportes originales como respaldo técnico.

## Slide 1. Monitoreo de Data Drift y Target Drift

**Mensaje:** El monitoreo permite identificar cambios en las distribuciones de los datos y detectar posibles riesgos de degradación del modelo.

- Evidently: comparación de datos históricos contra septiembre de 2026.
- 106,836 observaciones de referencia y 10,440 observaciones actuales.
- Cinco variables predictoras y la variable objetivo presentan drift.
- Proporción de variables con drift: 46.15 %, inferior al umbral global de 50 %.
- Incremento de 26.5 % en la demanda diaria promedio observada.

**Visual:** [Reporte Evidently](E13_drift.png) y [Drift scores por variable](../../figures/drift/evidently_drift_scores.png).

**Notas de exposición:** Aunque no se activó la alerta global de dataset drift, se identificaron cambios individuales que justifican mantener el monitoreo. El cambio en la distribución de la demanda constituye Target Drift, pero no demuestra por sí mismo la existencia de Concept Drift.

## Slide 2. A/B Testing y comparación de modelos

**Mensaje:** La comparación experimental permite evaluar modelos candidatos antes de tomar decisiones de promoción.

| Métrica | Champion local | Candidate local |
|---|---:|---:|
| MAE | 7.2366 | 7.2114 |
| RMSE | 12.2953 | 12.1144 |
| R² | 0.8830 | 0.8864 |

- Evaluación sobre el mismo conjunto de prueba de septiembre de 2026.
- Reducción de 0.35 % en MAE y 1.47 % en RMSE.
- Registro de métricas y artefactos mediante MLflow.
- Sin cambios de alias ni promoción del Candidate.

**Visual:** [Comparación A/B de modelos](E14_ab_testing.png).

**Notas de exposición:** El Candidate presentó mejoras marginales en las métricas evaluadas. La comparación fue offline, entre configuraciones locales de 100 y 200 árboles, y no corresponde a una evaluación directa contra el Champion de 300 árboles documentado en la validación final del Model Registry. No se realizó asignación de tráfico real ni evaluación de significancia estadística.

## Slide 3. Monitoreo de recursos y rendimiento operativo

**Mensaje:** Las pruebas de carga permiten identificar limitaciones de capacidad y rendimiento bajo diferentes niveles de concurrencia.

| Indicador | 10 usuarios | 25 usuarios | 50 usuarios |
|---|---:|---:|---:|
| Throughput (req/s) | 5.51 | 5.17 | 4.52 |
| Latencia P95 (s) | 1.50 | 5.10 | 12.00 |
| CPU promedio API (%) | 132.60 | 201.20 | 231.43 |
| Errores | 0 | 0 | 0 |

- 1,347 predicciones exitosas en los tres escenarios.
- Reducción aproximada de 18 % en throughput.
- Incremento de 8.32 veces en la latencia promedio.
- MLflow concentra aproximadamente el 84 % de la memoria observada.

**Visual:** [Latencia por escenario](E15a_latencia.png) y [Consumo de CPU](../../figures/finops/cpu_por_escenario.png).

**Notas de exposición:** Las pruebas realizadas mediante Locust y Docker Compose no registraron errores, pero mostraron una degradación considerable de la latencia al incrementar la concurrencia. Los resultados sugieren investigar posibles cuellos de botella computacionales y evaluar la separación de MLflow y el servicio de inferencia.

## Slide 4. FinOps y oportunidades de optimización

**Mensaje:** La eficiencia operativa del servicio influye directamente en el costo estimado por predicción.

- Infraestructura de referencia: AWS EC2 t3.xlarge.
- Capacidad: 4 vCPU y 16 GiB de memoria.
- Tarifa utilizada: USD 0.1664 por hora.
- Costo mensual estimado de cómputo: USD 121.47.

| Usuarios concurrentes | Costo por 1,000 predicciones (USD) |
|---|---:|
| 10 | 0.008389 |
| 25 | 0.008949 |
| 50 | 0.010227 |

El costo unitario estimado aumenta aproximadamente 21.9 % entre los escenarios de 10 y 50 usuarios.

**Visual:** [Costo estimado por 1,000 predicciones](E15_finops.png).

**Notas de exposición:** Una mayor concurrencia no necesariamente implica mayor eficiencia económica. La reducción del throughput incrementó el costo unitario estimado. Se recomienda optimizar los workers de la API, establecer objetivos de latencia y evaluar configuraciones alternativas de infraestructura. Los costos corresponden a una proyección basada en pruebas locales, no a un despliegue real en AWS.

## Slide 5. Model Card y gobernanza del modelo

**Mensaje:** La documentación y la gobernanza permiten mantener un modelo trazable, transparente y sujeto a criterios de uso responsable.

- Model Card del modelo P1 Champion v2.
- Random Forest de 300 árboles, con trazabilidad de datos, entrenamiento y registro.
- Documentación de usos previstos, métricas, riesgos y limitaciones.
- Consideraciones éticas, privacidad y representatividad de los datos.
- Criterios propuestos para mantenimiento, promoción de modelos y rollback.

**Visual:** [Model Card y trazabilidad](E16_model_card.png).

**Notas de exposición:** La Model Card consolida la información técnica y operativa del modelo siguiendo las recomendaciones de Mitchell et al., Gebru et al. y Hugging Face. Se distingue entre las funcionalidades implementadas y las políticas propuestas. La promoción de nuevos modelos debe considerar su rendimiento predictivo, compatibilidad operativa, costos y aprobación del equipo responsable.

[Model Card completa](../../../docs/model_card_p1.md).

## Cierre de la presentación

El proyecto MiBici integra los principales componentes del ciclo de vida de Machine Learning: preparación y versionado de datos, entrenamiento reproducible, seguimiento experimental, registro de modelos, despliegue, automatización, monitoreo y gobernanza.

Los resultados obtenidos permiten identificar oportunidades de mejora tanto en el rendimiento predictivo como en la eficiencia operativa. Como siguientes pasos se propone fortalecer la consistencia de los registros de modelos entre entornos, establecer objetivos de servicio, validar el rendimiento en infraestructura real y evaluar mecanismos de mantenimiento continuo bajo criterios de gobernanza.

## Evidencias complementarias

Los siguientes recursos pueden utilizarse como respaldo técnico durante la presentación o para responder preguntas:

- [Distribución de la variable objetivo](../../figures/drift/target_drift_demanda.png).
- [Distribución de variables con mayor drift](../../figures/drift/distribucion_variables_top_drift.png).
- [Resultados cuantitativos de A/B Testing](../../ab_testing/comparacion_modelos_p1.csv).
- [Consumo de memoria](../../figures/finops/memoria_por_escenario.png).
- [Resultados consolidados de rendimiento](../../finops/resumen_consolidado.csv).
- [Estimación de costos AWS](../../finops/costos_aws.csv).
- [Documentación de la validación final](README.md).