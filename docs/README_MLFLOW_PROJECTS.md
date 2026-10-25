# Guía de ejecución de MLflow Projects — MiBici Equipo 36

**Estado:** P1 validado: ingesta reproducible, experimentación, comparación con baseline, modelo nativo y Model Registry con alias `champion`.  
**Fecha de corte:** 24 de octubre de 2026.  
**Repositorio:** [maestria-mlops-equipo36/TC5061-Proyecto-Equipo36](https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36)  
**Rama de integración:** `feat/mlflow-project`  
**Rama de finalización P1:** `feat/p1-finalizacion`  
**Último commit P1 confirmado:** `8ab3e7b`.

> **Alcance de la validación:** ingesta y entrenamiento P1 ejecutados con `mlflow run` y `--env-manager local`, Python 3.12.13, MLflow 3.17.1 y tracking local `http://127.0.0.1:5000`. La creación automática del entorno desde `python_env.yaml` y la integración de P2/P3 al coordinador no se han validado. El Model Registry documentado reside en la instancia local utilizada para las pruebas.

## 1. Objetivo

Esta guía permite reproducir el flujo de **MLflow Projects** del Equipo 36: ingesta, entrenamiento, búsqueda de hiperparámetros, evaluación, trazabilidad y recuperación de modelos. Complementa el `README.md` principal.

- **P1 — Regresión:** estimar viajes iniciados en una estación durante un periodo futuro.
- **P2 — Clasificación:** identificar riesgo de falta de bicicletas o anclajes.
- **P3 — Clustering:** agrupar estaciones por patrones de uso.

La validación integral descrita aquí corresponde a **P1**. Las implementaciones de P2 y P3 existen por separado, pero su ejecución mediante el entry point `train` del coordinador sigue pendiente de integración y validación.
Actualmente **P1 y P3 están conectados al entry point `train`**. P2 conserva su implementación previa y su integración en el coordinador está en curso.

## 2. Archivos y responsabilidades

| Archivo | Función |
|---|---|
| `MLproject` | Define los entry points `ingest` y `train` y sus parámetros. |
| `python_env.yaml` | Declara Python 3.12.13 y dependencias; reconstrucción automática aún no validada. |
| `mibici/modeling/run_project.py` | Despacha P1 y recibe `question`, `model`, `n_estimators`, `max_depth` y `min_samples_leaf`. |
| `mibici/modeling/search_p1.py` | Búsqueda con corridas anidadas y selección desde corridas existentes. |
| `mibici/modeling/train.py` | Entrenamiento y evaluación de P1, métricas, gráficas y registro nativo del modelo. Contiene también lógica P3 independiente. |
| `mibici/dataset.py` | Ingesta, limpieza, Parquet, manifiesto y registro de este como artefacto MLflow. |
| `mibici/config.py` | Centraliza rutas y constantes. |
| `MLproject` | Define el proyecto y los entry points `ingest` y `train` (parametrizados para P1 y P3). |
| `python_env.yaml` | Declara Python y las dependencias que MLflow debe utilizar al construir un entorno; su reconstrucción automática todavía no se ha validado. |
| `mibici/modeling/run_project.py` | Recibe `question`, `model` e hiperparámetros; despacha el entrenamiento y búsqueda de P1 y P3. |
| `mibici/modeling/search_p1.py` | Búsqueda de tres configuraciones de P1 con corridas anidadas y recuperación del ganador sin entrenamiento. |
| `mibici/modeling/search_p3.py` | Búsqueda de cinco configuraciones de P3 (K-Means y GMM) con corridas anidadas (`nested=True`) y selección automática del ganador. |
| `mibici/modeling/baseline_reglas.py` | Referencia basada en reglas de intensidad (`viajes_diarios_por_anclaje`) para P3 integrada con MLflow Projects. |
| `mibici/modeling/train.py` | Contiene la implementación de P1 y P3; en P3 registra pipeline (`StandardScaler` + modelo) con firma (`signature`), `input_example`, `mlflow.log_input`, gráficas y configuración. |
| `mibici/dataset.py` | Implementa la ingesta y preparación de datos. |
| `mibici/config.py` | Centraliza rutas y otras constantes del proyecto. |
| `requirements.txt` | Dependencias del proyecto. |

### Entry points

```bash
python -m mibici.dataset --start-date 2025-10 --end-date 2026-09
python -m mibici.modeling.run_project --question p1 --model rf --n-estimators 100 --max-depth 12 --min-samples-leaf 5
```

Son comandos internos; para reproducir el proyecto se recomienda `mlflow run`, como se muestra más adelante.

## 3. Requisitos previos

1. Git y acceso al repositorio.
2. Python **3.12.13**, MLflow **3.17.1** y dependencias de `requirements.txt`.
3. Los CSV originales MiBici accesibles mediante `MIBICI_DATA_DIR`.
4. Servidor de tracking MLflow disponible y almacenamiento de artefactos accesible.
5. Espacio suficiente para procesar los 12 CSV y escribir los Parquet.

**No publicar en Git:** CSV originales, Parquet generados, modelos `.joblib`, bases SQLite, directorios `mlruns/`, `mlartifacts/`, credenciales ni datos locales.

## 4. Obtener el código y preparar el entorno

Para revisar los cambios de P1 antes de integrarlos:

```bash
git clone https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36.git
cd TC5061-Proyecto-Equipo36
git fetch origin
git switch feat/p1-finalizacion
```

Si ya existe un clon, revisar `git status` antes de cambiar de rama. Tras aprobar el PR y fusionarlo en `feat/mlflow-project`, utilizar esa rama de integración para las pruebas del equipo.

Entorno local macOS/Linux:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
python --version
mlflow --version
```

En Windows PowerShell: `.venv312\Scripts\Activate.ps1`.

**Límite conocido:** `--env-manager local` utiliza el entorno activo. No demuestra que MLflow reconstruya correctamente el entorno desde `python_env.yaml`.

## 5. Configurar los datos MiBici

Cada integrante debe configurar su propia carpeta de CSV, por ejemplo:

```bash
export MIBICI_DATA_DIR="/ruta/local/a/datos_mibici"
```

En PowerShell:

```powershell
$env:MIBICI_DATA_DIR = "C:\ruta\local\datos_mibici"
```

Comprobar el resultado:

```bash
python -c "from mibici.config import MIBICI_DATA_DIR; print(MIBICI_DATA_DIR)"
ls -lh data/processed/viajes_limpios.parquet
```

El periodo validado abarca **octubre de 2025 a septiembre de 2026**, con 12 CSV, **4,340,052 viajes consolidados** y **484 estaciones en la nomenclatura**. Para el entrenamiento P1 se seleccionaron **348 estaciones** según el criterio de cobertura.

## 6. Configurar el tracking de MLflow

La instancia local utilizada para las pruebas está disponible en:

```text
http://127.0.0.1:5000
```

La base SQLite centralizada del entorno de referencia se ubica en `~/tec/Operaciones ML/mlflow_tracking/mlflow.db`. La ubicación de artefactos puede ser distinta de la base. La URL `127.0.0.1` no es compartida entre integrantes.

Como alternativa para configurar un servidor de desarrollo independiente:

```bash
mkdir -p mlflow_tracking mlartifacts
mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/mlflow_tracking/mlflow.db" \
  --default-artifact-root "$(pwd)/mlartifacts" \
  --host 127.0.0.1 \
  --port 5000
```

No iniciar otro servidor si el puerto 5000 está ocupado. Configurar el tracking antes de ejecutar:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

## 7. Ejecutar P1 con MLflow Projects (validado)

```bash
mlflow run . -e train \
  -P question=p1 \
  -P model=rf \
  -P n_estimators=100 \
  -P max_depth=12 \
  -P min_samples_leaf=5 \
  --env-manager local \
  --experiment-name mibici-equipo36
```

Los tres hiperparámetros se probaron desde el entry point `train`. El entrenamiento usa una partición cronológica: **85,260** observaciones de entrenamiento, **21,576** de validación y **10,440** de prueba (septiembre de 2026).

**Corrida final validada:** `59273199ad7d4362bdf77087a508da93`, estado `FINISHED`.

| Métrica | Validación | Prueba |
|---|---:|---:|
| MAE | 5.0119 | 7.2366 |
| RMSE | 7.8936 | 12.2953 |

El modelo se reentrena con entrenamiento + validación antes de evaluar el conjunto de prueba. Ese modelo final es el que se registró de forma nativa en MLflow.

## 8. Búsqueda de hiperparámetros y selección P1

`mibici/modeling/search_p1.py` implementa búsqueda con corridas anidadas y selección por MAE de validación. Resultados históricos:

| Árboles | Profundidad | Mínimo por hoja | MAE validación | RMSE validación |
|---:|---:|---:|---:|---:|
| 100 | 8 | 5 | 5.1026 | 8.0282 |
| 100 | 12 | 5 | 5.0119 | 7.8936 |
| 300 | 12 | 5 | **5.0110** | **7.8904** |

Los resultados no necesariamente provienen del mismo padre. La recuperación desde `57a29674e7284c0cb34570bc2aa5f4a2` seleccionó 100/12/5 dentro de sus hijos elegibles. Para una búsqueda nueva:

```bash
python -m mibici.modeling.search_p1 --n-trials 3
```

Para recuperar sin reentrenar:

```bash
python -m mibici.modeling.search_p1 --reuse-parent-run-id 57a29674e7284c0cb34570bc2aa5f4a2
```

**Decisión operativa P1:** se publicó la configuración **100/12/5**. La configuración de 300 árboles reduce el MAE de validación en apenas **0.0009**; se priorizó el modelo de 100 árboles por su menor tamaño/complejidad. No se documentó una comparación cronometrada de costo de entrenamiento, por lo que esta es una decisión razonada, no una demostración empírica de tiempos.

### Comparación contra baseline de siete días

| Métrica | Baseline | Random Forest 100/12/5 | Reducción relativa |
|---|---:|---:|---:|
| MAE validación | 6.5526 | **5.0119** | 23.5% |
| RMSE validación | 10.7318 | **7.8936** | 26.4% |
| MAE prueba | 8.5673 | **7.2366** | 15.5% |
| RMSE prueba | 14.7710 | **12.2953** | 16.8% |

Baseline consultado en la corrida `64b9cd5d29ae405e94a32f6cff998d96`. Los errores son menores con Random Forest en validación y prueba.

## 9. Consultar resultados, artefactos y modelo nativo

Abrir el experimento `mibici-equipo36` en MLflow y seleccionar la corrida P1 correspondiente.

Artefactos de P1:

```text
figures/
  P1_random_forest_reales_vs_predichos.png
  P1_random_forest_importancia_caracteristicas.png
  P1_random_forest_distribucion_errores.png
models/
  P1_random_forest.joblib
```

Además del `.joblib`, la corrida final `59273199ad7d4362bdf77087a508da93` registra un **Logged Model nativo** con `mlflow.sklearn.log_model()`, firma de **12 características** y `input_example`.

- **Logged Model ID:** `m-5c010c8ea3094821a7292d4705052630`.
- Se autorizó explícitamente el tipo interno `sklearn.tree._tree.Tree` para la serialización con `skops`, únicamente por tratarse de un modelo generado en el entorno de confianza del equipo.
- Persiste una advertencia sobre columnas enteras y posibles valores faltantes en inferencia. No impidió el registro, pero requiere validación de entradas en un uso productivo.
## 9.5 Ejecutar P3 (Clustering de Estaciones) con MLflow Projects (validado)

La Pregunta 3 (P3) está integrada en el punto de entrada `train` de `MLproject` y calcula automáticamente `data/processed/features_p3_estaciones.parquet` si aún no existe después de `ingest`. Evalúa las 484 estaciones del sistema sobre las siete características estandarizadas (`ratio_flujo_neto`, `pct_salidas_pico_matutino`, `pct_salidas_pico_vespertino`, `pct_salidas_fin_de_semana`, `viajes_diarios_por_anclaje`, `latitud` y `longitud`).

### Comandos reproducibles con `mlflow run`

```bash
# 1. Referencia basada en reglas de intensidad (Percentiles 33 y 67)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=baseline \
  --env-manager local \
  --experiment-name mibici-equipo36

# 2. Baseline oficial de clustering: K-Means k=3
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=kmeans \
  -P n_clusters=3 \
  -P stage_of_work=baseline \
  --env-manager local \
  --experiment-name mibici-equipo36

# 3. Búsqueda de hiperparámetros con corridas anidadas (Nested Runs: 5 configuraciones)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . \
  -e train \
  -P question=p3 \
  -P model=search \
  --env-manager local \
  --experiment-name mibici-equipo36

# 4. Corridas candidatas individuales (K-Means k=4, K-Means k=5, GMM k=4 full y diag)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=4 -P stage_of_work=candidate --env-manager local --experiment-name mibici-equipo36

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=5 -P stage_of_work=candidate --env-manager local --experiment-name mibici-equipo36

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . -e train -P question=p3 -P model=gmm -P n_clusters=4 -P covariance_type=full -P stage_of_work=candidate --env-manager local --experiment-name mibici-equipo36

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . -e train -P question=p3 -P model=gmm -P n_clusters=4 -P covariance_type=diag -P stage_of_work=candidate --env-manager local --experiment-name mibici-equipo36
```

### Tabla comparativa de experimentos P3

| Corrida / Modelo | Rol (`stage_of_work`) | Hiperparámetros clave | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Tamaño de grupos (estaciones) | Mejora Silhouette vs. Baseline (k=3) |
|---|---|---|---:|---:|---:|---|---:|
| `P3_reglas_intensidad_k3_espacial` | `baseline` (reglas) | Percentiles 33 / 67 | 0.0592 | 4.5544 | 32.27 | `[160, 164, 160]` | -74.5% |
| `P3_kmeans_k3_espacial` | `baseline` | `k=3, n_init=10` | 0.2323 | 1.3457 | 117.79 | `[284, 185, 15]` | — (0.0%) |
| `P3_kmeans_k4_espacial` | `candidate` | `k=4, n_init=10` | 0.2481 | 1.2551 | 130.53 | `[103, 15, 222, 144]` | **+6.8%** |
| **`P3_kmeans_k5_espacial`** | **`candidate` (Ganador)** | **`k=5, n_init=10`** | **0.2721** | **1.1990** | **138.55** | **`[128, 133, 106, 102, 15]`** | **+17.1%** |
| `P3_gmm_k4_espacial` | `candidate` | `k=4, cov=full` | 0.2296 | 1.3782 | 113.25 | `[230, 108, 15, 131]` | -1.2% |
| `P3_gmm_diag_k4_espacial` | `candidate` | `k=4, cov=diag` | 0.2290 | 1.3729 | 114.09 | `[238, 91, 15, 140]` | -1.4% |

### Artefactos registrados en cada corrida candidata P3

Cada corrida candidata de P3 registra de forma estructurada:
- `model/`: Pipeline de `scikit-learn` (`StandardScaler` + algoritmo de clustering) registrado con `mlflow.sklearn.log_model`, incluyendo firma (`signature`) e `input_example`.
- `plots/` y `figures/`: Mapa geográfico de clusters (`*_mapa.png`) y perfil de comportamiento por grupo (`*_perfil.png`).
- `data/`: Manifiesto de versión de datos (`data_manifest.json`), tabla de estaciones con su cluster asignado (`*_estaciones_clusters.csv`) y registro formal de entrada con `mlflow.log_input`.
- `config/`: Archivo JSON (`*_config.json`) con hiperparámetros, características, semilla, versión de datos y resumen de métricas.

### Justificación operativa y de negocio del modelo seleccionado (P3)

Se selecciona **`P3_kmeans_k5_espacial` (K-Means con $k=5$)** como modelo candidato para pasar a producción (`mibici-equipo36-p3-grupos@champion`), y **`P3_kmeans_k4_espacial` (K-Means con $k=4$)** como `@challenger`, por las siguientes razones técnicas y operativas:
1. **Superioridad en cohesión y separación interna**: Obtuvo el mayor **Silhouette Score (`0.2721`, +17.1% respecto al baseline `k=3` y +359% respecto a las reglas de intensidad)**, el menor **Davies-Bouldin (`1.1990`, -10.9% respecto al baseline)** y el mayor **Calinski-Harabasz (`138.55`, +17.6%)**.
2. **Equilibrio operativo de cuadrillas y zonas de rebalanceo**: Mientras que `k=3` concentra 284 estaciones en un solo macro-grupo heterogéneo, `k=5` particiona la red principal en **cuatro grupos operativos altamente balanceados (128, 133, 106 y 102 estaciones)** más el corredor periférico/especializado de **15 estaciones** (Zapopan Norte / Periférico). Esto permite asignar rutas y políticas diferenciadas (emisoras residenciales matutinas, atractoras laborales/comerciales, vocación recreativa de fin de semana y alta rotación central) sin sobrecargar una sola zona logística.
3. **Estabilidad y bajo costo computacional**: El pipeline `StandardScaler + KMeans(k=5)` entrena e infiere sobre las 484 estaciones en menos de 0.2 segundos de CPU, garantizando determinismo (`random_seed=42`) y menor complejidad paramétrica frente a mezclas gaussianas (`GMM`), cuyas matrices de covarianza penalizan el BIC (`6712.01` y `7531.33`) sin mejorar la separación geométrica.

## 10. Ingesta desde MLflow Projects (declarada, pendiente de validación integral)

## 10. Ingesta desde MLflow Projects (validada)

```bash
mlflow run . -e ingest \
  -P start_date=2025-10 \
  -P end_date=2026-09 \
  --env-manager local \
  --experiment-name mibici-equipo36
```

**Validación realizada:** se ejecutó el entry point sobre los 12 CSV, con **4,340,052 viajes** y **484 estaciones**. Se compararon los SHA-256 de `viajes_limpios.parquet` y `estaciones.parquet` contra respaldos previos: ambos coincidieron exactamente. `diff -u` confirmó un manifiesto idéntico.

- **Run de reproducibilidad:** `62f1a7e198a84e4b9f589e452160e01b`.
- **Run con manifiesto como artefacto:** `f0166bff38444c73b77b1772c4048f9a`.
- **Artefacto confirmado en la interfaz:** `data/data_manifest.json`.
- **Versión de datos:** `mibici-2025-10_to_2026-09-v1`.
- **SHA-256 del Parquet consolidado:** `97d150752ce4600dde9a440add61519dffc2ece15107f170980eca63d5b2208d`.

**Precaución:** `dataset.py` sobrescribe `data/processed/estaciones.parquet`, `data/processed/viajes_limpios.parquet` y `data/data_manifest.json`. Respaldar antes de reingestar si es necesario conservar versiones previas.

## 11. Cambios publicados en GitHub

La estrategia de integración consiste en abrir PR desde `feat/p1-finalizacion` hacia **`feat/mlflow-project`**, no directamente a `main`.

| Commit | Cambio principal |
|---|---|
| `2245d0d` | Integración inicial de MLflow Projects. |
| `0aecd6e` | Reutilización de corrida MLflow Projects en P1. |
| `b5115d4` | Búsqueda de hiperparámetros y documentación de P1 en rama de integración. |
| `59a5aa2` | Registro del manifiesto de ingesta como artefacto MLflow. |
| `8ab3e7b` | Registro nativo de Random Forest con firma y ejemplo de entrada. |

Los dos últimos commits se publicaron en `feat/p1-finalizacion`. La creación de la versión en Registry y el alias `champion` fueron acciones en la instancia local de MLflow: **no se transportan automáticamente mediante Git**.

Antes de abrir el PR:

```bash
git status --short
git diff --check
git add README_MLFLOW_PROJECTS.md
git commit -m "docs(p1): update MLflow Projects validation and registry guide"
git push origin feat/p1-finalizacion
```

En GitHub, configurar **base** `feat/mlflow-project` y **compare** `feat/p1-finalizacion`; solicitar revisión antes de fusionar.

## 12. Limitaciones conocidas y próximos pasos

| Tema | Estado / siguiente acción |
|---|---|
| Ingesta P1 por MLflow Projects | **Validada**, con hashes y manifiesto como artefacto. |
| Entrenamiento Random Forest P1 | **Validado** con 100/12/5. |
| Búsqueda de hiperparámetros | Implementada con nested runs; resultados históricos conservados. |
| Comparación contra baseline | **Validada** mediante métricas de MLflow. |
| Logged Model con firma y ejemplo | **Validado**. |
| Model Registry P1 | **Versión 1 y alias `champion` creados** en instancia local. |
| Carga de `champion` | **Validada** mediante `mlflow.pyfunc.load_model()`. No se ejecutó una predicción de prueba tras cargarlo. |
| P2 y P3 desde `train` | Pendiente de integración y validación por los responsables. |
| Reconstrucción automática de entorno | Pendiente de validar sin `--env-manager local`. |
| Costos de entrenamiento | No se compararon formalmente los tiempos de 100 vs. 300 árboles. |
| Configuración | Revisar centralización de fechas y rutas portables. |
| Reproducción del Registry en otros equipos | Requiere infraestructura y artefactos accesibles, o automatización de registro. |
| P1 Random Forest, 100 árboles | **Validado** mediante MLflow Projects. |
| P1 Random Forest, 300 árboles | Ejecutado durante la búsqueda; MAE de validación observado: 5.0110. |
| Baseline P1 desde MLflow Projects | Disponible en el coordinador; pendiente de validación específica. |
| P3 Clustering desde `train` | **Validado** (baseline por reglas, K-Means `k=3,4,5`, GMM `full`/`diag`, búsqueda con *nested runs*, `log_model` con `signature` e `input_example`). |
| P2 desde `train` | Pendiente de integración en su rama correspondiente. |
| Ingesta mediante `mlflow run` | Declarada; pendiente de prueba integral. |
| Entorno automático | `python_env.yaml` existe, pero su reconstrucción sin `--env-manager local` no está validada. |
| Búsqueda de hiperparámetros | Implementada con corridas anidadas y recuperación sin reentrenar. |
| Registro de modelos | Pendiente de usar el formato MLflow Model, firma, ejemplo de entrada, versiones y alias `champion`. |
| Configuración | Revisar centralización de fechas y parámetros; evitar rutas específicas de un integrante. |
| Documentación | Mantener esta guía actualizada con cada funcionalidad validada. |

## 13. Solución rápida de problemas

**`Connection refused` al puerto 5000:** comprobar servidor y `MLFLOW_TRACKING_URI`.

**No aparece `viajes_limpios.parquet`:** verificar `MIBICI_DATA_DIR`, archivos CSV y ejecución de `ingest`.

**`ModuleNotFoundError`:** activar el entorno correcto e instalar `requirements.txt`.

**La corrida no tiene métricas:** comprobar que el coordinador haya ejecutado P1 y revisar su estado.

**Error `UntrustedTypesFoundException` al registrar Random Forest:** con `skops`, revisar el origen del modelo y, solo para modelos propios y confiables, indicar `skops_trusted_types=["sklearn.tree._tree.Tree"]` en `mlflow.sklearn.log_model()`.

**Advertencia de enteros en la firma:** validar que las entradas de inferencia cumplan los tipos esperados y gestionar nulos antes de predecir.

**No aparecen artefactos desde otra computadora:** comprobar la accesibilidad del artifact store; la base SQLite por sí sola no contiene los archivos del modelo.

**Error al crear versión de Registry con `source="models:/m-..."`:** utilizar la ubicación real del Logged Model (`client.get_logged_model(model_id).artifact_location`) y suministrar `model_id` a `create_model_version()`.

## 14. Cierre de P1 y transferencia al equipo

### Model Registry P1

- **Nombre registrado:** `mibici-equipo36-p1-demanda`.
- **Versión:** `1`.
- **Alias:** `champion`.
- **Run de origen:** `59273199ad7d4362bdf77087a508da93`.
- **Logged Model ID:** `m-5c010c8ea3094821a7292d4705052630`.

La carga del modelo ganador se verificó con:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
modelo = mlflow.pyfunc.load_model(
    "models:/mibici-equipo36-p1-demanda@champion"
)
print(modelo.metadata.signature)
```

La salida confirmó la firma de las **12 columnas de entrada** y una salida de predicción `float64`. Esta prueba verifica la recuperación y metadatos; una prueba adicional con `modelo.predict(...)` queda recomendada.

### Entrega y revisión

- Abrir PR `feat/p1-finalizacion` → `feat/mlflow-project`, adjuntando resumen de métricas y corridas.
- Solicitar revisión a otro integrante del equipo antes del merge.
- Conservar evidencia de ingesta, artefactos, métricas, firma y alias `champion`.
- No publicar la base SQLite ni artefactos en GitHub. Para compartir el entorno MLflow entre equipos, respaldar de manera consistente tanto metadatos como artefactos, o configurar un servidor accesible.
- Integrar posteriormente P2 y P3 y validar la creación automática del entorno.

**Conclusión:** P1 quedó validado funcionalmente en el entorno local de referencia. El cierre colaborativo requiere revisión del PR y su integración a la rama compartida.
