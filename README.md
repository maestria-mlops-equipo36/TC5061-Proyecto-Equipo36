# MiBici Data Science

**Proyecto integrador — Equipo 36 | Maestría, materia: Operaciones de Aprendizaje Automático (MLOps)**

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

**Integrantes:**
- Francisco Gómez Rubio A00710437 **GitHub Maintainer**
- Carolina Nicole González Leal A01284948 **ML Engineer (P1)**
- Alan Ortiz González A01840596 **ML Engineer (P2)**
- Alan Ricardo González Aguilar A01633872 **ML Engineer (P3)**

Repositorio de trabajo colaborativo para organizar, validar y posteriormente analizar datos abiertos de viajes del sistema MiBici. El proyecto utiliza la estructura de **Cookiecutter Data Science** y busca mantener separados el código versionado y los datos originales.

**Repositorio:** https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36

## 1. Datos utilizados

El equipo trabaja con **142 archivos CSV mensuales**, desde **diciembre de 2014 hasta septiembre de 2026**, que ocupan aproximadamente **2.9 GB**. Los datos originales se obtienen del portal de [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) o de la carpeta compartida que facilite el equipo.

**Los CSV no están incluidos en GitHub.** Cada integrante debe disponer de una copia local o de una carpeta de OneDrive sincronizada en su computadora. No es necesario moverlos a `data/raw/`: los scripts leen la ubicación indicada en `MIBICI_DATA_DIR`.

Todos los archivos mensuales deben estar directamente en una misma carpeta, con nombres como:

```text
datos_mibici/
├── datos_abiertos_2014_12.csv
├── datos_abiertos_2015_01.csv
├── ...
└── datos_abiertos_2026_09.csv
```

Si el equipo utiliza OneDrive, solicitar a otro integrante el acceso a la carpeta compartida; el repositorio **no** concede automáticamente acceso a OneDrive.

## 2. Clonar el repositorio

En una terminal de Windows (CMD, PowerShell) o macOS:

```bash
git clone https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36.git
cd TC5061-Proyecto-Equipo36
```

Se recomienda utilizar una versión de Python compatible con las dependencias del proyecto. Los experimentos documentados se ejecutaron con **Python 3.12** en el entorno local `.venv312` y MLflow 3.17.1. Se recomienda Python 3.12 y verificar las versiones efectivas de `requirements.txt` antes de reproducir las corridas.

## 3. Crear el entorno virtual e instalar dependencias

### Windows — CMD

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS — Terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La carpeta `.venv` es local y no se comparte por Git. Cada integrante debe crear su propio entorno virtual. Para salir de él: `deactivate`.

## 4. Configurar la ubicación de los datos

En la **raíz del repositorio** (junto a `README.md`), crear un archivo llamado `.env` con la variable `MIBICI_DATA_DIR`.

**Windows — ejemplo:**

```dotenv
MIBICI_DATA_DIR=C:/Users/usuario/OneDrive/Operaciones ML/datos_mibici
```

**macOS — ejemplo:**

```dotenv
MIBICI_DATA_DIR=/Users/usuario/Library/CloudStorage/OneDrive-Institucion/Operaciones ML/datos_mibici
```

Estas rutas son **ilustrativas**: sustituirlas por la ubicación real de la carpeta en cada equipo. Se recomienda usar `/` incluso en Windows para evitar problemas de escape en rutas. Si se usa OneDrive, asegurarse de que los CSV estén descargados y disponibles localmente.

El archivo `.env` está excluido mediante `.gitignore`: **no debe subirse a GitHub** ni contener tokens o contraseñas.

## 5. Ejecutar los procesos disponibles

Ejecutar desde la raíz del proyecto, con el entorno virtual activado.

### 5.1 Inventario de archivos mensuales

```bash
python -m mibici.dataset
```

Identifica los archivos CSV disponibles y genera:

```text
data/interim/inventario_mibici.csv
```

### 5.2 Validación de encabezados

```bash
python -m mibici.validate_schema
```

Revisa los nombres de columnas de los archivos mensuales y genera:

```text
data/interim/validacion_esquemas.csv
```

En la revisión inicial se verificaron **142 archivos** y se obtuvo correspondencia de esquemas después de normalizar variaciones de codificación en el encabezado `Año_de_nacimiento`. **Esta validación no implica que los valores de todas las filas sean correctos**; la calidad de los registros se analizará por separado.

### 5.3 Ingesta consolidada y versionado de datos

Ejecuta el pipeline de ingesta reproducible para descargar la nomenclatura, procesar los 12 meses seleccionados (octubre 2025 a septiembre 2026), exportar a Parquet y generar el manifiesto de datos con hashes SHA-256 (`data/data_manifest.json`):

```bash
python -m mibici.dataset
```

### 5.4 Ingeniería de características (Feature Engineering P3)

Calcula los vectores de comportamiento por estación (ratios de flujo neto, porcentajes en horas pico matutina y vespertina, rotación de anclajes y uso en fines de semana):

```bash
python -m mibici.features
```

Genera:
- `data/processed/features_p3_estaciones.parquet`

### 5.5 Servidor de MLflow y entorno de ejecución

Los tres problemas comparten el experimento de tracking **`mibici-equipo36`**. El servidor utilizado durante las pruebas estuvo disponible en `http://127.0.0.1:5000`, con base SQLite y artefactos locales gestionados fuera del repositorio. **Los runs no se suben automáticamente a GitHub**.

En macOS, desde la raíz del proyecto, con el entorno `.venv312` ya creado e instaladas las dependencias:

```bash
.venv312/bin/python -m pip show mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Iniciar por separado el servidor de MLflow configurado para el curso. Si se usa un servidor propio, asegurar que backend store y artefactos estén configurados de forma persistente; ejecutar simplemente `mlflow ui` podría abrir un almacén diferente del utilizado por las corridas.

### 5.6 P1 — Regresión: viajes por estación y día

**Pregunta:** ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro?

El objetivo es predecir la demanda diaria por estación, usando características históricas y división temporal. Se comparan un modelo de referencia y Random Forest.

```bash
# Referencia
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train \
  p1-regresion --modelo-tipo baseline

# Candidato
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train \
  p1-regresion --modelo-tipo random_forest
```

**Evidencia observada:** `P1_baseline` y `P1_random_forest` aparecen en la interfaz de MLflow. El baseline obtuvo MAE=6.5527 y RMSE=10.7318 en validación, y MAE=8.5673 y RMSE=14.7710 en prueba. Las figuras del candidato muestran reales frente a predichos, importancia de características y distribución de errores. Los valores numéricos del candidato deben consultarse en su run antes de afirmar una mejora cuantitativa.

### 5.7 P2 — Clasificación: riesgo en estación

**Pregunta:** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?

Se ejecutan `DummyClassifier` como referencia y `LogisticRegression` como candidato. La interpretación operativa depende de cómo se definió la etiqueta de riesgo (real o proxy).

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train_p2
```

**Evidencia observada:** runs `P2_Baseline_DummyClassifier` y `P2_Candidato_LogisticRegression`; candidato con F1 macro≈0.6910 y ROC AUC≈0.8652 en la ejecución mostrada. La matriz de confusión adjunta registra VN=126615, FP=26691, FN=5527 y VP=16262. Se generan una curva ROC y una matriz de confusión. Para reproducibilidad se debe comprobar que el código, la etiqueta y los datos coincidan con la versión evaluada.

### 5.8 P3 — Clustering: perfiles de estaciones

**Pregunta:** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?

Las características por estación se obtienen con:

```bash
env -u PYTHONPATH .venv312/bin/python -m mibici.features
```

Modelos comparados:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train --modelo-tipo kmeans --n-clusters 3

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train --modelo-tipo kmeans --n-clusters 4

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.train --modelo-tipo gmm --n-clusters 4
```

Se comparan perfiles y mapas de clusters; los indicadores internos incluyen silhouette, Davies-Bouldin y Calinski-Harabasz. La ejecución GMM debe verificarse en la lista de runs de la instancia usada.

#### 5.8.1 Referencia adicional por reglas de intensidad

Se incorpora una referencia sencilla que agrupa las estaciones según
`viajes_diarios_por_anclaje`:

- Uso bajo: valores menores o iguales al percentil 33.
- Uso medio: valores mayores al percentil 33 y menores o iguales al percentil 67.
- Uso alto: valores mayores al percentil 67.

Los umbrales se calculan a partir de las estaciones del archivo de entrada.
Esta referencia complementa el análisis; K-Means con k=3 se mantiene como
baseline oficial del proyecto.

##### Ejecución

Con el entorno del proyecto activo y desde la raíz del repositorio:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 env -u PYTHONPATH \
  .venv312/bin/python -m mibici.modeling.baseline_reglas
```

Se requiere el archivo `data/processed/features_p3_estaciones.parquet`,
generado mediante `.venv312/bin/python -m mibici.features` después de la ingesta.

##### Evaluación y registro

Las agrupaciones se evalúan sobre las mismas siete características
estandarizadas utilizadas por la configuración espacial de K-Means:
flujo neto, salidas en pico matutino, salidas en pico vespertino, salidas
en fin de semana, viajes diarios por anclaje, latitud y longitud.

Se registra la ejecución `P3_reglas_intensidad_k3_espacial` en el experimento
de MLflow configurado para el proyecto. Se guardan los umbrales, las
métricas, el número de estaciones por grupo y dos artefactos:
`estaciones_niveles.csv` y `metricas.json`.

##### Resultados observados

La referencia por reglas produjo 160 estaciones de uso bajo, 164 de uso
medio y 160 de uso alto. Los umbrales fueron aproximadamente 0.9913 y
3.3040 viajes diarios por anclaje.

| Métrica | Referencia por reglas | K-Means k=3 |
|---|---:|---:|
| Silhouette — mayor es mejor | 0.0592 | 0.2323 |
| Davies-Bouldin — menor es mejor | 4.5544 | 1.3457 |
| Calinski-Harabasz — mayor es mejor | 32.27 | 117.79 |

Los valores de K-Means corresponden a la ejecución previamente validada.
La comparación requiere utilizar el mismo conjunto de estaciones y las
mismas características y transformación.

K-Means obtiene mejores métricas internas que la regla de intensidad.
Estas métricas describen la estructura de los grupos, pero no demuestran
por sí solas su utilidad operativa. Los percentiles tampoco garantizan
grupos de igual tamaño cuando existen valores repetidos.

### 5.9 Verificación de runs y evidencias para evaluación

En MLflow abrir el experimento `mibici-equipo36` y comprobar al menos un run **por cada P1, P2 y P3**. Para cada run verificar: parámetros, métricas, etiquetas, artefactos y origen del código. Capturar cuatro evidencias legibles: (1) lista de runs con los tres problemas; (2) parámetros y etiquetas; (3) métricas; (4) artefactos y modelo, cuando aplique. Una captura de terminal demuestra ejecución, pero no sustituye las pestañas de MLflow.

> **Importante:** las instrucciones reflejan ejecuciones compartidas por el equipo, pero la presencia actual de cada script y run en `main` debe verificarse en GitHub. Los archivos ignorados (datos, modelos, credenciales y tracking) no deben agregarse para aparentar que están versionados.

### 5.10 Model Registry y ciclo de vida de los modelos

El proyecto utiliza **MLflow Model Registry** para gestionar los modelos seleccionados de las tres preguntas de negocio (P1, P2 y P3). Cada modelo se registra con un nombre consistente, una versión y metadatos de linaje que permiten relacionarlo con su entrenamiento y los datos utilizados.

Los modelos registrados son:

| Pregunta | Modelo registrado | Alias |
|---|---|---|
| P1 — Regresión | `mibici-equipo36-p1-demanda` | `champion` |
| P2 — Clasificación | `mibici-equipo36-p2-riesgo` | `champion` |
| P3 — Clustering | `mibici-equipo36-p3-grupos` | `champion` |

El alias `champion` identifica la versión seleccionada de cada modelo y permite acceder a ella sin depender de un número de versión específico. Su asignación no implica que exista un servicio de inferencia desplegado.

**Registro de modelos:**

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python -m mibici.modeling.register_models
```

**Verificación del registro:**

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
python -m mibici.modeling.verify_registry
```

Durante las pruebas locales se verificó el registro de los tres modelos y su carga mediante el alias `champion`.

Para consultar las configuraciones seleccionadas, el proceso de registro, la verificación de firmas y metadatos, y las URI de carga de los modelos, consultar la **sección 15: Model Registry y ciclo de vida de los modelos** en [`docs/README.md`](docs/README.md).

Los modelos registrados y sus artefactos permanecen en el servidor MLflow utilizado durante las pruebas; no se almacenan directamente en GitHub.

### 5.11 Pruebas automatizadas con PyTest

El proyecto incorpora pruebas automatizadas mediante **PyTest** para validar las funciones principales de los tres problemas de negocio, incluyendo casos normales, situaciones límite, integridad de datos y reproducibilidad de los entrenamientos.

Las pruebas se encuentran organizadas en la carpeta `tests/`:

| Archivo | Funcionalidad evaluada |
|---|---|
| `test_data.py` | Validaciones generales de datos, configuraciones y características |
| `test_regresion.py` | P1: demanda diaria, características históricas, validaciones y reproducibilidad de Random Forest |
| `test_clasificacion.py` | P2: cálculo de riesgo, separación temporal, prevención de fuga de información y reproducibilidad de regresión logística |
| `test_clustering.py` | P3: perfiles de estaciones, casos límite, entrenamiento de K-Means y GMM, y reproducibilidad |

#### Ejecución de las pruebas

Desde la raíz del repositorio, con el entorno virtual activado y las dependencias instaladas:

```bash
python -m pytest tests/ -v
```

También es posible ejecutar las pruebas de un problema específico:

```bash
python -m pytest tests/test_regresion.py -v
python -m pytest tests/test_clasificacion.py -v
python -m pytest tests/test_clustering.py -v
```

Las pruebas utilizan principalmente datos sintéticos para comprobar resultados conocidos sin depender de los archivos CSV originales ni de un servidor MLflow activo.

#### Reproducibilidad

Se incorporaron pruebas que entrenan dos modelos independientes utilizando los mismos datos, hiperparámetros y semilla aleatoria (`RANDOM_SEED = 42`).

Se verifica la consistencia de predicciones y métricas en:

- **P1:** Random Forest Regressor.
- **P2:** Logistic Regression.
- **P3:** K-Means y Gaussian Mixture Model.

En clustering, las agrupaciones se comparan mediante el *Adjusted Rand Index*, evitando depender de los números arbitrarios asignados a los clusters.

#### Resultados de ejecución

Las pruebas se ejecutaron localmente utilizando Python 3.12 y PyTest 9.1.1.

**Resultado obtenido:**

```text
34 passed, 2 warnings in 3.22s
```

Las dos advertencias corresponden a la deprecación del parámetro `penalty` de `LogisticRegression` en scikit-learn. No se registraron pruebas fallidas.

Este resultado valida los casos cubiertos por la suite, pero no sustituye las verificaciones de integración completa, calidad de datos reales o seguridad del repositorio.

## 6. Organización del proyecto

```text
TC5061-Proyecto-Equipo36/
├── data/
│   ├── external/         # Datos externos
│   ├── interim/          # Reportes y resultados intermedios (ignorados por Git)
│   ├── processed/        # Datos procesados (ignorados por Git)
│   └── raw/              # Datos originales, si se usan localmente (ignorados por Git)
├── docs/                 # Documentación MkDocs
├── mibici/               # Código fuente del proyecto
│   ├── config.py         # Rutas y variables de entorno
│   ├── dataset.py        # Inventario de archivos CSV
│   ├── validate_schema.py # Validación de encabezados
│   ├── features.py       # Ingeniería de características
│   ├── plots.py          # Visualizaciones
│   └── modeling/         # Entrenamiento y predicción
├── models/               # Modelos generados
├── notebooks/            # Cuadernos Jupyter
├── references/           # Diccionarios y referencias
├── reports/
│   └── figures/          # Figuras de análisis
├── tests/                # Pruebas automatizadas
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Los archivos `.gitkeep` conservan la estructura de carpetas vacías. Los datos de `data/`, los entornos virtuales y `.env` no se versionan.

## 7. Trabajo colaborativo con Git y GitHub

El repositorio es **público para consulta**, de modo que el profesor puede revisarlo mediante la URL sin invitación. Solo quienes tengan permisos de escritura en la organización podrán publicar cambios.

Se recomienda trabajar en una rama por tarea, por ejemplo:

```bash
git switch main
git pull origin main
git switch -c feature/validacion-calidad
```

Después de realizar y comprobar cambios:

```bash
git add .
git commit -m "Add data quality validation"
git push -u origin feature/validacion-calidad
```

Abrir un **Pull Request** en GitHub para revisar e integrar los cambios a `main`. Antes de ejecutar `git add .`, comprobar con `git status` que no se incluyan datos, credenciales o archivos personales.

Para actualizar una copia local antes de comenzar una nueva tarea:

```bash
git switch main
git pull origin main
```

**Reglas del equipo:** no subir CSV originales, tokens, `.env` ni `.venv`; no modificar directamente el trabajo de otra persona sin coordinarlo; documentar nuevos scripts y dependencias; evitar cambios directos en `main` una vez establecido el flujo de Pull Requests.

## 8. Alcance actual y siguientes pasos

**Implementado y ejecutado localmente:** estructura Cookiecutter Data Science, validación e ingesta de datos, ingeniería de características, P1 (baseline y Random Forest), P2 (DummyClassifier y LogisticRegression), P3 (K-Means k=3/k=4, GMM k=4 y referencia por reglas), con evidencias de tracking en MLflow.

Se implementó **MLflow Model Registry** para registrar y gestionar los modelos seleccionados de P1, P2 y P3 mediante versiones y aliases `champion`, incluyendo scripts para verificar su registro y carga.

También se incorporaron **pruebas automatizadas con PyTest** para validar las funciones principales de los tres problemas, incluyendo casos límite, separación temporal y reproducibilidad de los entrenamientos. La ejecución local de la suite completa obtuvo **34 pruebas aprobadas, sin fallos**.

**Pendiente de comprobar antes de entrega:** completar las verificaciones de calidad de datos con DeepChecks y los chequeos básicos de seguridad; confirmar que todos los scripts y cambios estén integrados en `main`; verificar que los runs correspondientes incluyan parámetros, métricas, etiquetas y artefactos; y asegurar que la documentación y las dependencias coincidan con el código de la versión entregada.

## 9. Licencia

Este repositorio incluye una licencia MIT (ver [`LICENSE`](LICENSE)). Los datos originales pueden estar sujetos a condiciones de uso distintas: consultar las condiciones publicadas por MiBici.
