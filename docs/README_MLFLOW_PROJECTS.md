# Guía de ejecución de MLflow Projects — MiBici Equipo 36

**Estado:** P1 y búsqueda de hiperparámetros con MLflow implementados; selección desde corridas existentes validada.  
**Fecha de corte:** 24 de octubre de 2026.  
**Repositorio:** [maestria-mlops-equipo36/TC5061-Proyecto-Equipo36](https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36)  
**Rama de trabajo:** `feat/mlflow-project`  
**Último commit previamente documentado:** `0aecd6e`. Cambios recientes pendientes de commit.

> **Alcance de la validación:** el entrenamiento P1 con Random Forest y 100 árboles se ejecutó correctamente con `--env-manager local` y un servidor MLflow accesible en `http://127.0.0.1:5000`. La creación automática del entorno desde `python_env.yaml`, la ejecución de ingesta mediante MLflow Projects y la integración de P2/P3 aún requieren pruebas adicionales.

## 1. Objetivo

Esta guía permite a los integrantes del Equipo 36 comprender y reproducir el flujo inicial de **MLflow Projects**: ejecutar el proyecto mediante comandos parametrizados, registrar experimentos y consultar métricas y artefactos. Complementa el `README.md` principal del repositorio y no sustituye la documentación de las tres preguntas de negocio.

- **P1 — Regresión:** predecir cuántos viajes se iniciarán en una estación o zona durante un periodo futuro.
- **P2 — Clasificación:** identificar riesgo de falta de bicicletas o de anclajes disponibles.
- **P3 — Clustering:** agrupar estaciones con patrones de uso similares.

Actualmente **solo P1 está conectado al entry point `train`**. P2 y P3 conservan sus implementaciones previas, pero su integración en el coordinador está pendiente.

## 2. Archivos y responsabilidades

| Archivo | Función |
|---|---|
| `MLproject` | Define el proyecto y los entry points `ingest` y `train`. |
| `python_env.yaml` | Declara Python y las dependencias que MLflow debe utilizar al construir un entorno; su reconstrucción automática todavía no se ha validado. |
| `mibici/modeling/run_project.py` | Recibe `question`, `model` y `n_estimators`; actualmente despacha el entrenamiento real de P1. |
| `mibici/modeling/search_p1.py` | Búsqueda de tres configuraciones de P1 con corridas anidadas y recuperación del ganador sin entrenamiento. |
| `mibici/modeling/train.py` | Contiene la implementación de P1 y P3; P1 acepta hiperparámetros de Random Forest y registra métricas y artefactos. |
| `mibici/dataset.py` | Implementa la ingesta y preparación de datos. |
| `mibici/config.py` | Centraliza rutas y otras constantes del proyecto. |
| `requirements.txt` | Dependencias del proyecto. |

### Entry points declarados

**Ingesta:**

```bash
python -m mibici.dataset --start-date 2025-10 --end-date 2026-09
```

**Entrenamiento:**

```bash
python -m mibici.modeling.run_project --question p1 --model rf --n-estimators 100
```

Estos comandos representan lo que invoca el archivo `MLproject`. No es necesario ejecutarlos directamente si se utiliza `mlflow run`.

## 3. Requisitos previos

1. Git y acceso al repositorio del equipo.
2. Python **3.12.13** para reproducir el entorno local con el que se validó P1.
3. MLflow **3.17.1** y las demás dependencias declaradas en `requirements.txt`.
4. Archivos de datos MiBici disponibles localmente y rutas configuradas para la computadora de cada integrante.
5. Un servidor MLflow accesible o una configuración de tracking equivalente.

**Importante:** los archivos CSV originales y los datos procesados no deben agregarse al repositorio si están excluidos por `.gitignore`. Tampoco deben publicarse credenciales, modelos generados o directorios de tracking locales.

## 4. Obtener el código y preparar el entorno

Desde una terminal:

```bash
git clone https://github.com/maestria-mlops-equipo36/TC5061-Proyecto-Equipo36.git
cd TC5061-Proyecto-Equipo36
git fetch origin
git switch feat/mlflow-project
```

> Si ya existe un clon local, no es necesario volver a clonarlo: desde ese repositorio, consultar primero `git status`, cambiar a la rama y ejecutar `git pull --ff-only` cuando no existan cambios locales incompatibles.

Crear y activar un entorno virtual (macOS/Linux):

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
mlflow --version
```

En Windows PowerShell, la activación habitual es:

```powershell
.venv312\Scripts\Activate.ps1
```

**Nota:** el nombre `.venv312` es una convención local; no es obligatorio. La instalación anterior utiliza `requirements.txt` en el entorno activo. Todavía **no** se ha comprobado que `mlflow run` sin `--env-manager local` pueda reconstruir el entorno automáticamente desde `python_env.yaml`.

## 5. Configurar los datos MiBici

La ingesta lee los archivos de origen desde una ubicación externa al repositorio. Cada integrante debe revisar `mibici/config.py` y la configuración de la variable de entorno `MIBICI_DATA_DIR` para apuntar a su propia carpeta de CSV.

Ejemplo para macOS/Linux, **reemplazando la ruta**:

```bash
export MIBICI_DATA_DIR="/ruta/local/a/datos_mibici"
```

En PowerShell:

```powershell
$env:MIBICI_DATA_DIR = "C:\ruta\local\datos_mibici"
```

Antes de entrenar P1, comprobar que existe el archivo procesado esperado:

```bash
ls -lh data/processed/viajes_limpios.parquet
```

Si no existe, debe prepararse mediante el flujo de ingesta correspondiente. El entry point `ingest` ya está declarado, pero **aún no se ha validado una ingesta completa ejecutada mediante MLflow Projects**.

## 6. Configurar el tracking de MLflow

En la validación del equipo se utilizó el servidor local:

```text
http://127.0.0.1:5000
```

Esta dirección se refiere a **la computadora que ejecuta el comando**. No es una URL compartida entre integrantes; cada persona debe levantar su propio servidor o recibir la dirección de uno compartido.

Para un servidor de desarrollo local con almacenamiento SQLite y artefactos en carpetas del proyecto, una opción es ejecutar desde la raíz del repositorio, en otra terminal:

```bash
mkdir -p mlflow_tracking mlartifacts
mlflow server \
  --backend-store-uri "sqlite:///$(pwd)/mlflow_tracking/mlflow.db" \
  --default-artifact-root "$(pwd)/mlartifacts" \
  --host 127.0.0.1 \
  --port 5000
```

> Esta es una **configuración sugerida**, no la misma ruta SQLite centralizada que se utilizó en la computadora donde se realizaron las pruebas. No iniciar dos servidores sobre el mismo puerto. Revisar que `mlflow_tracking/` y `mlartifacts/` estén excluidos de Git antes de utilizarlos. Si se emplea un servidor compartido, también debe configurarse correctamente el acceso a sus artefactos.

Abrir la interfaz en el navegador: `http://127.0.0.1:5000`.

## 7. Ejecutar P1 con MLflow Projects (validado)

Con el entorno virtual activo, los datos procesados disponibles y el servidor en funcionamiento, ejecutar desde la raíz del repositorio:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . \
  -e train \
  -P question=p1 \
  -P model=rf \
  -P n_estimators=100 \
  --env-manager local \
  --experiment-name mibici-equipo36
```

| Argumento | Significado |
|---|---|
| `-e train` | Selecciona el entry point de entrenamiento. |
| `-P question=p1` | Selecciona la pregunta P1. |
| `-P model=rf` | Utiliza Random Forest. |
| `-P n_estimators=100` | Entrena con 100 árboles. |
| `--env-manager local` | Usa el entorno Python ya activado; no lo reconstruye. |
| `--experiment-name mibici-equipo36` | Guarda la corrida en el experimento del equipo. |
| `MLFLOW_TRACKING_URI=...` | Indica el servidor donde registrar la ejecución. |

**Windows PowerShell:** la asignación de la variable de entorno se realiza por separado; luego se ejecuta el comando en una línea:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=100 --env-manager local --experiment-name mibici-equipo36
```

El modelo P1 también admite los hiperparámetros `max_depth` y `min_samples_leaf` en la función Python, con valores predeterminados **12** y **5**, respectivamente. **Aún no están expuestos como parámetros del entry point `train`**; por ahora, desde `mlflow run`, se modifica `n_estimators`.

### Ejecución de referencia confirmada

- **Run ID:** `69e641e3245f4070b5df3dffca72cc15`
- **Nombre de corrida:** `P1_random_forest`
- **Estado:** `Finished`
- **Registros usados:** 127,020 observaciones de 348 estaciones.
- **Partición temporal:** 85,260 entrenamiento; 21,576 validación; 10,440 prueba.

| Métrica | Validación | Prueba |
|---|---:|---:|
| MAE | 5.0119 | 7.2366 |
| RMSE | 7.8936 | 12.2953 |

Estos valores son **evidencia de la ejecución realizada**, no garantías de resultados idénticos si cambian los datos, dependencias o condiciones del entorno.

## 8. Búsqueda de hiperparámetros y selección P1

El script `mibici/modeling/search_p1.py` compara Random Forest con el criterio **menor MAE de validación (`val_mae`)**. Los resultados observados durante el desarrollo fueron:

| Árboles | Profundidad | Muestras mínimas por hoja | MAE validación |
|---:|---:|---:|---:|
| 100 | 8 | 5 | 5.1026 |
| 100 | 12 | 5 | 5.0119 |
| 300 | 12 | 5 | 5.0110 |

Estos resultados no necesariamente pertenecen todos a la misma corrida padre. La selección recuperada del padre indicado a continuación eligió **100 árboles, profundidad 12, MAE 5.0119**. El menor MAE conocido entre las tres configuraciones es 5.0110, correspondiente a 300 árboles; no debe confundirse con la selección de ese padre específico.

**Entrenar tres pruebas nuevas** (solo si hace falta repetir el experimento):

```bash
python -m mibici.modeling.search_p1 --n-trials 3
```

**Recuperar el mejor hijo de un padre existente sin entrenar:**

```bash
python -m mibici.modeling.search_p1 --reuse-parent-run-id 57a29674e7284c0cb34570bc2aa5f4a2
```

El identificador corresponde a la base local de referencia y debe sustituirse en otros entornos. El script consulta únicamente hijas terminadas de ese padre y crea una corrida `P1_MODELO_GANADOR_FINAL` con métricas de selección, *tags* de los hiperparámetros y el ID de la corrida hija elegida. **No registra un nuevo modelo en Model Registry.** Se comprobó esta recuperación sin reentrenar; no se ha vuelto a ejecutar una búsqueda completa con la última versión del script.

La implementación evita intentar cambiar parámetros inmutables de una corrida existente: la configuración seleccionada se registra mediante *tags*. Se recomienda conservar las corridas históricas, incluso las fallidas, como evidencia.

## 9. Consultar resultados y artefactos

En la interfaz MLflow:

1. Abrir el experimento `mibici-equipo36`.
2. Seleccionar la corrida `P1_random_forest` correspondiente.
3. Revisar **Overview** para las cuatro métricas, los parámetros y los tags.
4. Revisar **Artifacts** para los archivos generados.

Se confirmó la presencia de:

```text
figures/
  P1_random_forest_reales_vs_predichos.png
  P1_random_forest_importancia_caracteristicas.png
  P1_random_forest_distribucion_errores.png
models/
  P1_random_forest.joblib
```

**Interpretación inicial:** los errores son mayores en el conjunto de prueba que en validación. En importancia de variables destaca `promedio_mismo_dia_4_semanas`, seguida por `promedio_7_dias`. La distribución de errores se concentra cerca de cero, aunque presenta valores extremos.

**Distinción importante:** el archivo `.joblib` se almacena como artefacto, pero **todavía no se ha registrado un modelo en el formato nativo de MLflow con firma y ejemplo de entrada**, ni se ha publicado en Model Registry.

## 10. Ingesta desde MLflow Projects (declarada, pendiente de validación integral)

El archivo `MLproject` contiene el entry point `ingest` con los parámetros `start_date` y `end_date`:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
mlflow run . \
  -e ingest \
  -P start_date=2025-10 \
  -P end_date=2026-09 \
  --env-manager local \
  --experiment-name mibici-equipo36
```

**No marcar este flujo como validado todavía.** Requiere disponer de los CSV originales, revisar la configuración de rutas y verificar la generación de los datos procesados y el manifiesto.

## 11. Cambios publicados en GitHub

Se trabaja sobre `feat/mlflow-project` para que los demás integrantes revisen los cambios mediante Pull Request hacia `main`. Comprobar en GitHub si el PR ya existe antes de crear otro.

| Commit | Cambio principal |
|---|---|
| `2245d0d` | Integración inicial de `MLproject`, `python_env.yaml`, coordinador y parametrización de P1. |
| `0aecd6e` | Reutilización de la corrida de MLflow Projects al registrar métricas y artefactos de P1. |

Para actualizar la rama local antes de ejecutar:

```bash
git fetch origin
git switch feat/mlflow-project
git pull --ff-only origin feat/mlflow-project
```

**No subir a Git** bases SQLite, directorios `mlruns/`, `mlflow_tracking/`, `mlartifacts/`, datos privados, credenciales ni modelos generados. Revisar siempre `git status --short` antes de agregar archivos.

## 12. Limitaciones conocidas y próximos pasos

| Tema | Estado / siguiente acción |
|---|---|
| P1 Random Forest, 100 árboles | **Validado** mediante MLflow Projects. |
| P1 Random Forest, 300 árboles | Ejecutado durante la búsqueda; MAE de validación observado: 5.0110. |
| Baseline P1 desde MLflow Projects | Disponible en el coordinador; pendiente de validación específica. |
| P2 y P3 desde `train` | **No integrados**. Actualmente el coordinador informa que se incorporarán posteriormente; no debe interpretarse ese mensaje como entrenamiento exitoso. |
| Ingesta mediante `mlflow run` | Declarada; pendiente de prueba integral. |
| Entorno automático | `python_env.yaml` existe, pero su reconstrucción sin `--env-manager local` no está validada. |
| Búsqueda de hiperparámetros | Implementada con corridas anidadas y recuperación sin reentrenar. |
| Registro de modelos | Pendiente de usar el formato MLflow Model, firma, ejemplo de entrada, versiones y alias `champion`. |
| Configuración | Revisar centralización de fechas y parámetros; evitar rutas específicas de un integrante. |
| Documentación | Mantener esta guía actualizada con cada funcionalidad validada. |

## 13. Solución rápida de problemas

**`Connection refused` al puerto 5000:** comprobar que el servidor MLflow esté encendido y que `MLFLOW_TRACKING_URI` apunte a la dirección correcta.

**No aparece `viajes_limpios.parquet`:** revisar el directorio `data/processed/`, la configuración de `MIBICI_DATA_DIR` y la ejecución previa de la ingesta.

**`ModuleNotFoundError`:** confirmar que el entorno virtual está activo e instalar las dependencias de `requirements.txt`.

**La corrida aparece, pero sin métricas:** comprobar que el comando realmente llegó a ejecutar P1 y no solo el coordinador; consultar la salida de la terminal y el estado de la corrida.

**P2/P3 imprime un mensaje sin entrenar:** comportamiento actual conocido; todavía no están conectados al entry point `train`.

**No aparecen los artefactos desde otra computadora:** revisar dónde almacena los artefactos el servidor MLflow; una ruta local de otro integrante no necesariamente es accesible remotamente.

---

**Responsabilidad del equipo:** antes de considerar un paso como terminado, conservar el comando utilizado, el resultado observado, el identificador de corrida MLflow y el commit de GitHub asociado. Esta guía debe evolucionar junto con el Pull Request.


## 14. Cierre de P1 y transferencia al equipo

**Archivos para el commit en `feat/mlflow-project`:** `MLproject`, `mibici/modeling/run_project.py`, `mibici/modeling/train.py`, `mibici/modeling/search_p1.py` y este README. Revisar `git diff`, `git diff --check` y la sintaxis con `python -m py_compile` antes de confirmar. Hacer push y revisión por Pull Request; no fusionar sin aprobación del equipo.

**Entrega de MLflow:** la base de referencia se encuentra en `~/tec/Operaciones ML/mlflow_tracking/mlflow.db`. Esta base SQLite guarda los metadatos de corridas, métricas, parámetros y etiquetas, pero los archivos de artefactos (gráficas y modelos `.joblib`) pueden estar almacenados aparte. Respaldar la base con la API de copia de SQLite, identificar y copiar los artefactos correspondientes y verificar el paquete antes de entregarlo. **No subir a GitHub la base, los datos ni los artefactos generados.**

**Estado para los compañeros:** P1 tiene entrenamiento y evaluación registrados, búsqueda de hiperparámetros y selección recuperable sin entrenar. P2 y P3 conservan sus implementaciones propias, pero su integración al entry point `train` de MLflow Projects no está validada. La generación automática del entorno y la ingesta mediante MLflow Projects también están pendientes.
