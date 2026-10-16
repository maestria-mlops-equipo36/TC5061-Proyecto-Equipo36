# MiBici Data Science

**Proyecto integrador — Equipo 36 | Maestría, materia: Operaciones de Aprendizaje Automático (MLOps)**

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Repositorio de trabajo colaborativo para organizar, validar y posteriormente analizar datos abiertos de viajes del sistema MiBici. El proyecto utiliza la estructura de **Cookiecutter Data Science** y busca mantener separados el código versionado y los datos originales.

**Repositorio:** https://github.com/maestria-mlops-equipo36/mibici-data-science

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
git clone https://github.com/maestria-mlops-equipo36/mibici-data-science.git
cd mibici-data-science
```

Se recomienda utilizar una versión de Python compatible con las dependencias del proyecto. La configuración inicial del proyecto se realizó con Python **3.14.4**; la compatibilidad de todas las dependencias debe comprobarse en cada entorno.

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

Si no se detectan archivos, comprobar que `MIBICI_DATA_DIR` apunte a la carpeta correcta y que los nombres sigan el patrón `datos_abiertos_AAAA_MM.csv`.

## 6. Organización del proyecto

```text
mibici-data-science/
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

**Implementado:** estructura Cookiecutter, inventario mensual y validación de encabezados de los CSV.

**Pendiente:** validación de calidad a nivel de registros, preparación de datos, análisis exploratorio, desarrollo de modelos y seguimiento de experimentos, según avance el proyecto integrador.

## 9. Licencia

Este repositorio incluye una licencia MIT (ver [`LICENSE`](LICENSE)). Los datos originales pueden estar sujetos a condiciones de uso distintas: consultar las condiciones publicadas por MiBici.
