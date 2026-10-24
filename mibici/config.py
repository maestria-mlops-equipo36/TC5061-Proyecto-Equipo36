import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Ruta de los archivos históricos originales de MiBici
MIBICI_DATA_DIR = Path(
    os.getenv("MIBICI_DATA_DIR", str(RAW_DATA_DIR))
)

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configuración del proyecto y experimentos
RANDOM_SEED = 42
DEFAULT_START_DATE = "2025-10"
DEFAULT_END_DATE = "2026-09"
NOMENCLATURA_PATH = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
DATA_MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# MLFlow Tracking
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo36")
TEAM_NAME = "Equipo36"

# Configuración por defecto para P3 (Clustering)
P3_FEATURES_PATH = PROCESSED_DATA_DIR / "features_p3_estaciones.parquet"
P3_DEFAULT_MODEL = "kmeans"
P3_DEFAULT_N_CLUSTERS = 4
P3_DEFAULT_COVARIANCE_TYPE = "full"
P3_DEFAULT_INCLUIR_ESPACIAL = True
P3_FEATURE_COLS_BASE = [
    "ratio_flujo_neto",
    "pct_salidas_pico_matutino",
    "pct_salidas_pico_vespertino",
    "pct_salidas_fin_de_semana",
    "viajes_diarios_por_anclaje",
]
P3_FEATURE_COLS_ESPACIAL = P3_FEATURE_COLS_BASE + ["latitud", "longitud"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
