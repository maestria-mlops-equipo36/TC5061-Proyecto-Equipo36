import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
import typer
from loguru import logger
from tqdm import tqdm

from mibici.config import (
    MIBICI_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    DATA_MANIFEST_PATH,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
)

app = typer.Typer(help="Ingesta y preprocesamiento reproducible de datos de MiBici.")

NOMENCLATURA_URL = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
BASE_DOWNLOAD_URL = "https://mibici.net/site/assets/files/1118/datos_abiertos_{anio}_{mes:02d}.csv"


def calcular_hash_archivo(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo para trazabilidad y versionado."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def asegurar_nomenclatura(output_dir: Path = EXTERNAL_DATA_DIR) -> Path:
    """Descarga la nomenclatura oficial de estaciones si no existe localmente o es inválida."""
    output_dir.mkdir(parents=True, exist_ok=True)
    nomenclatura_file = output_dir / "nomenclatura_2026_09.csv"

    # Si ya existe y es un CSV válido no vacío, reutilizarlo
    if nomenclatura_file.exists() and nomenclatura_file.stat().st_size > 1024:
        try:
            df_check = pd.read_csv(nomenclatura_file, nrows=5)
            if "id" in df_check.columns or any("lat" in str(c).lower() for c in df_check.columns):
                logger.info(f"Reutilizando nomenclatura existente válida en: {nomenclatura_file}")
                return nomenclatura_file
        except Exception:
            logger.warning("El archivo existente no es un CSV válido. Reintentando descarga...")

    logger.info(f"Descargando nomenclatura desde {NOMENCLATURA_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml,text/csv,*/*",
    }

    descarga_exitosa = False

    # 1. Intentar con requests usando headers estándar
    try:
        resp = requests.get(NOMENCLATURA_URL, headers=headers, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1024:
            nomenclatura_file.write_bytes(resp.content)
            descarga_exitosa = True
            logger.success("Descarga completada con requests.")
        else:
            logger.warning(f"requests retornó código {resp.status_code}. Probando método alternativo...")
    except Exception as e:
        logger.warning(f"Error en descarga con requests ({e}). Probando método alternativo...")

    # 2. Mecanismo alternativo con curl / urllib si requests falló
    if not descarga_exitosa:
        import subprocess
        logger.info("Intentando descarga alternativa mediante curl...")
        try:
            subprocess.run(
                ["curl", "-s", "-L", "-A", headers["User-Agent"], "-o", str(nomenclatura_file), NOMENCLATURA_URL],
                check=True,
                timeout=30,
            )
            descarga_exitosa = True
            logger.success("Descarga completada con curl.")
        except Exception as e:
            logger.error(f"Fallo en descarga alternativa con curl: {e}")

    # Validar que el archivo descargado sea un CSV válido
    if not nomenclatura_file.exists() or nomenclatura_file.stat().st_size < 100:
        raise RuntimeError(
            f"No se pudo descargar o validar un archivo CSV correcto desde {NOMENCLATURA_URL}."
        )

    return nomenclatura_file


def leer_csv_con_fallback(archivo: Path) -> pd.DataFrame:
    """Lee un CSV mensual probando encodings habituales y normalizando columnas."""
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(archivo, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"No se pudo decodificar el archivo: {archivo}")

    # Normalizar nombres de columnas
    rename_dict = {}
    for col in df.columns:
        col_clean = col.strip().replace('"', "")
        if "nacimiento" in col_clean.lower():
            rename_dict[col] = "Anio_de_nacimiento"
        elif col_clean.lower() in ["viaje_id", "viaje id"]:
            rename_dict[col] = "Viaje_Id"
        elif col_clean.lower() in ["usuario_id", "usuario id"]:
            rename_dict[col] = "Usuario_Id"
        elif col_clean.lower() in ["genero", "género"]:
            rename_dict[col] = "Genero"
        elif col_clean.lower() in ["inicio_del_viaje", "inicio del viaje"]:
            rename_dict[col] = "Inicio_del_viaje"
        elif col_clean.lower() in ["fin_del_viaje", "fin del viaje"]:
            rename_dict[col] = "Fin_del_viaje"
        elif col_clean.lower() in ["origen_id", "origen id"]:
            rename_dict[col] = "Origen_Id"
        elif col_clean.lower() in ["destino_id", "destino id"]:
            rename_dict[col] = "Destino_Id"
        else:
            rename_dict[col] = col_clean

    df = df.rename(columns=rename_dict)
    return df


def filtrar_archivos_por_rango(
    input_path: Path, start_date: str, end_date: str
) -> List[Tuple[int, int, Path]]:
    """Filtra archivos mensuales que caigan dentro del rango [start_date, end_date] (formato YYYY-MM)."""
    archivos = sorted(input_path.glob("*.csv"))
    seleccionados = []
    
    start_tuple = tuple(map(int, start_date.split("-")))
    end_tuple = tuple(map(int, end_date.split("-")))

    for arc in archivos:
        match = re.fullmatch(r"datos_abiertos_(\d{4})_(\d{2})\.csv", arc.name, re.IGNORECASE)
        if match:
            anio, mes = int(match.group(1)), int(match.group(2))
            if start_tuple <= (anio, mes) <= end_tuple:
                seleccionados.append((anio, mes, arc))

    return seleccionados


@app.command()
def main(
    input_path: Path = MIBICI_DATA_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """
    Ingesta y consolida los datos mensuales de MiBici en formato Parquet,
    generando el data_manifest.json para versionado y reproducibilidad.
    """
    logger.info(f"Iniciando proceso de ingesta: rango {start_date} a {end_date}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Asegurar catálogo de estaciones
    nomenclatura_path = asegurar_nomenclatura()
    df_estaciones = pd.read_csv(nomenclatura_path)
    df_estaciones.columns = [c.strip().replace('"', "") for c in df_estaciones.columns]
    df_estaciones = df_estaciones.rename(
        columns={
            "id": "estacion_id",
            "name": "nombre",
            "latitude": "latitud",
            "longitude": "longitud",
            "dpcapacity": "capacidad",
        }
    )
    df_estaciones["estacion_id"] = pd.to_numeric(df_estaciones["estacion_id"], errors="coerce")
    df_estaciones["latitud"] = pd.to_numeric(df_estaciones["latitud"], errors="coerce")
    df_estaciones["longitud"] = pd.to_numeric(df_estaciones["longitud"], errors="coerce")
    df_estaciones["capacidad"] = pd.to_numeric(df_estaciones["capacidad"], errors="coerce")
    df_estaciones = df_estaciones.dropna(subset=["estacion_id"]).copy()
    df_estaciones["estacion_id"] = df_estaciones["estacion_id"].astype(int)

    estaciones_parquet = output_dir / "estaciones.parquet"
    df_estaciones.to_parquet(estaciones_parquet, index=False)
    logger.success(f"Estaciones guardadas en {estaciones_parquet} ({len(df_estaciones)} estaciones)")

    # 2. Filtrar archivos en rango
    archivos = filtrar_archivos_por_rango(input_path, start_date, end_date)
    if not archivos:
        logger.error(f"No se encontraron archivos en {input_path} para el rango {start_date} a {end_date}")
        raise typer.Exit(code=1)

    logger.info(f"Procesando {len(archivos)} archivos mensuales...")
    
    dfs = []
    manifest_archivos = []
    total_viajes = 0

    for anio, mes, archivo in tqdm(archivos, desc="Procesando meses"):
        hash_csv = calcular_hash_archivo(archivo)
        df_mes = leer_csv_con_fallback(archivo)

        # Parsear tipos y limpiar valores faltantes esenciales
        df_mes["Origen_Id"] = pd.to_numeric(df_mes["Origen_Id"], errors="coerce")
        df_mes["Destino_Id"] = pd.to_numeric(df_mes["Destino_Id"], errors="coerce")
        df_mes = df_mes.dropna(subset=["Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje"]).copy()
        
        df_mes["Origen_Id"] = df_mes["Origen_Id"].astype(int)
        df_mes["Destino_Id"] = df_mes["Destino_Id"].astype(int)
        df_mes["Inicio_del_viaje"] = pd.to_datetime(df_mes["Inicio_del_viaje"], errors="coerce")
        df_mes["Fin_del_viaje"] = pd.to_datetime(df_mes["Fin_del_viaje"], errors="coerce")
        df_mes = df_mes.dropna(subset=["Inicio_del_viaje", "Fin_del_viaje"]).copy()

        # Calcular duración del viaje en minutos
        df_mes["duracion_min"] = (df_mes["Fin_del_viaje"] - df_mes["Inicio_del_viaje"]).dt.total_seconds() / 60.0
        
        # Filtrar viajes ilógicos/outliers (menores a 1 min o mayores a 8 horas)
        df_mes = df_mes[(df_mes["duracion_min"] >= 1.0) & (df_mes["duracion_min"] <= 480.0)].copy()

        num_registros = len(df_mes)
        total_viajes += num_registros
        dfs.append(df_mes)

        manifest_archivos.append({
            "archivo": archivo.name,
            "anio": anio,
            "mes": mes,
            "sha256": hash_csv,
            "registros_validos": num_registros,
        })

    # 3. Concatenar y guardar en Parquet
    df_total = pd.concat(dfs, ignore_index=True)
    viajes_parquet = output_dir / "viajes_limpios.parquet"
    df_total.to_parquet(viajes_parquet, index=False)
    logger.success(f"Viajes consolidados guardados en {viajes_parquet} ({len(df_total)} registros)")

    # 4. Crear manifiesto versionable en Git
    manifest_data = {
        "dataset_version": f"mibici-{start_date}_to_{end_date}-v1",
        "rango_fechas": {"inicio": start_date, "fin": end_date},
        "total_meses": len(archivos),
        "total_viajes": total_viajes,
        "total_estaciones_nomenclatura": len(df_estaciones),
        "hash_consolidado_parquet": calcular_hash_archivo(viajes_parquet),
        "archivos_origen": manifest_archivos,
    }

    with open(DATA_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.success(f"Manifiesto de datos generado exitosamente en: {DATA_MANIFEST_PATH}")


if __name__ == "__main__":
    app()