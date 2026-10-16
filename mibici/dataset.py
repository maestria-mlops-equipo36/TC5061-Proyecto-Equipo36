from pathlib import Path
import re

import pandas as pd
import typer
from loguru import logger

from mibici.config import MIBICI_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = MIBICI_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "inventario_mibici.csv",
):
    logger.info(f"Directorio de origen: {input_path}")

    if not input_path.is_dir():
        logger.error(f"No existe el directorio: {input_path}")
        raise typer.Exit(code=1)

    archivos = sorted(input_path.glob("*.csv"))

    logger.info(f"Archivos encontrados: {len(archivos)}")

    registros = []

    for archivo in archivos:
        coincidencia = re.fullmatch(
            r"datos_abiertos_(\d{4})_(\d{2})\.csv",
            archivo.name,
            flags=re.IGNORECASE,
        )

        anio = None
        mes = None

        if coincidencia:
            anio = int(coincidencia.group(1))
            mes = int(coincidencia.group(2))

        registros.append({
            "archivo": archivo.name,
            "anio": anio,
            "mes": mes,
            "tamano_mb": round(
                archivo.stat().st_size / (1024 ** 2), 2
            ),
        })

    df = pd.DataFrame(
        registros,
        columns=["archivo", "anio", "mes", "tamano_mb"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Inventario generado: {output_path}")
    logger.info(f"Total de archivos: {len(df)}")
    logger.info(f"Tamaño total: {df['tamano_mb'].sum():.2f} MB")


if __name__ == "__main__":
    app()