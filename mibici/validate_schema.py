import pandas as pd
from loguru import logger

from mibici.config import MIBICI_DATA_DIR, INTERIM_DATA_DIR

def leer_columnas(archivo):
    for encoding in ["utf-8-sig", "cp1252"]:
        try:
            columnas = pd.read_csv(
                archivo,
                nrows=0,
                encoding=encoding,
            ).columns.tolist()

            return columnas, encoding

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"No se pudo interpretar el archivo: {archivo.name}"
    )
    
def normalizar_columnas(columnas):
    columnas_esperadas = [
        "Viaje_Id",
        "Usuario_Id",
        "Genero",
        "Año_de_nacimiento",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    columnas_normalizadas = []

    for columna in columnas:
        if "nacimiento" in columna.lower():
            columnas_normalizadas.append("Año_de_nacimiento")
        else:
            columnas_normalizadas.append(columna)

    return columnas_normalizadas
def main():
    archivos = sorted(MIBICI_DATA_DIR.glob("*.csv"))

    if not archivos:
        logger.error("No se encontraron archivos CSV.")
        return

    """columnas_referencia = pd.read_csv(
        archivos[0],
        nrows=0,
        encoding="latin-1"
    ).columns.tolist()
    """
    columnas_referencia, _ = leer_columnas(archivos[0])
    columnas_referencia = normalizar_columnas(columnas_referencia)
    resultados = []

    for archivo in archivos:
        """
        columnas = pd.read_csv(
            archivo,
            nrows=0,
            encoding="latin-1"
        ).columns.tolist()
        """
        columnas, encoding = leer_columnas(archivo)
        columnas = normalizar_columnas(columnas)
        resultados.append({
            "archivo": archivo.name,
            "numero_columnas": len(columnas),
            "estructura_valida": columnas == columnas_referencia,
            "encoding": encoding,
            "columnas": ", ".join(columnas),
        })

    df = pd.DataFrame(resultados)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    salida = INTERIM_DATA_DIR / "validacion_esquemas.csv"
    df.to_csv(salida, index=False)

    logger.info(f"Archivos revisados: {len(df)}")
    logger.info(
        f"Estructuras válidas: {df['estructura_valida'].sum()}"
    )
    logger.info(
        f"Estructuras diferentes: {(~df['estructura_valida']).sum()}"
    )

    if not df["estructura_valida"].all():
        logger.warning("Archivos con estructura diferente:")
        print(
            df.loc[
                ~df["estructura_valida"],
                ["archivo", "numero_columnas", "columnas"],
            ].to_string(index=False)
        )

    logger.success(f"Reporte generado: {salida}")


if __name__ == "__main__":
    main()