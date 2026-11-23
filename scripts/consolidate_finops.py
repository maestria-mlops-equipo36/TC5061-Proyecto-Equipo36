import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE = Path("reports/finops")
ESCENARIOS = (10, 25, 50)


def leer_locust(usuarios):
    archivo = BASE / f"locust_{usuarios}u_finops_stats.csv"

    with archivo.open(newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))

    prediccion = next(
        fila for fila in filas
        if fila["Type"] == "POST" and fila["Name"] == "/predict"
    )

    solicitudes = int(prediccion["Request Count"])
    errores = int(prediccion["Failure Count"])

    return {
        "predicciones_exitosas": solicitudes - errores,
        "errores": errores,
        "throughput_predict_rps": float(prediccion["Requests/s"]),
        "latencia_promedio_ms": float(prediccion["Average Response Time"]),
        "latencia_p95_ms": float(prediccion["95%"]),
    }


def leer_recursos(usuarios):
    archivo = BASE / f"recursos_{usuarios}u.csv"
    contenedores = defaultdict(list)

    with archivo.open(newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            contenedores[fila["container_name"]].append(fila)

    resultado = {}

    for servicio in ("api", "mlflow"):
        nombres = [
            nombre for nombre in contenedores
            if nombre.endswith(f"-{servicio}-1")
        ]

        if len(nombres) != 1:
            raise ValueError(
                f"Se esperaba un contenedor {servicio}: {nombres}"
            )

        filas = contenedores[nombres[0]]
        cpu = [float(f["cpu_percent"]) for f in filas]
        memoria = [float(f["memory_used_mib"]) for f in filas]

        resultado[f"cpu_promedio_{servicio}_pct"] = round(mean(cpu), 2)
        resultado[f"cpu_maxima_{servicio}_pct"] = round(max(cpu), 2)
        resultado[f"memoria_promedio_{servicio}_mib"] = round(
            mean(memoria), 2
        )
        resultado[f"memoria_maxima_{servicio}_mib"] = round(
            max(memoria), 2
        )
        resultado[f"muestras_{servicio}"] = len(filas)

    resultado["memoria_total_promedio_mib"] = round(
        resultado["memoria_promedio_api_mib"]
        + resultado["memoria_promedio_mlflow_mib"],
        2,
    )

    return resultado


def main():
    resultados = []

    for usuarios in ESCENARIOS:
        fila = {"usuarios": usuarios}
        fila.update(leer_locust(usuarios))
        fila.update(leer_recursos(usuarios))
        resultados.append(fila)

    destino = BASE / "resumen_consolidado.csv"

    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f, fieldnames=list(resultados[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nArchivo generado: {destino}")
    print("\nRESUMEN FINOPS")
    print("-" * 78)

    for fila in resultados:
        print(
            f"{fila['usuarios']:>2} usuarios | "
            f"Predicciones: {fila['predicciones_exitosas']:>3} | "
            f"RPS: {fila['throughput_predict_rps']:.2f} | "
            f"P95: {fila['latencia_p95_ms']:.0f} ms | "
            f"CPU API: {fila['cpu_promedio_api_pct']:.2f}% | "
            f"RAM total: {fila['memoria_total_promedio_mib']:.2f} MiB"
        )


if __name__ == "__main__":
    main()
