import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

BASE = Path("reports/finops")
DESTINO = Path("reports/figures/finops")
DESTINO.mkdir(parents=True, exist_ok=True)


def leer_csv(nombre):
    with (BASE / nombre).open(
        newline="", encoding="utf-8-sig"
    ) as archivo:
        return list(csv.DictReader(archivo))


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(DESTINO / nombre, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Generada: {DESTINO / nombre}")


datos = leer_csv("resumen_consolidado.csv")
costos = leer_csv("costos_aws.csv")

usuarios = [int(f["usuarios"]) for f in datos]
posiciones = list(range(len(usuarios)))
etiquetas = [str(u) for u in usuarios]

# 1. CPU promedio
fig, ax = plt.subplots(figsize=(9, 5))
ancho = 0.35

ax.bar(
    [p - ancho / 2 for p in posiciones],
    [float(f["cpu_promedio_api_pct"]) for f in datos],
    width=ancho,
    label="FastAPI",
)
ax.bar(
    [p + ancho / 2 for p in posiciones],
    [float(f["cpu_promedio_mlflow_pct"]) for f in datos],
    width=ancho,
    label="MLflow",
)

ax.set_xticks(posiciones, etiquetas)
ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("CPU promedio (%)")
ax.set_title("MiBici - Consumo promedio de CPU")
ax.legend()
ax.grid(axis="y", alpha=0.25)
guardar("cpu_por_escenario.png")

# 2. Memoria promedio
fig, ax = plt.subplots(figsize=(9, 5))

api = [float(f["memoria_promedio_api_mib"]) for f in datos]
mlflow = [
    float(f["memoria_promedio_mlflow_mib"]) for f in datos
]

ax.bar(etiquetas, api, label="FastAPI")
ax.bar(etiquetas, mlflow, bottom=api, label="MLflow")

ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("Memoria promedio (MiB)")
ax.set_title("MiBici - Memoria promedio por escenario")
ax.legend()
ax.grid(axis="y", alpha=0.25)
guardar("memoria_por_escenario.png")

# 3. Latencia
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    usuarios,
    [float(f["latencia_promedio_ms"]) / 1000 for f in datos],
    marker="o",
    label="Promedio",
)
ax.plot(
    usuarios,
    [float(f["latencia_p95_ms"]) / 1000 for f in datos],
    marker="s",
    label="P95",
)

ax.set_xticks(usuarios)
ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("Latencia (segundos)")
ax.set_title("MiBici - Latencia de predicciones")
ax.legend()
ax.grid(alpha=0.25)
guardar("latencia_por_escenario.png")

# 4. Costos AWS
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    [str(f["usuarios"]) for f in costos],
    [
        float(f["costo_ec2_1000_predicciones_usd"])
        for f in costos
    ],
)

ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("USD por 1,000 predicciones")
ax.set_title("MiBici - Costo EC2 estimado por 1,000 predicciones")
ax.grid(axis="y", alpha=0.25)
guardar("costo_aws_por_1000.png")

print("\nGraficas FinOps generadas correctamente.")
