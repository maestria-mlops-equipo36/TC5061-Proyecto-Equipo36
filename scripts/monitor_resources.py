
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_percentage(value):
    """Convierte '12.5%' en 12.5."""
    return float(value.strip().replace("%", ""))

import re

def parse_memory(value):
    """Convierte valores de memoria de Docker a MiB."""
    value = value.strip()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)",
        value
    )

    if not match:
        raise ValueError(f"Formato de memoria no reconocido: {value}")

    number = float(match.group(1))
    unit = match.group(2)

    factors = {
        "B": 1 / (1024 ** 2),
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "TiB": 1024 ** 2,
        "kB": 1000 / (1024 ** 2),
        "MB": 1000 ** 2 / (1024 ** 2),
        "GB": 1000 ** 3 / (1024 ** 2),
    }

    if unit not in factors:
        raise ValueError(f"Unidad de memoria no soportada: {unit}")

    return number * factors[unit]

def collect_stats():
    """Obtiene estadísticas de contenedores activos."""
    command = [
        "docker", "stats", "--no-stream",
        "--format", "{{json .}}"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    records = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        item = json.loads(line)
        memory_used, memory_limit = item["MemUsage"].split("/")

        records.append({
            "container_id": item["ID"],
            "container_name": item["Name"],
            "cpu_percent": parse_percentage(item["CPUPerc"]),
            "memory_used_mib": round(parse_memory(memory_used), 3),
            "memory_limit_mib": round(parse_memory(memory_limit), 3),
            "memory_percent": parse_percentage(item["MemPerc"]),
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Monitoreo de recursos Docker para MiBici"
    )

    parser.add_argument(
        "--duration", type=int, default=60,
        help="Duración del monitoreo en segundos"
    )
    parser.add_argument(
        "--interval", type=float, default=2,
        help="Intervalo de muestreo en segundos"
    )
    parser.add_argument(
        "--scenario", default="baseline",
        help="Nombre del escenario de carga"
    )
    parser.add_argument(
        "--output",
        default="reports/finops/resource_usage.csv",
        help="Ruta del archivo CSV"
    )

    args = parser.parse_args()

    if args.duration <= 0 or args.interval <= 0:
        parser.error("La duración y el intervalo deben ser positivos")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp_utc",
        "scenario",
        "container_id",
        "container_name",
        "cpu_percent",
        "memory_used_mib",
        "memory_limit_mib",
        "memory_percent",
    ]

    print(f"Escenario: {args.scenario}")
    print(f"Duración: {args.duration} segundos")
    print(f"Intervalo: {args.interval} segundos")
    print(f"Archivo: {output}")

    deadline = time.monotonic() + args.duration
    samples = 0

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        while time.monotonic() < deadline:
            timestamp = datetime.now(timezone.utc).isoformat()

            try:
                records = collect_stats()
            except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as exc:
                print(f"Error al obtener estadísticas: {exc}")
                break

            for record in records:
                writer.writerow({
                    "timestamp_utc": timestamp,
                    "scenario": args.scenario,
                    **record,
                })

            file.flush()
            samples += 1
            print(
                f"Muestra {samples}: "
                f"{len(records)} contenedores"
            )

            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(args.interval, remaining))

    print(f"Monitoreo finalizado. Muestras: {samples}")


if __name__ == "__main__":
    main()
