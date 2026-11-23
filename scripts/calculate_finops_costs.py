import argparse
import csv
from pathlib import Path

BASE = Path("reports/finops")


def main():
    parser = argparse.ArgumentParser(
        description="Estimacion FinOps AWS para MiBici"
    )
    parser.add_argument(
        "--precio-hora",
        type=float,
        default=0.1664,
        help="Tarifa EC2 en USD por hora",
    )
    parser.add_argument(
        "--horas-mes",
        type=float,
        default=730,
    )
    args = parser.parse_args()

    if args.precio_hora <= 0 or args.horas_mes <= 0:
        parser.error("El precio y las horas deben ser positivos")

    origen = BASE / "resumen_consolidado.csv"
    destino = BASE / "costos_aws.csv"

    with origen.open(newline="", encoding="utf-8-sig") as f:
        escenarios = list(csv.DictReader(f))

    resultados = []

    for escenario in escenarios:
        usuarios = int(escenario["usuarios"])
        rps = float(escenario["throughput_predict_rps"])

        if rps <= 0:
            raise ValueError(
                f"Throughput invalido para {usuarios} usuarios"
            )

        predicciones_hora = rps * 3600
        costo_hora = args.precio_hora

        costo_1000 = (
            costo_hora / predicciones_hora
        ) * 1000

        resultados.append({
            "usuarios": usuarios,
            "instancia_aws": "t3.xlarge",
            "region": "us-east-1",
            "predicciones_exitosas_prueba": int(
                escenario["predicciones_exitosas"]
            ),
            "throughput_predict_rps": round(rps, 4),
            "predicciones_hora_estimadas": round(
                predicciones_hora, 2
            ),
            "costo_ec2_hora_usd": round(costo_hora, 6),
            "costo_ec2_mes_usd": round(
                costo_hora * args.horas_mes, 2
            ),
            "costo_ec2_1000_predicciones_usd": round(
                costo_1000, 6
            ),
            "cpu_promedio_api_pct": escenario[
                "cpu_promedio_api_pct"
            ],
            "memoria_total_promedio_mib": escenario[
                "memoria_total_promedio_mib"
            ],
        })

    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(
            f, fieldnames=resultados[0].keys()
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    print("\nESTIMACION FINOPS AWS - MIBICI")
    print("=" * 72)
    print("Instancia: t3.xlarge | Region: us-east-1")
    print(f"Tarifa EC2: USD {args.precio_hora:.4f}/hora")
    print(f"Horas mensuales: {args.horas_mes:.0f}")
    print("-" * 72)

    for fila in resultados:
        print(
            f"{fila['usuarios']:>2} usuarios | "
            f"{fila['predicciones_hora_estimadas']:>9,.0f} pred/h | "
            f"USD {fila['costo_ec2_hora_usd']:.4f}/h | "
            f"USD {fila['costo_ec2_1000_predicciones_usd']:.6f}/1000"
        )

    print(f"\nArchivo generado: {destino}")
    print(
        "\nNota: estimacion de computo EC2, no facturacion real."
    )


if __name__ == "__main__":
    main()
