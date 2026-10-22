import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Coordinador de entrenamientos MiBici"
    )

    parser.add_argument(
        "--question",
        choices=["p1", "p2", "p3"],
        required=True,
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--n-estimators", type=int, default=100)

    args = parser.parse_args()

    print("Proyecto: MiBici Equipo 36")
    print(f"Pregunta: {args.question}")
    print(f"Modelo: {args.model}")
    print(f"Número de árboles: {args.n_estimators}")

    if args.question == "p1":
        from mibici.modeling.train import p1_regresion

        if args.model not in ["rf", "baseline"]:
            parser.error("Para P1 utiliza 'rf' o 'baseline'.")

        modelo_tipo = (
            "random_forest" if args.model == "rf" else "baseline"
        )

        p1_regresion(
            modelo_tipo=modelo_tipo,
            n_estimators=args.n_estimators,
        )

    else:
        print(
            f"El entrenamiento de {args.question} "
            "se integrará posteriormente."
        )


if __name__ == "__main__":
    main()
