import argparse

from mibici.config import (
    P3_DEFAULT_COVARIANCE_TYPE,
    P3_DEFAULT_INCLUIR_ESPACIAL,
    P3_DEFAULT_N_CLUSTERS,
)


def _parse_bool(valor: str) -> bool:
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip().lower()
    if texto in ("true", "1", "yes", "si", "sí"):
        return True
    if texto in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"Valor booleano inválido: {valor}. Usa 'true' o 'false'."
    )


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
    # Parámetros de P1
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    # Parámetros de P2 
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--penalty", type=str, default="l2")
    # Parámetros de P3 (valores por defecto desde mibici.config)
    parser.add_argument(
        "--n-clusters", type=int, default=P3_DEFAULT_N_CLUSTERS
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default=P3_DEFAULT_COVARIANCE_TYPE,
        choices=["full", "tied", "diag", "spherical"],
    )
    parser.add_argument(
        "--incluir-espacial",
        type=_parse_bool,
        default=P3_DEFAULT_INCLUIR_ESPACIAL,
    )
    parser.add_argument(
        "--stage-of-work",
        type=str,
        default="auto",
        choices=["auto", "baseline", "search", "candidate"],
    )

    args = parser.parse_args()
    stage = None if args.stage_of_work == "auto" else args.stage_of_work

    print("Proyecto: MiBici Equipo 36")
    print(f"Pregunta: {args.question}")
    print(f"Modelo: {args.model}")

    

    if args.question == "p1":
        from mibici.modeling.train import p1_regresion

        print(f"Número de árboles: {args.n_estimators}")

        if args.model not in ["rf", "baseline", "search"]:
            parser.error("Para P1 utiliza 'rf', 'baseline' o 'search'.")

        if args.model == "search":
            import os
            import subprocess
            import sys

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mibici.modeling.search_p1",
                    "--tracking-uri",
                    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                ],
                check=True,
            )
        else:
            modelo_tipo = (
                "random_forest" if args.model == "rf" else "baseline"
            )

            p1_regresion(
                modelo_tipo=modelo_tipo,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
            )

    
    elif args.question == "p2":
        from mibici.modeling.search_p2 import ejecutar_busqueda_p2
        from mibici.modeling.train_p2 import p2_clasificacion

        modelo_p2 = args.model.strip().lower()
        print(f"Configuración P2: C={args.c_param}, solver={args.solver}, penalty={args.penalty}")

        if modelo_p2 in ("baseline", "dummy"):
            p2_clasificacion(
                modelo_tipo="baseline",
                stage_of_work=stage or "baseline",
            )
        elif modelo_p2 in ("lr", "logistic_regression", "candidate"):
            p2_clasificacion(
                modelo_tipo="logistic_regression",
                c_param=args.c_param,
                solver=args.solver,
                penalty=args.penalty,
                stage_of_work=stage or "candidate",
            )
        elif modelo_p2 == "search":
            ejecutar_busqueda_p2()
        else:
            parser.error(
                "Para P2 utiliza 'baseline', 'lr', 'logistic_regression' o 'search'."
            )

    elif args.question == "p3":
        from mibici.modeling.baseline_reglas import ejecutar_baseline_reglas_p3
        from mibici.modeling.search_p3 import ejecutar_busqueda_p3
        from mibici.modeling.train import p3_clustering

        modelo_p3 = args.model.strip().lower()
        print(
            f"Configuración P3: n_clusters={args.n_clusters}, "
            f"covariance_type={args.covariance_type}, "
            f"incluir_espacial={args.incluir_espacial}"
        )

        if modelo_p3 in ("baseline", "reglas", "reglas_intensidad"):
            ejecutar_baseline_reglas_p3(
                stage_of_work=stage or "baseline",
            )
        elif modelo_p3 in ("kmeans", "gmm"):
            p3_clustering(
                modelo_tipo=modelo_p3,
                n_clusters=args.n_clusters,
                covariance_type=args.covariance_type,
                incluir_espacial=args.incluir_espacial,
                stage_of_work=stage,
            )
        elif modelo_p3 == "search":
            ejecutar_busqueda_p3()
        else:
            parser.error(
                "Para P3 utiliza 'baseline', 'reglas', 'kmeans', 'gmm' o 'search'."
            )

    else:
        print(
            f"El entrenamiento de {args.question} "
            "se integrará posteriormente."
        )


if __name__ == "__main__":
    main()
