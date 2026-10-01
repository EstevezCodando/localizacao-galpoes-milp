"""Treina e avalia os modelos de demanda com validação espacial.

Uso:  uv run python -m alocacao_capacitada.ml.run_demand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.ml.demand_model import (
    cross_predict,
    decile_calibration,
    evaluate_predictions,
    permutation_importance,
)
from alocacao_capacitada.ml.features import load_modeling_table
from alocacao_capacitada.territory.osm_roads import UF_FOCO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--municipios", type=Path, default=Path("data/reference/municipios.parquet")
    )
    parser.add_argument(
        "--painel", type=Path, default=Path("data/reference/olist_painel_municipal.parquet")
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--foco",
        action="store_true",
        help="só S, SE e CO, com variáveis do OpenStreetMap (saída em results/foco)",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("results") / ("foco" if args.foco else "")

    table = load_modeling_table(
        args.municipios,
        args.painel,
        focus=UF_FOCO if args.foco else None,
        osm_dir=args.municipios.parent if args.foco else None,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for scheme in ("uf", "regiao"):
        pred = cross_predict(table, scheme, with_osm=args.foco)
        pred.to_parquet(args.out / f"ml_demanda_previsoes_{scheme}.parquet", index=False)
        metrics = evaluate_predictions(pred).assign(validacao=scheme)
        frames.append(metrics)
        if scheme == "uf":
            decile_calibration(pred, "gbm_completo").to_csv(
                args.out / "ml_demanda_calibracao_gbm_uf.csv", index=False
            )
            decile_calibration(pred, "pop_proporcional").to_csv(
                args.out / "ml_demanda_calibracao_base_uf.csv", index=False
            )
    permutation_importance(table).to_csv(args.out / "ml_demanda_importancia.csv", index=False)
    if args.foco:
        permutation_importance(table, model_name="gbm_osm").to_csv(
            args.out / "ml_demanda_importancia_osm.csv", index=False
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.out / "ml_demanda_metricas.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
