"""Demanda incerta com protocolo corrigido: réplicas, choque comum, folga de capacidade e IC.

Uso:  uv run python -m alocacao_capacitada.analysis.run_stochastic --slack 1.1 --rho 0.5
Cada execução escreve results/incerteza_<slack>_<rho>.csv (uma linha por réplica e σ).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from alocacao_capacitada.analysis.stochastic import assess
from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.lake.freight import load_freight_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--freight", type=Path, default=Path("data/reference/antt_tabela_a.csv"))
    parser.add_argument("--size", default="30x10")
    parser.add_argument("--slack", type=float, default=1.5)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 0.8])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    nodes = load_geo_nodes(args.raw_dir)
    freight = load_freight_model(args.freight, axles=2, orders_per_vehicle=50)
    n_demand, n_fac = (int(v) for v in args.size.split("x"))
    config = InstanceConfig(n_demand=n_demand, n_facilities=n_fac, freight=freight)
    instance = build_instance(nodes, replace(config, capacity_slack=args.slack))
    rows = []
    for sigma in args.sigmas:
        for rep in range(args.reps):
            r = assess(
                instance,
                sigma,
                args.n_train,
                args.n_test,
                args.time_limit,
                seed=1000 * rep + 7,
                rho=args.rho,
            )
            lo, hi = r.vss_ci95
            rows.append(
                {
                    "folga": args.slack,
                    "rho": args.rho,
                    "sigma": sigma,
                    "replica": rep,
                    "RP": r.rp,
                    "EEV": r.eev,
                    "WS": r.ws,
                    "VSS": r.vss,
                    "VSS_ic95_lo": lo,
                    "VSS_ic95_hi": hi,
                    "VSS_pct": 100 * r.vss / r.eev,
                    "EVPI": r.evpi,
                    "EVPI_pct": 100 * r.evpi / r.rp,
                    "centros_estocastico": r.n_open_stochastic,
                    "centros_deterministico": r.n_open_deterministic,
                    "mesma_politica": r.same_policy,
                    "gap_max_solver": r.max_solver_gap,
                }
            )
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(
                args.out / f"incerteza_folga{args.slack}_rho{args.rho}.csv", index=False
            )


if __name__ == "__main__":
    main()
