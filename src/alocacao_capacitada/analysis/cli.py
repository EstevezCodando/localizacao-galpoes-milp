"""Gera as tabelas de sensibilidade e de demanda incerta.

Uso:  uv run python -m alocacao_capacitada.analysis.cli --size 50x15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.analysis.sensitivity import run_sensitivity
from alocacao_capacitada.analysis.stochastic import assess
from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.lake.freight import load_freight_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--freight", type=Path, default=Path("data/reference/antt_tabela_a.csv"))
    parser.add_argument("--size", default="50x15")
    parser.add_argument("--stochastic-size", default="30x10")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--skip-stochastic", action="store_true")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    nodes = load_geo_nodes(args.raw_dir)
    freight = load_freight_model(args.freight, axles=2, orders_per_vehicle=50)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_demand, n_fac = (int(v) for v in args.size.split("x"))
    base = InstanceConfig(n_demand=n_demand, n_facilities=n_fac, freight=freight)
    sens = run_sensitivity(nodes, base, max(args.time_limit, 60.0), args.iterations)
    sens.to_csv(args.out_dir / "sensibilidade.csv", index=False)
    print(sens.round(3).to_string(index=False))

    if args.skip_stochastic:
        return
    n_demand, n_fac = (int(v) for v in args.stochastic_size.split("x"))
    instance = build_instance(
        nodes, InstanceConfig(n_demand=n_demand, n_facilities=n_fac, freight=freight)
    )
    rows = []
    for sigma in args.sigmas:
        rep = assess(instance, sigma=sigma, time_limit_s=args.time_limit)
        rows.append(
            {
                "sigma": sigma,
                "RP": rep.rp,
                "EEV": rep.eev,
                "WS": rep.ws,
                "VSS": rep.vss,
                "VSS_pct": 100 * rep.vss / rep.eev,
                "EVPI": rep.evpi,
                "EVPI_pct": 100 * rep.evpi / rep.rp,
                "centros_estocastico": rep.n_open_stochastic,
                "centros_deterministico": rep.n_open_deterministic,
            }
        )
    stoch = pd.DataFrame(rows)
    stoch.to_csv(args.out_dir / "demanda_incerta.csv", index=False)
    print(stoch.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
