"""Efeito do custo fixo real (aluguel de galpão) na decisão: premissa antiga vs. aluguel coletado.

Uso:  uv run python -m alocacao_capacitada.analysis.rent_experiment --sizes 50x15 100x30 200x50
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import GeoNodes, load_geo_nodes
from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.lake.freight import load_freight_model
from alocacao_capacitada.lake.rent import load_rent_table, rent_lookup
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation


def run_case(
    nodes: GeoNodes, config: InstanceConfig, label: str, size: str, time_limit_s: float
) -> tuple[dict[str, object], pd.DataFrame]:
    instance = build_instance(nodes, config)
    solver = LnsSolver(LocalSearch(NearestFeasibleGreedy()), seed=0)
    solution = solver.solve(instance, time_limit_s).solution
    ev = evaluate(instance, solution)
    lp = solve_lp_relaxation(instance)
    opened = solution.open_facilities()
    supply = nodes.supply.head(config.n_facilities)
    states = supply["state"].astype(str).tolist()
    rents = rent_lookup(config.rent_cities, states) if config.rent_cities is not None else {}
    hubs = pd.DataFrame(
        {
            "instancia": size,
            "cenario": label,
            "prefixo_cep": [instance.facility_ids[i] for i in opened],
            "uf": [states[i] for i in opened],
            "aluguel_rs_m2": [rents[states[i]][0] if rents else None for i in opened],
            "aluguel_imputado": [rents[states[i]][1] if rents else None for i in opened],
            "custo_fixo": [float(instance.fixed_cost[i]) for i in opened],
        }
    )
    row = {
        "instancia": size,
        "cenario": label,
        "custo_total": ev.total_cost,
        "custo_fixo": ev.fixed_cost,
        "custo_transporte": ev.transport_cost,
        "pct_fixo": 100 * ev.fixed_cost / ev.total_cost,
        "centros_abertos": ev.n_open,
        "demanda_nao_atendida": ev.unserved_demand,
        "gap_vs_pl": (ev.total_cost - lp.lower_bound) / ev.total_cost,
    }
    return row, hubs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--freight", type=Path, default=Path("data/reference/antt_tabela_a.csv"))
    parser.add_argument(
        "--rent", type=Path, default=Path("data/reference/aluguel_galpao_cidades.csv")
    )
    parser.add_argument("--sizes", nargs="+", default=["50x15", "100x30", "200x50"])
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    nodes = load_geo_nodes(args.raw_dir)
    freight = load_freight_model(args.freight, axles=2, orders_per_vehicle=50)
    cities = load_rent_table(args.rent)
    rows, hub_frames = [], []
    for size in args.sizes:
        n_demand, n_fac = (int(v) for v in size.split("x"))
        base = InstanceConfig(n_demand=n_demand, n_facilities=n_fac, freight=freight)
        cases = {"premissa antiga": base}
        cases["área fixa 5000 m² (inconsistente com o volume)"] = replace(
            base, rent_cities=cities, hub_area_m2=5000.0
        )
        for density in (2.0, 5.0, 10.0, 25.0):
            cases[f"aluguel real, {density:g} pedidos/m²/mês"] = replace(
                base, rent_cities=cities, orders_per_m2_month=density
            )
        for label, config in cases.items():
            row, hubs = run_case(nodes, config, label, size, args.time_limit)
            rows.append(row)
            hub_frames.append(hubs)
            n_open, cost = row["centros_abertos"], row["custo_total"]
            print(f"{size} | {label}: {n_open} centros, custo {cost:.0f}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "aluguel_real.csv", index=False)
    pd.concat(hub_frames).to_csv(args.out_dir / "aluguel_real_centros.csv", index=False)


if __name__ == "__main__":
    main()
