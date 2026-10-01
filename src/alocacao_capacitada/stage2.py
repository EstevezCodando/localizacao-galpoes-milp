"""Etapa 2: LNS com várias sementes, MILP frio vs. com warm start e limite do PL.

Uso:  uv run python -m alocacao_capacitada.stage2 --sizes 100x30 200x50 --time-limit 60 --seeds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.benchmark import run_benchmark
from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.lake.freight import load_freight_model
from alocacao_capacitada.solvers.base import Solver
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver


def stage2_solvers(seeds: int) -> list[Solver]:
    start = LocalSearch(NearestFeasibleGreedy())
    lns = [LnsSolver(start, seed=s) for s in range(seeds)]
    return [NearestFeasibleGreedy(), start, *lns, MilpSolver(), MilpSolver(warm_start=start)]


def run_instance(instance: Instance, label: str, time_limit_s: float, seeds: int) -> pd.DataFrame:
    table = run_benchmark(instance, label, stage2_solvers(seeds), time_limit_s)
    # Nomes de LNS iguais para todas as sementes: a coluna `semente` os distingue.
    seed_idx = iter(range(seeds))
    table["semente"] = [next(seed_idx) if m.startswith("lns") else None for m in table["metodo"]]
    lp = solve_lp_relaxation(instance)
    table["limite_pl"] = lp.lower_bound
    best = table["custo_total"].min()
    table["gap_vs_melhor"] = (table["custo_total"] - best) / best
    table["gap_ub_vs_pl"] = (table["custo_total"] - lp.lower_bound) / table["custo_total"]
    table["excesso_lb_vs_pl"] = (table["custo_total"] - lp.lower_bound) / lp.lower_bound
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--freight", type=Path, default=Path("data/reference/antt_tabela_a.csv"))
    parser.add_argument("--sizes", nargs="+", default=["100x30"])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/stage2.csv"))
    args = parser.parse_args()

    nodes = load_geo_nodes(args.raw_dir)
    freight = load_freight_model(args.freight, axles=2, orders_per_vehicle=50)
    frames = []
    for size in args.sizes:
        n_demand, n_fac = (int(v) for v in size.lower().split("x"))
        config = InstanceConfig(n_demand=n_demand, n_facilities=n_fac, freight=freight)
        instance = build_instance(nodes, config)
        frames.append(run_instance(instance, size, args.time_limit, args.seeds))
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    cols = ["instancia", "metodo", "semente", "status", "custo_total", "centros_abertos"]
    cols += ["tempo_s", "gap_vs_melhor", "gap_ub_vs_pl", "excesso_lb_vs_pl"]
    print(table[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
