"""Harness de comparação: mesmos dados, mesma função de avaliação, vários métodos.

Uso:  uv run alocacao-capacitada --raw-dir data/raw --sizes 20x5 50x15 100x30 --time-limit 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.base import Solver, excess_lb, gap_ub
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.milp import MilpSolver


def default_solvers() -> list[Solver]:
    return [NearestFeasibleGreedy(), LocalSearch(NearestFeasibleGreedy()), MilpSolver()]


def run_benchmark(
    instance: Instance, label: str, solvers: list[Solver], time_limit_s: float
) -> pd.DataFrame:
    """Executa cada solver e mede custo, viabilidade, tempo, limite inferior e gap."""
    rows = []
    for solver in solvers:
        result = solver.solve(instance, time_limit_s)
        ev = evaluate(instance, result.solution)
        rows.append(
            {
                "instancia": label,
                "metodo": solver.name,
                "status": result.status,
                "viavel": ev.feasible,
                "custo_total": ev.total_cost,
                "custo_fixo": ev.fixed_cost,
                "custo_transporte": ev.transport_cost,
                "demanda_nao_atendida": ev.unserved_demand,
                "centros_abertos": ev.n_open,
                "tempo_s": result.runtime_s,
                "limite_inferior": result.lower_bound,
                "gap_ub": gap_ub(ev.total_cost, result.lower_bound),
                "excesso_lb": excess_lb(ev.total_cost, result.lower_bound),
            }
        )
    return pd.DataFrame(rows)


def _parse_size(text: str) -> tuple[int, int]:
    n_demand, n_fac = text.lower().split("x")
    return int(n_demand), int(n_fac)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sizes", nargs="+", default=["20x5", "50x15", "100x30"])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()

    nodes = load_geo_nodes(args.raw_dir)
    frames = []
    for size in args.sizes:
        n_demand, n_fac = _parse_size(size)
        instance = build_instance(nodes, InstanceConfig(n_demand=n_demand, n_facilities=n_fac))
        frames.append(run_benchmark(instance, size, default_solvers(), args.time_limit))
    table = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
