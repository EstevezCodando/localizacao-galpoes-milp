"""Compara os métodos com os ótimos publicados de Holmberg (71 instâncias, fonte única).

Uso:  uv run python -m alocacao_capacitada.validation.run_holmberg --milp-time 20 --lns-time 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.base import Solver
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver
from alocacao_capacitada.validation.holmberg import load_holmberg_instance, load_holmberg_optima


def _measure(instance: Instance, solver: Solver, limit: float, optimum: float) -> dict[str, object]:
    result = solver.solve(instance, limit)
    ev = evaluate(instance, result.solution)
    return {
        "custo": ev.total_cost,
        "viavel": ev.feasible and ev.unserved_demand == 0,
        "gap_vs_otimo": (ev.total_cost - optimum) / optimum,
        "tempo_s": result.runtime_s,
        "status": result.status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/bronze/holmberg"))
    parser.add_argument("--milp-time", type=float, default=20.0)
    parser.add_argument("--lns-time", type=float, default=10.0)
    parser.add_argument("--free-size", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("results/validacao_holmberg.csv"))
    args = parser.parse_args()

    optima = load_holmberg_optima(args.dir / "Holmberg_Solution_Values.txt")
    rows = []
    for _, ref in optima.iterrows():
        instance = load_holmberg_instance(args.dir / "instances" / str(ref["name"]))
        opt = float(ref["optimal"])
        start = LocalSearch(NearestFeasibleGreedy())
        methods: list[tuple[str, Solver, float]] = [
            ("guloso", NearestFeasibleGreedy(), 1.0),
            ("busca_local", start, 10.0),
            ("lns", LnsSolver(start, seed=0, free_size=args.free_size), args.lns_time),
            ("milp", MilpSolver(), args.milp_time),
        ]
        row: dict[str, object] = {"instancia": ref["name"], "m": ref["m"], "n": ref["n"]}
        row["otimo_publicado"] = opt
        row["gap_pl_vs_otimo"] = (opt - solve_lp_relaxation(instance).lower_bound) / opt
        for label, solver, limit in methods:
            for key, value in _measure(instance, solver, limit, opt).items():
                row[f"{label}_{key}"] = value
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)  # salva a cada instância
        lns_gap, milp_gap = row["lns_gap_vs_otimo"], row["milp_gap_vs_otimo"]
        print(f"{row['instancia']}: lns {lns_gap:.4%}  milp {milp_gap:.4%}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
