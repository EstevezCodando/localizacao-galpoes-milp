"""Tabela de validação: modelo vs. ótimos publicados e SSCFLP (fonte única) nas mesmas instâncias.

Uso:  uv run python -m alocacao_capacitada.validation.run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver
from alocacao_capacitada.validation.orlib import (
    load_orlib_instance,
    load_published_optima,
    solve_splittable,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/bronze/orlib"))
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("results/validacao_orlib.csv"))
    args = parser.parse_args()

    published = load_published_optima(args.dir / "capopt.txt")
    rows = []
    for path in sorted(args.dir.glob("cap[0-9]*.txt")):
        name = path.stem
        instance = load_orlib_instance(path)
        split_value, split_status = solve_splittable(instance, args.time_limit)
        row = {
            "instancia": name,
            "m_x_n": f"{instance.n_facilities}x{instance.n_demand}",
            "otimo_publicado_divisivel": published[name],
            "meu_modelo_divisivel": split_value,
            "diferenca_rel": abs(split_value - published[name]) / published[name],
            "status_divisivel": split_status,
        }
        # Fonte única só faz sentido se todo cliente couber em algum centro.
        if instance.demand.max() <= instance.capacity.max():
            milp = MilpSolver().solve(instance, args.time_limit)
            lns = LnsSolver(LocalSearch(NearestFeasibleGreedy()), seed=0, free_size=12)
            lns_solution = lns.solve(instance, 10).solution
            row |= {
                "sscflp_milp": evaluate(instance, milp.solution).total_cost,
                "sscflp_milp_status": milp.status,
                "sscflp_lns_10s": evaluate(instance, lns_solution).total_cost,
                "sscflp_limite_pl": solve_lp_relaxation(instance).lower_bound,
            }
        else:
            row["sscflp_milp_status"] = "N/A: cliente maior que qualquer centro"
        rows.append(row)
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
