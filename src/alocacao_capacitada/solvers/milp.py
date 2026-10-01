"""Modelo exato (MILP) do SSCFLP por branch-and-cut (SCIP via OR-Tools).

Com `warm_start`, a solução de outro solver entra como dica (`SetHint`): o solver
começa com um limite superior bom, poda mais cedo e tende a fechar o gap mais rápido.
"""

from __future__ import annotations

import time

import numpy as np
from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.solvers.base import Solver, SolveResult
from alocacao_capacitada.solvers.model import build_model

_STATUS_LABELS = {
    pywraplp.Solver.OPTIMAL: "OTIMO",
    pywraplp.Solver.FEASIBLE: "VIAVEL_LIMITE_TEMPO",
    pywraplp.Solver.INFEASIBLE: "INVIAVEL",
}


class MilpSolver:
    def __init__(self, backend: str = "SCIP", warm_start: Solver | None = None) -> None:
        self._backend = backend
        self._warm_start = warm_start
        self.name = "milp_scip" if warm_start is None else f"milp_scip+{warm_start.name}"

    def solve(self, instance: Instance, time_limit_s: float) -> SolveResult:
        start = time.perf_counter()
        m, n = instance.n_facilities, instance.n_demand
        model = build_model(instance, self._backend, integer=True)
        model.solver.SetNumThreads(1)

        if self._warm_start is not None:
            available = max(0.0, time_limit_s - (time.perf_counter() - start))
            hint = self._warm_start.solve(instance, min(available, 5.0)).solution.assignment
            opened = set(np.unique(hint[hint != UNASSIGNED]).tolist())
            variables = [*model.y] + [v for row in model.x for v in row]
            values = [1.0 if i in opened else 0.0 for i in range(m)]
            values += [1.0 if hint[j] == i else 0.0 for i in range(m) for j in range(n)]
            model.solver.SetHint(variables, values)

        remaining = max(0.001, time_limit_s - (time.perf_counter() - start))
        model.solver.SetTimeLimit(max(1, int(remaining * 1000)))
        status = model.solver.Solve()
        label = _STATUS_LABELS.get(status, f"STATUS_{status}")
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(f"MILP sem solução viável: {label}")

        assignment = np.full(n, UNASSIGNED, dtype=np.int64)
        for j in range(n):
            for i in range(m):
                if model.x[i][j].solution_value() > 0.5:
                    assignment[j] = i
        return SolveResult(
            Solution(assignment),
            time.perf_counter() - start,
            label,
            lower_bound=float(model.objective.BestBound()),
        )
