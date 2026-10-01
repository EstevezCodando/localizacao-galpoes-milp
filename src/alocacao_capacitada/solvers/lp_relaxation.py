"""Relaxação linear (PL) do SSCFLP: limite inferior e preços-sombra de capacidade.

O valor do PL é um limite inferior válido para o MILP.

Atenção ao dual da restrição de capacidade `sum_j d_j x_ij - Q_i y_i <= 0`: Q_i é o
coeficiente de `y_i`, não o lado direito. Logo o dual (`capacity_dual`) NÃO é o valor de
uma unidade extra de capacidade. Pelo teorema do envelope, a sensibilidade local do custo
a Q_i é `dual * y_i` (`capacity_marginal_value`), conferida por diferença finita em
`tests/test_stage2.py`. Ambos são da relaxação, não do problema inteiro, e valem apenas
para perturbações pequenas (base não degenerada).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.model import build_model


@dataclass(frozen=True)
class LpResult:
    lower_bound: float
    capacity_dual: np.ndarray  # dual da linha de capacidade, shape (m,); NÃO é d(custo)/dQ
    capacity_marginal_value: np.ndarray  # d(custo)/dQ_i = dual * y (relaxação), shape (m,)
    y_fraction: np.ndarray  # grau de abertura fracionário de cada centro


def solve_lp_relaxation(instance: Instance) -> LpResult:
    model = build_model(instance, "GLOP", integer=False)
    status = model.solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"PL não ótimo (status {status})")
    dual = np.array([c.dual_value() for c in model.capacity])
    y = np.array([v.solution_value() for v in model.y])
    return LpResult(
        lower_bound=float(model.objective.Value()),
        capacity_dual=dual,
        capacity_marginal_value=dual * y,
        y_fraction=y,
    )
