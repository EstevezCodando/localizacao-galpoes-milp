"""Validação externa: instâncias clássicas da OR-Library (Beasley) com ótimos publicados.

Os valores de `capopt` são da variante de DEMANDA DIVISÍVEL (CFLP). Por isso:
  * validamos o modelo com `split_demand=True` contra o ótimo publicado (igualdade);
  * a variante de fonte única (SSCFLP) deve custar >= esse valor (sanidade adicional).

Formato: "m n"; m linhas "capacidade custo_fixo"; n blocos "demanda" + m custos de
atribuição (custo TOTAL de atender a demanda inteira do cliente por aquele centro).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.model import build_model

_BIG_PENALTY = 1e6  # força atender toda a demanda (o CFLP clássico não admite não atender)


def load_orlib_instance(path: Path) -> Instance:
    tokens = path.read_text(encoding="utf-8").split()
    m, n = int(tokens[0]), int(tokens[1])
    pos = 2
    capacity = np.zeros(m)
    fixed = np.zeros(m)
    for i in range(m):
        capacity[i], fixed[i] = float(tokens[pos]), float(tokens[pos + 1])
        pos += 2
    demand = np.zeros(n)
    total_cost = np.zeros((m, n))
    for j in range(n):
        demand[j] = float(tokens[pos])
        pos += 1
        total_cost[:, j] = [float(v) for v in tokens[pos : pos + m]]
        pos += m
    if pos != len(tokens):
        raise ValueError(f"{path.name}: {len(tokens) - pos} valores sobrando; formato inesperado")
    unit_cost = total_cost / np.maximum(demand[None, :], 1e-12)
    return Instance(
        facility_ids=tuple(f"W{i}" for i in range(m)),
        demand_ids=tuple(f"C{j}" for j in range(n)),
        demand=demand,
        capacity=capacity,
        fixed_cost=fixed,
        unit_cost=unit_cost,
        unserved_penalty=_BIG_PENALTY,
    )


def load_published_optima(path: Path) -> dict[str, float]:
    optima: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2:
            optima[parts[0]] = float(parts[1])
    return optima


def solve_splittable(instance: Instance, time_limit_s: float) -> tuple[float, str]:
    """Ótimo do CFLP clássico (demanda divisível). Devolve (valor, status)."""
    model = build_model(instance, "SCIP", integer=True, split_demand=True)
    model.solver.SetTimeLimit(int(time_limit_s * 1000))
    status = model.solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"sem solução (status {status})")
    label = "OTIMO" if status == pywraplp.Solver.OPTIMAL else "VIAVEL_LIMITE_TEMPO"
    return float(model.objective.Value()), label
