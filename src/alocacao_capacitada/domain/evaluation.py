"""Avaliação independente de soluções: custo, serviço e viabilidade.

Todo solver é medido pela MESMA função, evitando que um método "se auto-avalie".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution

_TOL = 1e-6


@dataclass(frozen=True)
class Evaluation:
    fixed_cost: float
    transport_cost: float
    penalty_cost: float
    unserved_demand: float
    n_open: int
    max_utilization: float
    violations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_cost(self) -> float:
        return self.fixed_cost + self.transport_cost + self.penalty_cost

    @property
    def feasible(self) -> bool:
        return not self.violations


def evaluate(instance: Instance, solution: Solution) -> Evaluation:
    a = solution.assignment
    violations: list[str] = []
    if a.shape != (instance.n_demand,):
        raise ValueError("assignment com tamanho incompatível com a instância")
    if ((a < UNASSIGNED) | (a >= instance.n_facilities)).any():
        violations.append("índice de centro inválido")
        raise ValueError("; ".join(violations))

    served = a != UNASSIGNED
    nodes = np.flatnonzero(served)
    facilities = a[served]

    load = np.bincount(facilities, weights=instance.demand[nodes], minlength=instance.n_facilities)
    over = load > instance.capacity + _TOL
    for i in np.flatnonzero(over):
        violations.append(
            f"capacidade excedida no centro {instance.facility_ids[i]}: "
            f"{load[i]:.1f} > {instance.capacity[i]:.1f}"
        )

    opened = np.unique(facilities)
    transport = float((instance.demand[nodes] * instance.unit_cost[facilities, nodes]).sum())
    unserved = float(instance.demand[~served].sum())
    util = load[opened] / np.maximum(instance.capacity[opened], _TOL)
    costs = (
        float(instance.fixed_cost[opened].sum()),
        transport,
        unserved * instance.unserved_penalty,
    )
    if not np.isfinite(costs).all():
        raise ValueError("custo não finito: confira os dados da instância")
    return Evaluation(
        fixed_cost=float(instance.fixed_cost[opened].sum()),
        transport_cost=transport,
        penalty_cost=unserved * instance.unserved_penalty,
        unserved_demand=unserved,
        n_open=int(opened.size),
        max_utilization=float(util.max()) if util.size else 0.0,
        violations=tuple(violations),
    )
