"""Exemplo didático: 3 centros candidatos e 5 regiões, pequeno o bastante para conferir à mão.

Mostra, na mesma instância, o que cada método devolve e por que eles diferem:
guloso, busca local, MILP (ótimo), relaxação linear (limite inferior) e uma enumeração
completa (4^5 = 1024 atribuições) que confirma o ótimo sem usar solver.

Uso:  uv run python examples/exemplo_didatico.py
"""

from __future__ import annotations

from itertools import product

import numpy as np

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver

FACILITIES = ("A", "B", "C")
REGIONS = ("R1", "R2", "R3", "R4", "R5")


def build_instance() -> Instance:
    return Instance(
        facility_ids=FACILITIES,
        demand_ids=REGIONS,
        demand=np.array([4.0, 8.0, 8.0, 4.0, 2.0]),  # pedidos por região
        capacity=np.array([14.0, 14.0, 14.0]),  # pedidos que cada centro aguenta
        fixed_cost=np.array([33.0, 35.0, 38.0]),  # custo de abrir cada centro
        unit_cost=np.array(
            [  # custo por pedido de levar do centro (linha) à região (coluna)
                [6.0, 6.0, 8.0, 8.0, 8.0],
                [7.0, 6.0, 8.0, 1.0, 1.0],
                [7.0, 4.0, 6.0, 4.0, 8.0],
            ]
        ),
        unserved_penalty=50.0,  # custo por pedido que fica sem atendimento
    )


def describe(inst: Instance, solution: Solution) -> str:
    parts = []
    for centre in solution.open_facilities():
        regions = [REGIONS[j] for j in range(inst.n_demand) if solution.assignment[j] == centre]
        parts.append(f"{FACILITIES[centre]}→{'+'.join(regions)}")
    return "; ".join(parts) if parts else "nenhum centro"


def brute_force(inst: Instance) -> tuple[float, Solution]:
    """Enumera todas as atribuições (cada região vai a A, B, C ou fica sem atendimento)."""
    best_cost, best = float("inf"), None
    for choice in product([UNASSIGNED, 0, 1, 2], repeat=inst.n_demand):
        solution = Solution(np.array(choice))
        ev = evaluate(inst, solution)
        if ev.feasible and ev.total_cost < best_cost:
            best_cost, best = ev.total_cost, solution
    assert best is not None
    return best_cost, best


def main() -> None:
    inst = build_instance()
    print(f"Demanda total: {inst.total_demand:.0f} pedidos; capacidade por centro: 14\n")

    rows = []
    for name, solver in (
        ("Guloso (mais próximo)", NearestFeasibleGreedy()),
        ("Busca local", LocalSearch(NearestFeasibleGreedy())),
        ("MILP (ótimo)", MilpSolver()),
    ):
        solution = solver.solve(inst, 10).solution
        ev = evaluate(inst, solution)
        rows.append(
            (name, ev.total_cost, ev.fixed_cost, ev.transport_cost, describe(inst, solution))
        )
    for name, total, fixed, transport, text in rows:
        print(
            f"{name:24s} custo {total:6.1f}  (fixo {fixed:5.1f} + frete {transport:5.1f})  {text}"
        )

    cost, best = brute_force(inst)
    print(f"{'Enumeração (1024 casos)':24s} custo {cost:6.1f}  {describe(inst, best)}")

    lp = solve_lp_relaxation(inst)
    print(f"\nRelaxação linear: limite inferior = {lp.lower_bound:.2f}")
    print(
        "  abertura fracionária y =",
        {k: float(v) for k, v in zip(FACILITIES, lp.y_fraction.round(3), strict=True)},
    )
    print(
        "  preços-sombra de capacidade =",
        {k: float(v) for k, v in zip(FACILITIES, lp.capacity_dual.round(3), strict=True)},
    )
    gap = (cost - lp.lower_bound) / cost
    print(f"  gap do limite em relação ao ótimo = {gap:.1%}")


if __name__ == "__main__":
    main()
