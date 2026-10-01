"""Construção única do modelo do SSCFLP, reutilizada pelo MILP e pela relaxação linear.

min  sum_i f_i y_i + sum_ij d_j c_ij x_ij + p sum_j d_j (1 - sum_i x_ij)
s.a. sum_i x_ij <= 1                  para todo j   (single-source; pode ficar sem atendimento)
     sum_j d_j x_ij <= Q_i y_i        para todo i   (capacidade; liga x a y)
     x_ij <= y_i                      para todo i,j (reforço: relaxação linear mais justa)
     x, y binários  (no LP: 0 <= x, y <= 1)
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.instance import Instance


@dataclass
class SscflpModel:
    solver: pywraplp.Solver
    y: list[pywraplp.Variable]
    x: list[list[pywraplp.Variable]]
    capacity: list[pywraplp.Constraint]
    objective: pywraplp.Objective


def build_model(
    instance: Instance, backend: str, integer: bool, split_demand: bool = False
) -> SscflpModel:
    """`split_demand=True` deixa x contínuo (demanda divisível): variante clássica do CFLP."""
    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        raise RuntimeError(f"backend {backend} indisponível")
    m, n = instance.n_facilities, instance.n_demand

    def binary(name: str) -> pywraplp.Variable:
        return solver.BoolVar(name) if integer else solver.NumVar(0.0, 1.0, name)

    def share(name: str) -> pywraplp.Variable:
        return binary(name) if not split_demand else solver.NumVar(0.0, 1.0, name)

    y = [binary(f"y{i}") for i in range(m)]
    x = [[share(f"x{i}_{j}") for j in range(n)] for i in range(m)]

    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(m)) <= 1)
    capacity = []
    for i in range(m):
        capacity.append(
            solver.Add(
                sum(float(instance.demand[j]) * x[i][j] for j in range(n))
                <= float(instance.capacity[i]) * y[i]
            )
        )
        for j in range(n):
            solver.Add(x[i][j] <= y[i])

    p = instance.unserved_penalty
    objective = solver.Objective()
    objective.SetOffset(p * instance.total_demand)
    for i in range(m):
        objective.SetCoefficient(y[i], float(instance.fixed_cost[i]))
        for j in range(n):
            dj = float(instance.demand[j])
            objective.SetCoefficient(x[i][j], dj * (float(instance.unit_cost[i, j]) - p))
    objective.SetMinimization()
    return SscflpModel(solver, y, x, capacity, objective)
