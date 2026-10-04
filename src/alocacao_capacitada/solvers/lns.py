"""Large Neighborhood Search (matheurística): destrói parte da solução e a reotimiza com MILP.

A cada iteração libera-se um subconjunto de nós (vizinhança grande), fixa-se o resto
e resolve-se exatamente o subproblema. Só se aceita melhoria estrita; portanto a
solução nunca piora e permanece viável (a capacidade residual é respeitada pelo MILP).

Estratégias de destruição (alternadas):
  * relacionada: um nó semente e os nós de perfil de custo mais parecido (vizinhança geográfica);
  * por centro: todos os nós de 1 ou 2 centros abertos (permite fechar/reabrir centros).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.solvers.base import Solver, SolveResult

_EPS = 1e-6


class LnsSolver:
    def __init__(
        self,
        initial: Solver,
        seed: int = 0,
        free_size: int = 50,
        sub_time_limit_s: float = 2.0,
        max_free_size: int | None = None,
        patience: int = 5,
        max_iterations: int | None = None,
        on_progress: Callable[[int, float, float], None] | None = None,
    ) -> None:
        """`max_free_size` ativa a vizinhança adaptativa: após `patience` iterações sem melhora
        o tamanho cresce 25% (até o máximo); ao melhorar, recua à metade da distância até o piso.
        Vizinhanças pequenas não enxergam trocas que exigem mexer em muitos nós ao mesmo tempo,
        o que pesa em instâncias de capacidade apertada."""
        self._initial = initial
        self._seed = seed
        self._free_size = free_size
        self._max_free_size = max_free_size or free_size
        self._patience = patience
        self._max_iterations = max_iterations
        self._on_progress = on_progress
        self._sub_time_limit_s = sub_time_limit_s
        self.name = f"lns({initial.name})"

    def solve(self, instance: Instance, time_limit_s: float) -> SolveResult:
        start = time.perf_counter()
        deadline = start + time_limit_s
        rng = np.random.default_rng(self._seed)
        current = self._initial.solve(instance, min(time_limit_s, 5.0)).solution.assignment.copy()
        best_cost = evaluate(instance, Solution(current)).total_cost
        if self._on_progress is not None:
            self._on_progress(0, time.perf_counter() - start, best_cost)
        iteration = 0
        size, failures = self._free_size, 0
        while time.perf_counter() < deadline and (
            self._max_iterations is None or iteration < self._max_iterations
        ):
            free = self._destroy(instance, current, rng, iteration, size)
            iteration += 1
            budget = min(self._sub_time_limit_s, deadline - time.perf_counter())
            if budget <= 0:
                break
            candidate = _resolve(instance, current, free, budget)
            cost = None if candidate is None else evaluate(instance, Solution(candidate)).total_cost
            if candidate is not None and cost is not None and cost < best_cost - _EPS:
                current, best_cost = candidate, cost
                size, failures = (size + self._free_size) // 2, 0
            else:
                failures += 1
                if failures >= self._patience:
                    size, failures = min(int(size * 1.25) + 1, self._max_free_size), 0
            if self._on_progress is not None:
                self._on_progress(iteration, time.perf_counter() - start, best_cost)
        stopped = self._max_iterations is not None and iteration >= self._max_iterations
        status = "LIMITE_ITERACOES" if stopped else "LIMITE_TEMPO"
        return SolveResult(Solution(current), time.perf_counter() - start, status)

    def _destroy(
        self,
        inst: Instance,
        current: np.ndarray,
        rng: np.random.Generator,
        iteration: int,
        size: int,
    ) -> np.ndarray:
        k = min(size, inst.n_demand)
        opened = np.unique(current[current != UNASSIGNED])
        if iteration % 2 == 1 and opened.size:
            chosen = rng.choice(opened, size=min(2, opened.size), replace=False)
            free = np.flatnonzero(np.isin(current, chosen))
            if free.size > k:
                free = rng.choice(free, size=k, replace=False)
            return np.asarray(free)
        seed_node = int(rng.integers(inst.n_demand))
        profile = inst.unit_cost[:, [seed_node]]
        closeness = np.abs(inst.unit_cost - profile).sum(axis=0)
        return np.argsort(closeness, kind="stable")[:k]


def _resolve(
    inst: Instance, current: np.ndarray, free: np.ndarray, time_limit_s: float
) -> np.ndarray | None:
    """Reotimiza os nós `free` mantendo os demais fixos; devolve a atribuição completa."""
    started = time.perf_counter()
    m = inst.n_facilities
    fixed_mask = np.ones(inst.n_demand, dtype=bool)
    fixed_mask[free] = False
    fixed_nodes = np.flatnonzero(fixed_mask & (current != UNASSIGNED))
    residual = inst.capacity.astype(float).copy()
    used = np.zeros(m, dtype=bool)
    for j in fixed_nodes:
        residual[current[j]] -= inst.demand[j]
        used[current[j]] = True

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP indisponível")
    solver.SetNumThreads(1)
    p = inst.unserved_penalty
    x = {(i, j): solver.BoolVar("") for i in range(m) for j in free}
    # Centros já usados por nós fixos estão abertos (custo fixo já pago): sem variável y.
    y = {i: solver.BoolVar("") for i in range(m) if not used[i]}

    for j in free:
        solver.Add(sum(x[i, j] for i in range(m)) <= 1)
    for i in range(m):
        load = sum(float(inst.demand[j]) * x[i, j] for j in free)
        solver.Add(load <= max(residual[i], 0.0) * (1 if used[i] else y[i]))
        if not used[i]:
            for j in free:
                solver.Add(x[i, j] <= y[i])

    objective = solver.Objective()
    for i, var in y.items():
        objective.SetCoefficient(var, float(inst.fixed_cost[i]))
    for (i, j), var in x.items():
        objective.SetCoefficient(var, float(inst.demand[j]) * (float(inst.unit_cost[i, j]) - p))
    objective.SetMinimization()

    remaining = time_limit_s - (time.perf_counter() - started)
    if remaining <= 0:
        return None
    solver.SetTimeLimit(max(int(remaining * 1000), 1))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None
    out = current.copy()
    out[free] = UNASSIGNED
    for (i, j), var in x.items():
        if var.solution_value() > 0.5:
            out[j] = i
    return out
