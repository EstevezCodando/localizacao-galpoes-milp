"""Busca local sobre uma solução inicial: realocar, trocar e fechar centros.

Vizinhanças (todas preservam a viabilidade de capacidade):
  * relocate: move um nó para outro centro (ou para "sem atendimento");
  * swap: troca dois nós de centros diferentes;
  * close: esvazia um centro, realocando cada nó ao melhor centro já aberto.
Aceita apenas melhorias estritas (descida), até ótimo local ou limite de tempo.
"""

from __future__ import annotations

import time

import numpy as np

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.solvers.base import Solver, SolveResult

_EPS = 1e-9


class _State:
    """Estado mutável com carga e contagem por centro (deltas em O(1))."""

    def __init__(self, instance: Instance, assignment: np.ndarray) -> None:
        self.inst = instance
        self.a = assignment.copy()
        self.load = np.zeros(instance.n_facilities)
        self.count = np.zeros(instance.n_facilities, dtype=np.int64)
        for j, i in enumerate(self.a):
            if i != UNASSIGNED:
                self.load[i] += instance.demand[j]
                self.count[i] += 1

    def clone(self) -> _State:
        return _State(self.inst, self.a)

    def node_cost(self, j: int, i: int) -> float:
        d = self.inst.demand[j]
        if i == UNASSIGNED:
            return float(d * self.inst.unserved_penalty)
        return float(d * self.inst.unit_cost[i, j])

    def fits(self, j: int, i: int) -> bool:
        return bool(self.load[i] + self.inst.demand[j] <= self.inst.capacity[i])

    def move(self, j: int, new: int) -> None:
        old = int(self.a[j])
        d = self.inst.demand[j]
        if old != UNASSIGNED:
            self.load[old] -= d
            self.count[old] -= 1
        if new != UNASSIGNED:
            self.load[new] += d
            self.count[new] += 1
        self.a[j] = new

    def relocate_delta(self, j: int, new: int) -> float:
        old = int(self.a[j])
        delta = self.node_cost(j, new) - self.node_cost(j, old)
        if old != UNASSIGNED and self.count[old] == 1:
            delta -= float(self.inst.fixed_cost[old])
        if new != UNASSIGNED and self.count[new] == 0:
            delta += float(self.inst.fixed_cost[new])
        return delta


class LocalSearch:
    """Melhora a solução de um solver construtivo (composição, não herança)."""

    def __init__(self, initial: Solver) -> None:
        self._initial = initial
        self.name = f"busca_local({initial.name})"

    def solve(self, instance: Instance, time_limit_s: float) -> SolveResult:
        start = time.perf_counter()
        deadline = start + time_limit_s
        state = _State(instance, self._initial.solve(instance, 0.0).solution.assignment)
        status = "OTIMO_LOCAL"
        while True:
            if time.perf_counter() > deadline:
                status = "LIMITE_TEMPO"
                break
            improved = (
                self._relocate_pass(state, deadline)
                or self._swap_pass(state, deadline)
                or self._close_pass(state, deadline)
            )
            if not improved:
                break
        return SolveResult(Solution(state.a.copy()), time.perf_counter() - start, status)

    @staticmethod
    def _relocate_pass(s: _State, deadline: float) -> bool:
        inst, any_gain = s.inst, False
        for j in range(inst.n_demand):
            if time.perf_counter() > deadline:
                break
            best_delta, best_target = -_EPS, None
            for i in [*range(inst.n_facilities), UNASSIGNED]:
                if i == s.a[j] or (i != UNASSIGNED and not s.fits(j, i)):
                    continue
                delta = s.relocate_delta(j, i)
                if delta < best_delta:
                    best_delta, best_target = delta, i
            if best_target is not None:
                s.move(j, best_target)
                any_gain = True
        return any_gain

    @staticmethod
    def _swap_pass(s: _State, deadline: float) -> bool:
        inst, any_gain = s.inst, False
        served = np.flatnonzero(s.a != UNASSIGNED)
        for idx, node_j in enumerate(served):
            j = int(node_j)
            if time.perf_counter() > deadline:
                break
            for node_k in served[idx + 1 :]:
                k = int(node_k)
                a, b = int(s.a[j]), int(s.a[k])
                if a == b:
                    continue
                dj, dk = inst.demand[j], inst.demand[k]
                if s.load[a] - dj + dk > inst.capacity[a] or s.load[b] - dk + dj > inst.capacity[b]:
                    continue
                # A troca não altera a contagem por centro: custo fixo inalterado.
                delta = (
                    s.node_cost(j, b) + s.node_cost(k, a) - s.node_cost(j, a) - s.node_cost(k, b)
                )
                if delta < -_EPS:
                    s.move(j, b)
                    s.move(k, a)
                    any_gain = True
                    break
        return any_gain

    @staticmethod
    def _close_pass(s: _State, deadline: float) -> bool:
        inst = s.inst
        for i in np.flatnonzero(s.count > 0):
            if time.perf_counter() > deadline:
                break
            trial = s.clone()
            gain = float(inst.fixed_cost[i])
            nodes = np.flatnonzero(s.a == i)
            for node in nodes[np.argsort(-inst.demand[nodes], kind="stable")]:
                j = int(node)
                cost_old = trial.node_cost(j, int(i))
                trial.move(j, UNASSIGNED)
                options = [(trial.node_cost(j, UNASSIGNED), UNASSIGNED)]
                options += [
                    (trial.node_cost(j, int(t)), int(t))
                    for t in np.flatnonzero(trial.count > 0)
                    if t != i and trial.fits(j, int(t))
                ]
                cost_new, target = min(options)
                trial.move(j, target)
                gain += cost_old - cost_new
            if gain > _EPS:
                s.a, s.load, s.count = trial.a, trial.load, trial.count
                return True
        return False
