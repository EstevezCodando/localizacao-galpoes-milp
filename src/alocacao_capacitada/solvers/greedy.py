"""Baseline: cada nó vai ao centro viável de menor custo de transporte.

Representa a regra operacional simples ("centro mais próximo com capacidade")
contra a qual os métodos mais sofisticados precisam ser comparados.
Custo fixo é ignorado na escolha: todo centro é tratado como já aberto.
"""

from __future__ import annotations

import time

import numpy as np

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.solvers.base import SolveResult


class NearestFeasibleGreedy:
    name = "guloso_mais_proximo"

    def solve(self, instance: Instance, time_limit_s: float = 0.0) -> SolveResult:
        start = time.perf_counter()
        residual = instance.capacity.astype(float).copy()
        assignment = np.full(instance.n_demand, UNASSIGNED, dtype=np.int64)
        # Nós maiores primeiro: são os mais difíceis de encaixar (first-fit decreasing).
        for j in np.argsort(-instance.demand, kind="stable"):
            order = np.argsort(instance.unit_cost[:, j], kind="stable")
            for i in order:
                if instance.demand[j] <= residual[i]:
                    assignment[j] = i
                    residual[i] -= instance.demand[j]
                    break
        return SolveResult(Solution(assignment), time.perf_counter() - start, "HEURISTICA")
