"""Baselines heurísticos fortes: o LNS matheurístico e a busca local do projeto principal,
adaptados à variante estrita por uma penalidade de não atendimento proibitiva (qualquer
solução com cliente sem centro é rejeitada pelo validador estrito)."""

from __future__ import annotations

import time

import numpy as np

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.pesquisa.hibrido import Saida
from alocacao_capacitada.pesquisa.problema import Problema, validar
from alocacao_capacitada.pesquisa.tempos import cronometrar
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch


def para_instance(prob: Problema) -> Instance:
    unit = prob.cost / prob.demand[None, :]
    # por unidade; qualquer solução com cliente sem centro custa > 2x qualquer solução completa
    penal = max(1., float(2 * (prob.fixed.sum() + prob.cost.max(axis=0).sum()) / prob.demand.min()))
    return Instance(
        facility_ids=tuple(f"F{i}" for i in range(prob.m)),
        demand_ids=tuple(f"C{j}" for j in range(prob.n)),
        demand=prob.demand, capacity=prob.capacity, fixed_cost=prob.fixed,
        unit_cost=unit, unserved_penalty=penal,
    )


@cronometrar("t_heuristica")
def lns(prob: Problema, tempo: float, seed: int = 0, free_size: int = 50) -> Saida:
    t0 = time.perf_counter()
    if tempo <= 0:
        return Saida("lns", np.inf, False, 0.0, 0.0, 0, 1.0, False, False)
    inst = para_instance(prob)
    traj: list[tuple[float, float]] = []

    def prog(_it: int, _t: float, cost: float) -> None:
        if not traj or cost < traj[-1][1] - 1e-9:
            traj.append((time.perf_counter() - t0, cost))

    solver = LnsSolver(LocalSearch(NearestFeasibleGreedy()), seed=seed, free_size=free_size,
                       max_free_size=2 * free_size, on_progress=prog)
    restante = tempo - (time.perf_counter() - t0)
    if restante <= 0:
        return Saida("lns", np.inf, False, time.perf_counter() - t0,
                     0.0, 0, 1.0, False, False)
    res = solver.solve(inst, restante)
    a = res.solution.assignment
    v = validar(prob, a.astype(np.int64)) if (a >= 0).all() else None
    # pontos com cliente sem centro não são soluções da variante estrita
    limite = inst.unserved_penalty * float(prob.demand.min()) / 2
    traj = [(t, c) for t, c in traj if c < limite]
    return Saida("lns", v.objetivo if v else np.inf, bool(v and v.valida),
                 time.perf_counter() - t0, 0.0, 1, 1.0, False, False, traj,
                 atribuicao=a.astype(np.int64) if v else None)
