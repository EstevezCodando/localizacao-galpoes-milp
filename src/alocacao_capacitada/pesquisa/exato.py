"""Solver completo (SCIP 10 via PySCIPOpt) e relaxação linear (HiGHS) do SSCFLP estrito.

Registra tudo o que o protocolo (doc 10) pede: tempo de montagem e de solução, status, bound e
seu ESCOPO (original ou reduzido), nós de B&B, trajetória do incumbente (para tempo até alvo) e
o pool de soluções (para rótulos). Uma thread, semente fixa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import highspy
import numpy as np
from pyscipopt import SCIP_EVENTTYPE, Eventhdlr, Model

from alocacao_capacitada.pesquisa.problema import Problema, validar
from alocacao_capacitada.pesquisa.tempos import cronometrar, registrar


class _Trajetoria(Eventhdlr):  # type: ignore[misc]
    def __init__(self) -> None:
        self.pontos: list[tuple[float, float]] = []
        self.t0 = 0.0

    def eventinit(self) -> None:
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        if self.model.getNSols() > 0:  # solução inicial aceita antes da captura de eventos
            self.pontos.append((time.perf_counter() - self.t0, self.model.getPrimalbound()))

    def eventexit(self) -> None:
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event: object) -> None:
        sol = self.model.getBestSol()
        self.pontos.append((time.perf_counter() - self.t0, self.model.getSolObjVal(sol)))


@dataclass
class Resultado:
    status: str
    atribuicao: np.ndarray | None
    objetivo: float  # recalculado pelo validador sobre o problema ORIGINAL
    bound: float | None
    bound_escopo: str  # "original" | "reduzido"
    t_montagem: float
    t_solver: float
    t_total: float
    nos: int
    trajetoria: list[tuple[float, float]] = field(default_factory=list)
    # (obj, y em índices originais)
    pool: list[tuple[float, np.ndarray]] = field(default_factory=list)
    valida: bool = False
    t_validacao: float = 0.0


def resolver(
    prob: Problema,
    tempo_s: float,
    centros: np.ndarray | None = None,
    pares_permitidos: np.ndarray | None = None,
    dica: np.ndarray | None = None,
    prioridade: np.ndarray | None = None,
    gap_alvo: float = 0.0,
    pool: bool = False,
    seed: int = 0,
    emphasis: str | None = None,
) -> Resultado:
    """Resolve o SSCFLP com SCIP.

    centros: índices originais mantidos (poda). Sem poda o bound tem escopo "original".
    pares_permitidos: máscara (m, n) no espaço ORIGINAL; pares proibidos ficam com ub = 0.
    dica: atribuição completa (índices originais) passada como solução inicial (warm start).
    prioridade: prioridade de branching por centro (índices originais), orientação sem poda.
    """
    t0 = time.perf_counter()
    deadline = t0 + max(tempo_s, 0.0)

    def esgotado(montagem: float = 0.0) -> Resultado:
        # Preserva um incumbente já conhecido, sem iniciar outro solver.
        v = validar(prob, dica) if dica is not None else None
        ok = bool(v and v.valida)
        elapsed = time.perf_counter() - t0
        return Resultado("budget_exhausted", dica if ok else None,
                         v.objetivo if ok else np.inf, None,
                         "original" if centros is None and pares_permitidos is None else "reduzido",
                         montagem, 0.0, elapsed, 0,
                         trajetoria=[(elapsed, v.objetivo)] if ok else [], valida=ok)

    if tempo_s <= 0:
        return esgotado()
    idx = np.arange(prob.m) if centros is None else np.sort(np.asarray(centros, dtype=int))
    sub = prob if centros is None else prob.restrito(idx)
    m, n = sub.m, sub.n
    model = Model()
    model.hideOutput()
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("randomization/randomseedshift", seed)
    model.setParam("timing/clocktype", 2)  # relógio de parede, como o protocolo exige
    if gap_alvo > 0:
        model.setParam("limits/gap", gap_alvo)
    if emphasis == "feasibility":
        from pyscipopt import SCIP_PARAMEMPHASIS

        model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
    y = model.addMatrixVar(m, vtype="B", name="y")
    ub = None
    if pares_permitidos is not None:
        ub = np.asarray(pares_permitidos, dtype=float)[idx]
    x = model.addMatrixVar((m, n), vtype="B", name="x", ub=ub if ub is not None else 1.0)
    model.addMatrixCons(x.sum(axis=0) == 1)
    model.addMatrixCons((x * sub.demand[None, :]).sum(axis=1) <= sub.capacity * y)
    model.addMatrixCons(x - y.reshape(m, 1) <= 0)
    model.setObjective((sub.fixed * y).sum() + (sub.cost * x).sum())
    if prioridade is not None:
        pr = np.asarray(prioridade)[idx]
        for i in range(m):
            model.chgVarBranchPriority(y[i], int(pr[i]))
    if dica is not None:
        pos = {int(o): k for k, o in enumerate(idx)}
        if all(int(i) in pos for i in np.unique(dica)):
            sol = model.createSol()
            abertos = {pos[int(i)] for i in np.unique(dica)}
            for i in range(m):
                model.setSolVal(sol, y[i], 1.0 if i in abertos else 0.0)
            for j in range(n):
                model.setSolVal(sol, x[pos[int(dica[j])], j], 1.0)
            model.addSol(sol, free=True)
    traj = _Trajetoria()
    model.includeEventhdlr(traj, "traj", "trajetoria do incumbente")
    t_montagem = time.perf_counter() - t0
    registrar("t_montagem", t_montagem)
    traj.t0 = t0
    restante = deadline - time.perf_counter()
    if restante <= 0:
        return esgotado(t_montagem)
    # Reserva para extração/validação. Chamadas nativas não são preemptíveis;
    # o executor registra todo estouro, sem ocultá-lo no orçamento.
    model.setParam("limits/time", restante * 0.95)
    t_solve = time.perf_counter()
    model.optimize()
    t_solver = time.perf_counter() - t_solve
    registrar("t_solver", t_solver)
    t_validacao = time.perf_counter()
    status = model.getStatus()
    nos = int(model.getNNodes())
    bound = float(model.getDualbound()) if status != "infeasible" else None
    atribuicao = None
    lista_pool: list[tuple[float, np.ndarray]] = []
    if model.getNSols() > 0:
        best = model.getBestSol()
        xv = np.array([[model.getSolVal(best, x[i, j]) for j in range(n)] for i in range(m)])
        atribuicao = idx[np.argmax(xv, axis=0)].astype(np.int64)
        if pool:
            for s in model.getSols():
                yv = np.array([model.getSolVal(s, y[i]) for i in range(m)]) > 0.5
                yo = np.zeros(prob.m, dtype=bool)
                yo[idx[yv]] = True
                lista_pool.append((float(model.getSolObjVal(s)), yo))
    val = validar(prob, atribuicao) if atribuicao is not None else None
    registrar("t_validacao", time.perf_counter() - t_validacao)
    return Resultado(
        status=status,
        atribuicao=atribuicao,
        objetivo=val.objetivo if val is not None else np.inf,
        bound=bound,
        bound_escopo="original" if centros is None and pares_permitidos is None else "reduzido",
        t_montagem=t_montagem,
        t_solver=t_solver,
        t_total=time.perf_counter() - t0,
        nos=nos,
        trajetoria=traj.pontos,
        pool=lista_pool,
        valida=bool(val is not None and val.valida),
        t_validacao=time.perf_counter() - t_validacao,
    )


@dataclass(frozen=True)
class Relaxacao:
    bound: float
    y: np.ndarray  # (m,)
    x: np.ndarray  # (m, n)
    rc_y: np.ndarray  # custo reduzido de y_i
    rc_x: np.ndarray  # custo reduzido de x_ij
    dual_atend: np.ndarray  # dual da linha de atendimento de cada cliente (n,)
    dual_cap: np.ndarray  # dual da linha de capacidade (m,)
    tempo: float


@cronometrar("t_pl")
def relaxacao_linear(prob: Problema, pares_permitidos: np.ndarray | None = None,
                     tempo_s: float | None = None) -> Relaxacao:
    """PL forte (com x_ij <= y_i) pelo HiGHS, montado em CSR. Bound válido do ORIGINAL quando
    `pares_permitidos` é None."""
    t0 = time.perf_counter()
    if tempo_s is not None and tempo_s <= 0:
        raise TimeoutError("Orçamento da PL esgotado antes da montagem")
    m, n = prob.m, prob.n
    nx = m * n
    ncol = m + nx
    cost = np.concatenate([prob.fixed, prob.cost.ravel()])
    upper = np.ones(ncol)
    if pares_permitidos is not None:
        upper[m:] = np.asarray(pares_permitidos, dtype=float).ravel()
    xi = m + np.arange(nx).reshape(m, n)  # índice da coluna x_ij
    # linhas: n atendimento, m capacidade, m*n ligação
    starts = [0]
    index: list[np.ndarray] = []
    value: list[np.ndarray] = []
    for j in range(n):
        index.append(xi[:, j])
        value.append(np.ones(m))
        starts.append(starts[-1] + m)
    for i in range(m):
        index.append(np.concatenate([xi[i], [i]]))
        value.append(np.concatenate([prob.demand, [-prob.capacity[i]]]))
        starts.append(starts[-1] + n + 1)
    link_idx = np.column_stack([xi.ravel(), np.repeat(np.arange(m), n)])
    link_val = np.tile([1.0, -1.0], (nx, 1))
    ind = np.concatenate([*index, link_idx.ravel()])
    val = np.concatenate([*value, link_val.ravel()])
    start = np.concatenate([np.array(starts), starts[-1] + 2 * np.arange(1, nx + 1)])
    nrow = n + m + nx
    lp = highspy.HighsLp()
    lp.num_col_ = ncol
    lp.num_row_ = nrow
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(ncol)
    lp.col_upper_ = upper
    lp.row_lower_ = np.concatenate([np.ones(n), np.full(m + nx, -highspy.kHighsInf)])
    lp.row_upper_ = np.concatenate([np.ones(n), np.zeros(m + nx)])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = start.astype(np.int32)
    lp.a_matrix_.index_ = ind.astype(np.int32)
    lp.a_matrix_.value_ = val
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    if tempo_s is not None:
        restante = tempo_s - (time.perf_counter() - t0)
        if restante <= 0:
            raise TimeoutError("Orçamento da PL esgotado na montagem")
        h.setOptionValue("time_limit", restante * 0.95)
    h.run()
    if h.getModelStatus() == highspy.HighsModelStatus.kTimeLimit:
        # Não usar duais/custos reduzidos sem otimalidade para certificar poda.
        raise TimeoutError("PL interrompida pelo limite de tempo; sem certificado")
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"PL não ótimo: {h.getModelStatus()}")
    sol = h.getSolution()
    col = np.array(sol.col_value)
    dual = np.array(sol.col_dual)
    row_dual = np.array(sol.row_dual)
    return Relaxacao(
        bound=float(h.getInfo().objective_function_value),
        y=col[:m],
        x=col[m:].reshape(m, n),
        rc_y=dual[:m],
        rc_x=dual[m:].reshape(m, n),
        dual_atend=row_dual[:n],
        dual_cap=row_dual[n : n + m],
        tempo=time.perf_counter() - t0,
    )
