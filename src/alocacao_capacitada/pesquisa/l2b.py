"""Frentes P02 (Gasse et al., NeurIPS 2019) e P03 (Gupta et al., NeurIPS 2020): aprender a
ramificar imitando strong branching. Reconstrução R3 em PySCIPOpt 6 / SCIP 10 — o repositório
oficial depende do Ecole e de PySCIPOpt antigo (sem suporte a Python 3.12/Windows).

Instâncias: família `facilities` do learn2branch = CFLP de Cornuéjols et al. (1991) com
atribuição CONTÍNUA (demanda fracionável), razão de capacidade 5, restrição de cobertura de
capacidade e reforço x_ij <= y_j. Reimplementada a partir da descrição do gerador.

Diferenças em relação a P02 (registradas no relatório):
  * atributos por candidato (estilo Khalil et al., 2016, que P02 usa como baseline) e GNN no
    grafo LOGÍSTICO centro–cliente com valores do PL nas arestas, em vez do grafo bipartido
    variável–restrição completo (extraí-lo em Python a cada nó custaria mais que o nó);
  * o professor é o strong branching completo do SCIP via getVarStrongbranch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pyscipopt import SCIP_RESULT, Branchrule, Model


@dataclass(frozen=True)
class Cflp:
    cost: np.ndarray  # (m, n) custo total
    fixed: np.ndarray
    cap: np.ndarray
    dem: np.ndarray


def gerar_facilities(n_cli: int, n_fac: int, seed: int, ratio: float = 5.0) -> Cflp:
    rng = np.random.default_rng(seed)
    cx, cy = rng.random(n_cli), rng.random(n_cli)
    fx, fy = rng.random(n_fac), rng.random(n_fac)
    dem = rng.integers(5, 36, n_cli).astype(float)
    cap = rng.integers(10, 161, n_fac).astype(float)
    fixed = (rng.integers(100, 111, n_fac) * np.sqrt(cap) + rng.integers(0, 91, n_fac)).astype(int)
    cap = np.floor(cap * ratio * dem.sum() / cap.sum())
    dist = np.sqrt((fx[:, None] - cx[None]) ** 2 + (fy[:, None] - cy[None]) ** 2)
    return Cflp(dist * 10 * dem[None, :], fixed.astype(float), cap, dem)


def modelo(inst: Cflp, tempo: float, seed: int = 0) -> tuple[Model, list, object]:  # type: ignore[type-arg]
    m, n = inst.cost.shape
    md = Model()
    md.hideOutput()
    md.setParam("parallel/maxnthreads", 1)
    md.setParam("timing/clocktype", 2)
    md.setParam("limits/time", tempo)
    md.setParam("randomization/randomseedshift", seed)
    y = md.addMatrixVar(m, vtype="B", name="y")
    x = md.addMatrixVar((m, n), vtype="C", lb=0, ub=1, name="x")
    md.addMatrixCons(x.sum(axis=0) >= 1)
    md.addMatrixCons((x * inst.dem[None, :]).sum(axis=1) <= inst.cap * y)
    md.addCons((inst.cap * y).sum() >= inst.dem.sum())
    md.addMatrixCons(x - y.reshape(m, 1) <= 0)
    md.setObjective((inst.fixed * y).sum() + (inst.cost * x).sum())
    return md, [y[i] for i in range(m)], x


def atributos_candidatos(md: Model, ys: list, x: object, inst: Cflp, cands: list) -> np.ndarray:  # type: ignore[type-arg]
    """Atributos dinâmicos e estáticos por variável candidata y_j (todos baratos)."""
    m, n = inst.cost.shape
    idx = {v.name.removeprefix("t_"): k for k, v in enumerate(ys)}
    out = []
    lp = md.getLPObjVal()
    for v in cands:
        j = idx[v.name.removeprefix("t_")]
        val = v.getLPSol()
        frac = val - np.floor(val)
        pc_down = md.getVarPseudocost(v, 0) if hasattr(md, "getVarPseudocost") else 0.0
        pc_up = md.getVarPseudocost(v, 1) if hasattr(md, "getVarPseudocost") else 0.0
        out.append([
            frac, min(frac, 1 - frac), val,
            inst.fixed[j] / inst.fixed.mean(), inst.cap[j] / inst.cap.mean(),
            md.getVarRedcost(v) / max(abs(lp), 1.0),
            pc_down, pc_up, pc_down * pc_up,
            md.getDepth() / 10.0, len(cands) / m,
        ])
    return np.array(out, dtype=np.float32)


class RegraAprendida(Branchrule):  # type: ignore[misc]
    """Pontua candidatos com o modelo e ramifica no melhor. Mede o tempo de inferência."""

    def __init__(self, ys: list, x: object, inst: Cflp, scorer: object) -> None:  # type: ignore[type-arg]
        self.ys, self.x, self.inst, self.scorer = ys, x, inst, scorer
        self.t_inf = 0.0
        self.chamadas = 0

    def branchexeclp(self, allowaddcons: bool) -> dict[str, object]:
        cands, *_ = self.model.getLPBranchCands()
        t0 = time.perf_counter()
        f = atributos_candidatos(self.model, self.ys, self.x, self.inst, cands)
        s = self.scorer.predict(f)  # type: ignore[attr-defined]
        self.t_inf += time.perf_counter() - t0
        self.chamadas += 1
        self.model.branchVar(cands[int(np.argmax(s))])
        return {"result": SCIP_RESULT.BRANCHED}


class Coletor(Branchrule):  # type: ignore[misc]
    """Coleta (atributos, score de strong branching) em nós amostrados com prob `p_sb`; ramifica
    pelo strong branching nesses nós e devolve DIDNOTRUN nos demais (o SCIP usa o relpscost).
    Mesmo esquema de exploração de P02 (professor em uma fração dos nós)."""

    def __init__(self, ys: list, x: object, inst: Cflp, p_sb: float, seed: int) -> None:  # type: ignore[type-arg]
        self.ys, self.x, self.inst, self.p = ys, x, inst, p_sb
        self.rng = np.random.default_rng(seed)
        self.amostras: list[tuple[np.ndarray, np.ndarray]] = []

    def branchexeclp(self, allowaddcons: bool) -> dict[str, object]:
        if self.rng.random() > self.p:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        md = self.model
        cands, *_ = md.getLPBranchCands()
        lp = md.getLPObjVal()
        f = atributos_candidatos(md, self.ys, self.x, self.inst, cands)
        md.startStrongbranch()
        sc = []
        for v in cands:
            r = md.getVarStrongbranch(v, 1000, idempotent=True)
            down, up = r[0], r[1]
            dg = max(down - lp, 1e-6) if r[2] else 1e-6
            ug = max(up - lp, 1e-6) if r[3] else 1e-6
            if r[4]:
                dg = 1e6
            if r[5]:
                ug = 1e6
            sc.append(dg * ug)
        md.endStrongbranch()
        sc_a = np.array(sc)
        self.amostras.append((f, sc_a))
        md.branchVar(cands[int(np.argmax(sc_a))])
        return {"result": SCIP_RESULT.BRANCHED}


def resolver(inst: Cflp, tempo: float, regra: str, scorer: object | None = None,
             seed: int = 0) -> dict[str, float]:
    md, ys, x = modelo(inst, tempo, seed)
    br = None
    if regra in ("relpscost", "pscost", "fullstrong", "mostinf"):
        md.setParam(f"branching/{regra}/priority", 100_000)
    elif regra == "aprendida":
        br = RegraAprendida(ys, x, inst, scorer)
        md.includeBranchrule(br, "aprendida", "imitação de SB", priority=200_000, maxdepth=-1,
                             maxbounddist=1.0)
    t0 = time.perf_counter()
    md.optimize()
    return {"tempo": time.perf_counter() - t0, "nos": md.getNNodes(), "status": md.getStatus(),
            "obj": md.getObjVal() if md.getNSols() else np.nan, "bound": md.getDualbound(),
            "gap": md.getGap(), "t_inferencia": br.t_inf if br else 0.0,
            "chamadas": br.chamadas if br else 0}


def coletar(inst: Cflp, p_sb: float, seed: int, tempo: float = 120.0
            ) -> list[tuple[np.ndarray, np.ndarray]]:
    md, ys, x = modelo(inst, tempo, seed)
    col = Coletor(ys, x, inst, p_sb, seed)
    md.includeBranchrule(col, "coletor", "strong branching amostrado", priority=200_000,
                         maxdepth=-1, maxbounddist=1.0)
    md.optimize()
    return col.amostras
