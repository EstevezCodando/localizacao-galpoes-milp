"""Frente P06 — Qian, Morris, Jegelka & Sohler (ICML 2026): aprender a aproximar o Uniform
Facility Location (UniFL) com GNN. Reconstrução metodológica (R3): código oficial não confirmado.

UniFL: pontos P num espaço métrico; todo ponto é cliente e candidato; custo de abertura
uniforme f; sem capacidade.  min f |S| + sum_j min_{i in S} d(i, j).

Implementado aqui:
  * ótimo exato (SCIP, formulação forte);
  * Mettu–Plaxton (MP, 3-aproximação): raio r_i com sum_j max(0, r_i - d_ij) = f; percorre
    em ordem crescente de r e abre i se nenhum aberto estiver a menos de 2 r_i;
  * busca local add/drop/swap (heurística clássica forte, Arya et al. 2004);
  * MPNN não supervisionada: probabilidades de abertura p_i; perda = custo ESPERADO sob
    aberturas independentes (diferenciável, sem rótulos), como no artigo. Decodificação por
    limiar + reparo; opcionalmente polida pela busca local (tempo contabilizado).

As garantias de aproximação do artigo valem para UniFL sob as hipóteses dele e NÃO se
transferem para o SSCFLP com capacidade (doc 03).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn

torch.set_num_threads(1)


@dataclass(frozen=True)
class UniFL:
    pts: np.ndarray
    f: float
    familia: str

    @property
    def n(self) -> int:
        return int(self.pts.shape[0])

    def dist(self) -> np.ndarray:
        return cdist(self.pts, self.pts)


def gerar_unifl(n: int, familia: str, seed: int, k_alvo: float | None = None) -> UniFL:
    rng = np.random.default_rng(seed)
    if familia == "uniforme":
        pts = rng.uniform(0, 1, (n, 2))
    else:
        c = rng.uniform(0.1, 0.9, (int(rng.integers(4, 9)), 2))
        pts = np.clip(c[rng.integers(len(c), size=n)] + rng.normal(0, 0.05, (n, 2)), 0, 1)
    k = k_alvo or np.sqrt(n)
    return UniFL(pts, float(0.19 * n / k**1.5), familia)


def custo(inst: UniFL, abertos: np.ndarray, d: np.ndarray | None = None) -> float:
    if abertos.size == 0:
        return np.inf
    d = inst.dist() if d is None else d
    return float(inst.f * abertos.size + d[abertos].min(axis=0).sum())


def raios_mp(d: np.ndarray, f: float) -> np.ndarray:
    ds = np.sort(d, axis=1)
    cum = np.cumsum(ds, axis=1)
    k = np.arange(1, d.shape[1] + 1)
    r = (f + cum) / k  # candidato com k pontos dentro da bola
    nxt = np.concatenate([ds[:, 1:], np.full((d.shape[0], 1), np.inf)], axis=1)
    ok = (r >= ds - 1e-12) & (r <= nxt + 1e-12)
    return r[np.arange(d.shape[0]), ok.argmax(axis=1)]


def mettu_plaxton(inst: UniFL, d: np.ndarray | None = None) -> np.ndarray:
    d = inst.dist() if d is None else d
    r = raios_mp(d, inst.f)
    abertos: list[int] = []
    for i in np.argsort(r, kind="stable"):
        if not abertos or d[i, abertos].min() > 2 * r[i]:
            abertos.append(int(i))
    return np.array(sorted(abertos))


def busca_local(inst: UniFL, ini: np.ndarray, d: np.ndarray | None = None,
                tempo: float = 60.0) -> np.ndarray:
    """Primeira melhora em add/drop/swap, vetorizada pelos custos de menor e 2ª menor distância."""
    d = inst.dist() if d is None else d
    n = inst.n
    S = set(int(i) for i in ini)
    t0 = time.perf_counter()
    melhorou = True
    while melhorou and time.perf_counter() - t0 < tempo:
        melhorou = False
        lst = np.array(sorted(S))
        sub = d[lst]
        order = np.argsort(sub, axis=0)
        d1 = sub[order[0], np.arange(n)]
        d2 = sub[order[1], np.arange(n)] if lst.size > 1 else np.full(n, np.inf)
        atual = inst.f * lst.size + d1.sum()
        # add: ganho de abrir k
        add = inst.f + np.minimum(d, d1[None, :]).sum(axis=1)
        add[lst] = np.inf
        k = int(np.argmin(add))
        if add[k] - d1.sum() < -1e-9:
            S.add(k)
            melhorou = True
            continue
        # drop
        if lst.size > 1:
            for pos, i in enumerate(lst):
                serv = order[0] == pos
                delta = -inst.f + (d2[serv] - d1[serv]).sum()
                if delta < -1e-9:
                    S.remove(int(i))
                    melhorou = True
                    break
            if melhorou:
                continue
        # swap (i sai, k entra): custo exato vetorizado por i
        for pos, i in enumerate(lst):
            serv = order[0] == pos
            base = d1.copy()
            base[serv] = d2[serv]  # sem i
            novo = np.minimum(d, base[None, :]).sum(axis=1)
            novo[lst] = np.inf
            k = int(np.argmin(novo))
            if novo[k] + inst.f * lst.size < atual - 1e-9:
                S.remove(int(i))
                S.add(k)
                melhorou = True
                break
    return np.array(sorted(S))


def exato(inst: UniFL, tempo: float = 300.0) -> tuple[float, float, np.ndarray, str]:
    from pyscipopt import Model

    d = inst.dist()
    n = inst.n
    m = Model()
    m.hideOutput()
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("limits/time", tempo)
    y = m.addMatrixVar(n, vtype="B")
    x = m.addMatrixVar((n, n), vtype="C", lb=0, ub=1)
    m.addMatrixCons(x.sum(axis=0) == 1)
    m.addMatrixCons(x - y.reshape(n, 1) <= 0)
    m.setObjective(inst.f * y.sum() + (d * x).sum())
    # solução inicial da busca local acelera e é legítima (heurística clássica)
    ini = busca_local(inst, mettu_plaxton(inst, d), d)
    sol = m.createSol()
    for i in range(n):
        m.setSolVal(sol, y[i], 1.0 if i in set(ini.tolist()) else 0.0)
    near = ini[np.argmin(d[ini], axis=0)]
    for j in range(n):
        m.setSolVal(sol, x[near[j], j], 1.0)
    m.addSol(sol, free=True)
    m.optimize()
    best = m.getBestSol()
    yv = np.array([m.getSolVal(best, y[i]) for i in range(n)]) > 0.5
    return float(m.getObjVal()), float(m.getDualbound()), np.flatnonzero(yv), m.getStatus()


# --------------------------------------------------------------------------- MPNN
def _feats(inst: UniFL, d: np.ndarray, k: int, com_raio: bool) -> tuple[torch.Tensor, ...]:
    n = inst.n
    kk = min(k, n - 1)
    nb = np.argsort(d, axis=1)[:, 1 : kk + 1]
    dn = d[np.arange(n)[:, None], nb] / inst.f  # distâncias em unidades de f
    x = [dn.mean(axis=1), dn[:, 0], dn[:, -1], np.full(n, np.log(n))]
    if com_raio:
        x.append(raios_mp(d, inst.f) / inst.f)
    X = np.column_stack(x).astype(np.float32)
    src = np.repeat(np.arange(n), kk)
    dst = nb.ravel()
    e = dn.ravel().astype(np.float32)[:, None]
    return (torch.from_numpy(X), torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(e))


class MPNN(nn.Module):
    def __init__(self, fin: int, dim: int = 64, camadas: int = 4, estavel: bool = False) -> None:
        super().__init__()
        self.estavel = estavel
        self.enc = nn.Sequential(nn.Linear(fin, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.ReLU(),
                                                nn.Linear(dim, dim)) for _ in range(camadas)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(),
                                                nn.Linear(dim, dim)) for _ in range(camadas)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(camadas)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, X: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                e: torch.Tensor) -> torch.Tensor:
        h = self.enc(X)
        n = h.shape[0]
        for msg, upd, nrm in zip(self.msg, self.upd, self.norm):
            mm = msg(torch.cat([h[src], h[dst], e], 1))
            s = torch.zeros(n, h.shape[1]).index_add_(0, src, mm)
            mx = torch.full((n, h.shape[1]), -1e9).index_reduce_(0, src, mm, "amax")
            h = nrm(h + upd(torch.cat([h, s / 16.0, mx], 1)))
        p = torch.sigmoid(self.head(h).squeeze(1))
        # versão estável (desvio documentado): piso/teto mantêm o gradiente vivo e evitam o
        # colapso por saturação observado em 3 de 4 sementes da versão original
        return 0.01 + 0.98 * p if self.estavel else p


def custo_esperado(p: torch.Tensor, inst: UniFL, d: np.ndarray, kc: int = 24) -> torch.Tensor:
    """E[custo] com aberturas independentes; cada cliente olha seus kc candidatos mais próximos
    (incluindo ele mesmo, distância 0) e paga M = 2 * d_kc se nenhum estiver aberto."""
    n = inst.n
    kc = min(kc, n)
    nb = np.argsort(d, axis=1)[:, :kc]  # (n, kc), nb[j,0] = j
    c = torch.from_numpy(d[np.arange(n)[:, None], nb]).float()
    pj = p[torch.from_numpy(nb)]  # (n, kc)
    surv = torch.cumprod(torch.cat([torch.ones(n, 1), 1 - pj], 1), 1)  # P(nenhum dos l<k)
    exp_serv = (c * pj * surv[:, :-1]).sum(1) + 2 * c[:, -1] * surv[:, -1]
    return (inst.f * p.sum() + exp_serv.sum()) / (inst.f * np.sqrt(n))


def treinar_mpnn(n_treino: int = 200, tamanhos: tuple[int, ...] = (100, 200), epocas: int = 30,
                 com_raio: bool = False, seed: int = 0, k: int = 16,
                 estavel: bool = False) -> tuple[MPNN, float]:
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    insts = [gerar_unifl(int(tamanhos[s % len(tamanhos)]), ("uniforme", "clusters")[s % 2],
                         10_000 + s) for s in range(n_treino)]
    prep = [(i, i.dist()) for i in insts]
    prep = [(i, d, _feats(i, d, k, com_raio)) for i, d in prep]
    model = MPNN(prep[0][2][0].shape[1], estavel=estavel)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4 if estavel else 1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epocas):
        for idx in rng.permutation(len(prep)):
            inst, d, (X, s, t, e) = prep[idx]
            opt.zero_grad()
            loss = custo_esperado(model(X, s, t, e), inst, d)
            loss.backward()
            if estavel:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model, time.perf_counter() - t0


def decodificar(model: MPNN, inst: UniFL, d: np.ndarray, com_raio: bool, k: int = 16,
                limiar: float = 0.5) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = model(*_feats(inst, d, k, com_raio)).numpy()
    S = np.flatnonzero(p >= limiar)
    if S.size == 0:
        S = np.array([int(np.argmax(p))])
    # reparo guloso barato: fecha centros que não pagam o próprio custo fixo
    while S.size > 1:
        sub = d[S]
        o = np.argsort(sub, axis=0)
        d1, d2 = sub[o[0], np.arange(inst.n)], sub[o[1], np.arange(inst.n)]
        delta = np.array([-inst.f + (d2[o[0] == a] - d1[o[0] == a]).sum() for a in range(S.size)])
        a = int(np.argmin(delta))
        if delta[a] >= 0:
            break
        S = np.delete(S, a)
    return S
