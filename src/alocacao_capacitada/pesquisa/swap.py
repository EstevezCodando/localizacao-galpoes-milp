"""Frentes P04 (Guo, Xu & Jin 2023) e P05 (Su et al., SIGSPATIAL 2024): trocas guiadas por
aprendizado em localização sobre redes. Reconstrução metodológica (R3).

Problema: p-mediana ponderada num grafo viário sintético (distância = caminho mínimo).
  min sum_j w_j min_{i in S} d(i, j),  |S| = p.

Busca por trocas (Teitz & Bart, 1968): a cada iteração, avalia trocas (i sai, k entra) e aplica
a melhor que melhora. Custo de uma iteração completa: p * (n - p) avaliações exatas.

O mecanismo dos artigos é usar uma política aprendida para propor trocas (P04 por RL; P05 por
GNN guiada por conhecimento) e avaliar muito menos candidatos. Aqui, por imitação supervisionada
(substituição documentada do PPO de P04): um modelo pontua centros que entram e que saem e só as
`k_out x k_in` trocas mais promissoras são avaliadas exatamente; se nenhuma melhora, faz-se UMA
varredura completa (fallback contado) antes de parar. Isso preserva o ótimo local da busca
completa como critério de parada, então qualquer diferença de qualidade vem da trajetória.

Competidor clássico com o MESMO filtro: pontuar a entrada pelo ganho de adição
g_add(k) = sum_j w_j max(0, d1_j - d_kj) e a saída pela perda de remoção
l_rem(i) = sum_{j servidos por i} w_j (d2_j - d1_j). É o "conhecimento" do qual a rede aprende.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PMediana:
    d: np.ndarray  # (n, n) caminhos mínimos
    w: np.ndarray  # (n,) pesos
    p: int
    xy: np.ndarray
    familia: str

    @property
    def n(self) -> int:
        return int(self.w.size)


def gerar_rede(n: int, p: int, familia: str, seed: int) -> PMediana:
    """Grafo geométrico aleatório conexo (k vizinhos + desvio de 0-40% por aresta).
    `cidade`: pesos concentrados em polos; `uniforme`: pesos U(1, 10)."""
    rng = np.random.default_rng(seed)
    if familia == "cidade":
        c = rng.uniform(0.15, 0.85, (int(rng.integers(3, 7)), 2))
        xy = np.clip(c[rng.integers(len(c), size=n)] + rng.normal(0, 0.12, (n, 2)), 0, 1)
        w = 1 + 9 * np.exp(-((xy[:, None] - c[None]) ** 2).sum(2).min(1) / 0.01)
    else:
        xy = rng.uniform(0, 1, (n, 2))
        w = rng.uniform(1, 10, n)
    tree = cKDTree(xy)
    dist, nb = tree.query(xy, k=6)
    rows = np.repeat(np.arange(n), 5)
    cols = nb[:, 1:].ravel()
    val = dist[:, 1:].ravel() * rng.uniform(1.0, 1.4, rows.size)
    g = csr_matrix((val, (rows, cols)), shape=(n, n))
    g = g.maximum(g.T)  # não dirigido
    d = dijkstra(g, directed=False)
    if np.isinf(d).any():  # liga componentes pelo par mais próximo (garante conexidade)
        comp = np.isinf(d[0])
        a, b = np.flatnonzero(~comp), np.flatnonzero(comp)
        dd = np.linalg.norm(xy[a][:, None] - xy[b][None], axis=2)
        i, j = np.unravel_index(np.argmin(dd), dd.shape)
        g = g.tolil()
        g[a[i], b[j]] = g[b[j], a[i]] = dd[i, j] * 1.2
        d = dijkstra(g.tocsr(), directed=False)
    return PMediana(d, w, p, xy, familia)


def custo(inst: PMediana, S: np.ndarray) -> float:
    return float((inst.w * inst.d[S].min(axis=0)).sum())


def _estado(inst: PMediana, S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = inst.d[S]
    o = np.argsort(sub, axis=0)
    n = inst.n
    return o[0], sub[o[0], np.arange(n)], sub[o[1], np.arange(n)]


def gain_add(inst: PMediana, d1: np.ndarray) -> np.ndarray:
    return (inst.w[None, :] * np.maximum(0.0, d1[None, :] - inst.d)).sum(axis=1)


def loss_rem(inst: PMediana, S: np.ndarray, a1: np.ndarray, d1: np.ndarray,
             d2: np.ndarray) -> np.ndarray:
    return np.array([(inst.w * (d2 - d1))[a1 == pos].sum() for pos in range(S.size)])


def delta_trocas(inst: PMediana, S: np.ndarray, a1: np.ndarray, d1: np.ndarray, d2: np.ndarray,
                 outs: np.ndarray, ins: np.ndarray) -> np.ndarray:
    """Variação exata de custo de cada troca (outs[a] sai, ins[b] entra): matriz (|outs|, |ins|)."""
    atual = (inst.w * d1).sum()
    res = np.empty((outs.size, ins.size))
    dk = inst.d[ins]  # (|ins|, n)
    for a, pos in enumerate(outs):
        base = np.where(a1 == pos, d2, d1)
        res[a] = (inst.w[None, :] * np.minimum(dk, base[None, :])).sum(axis=1) - atual
    return res


def inicial_guloso(inst: PMediana) -> np.ndarray:
    S: list[int] = []
    d1 = np.full(inst.n, np.inf)
    for _ in range(inst.p):
        g = (inst.w[None, :] * np.minimum(inst.d, d1[None, :])).sum(axis=1)
        g[S] = np.inf
        k = int(np.argmin(g))
        S.append(k)
        d1 = np.minimum(d1, inst.d[k])
    return np.array(S)


_CACHE: dict[int, np.ndarray] = {}


def _loc(inst: PMediana) -> np.ndarray:
    """Distância média aos 10 vizinhos: estática, calculada uma vez por instância."""
    key = id(inst.d)
    if key not in _CACHE:
        _CACHE.clear()
        # 11 menores incluem a própria (distância 0): soma/10 = média dos 10 vizinhos
        _CACHE[key] = np.partition(inst.d, 10, axis=1)[:, :11].sum(axis=1) / 10
    return _CACHE[key]


def atributos(inst: PMediana, S: np.ndarray, a1: np.ndarray, d1: np.ndarray,
              d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Atributos por candidato a ENTRAR (n,F) e por centro a SAIR (p,G), normalizados pela
    instância (escala = custo médio por cliente)."""
    esc = float((inst.w * d1).sum() / inst.w.sum()) + 1e-12
    ga = gain_add(inst, d1) / (esc * inst.w.sum())
    lr = loss_rem(inst, S, a1, d1, d2) / (esc * inst.w.sum())
    near_open = inst.d[:, S].min(axis=1) / esc
    loc = _loc(inst) / esc
    dens = (inst.w[None, :] * (inst.d < esc)).sum(axis=1) / inst.w.sum()
    carga = np.bincount(a1, weights=inst.w, minlength=S.size) / inst.w.sum()
    # quanto o melhor "par" de saída custaria ao entrar k: perda do centro aberto mais próximo
    viz = np.argmin(inst.d[:, S], axis=1)
    lr_viz = lr[viz]
    f_in = np.column_stack([ga, near_open, loc, dens, lr_viz, ga - lr_viz,
                            np.full(inst.n, np.log(inst.n)), np.full(inst.n, inst.p / inst.n)])
    f_out = np.column_stack([lr, carga, loc[S], dens[S], np.full(S.size, np.log(inst.n))])
    f_in[S] = 0.0
    return f_in.astype(np.float32), f_out.astype(np.float32)


@dataclass
class Execucao:
    custo: float
    tempo: float
    iteracoes: int
    avaliacoes: int  # trocas avaliadas exatamente
    fallbacks: int


def busca_trocas(inst: PMediana, S0: np.ndarray, filtro: str = "completo", k_out: int = 4,
                 k_in: int = 16, modelo: object | None = None, tempo: float = 120.0,
                 fallback: bool = True) -> Execucao:
    """filtro: 'completo' | 'classico' | 'aprendido' | 'aleatorio'. Sem `fallback` a busca para
    quando nenhuma troca filtrada melhora (regime dos artigos: mais rápida, sem garantia de
    ótimo local da vizinhança completa)."""
    t0 = time.perf_counter()
    S = np.array(S0)
    it = aval = fb = 0
    rng = np.random.default_rng(0)
    while time.perf_counter() - t0 < tempo:
        it += 1
        a1, d1, d2 = _estado(inst, S)
        fora = np.setdiff1d(np.arange(inst.n), S)
        if filtro == "completo":
            outs, ins = np.arange(S.size), fora
        else:
            if filtro == "classico":
                s_in = gain_add(inst, d1)
                s_out = -loss_rem(inst, S, a1, d1, d2)
            elif filtro == "aleatorio":
                s_in, s_out = rng.random(inst.n), rng.random(S.size)
            else:
                f_in, f_out = atributos(inst, S, a1, d1, d2)
                s_in, s_out = modelo.score(f_in, f_out)  # type: ignore[attr-defined]
            s_in = np.asarray(s_in, dtype=float).copy()
            s_in[S] = -np.inf
            outs = np.argsort(-np.asarray(s_out))[: min(k_out, S.size)]
            ins = np.argsort(-s_in)[: min(k_in, fora.size)]
        delta = delta_trocas(inst, S, a1, d1, d2, outs, ins)
        aval += delta.size
        a, b = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[a, b] < -1e-9:
            S = S.copy()
            S[outs[a]] = ins[b]
            continue
        if filtro == "completo" or not fallback:
            break
        # fallback: uma varredura completa; se ainda não melhora, é ótimo local verdadeiro
        fb += 1
        delta = delta_trocas(inst, S, a1, d1, d2, np.arange(S.size), fora)
        aval += delta.size
        a, b = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[a, b] < -1e-9:
            S = S.copy()
            S[a] = fora[b]
            continue
        break
    return Execucao(custo(inst, S), time.perf_counter() - t0, it, aval, fb)


def coletar_imitacao(inst: PMediana, S0: np.ndarray, max_it: int = 60) -> list[tuple[np.ndarray, ...]]:
    """Trajetória da busca completa; em cada estado, rótulos = melhor ganho exato por candidato
    que entra (máx sobre saídas) e por centro que sai (máx sobre entradas)."""
    S = np.array(S0)
    out = []
    for _ in range(max_it):
        a1, d1, d2 = _estado(inst, S)
        fora = np.setdiff1d(np.arange(inst.n), S)
        delta = delta_trocas(inst, S, a1, d1, d2, np.arange(S.size), fora)
        f_in, f_out = atributos(inst, S, a1, d1, d2)
        esc = float((inst.w * d1).sum())
        y_in = np.zeros(inst.n, dtype=np.float32)
        y_in[fora] = -delta.min(axis=0) / esc
        y_out = (-delta.min(axis=1) / esc).astype(np.float32)
        out.append((f_in[fora], y_in[fora], f_out, y_out))
        a, b = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[a, b] >= -1e-9:
            break
        S = S.copy()
        S[a] = fora[b]
    return out


@dataclass
class ModeloTrocas:
    m_in: object
    m_out: object

    def score(self, f_in: np.ndarray, f_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.m_in.predict(f_in), self.m_out.predict(f_out)  # type: ignore[attr-defined]


def treinar_modelo(amostras: list[tuple[np.ndarray, ...]], seed: int = 0) -> ModeloTrocas:
    import lightgbm as lgb

    Xi = np.vstack([a[0] for a in amostras])
    yi = np.concatenate([a[1] for a in amostras])
    Xo = np.vstack([a[2] for a in amostras])
    yo = np.concatenate([a[3] for a in amostras])
    kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed,
              verbose=-1, n_jobs=1)
    return ModeloTrocas(lgb.LGBMRegressor(**kw).fit(Xi, yi), lgb.LGBMRegressor(**kw).fit(Xo, yo))


def exato_pmediana(inst: PMediana, tempo: float = 120.0, S0: np.ndarray | None = None
                   ) -> tuple[float, float, str]:
    from pyscipopt import Model

    n = inst.n
    m = Model()
    m.hideOutput()
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("limits/time", tempo)
    y = m.addMatrixVar(n, vtype="B")
    x = m.addMatrixVar((n, n), vtype="C", lb=0, ub=1)
    m.addMatrixCons(x.sum(axis=0) == 1)
    m.addMatrixCons(x - y.reshape(n, 1) <= 0)
    m.addCons(y.sum() == inst.p)
    m.setObjective((inst.d * inst.w[None, :] * x).sum())
    if S0 is not None:
        sol = m.createSol()
        for i in range(n):
            m.setSolVal(sol, y[i], 1.0 if i in set(S0.tolist()) else 0.0)
        near = S0[np.argmin(inst.d[S0], axis=0)]
        for j in range(n):
            m.setSolVal(sol, x[near[j], j], 1.0)
        m.addSol(sol, free=True)
    m.optimize()
    return float(m.getObjVal()), float(m.getDualbound()), m.getStatus()
