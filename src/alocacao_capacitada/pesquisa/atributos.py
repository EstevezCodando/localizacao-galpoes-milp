"""Atributos de centros, clientes e arestas para os ranqueadores aprendidos.

Só usam dados disponíveis no momento da decisão (custos, capacidades, demandas). Nada de ótimo,
BKS ou duais de resoluções caras (doc 07). A variante `com_pl` acrescenta a relaxação linear e
o custo dela entra no tempo online.

Invariância: atributos são normalizados por estatísticas DA PRÓPRIA instância, para que um
modelo treinado em 50x200 possa ser aplicado em 100x400 (teste de escala).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alocacao_capacitada.pesquisa.exato import Relaxacao
from alocacao_capacitada.pesquisa.problema import Problema
from alocacao_capacitada.pesquisa.tempos import cronometrar


def razao_servico(prob: Problema, order: np.ndarray | None = None) -> np.ndarray:
    """Custo médio por unidade de demanda se o centro i abrisse sozinho e atendesse seus clientes
    mais baratos (por unidade) até lotar: (f_i + sum c_ij) / sum d_j. Critério clássico do
    guloso de adição (Kuehn & Hamburger, 1963; Jacobsen, 1983). Menor é melhor."""
    unit = prob.cost / prob.demand[None, :]
    if order is None:
        order = np.argsort(unit, axis=1)
    out = np.empty(prob.m)
    for i in range(prob.m):
        d = prob.demand[order[i]]
        cum = np.cumsum(d)
        k = int(np.searchsorted(cum, prob.capacity[i], side="right"))
        k = max(k, 1)
        served = d[:k].sum()
        out[i] = (prob.fixed[i] + prob.cost[i, order[i, :k]].sum()) / served
    return out


@dataclass(frozen=True)
class Grafo:
    """Grafo bipartido centro–cliente esparsificado por k vizinhos (nos dois sentidos)."""

    x_fac: np.ndarray  # (m, F)
    x_cli: np.ndarray  # (n, G)
    src_fac: np.ndarray  # (E,) índice do centro em cada aresta
    dst_cli: np.ndarray  # (E,) índice do cliente
    x_ed: np.ndarray  # (E, H)


@cronometrar("t_atributos")
def grafo(prob: Problema, k_cli: int = 10, k_fac: int = 10, lp: Relaxacao | None = None) -> Grafo:
    m, n = prob.m, prob.n
    unit = prob.cost / prob.demand[None, :]  # custo por unidade (≈ distância)
    escala = float(np.median(unit)) or 1.0
    u = unit / escala
    dbar = float(prob.demand.mean())
    qbar = float(prob.capacity.mean()) or 1.0
    fbar = float(prob.fixed.mean()) or 1.0
    # arestas: k centros mais baratos de cada cliente ∪ k clientes mais baratos de cada centro
    kc, kf = min(k_cli, m), min(k_fac, n)
    order_f = np.argsort(u, axis=0)
    order_c = np.argsort(u, axis=1)
    near_f = order_f[:kc]  # (kc, n)
    near_c = order_c[:, :kf]  # (m, kf)
    mask = np.zeros((m, n), dtype=bool)
    mask[near_f, np.arange(n)[None, :]] = True
    mask[np.arange(m)[:, None], near_c] = True
    src, dst = np.nonzero(mask)
    rank_for_cli = np.empty_like(order_f)
    rank_for_fac = np.empty_like(order_c)
    np.put_along_axis(rank_for_cli, order_f, np.arange(m)[:, None], axis=0)
    np.put_along_axis(rank_for_fac, order_c, np.arange(n)[None, :], axis=1)
    minc = u.min(axis=0)
    sorted_u = np.take_along_axis(u, order_f, axis=0)
    regret = sorted_u[1] - sorted_u[0] if m > 1 else np.zeros(n)
    rs = razao_servico(prob, order_c) / escala
    # densidade de demanda que tem i entre seus 3 mais baratos
    top3 = order_f[: min(3, m)]
    dem_top3 = np.zeros(m)
    np.add.at(dem_top3, top3.ravel(), np.repeat(prob.demand[None, :], top3.shape[0], 0).ravel())
    # concorrência: centros com perfil de custo parecido (sem coordenadas: vale p/ Holmberg)
    from scipy.spatial.distance import cdist

    perfil = cdist(u, u, metric="cityblock") / n
    viz = np.sort(perfil, axis=1)[:, 1] if m > 1 else np.zeros(m)
    conc = (perfil < np.median(viz) * 1.5).sum(axis=1) - 1
    razao = prob.razao_capacidade
    x_fac = np.column_stack([
        prob.capacity / qbar,
        prob.fixed / fbar,
        np.divide(prob.fixed, prob.capacity, out=np.zeros(m), where=prob.capacity > 0)
        / (fbar / qbar),
        rs,
        rs / (float(rs.mean()) or 1.0),
        np.argsort(np.argsort(rs)) / m,  # posição relativa no ranking clássico
        np.take_along_axis(u, near_c, axis=1).mean(axis=1),
        dem_top3 / (prob.demand.sum() / m),
        prob.capacity / (prob.demand.sum() / m),
        conc / max(m, 1),
        np.full(m, razao),
        np.full(m, np.log(m)),
        np.full(m, np.log(n / m)),
    ])
    x_cli = np.column_stack([
        prob.demand / dbar,
        prob.demand / qbar,
        minc,
        regret,
        (u <= minc[None, :] * 1.25).sum(axis=0) / m,
    ])
    x_ed = np.column_stack([
        u[src, dst],
        u[src, dst] - minc[dst],
        rank_for_cli[src, dst] / m,
        rank_for_fac[src, dst] / n,
        np.divide(prob.demand[dst], prob.capacity[src], out=np.zeros(src.size),
                  where=prob.capacity[src] > 0),
    ])
    if lp is not None:
        x_fac = np.column_stack([x_fac, lp.y, lp.rc_y / fbar, lp.x.sum(axis=1) / n])
        x_ed = np.column_stack([x_ed, lp.x[src, dst], lp.rc_x[src, dst] / (escala * dbar)])
    # Capacidade zero permanece explícita em x_fac; razão indefinida usa sentinela zero.
    blocks = [a.astype(np.float32) for a in (x_fac, x_cli, x_ed)]
    if not all(np.isfinite(a).all() for a in blocks):
        raise ValueError("Atributos não finitos após normalização")
    return Grafo(blocks[0], blocks[1], src, dst, blocks[2])


def tabular(g: Grafo) -> np.ndarray:
    """Mesmas informações do grafo, agregadas por centro (média/mín/máx sobre os vizinhos):
    entrada do modelo tabular e do MLP (ablação H2: relação vs agregados)."""
    m = g.x_fac.shape[0]
    feats = [g.x_fac]
    nb_cli = g.x_cli[g.dst_cli]
    for block in (nb_cli, g.x_ed):
        F = block.shape[1]
        s = np.zeros((m, F))
        mx = np.full((m, F), -np.inf)
        mn = np.full((m, F), np.inf)
        cnt = np.bincount(g.src_fac, minlength=m).astype(float)[:, None]
        np.add.at(s, g.src_fac, block)
        np.maximum.at(mx, g.src_fac, block)
        np.minimum.at(mn, g.src_fac, block)
        cnt[cnt == 0] = 1
        mx[np.isinf(mx)] = 0
        mn[np.isinf(mn)] = 0
        feats += [s / cnt, mx, mn]
    feats.append(np.bincount(g.src_fac, minlength=m)[:, None] / g.x_cli.shape[0])
    return np.column_stack(feats).astype(np.float32)
