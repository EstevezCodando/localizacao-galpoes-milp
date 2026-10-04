"""Ótimo por enumeração exaustiva das atribuições (m^n), independente de qualquer solver.

Serve só para microinstâncias (doc 12, sprint 2: "microinstâncias concordam com enumeração").
"""

from __future__ import annotations

import numpy as np

from alocacao_capacitada.pesquisa.problema import Problema


def otimo_enumerado(prob: Problema, limite: int = 2_000_000) -> tuple[float, np.ndarray | None]:
    m, n = prob.m, prob.n
    total = m**n
    if total > limite:
        raise ValueError(f"{total} atribuições: grande demais para enumerar")
    melhor, arg = np.inf, None
    bloco = 200_000
    for ini in range(0, total, bloco):
        cod = np.arange(ini, min(ini + bloco, total))
        a = np.empty((cod.size, n), dtype=np.int64)
        r = cod.copy()
        for j in range(n):
            a[:, j] = r % m
            r //= m
        carga = np.zeros((cod.size, m))
        usado = np.zeros((cod.size, m), dtype=bool)
        atend = np.zeros(cod.size)
        for j in range(n):
            np.add.at(carga, (np.arange(cod.size), a[:, j]), prob.demand[j])
            usado[np.arange(cod.size), a[:, j]] = True
            atend += prob.cost[a[:, j], j]
        ok = (carga <= prob.capacity[None, :] + 1e-9).all(axis=1)
        obj = atend + usado.astype(float) @ prob.fixed
        obj[~ok] = np.inf
        k = int(np.argmin(obj))
        if obj[k] < melhor:
            melhor, arg = float(obj[k]), a[k].copy()
    return melhor, arg
