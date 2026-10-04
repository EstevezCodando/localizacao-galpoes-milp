"""Geradores de instâncias sintéticas controladas (doc 06, camada B).

Base de custos: Cornuéjols, Sridharan & Thizy (1991), a mesma usada pelo gerador `facilities`
do learn2branch (Gasse et al., 2019), adaptada para fonte única:

  demanda d_j ~ U(5, 35); capacidade bruta s_i ~ U(10, 160), reescalada para sum(Q) = r * sum(d);
  custo fixo f_i = (U(0, 90) + U(100, 110) * sqrt(Q_i)) * escala_fixo;
  custo total c_ij = 10 * dist_ij * d_j   (dist euclidiana no quadrado unitário).

Famílias espaciais: `uniforme`, `clusters` (mistura gaussiana) e `corredor` (pontos ao longo
de poucos segmentos, imitando eixos viários). `assimetrico=True` multiplica cada par por um
fator de desvio viário em [1.1, 1.8] com correlação espacial (custo deixa de ser métrico puro).

Viabilidade de fonte única não é garantida por sum(Q) >= sum(d) (doc 05): cada instância passa
por um teste FFD; se o FFD falha, `viavel_fonte_unica` resolve um MILP de viabilidade curto.
Instâncias inviáveis ou indeterminadas são REJEITADAS e contadas (viés registrado).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alocacao_capacitada.pesquisa.problema import Problema

FAMILIAS = ("uniforme", "clusters", "corredor")


@dataclass(frozen=True)
class Config:
    familia: str
    m: int
    n: int
    razao: float  # capacidade total / demanda total
    escala_fixo: float = 1.0
    assimetrico: bool = False


def _pontos(familia: str, k: int, rng: np.random.Generator, centros: np.ndarray) -> np.ndarray:
    if familia == "uniforme":
        return rng.uniform(0, 1, (k, 2))
    if familia == "clusters":
        lab = rng.integers(len(centros), size=k)
        return np.clip(centros[lab] + rng.normal(0, 0.06, (k, 2)), 0, 1)
    if familia == "corredor":
        # centros[2s], centros[2s+1] são extremos de cada segmento
        seg = rng.integers(len(centros) // 2, size=k)
        t = rng.uniform(0, 1, k)[:, None]
        a, b = centros[2 * seg], centros[2 * seg + 1]
        return np.clip(a + t * (b - a) + rng.normal(0, 0.025, (k, 2)), 0, 1)
    raise ValueError(f"família desconhecida: {familia}")


def _ffd(demand: np.ndarray, capacity: np.ndarray) -> bool:
    """First-fit decreasing (na ordem de capacidade decrescente): suficiente, não necessário."""
    resid = np.sort(capacity)[::-1].astype(float)
    for d in np.sort(demand)[::-1]:
        ok = np.flatnonzero(resid >= d - 1e-9)
        if ok.size == 0:
            return False
        # best fit: menor resíduo que cabe
        k = ok[np.argmin(resid[ok])]
        resid[k] -= d
    return True


def viavel_fonte_unica(demand: np.ndarray, capacity: np.ndarray, tempo_s: float = 5.0) -> bool | None:
    """True/False quando decidido; None se o MILP de viabilidade não concluir no tempo."""
    if capacity.sum() < demand.sum() - 1e-9 or demand.max() > capacity.max() + 1e-9:
        return False
    if _ffd(demand, capacity):
        return True
    from pyscipopt import Model

    m, n = capacity.size, demand.size
    model = Model()
    model.hideOutput()
    model.setParam("limits/time", tempo_s)
    model.setParam("parallel/maxnthreads", 1)
    x = model.addMatrixVar((m, n), vtype="B")
    model.addMatrixCons(x.sum(axis=0) == 1)
    model.addMatrixCons((x * demand[None, :]).sum(axis=1) <= capacity)
    model.setObjective(0)
    model.optimize()
    status = model.getStatus()
    if model.getNSols() > 0:
        return True
    if status == "infeasible":
        return False
    return None


def gerar(cfg: Config, seed: int, max_tentativas: int = 50) -> tuple[Problema, int]:
    """Devolve a instância e o número de rejeições até obtê-la."""
    rejeitadas = 0
    for tentativa in range(max_tentativas):
        rng = np.random.default_rng([seed, tentativa])
        if cfg.familia == "clusters":
            base = rng.uniform(0.1, 0.9, (int(rng.integers(3, 7)), 2))
        elif cfg.familia == "corredor":
            base = rng.uniform(0, 1, (2 * int(rng.integers(2, 4)), 2))
        else:
            base = np.zeros((1, 2))
        xy_cli = _pontos(cfg.familia, cfg.n, rng, base)
        # metade dos candidatos segue a distribuição dos clientes, metade é uniforme
        k = cfg.m // 2
        xy_fac = np.vstack([_pontos(cfg.familia, k, rng, base), rng.uniform(0, 1, (cfg.m - k, 2))])
        rng.shuffle(xy_fac)
        demand = rng.uniform(5, 35, cfg.n)
        raw = rng.uniform(10, 160, cfg.m)
        capacity = raw * cfg.razao * demand.sum() / raw.sum()
        fixed = (rng.uniform(0, 90, cfg.m) + rng.uniform(100, 110, cfg.m) * np.sqrt(capacity))
        fixed = fixed * cfg.escala_fixo
        dist = np.linalg.norm(xy_fac[:, None, :] - xy_cli[None, :, :], axis=2)
        if cfg.assimetrico:
            # desvio viário espacialmente correlacionado: função suave das coordenadas
            phase = rng.uniform(0, 2 * np.pi, 4)
            def _campo(p: np.ndarray) -> np.ndarray:
                return 0.5 + 0.5 * np.sin(6 * p[:, 0] + phase[0]) * np.cos(5 * p[:, 1] + phase[1])
            detour = 1.1 + 0.7 * np.sqrt(np.outer(_campo(xy_fac), _campo(xy_cli)))
            dist = dist * detour
        cost = 10.0 * dist * demand[None, :]
        viavel = viavel_fonte_unica(demand, capacity)
        if viavel is not True:
            rejeitadas += 1
            continue
        nome = f"{cfg.familia}{'_asm' if cfg.assimetrico else ''}_m{cfg.m}_n{cfg.n}_r{cfg.razao}_s{seed}"
        prob = Problema(
            nome=nome,
            familia=cfg.familia + ("_asm" if cfg.assimetrico else ""),
            fixed=fixed,
            capacity=capacity,
            demand=demand,
            cost=cost,
            xy_fac=xy_fac,
            xy_cli=xy_cli,
            meta={"razao": cfg.razao, "seed": seed, "tentativa": tentativa,
                  "escala_fixo": cfg.escala_fixo},
        )
        return prob, rejeitadas
    raise RuntimeError(f"nenhuma instância viável em {max_tentativas} tentativas: {cfg}")
