"""LNS orientado por clusters para o SSCFLP estrito (decomposição com coordenação global).

Os clusters NÃO substituem clientes por agregados: cada cliente mantém demanda e atribuição
individuais (fonte única preservada). Os grupos só organizam a busca:

  1. solução global viável (construtivo guloso com custo fixo rateado);
  2. grupos por compatibilidade de perfil de custo (k-means sobre os custos unitários aos centros
     disponíveis), com tamanho-alvo em clientes;
  3. movimentos, sempre com o RESTANTE FIXO e capacidade RESIDUAL descontada:
       interior   reotimiza um grupo;
       fronteira  reotimiza a união de dois grupos vizinhos (transferências entre grupos);
       liberacao  libera todos os clientes de 1-2 centros caros e dos centros vizinhos, permitindo
                  fechar um centro cujos clientes estariam presos no interior de grupos.
     O custo fixo de um centro já usado por clientes fixos está pago (sem variável y); só centros
     sem nenhum cliente fixo têm y no subproblema. Logo o custo fixo é cobrado uma única vez no
     global e a união dos subproblemas é sempre viável (o subproblema vê a capacidade residual).

Aceita só melhoria estrita; a solução global permanece viável a cada passo (validada no fim).
Relógio único: todos os tempos são medidos de `t0` = início da chamada, inclusive construção,
agrupamento, montagem dos subproblemas e validação; nenhum subproblema começa após o prazo.

Referências: Ahuja, Orlin, Pallottino, Scaparra & Scutellà (2004), multi-exchange para o SSCFLP;
Guastaroba & Speranza (2014), kernel search; Kong (2021), matheurística LNS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pyscipopt import Model
from scipy.spatial.distance import cdist

from alocacao_capacitada.pesquisa.problema import Problema, validar

_EPS = 1e-6


@dataclass
class SaidaDecomp:
    atribuicao: np.ndarray
    objetivo: float
    valida: bool
    t_total: float
    trajetoria: list[tuple[float, float]] = field(default_factory=list)
    movimentos: dict[str, list[int]] = field(default_factory=dict)  # tipo -> [tentados, aceitos]
    t_sub: list[float] = field(default_factory=list)  # tempo de cada subproblema
    n_grupos: int = 0
    estouro: float = 0.0  # tempo além do prazo (chamadas nativas não são preemptíveis)


def construtivo(prob: Problema) -> np.ndarray:
    """Guloso: clientes em ordem decrescente de demanda; cada um vai ao centro com menor
    c_ij + f_i * d_j / Q_i (fixo rateado se ainda fechado) que ainda comporta d_j.
    Se ninguém comporta, falha explicitamente; isso não prova inviabilidade do problema."""
    m, n = prob.m, prob.n
    resid = prob.capacity.astype(float).copy()
    aberto = np.zeros(m, dtype=bool)
    a = np.full(n, -1, dtype=np.int64)
    for j in np.argsort(-prob.demand, kind="stable"):
        custo = prob.cost[:, j] + np.where(aberto, 0.0, prob.fixed * prob.demand[j] /
                                           np.maximum(prob.capacity, _EPS))
        custo[resid < prob.demand[j] - _EPS] = np.inf
        i = int(np.argmin(custo))
        if not np.isfinite(custo[i]):
            raise RuntimeError("construtivo: nenhum centro comporta o cliente (reparo necessário)")
        a[j] = i
        resid[i] -= prob.demand[j]
        aberto[i] = True
    return a


def grupos(prob: Problema, tamanho: int, seed: int = 0) -> list[np.ndarray]:
    """K-means sobre o perfil de custo unitário a todos os centros."""
    from sklearn.cluster import KMeans

    if tamanho <= 0:
        raise ValueError("tamanho deve ser positivo")
    unit = prob.cost / prob.demand[None, :]
    k = max(1, int(round(prob.n / tamanho)))
    if k == 1:
        return [np.arange(prob.n)]
    escala = float(np.median(unit))
    X = np.log1p(unit.T / (escala if escala > 0 else 1.0))  # (n, m)
    k = min(k, np.unique(X, axis=0).shape[0])
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(X)
    return [np.flatnonzero(lab == g) for g in range(k) if (lab == g).any()]


def _vizinhos_grupos(prob: Problema, gs: list[np.ndarray]) -> np.ndarray:
    """Distância entre grupos = distância entre perfis médios de custo (para a fronteira)."""
    unit = prob.cost / prob.demand[None, :]
    cent = np.stack([unit[:, g].mean(axis=1) for g in gs])
    d = cdist(cent, cent, metric="cityblock")
    np.fill_diagonal(d, np.inf)
    return d


def reotimizar(prob: Problema, a: np.ndarray, livres: np.ndarray, tempo: float,
               k_cand: int = 10) -> np.ndarray | None:
    """Reotimiza os clientes `livres` com o restante fixo. Candidatos de cada livre: seus k_cand
    centros mais baratos + o centro atual (garante que a solução atual é viável e é dada como
    dica). Devolve a atribuição completa ou None."""
    prazo = time.perf_counter() + tempo
    if not np.isfinite(tempo) or tempo < 0:
        raise ValueError("tempo deve ser finito e não negativo")
    if not validar(prob, a).valida:
        raise ValueError("a atribuição inicial deve ser viável")
    a = np.asarray(a, dtype=np.int64)
    livres = np.asarray(livres)
    if (livres.ndim != 1 or not np.issubdtype(livres.dtype, np.integer)
            or np.any(livres < 0) or np.any(livres >= prob.n)
            or np.unique(livres).size != livres.size):
        raise ValueError("livres deve conter índices inteiros distintos e válidos")
    if k_cand < 1:
        raise ValueError("k_cand deve ser positivo")
    if livres.size == 0 or time.perf_counter() >= prazo:
        return a.copy()
    m = prob.m
    fixos = np.ones(prob.n, dtype=bool)
    fixos[livres] = False
    carga_fixa = np.bincount(a[fixos], weights=prob.demand[fixos], minlength=m)
    pago = np.bincount(a[fixos], minlength=m) > 0
    resid = prob.capacity - carga_fixa
    near = np.argsort(prob.cost[:, livres], axis=0)[: min(k_cand, m)]
    pares = {(int(i), int(j)) for col, j in enumerate(livres) for i in near[:, col]}
    pares |= {(int(a[j]), int(j)) for j in livres}
    md = Model()
    md.hideOutput()
    md.setParam("parallel/maxnthreads", 1)
    md.setParam("timing/clocktype", 2)
    x = {p: md.addVar(vtype="B") for p in pares}
    usados = {i for i, _ in pares}
    y = {i: md.addVar(vtype="B") for i in usados if not pago[i]}
    por_cli: dict[int, list] = {}
    por_fac: dict[int, list] = {}
    for (i, j), v in x.items():
        por_cli.setdefault(j, []).append(v)
        por_fac.setdefault(i, []).append((j, v))
    for j in livres:
        md.addCons(sum(por_cli[int(j)]) == 1)
    for i, lst in por_fac.items():
        carga = sum(float(prob.demand[j]) * v for j, v in lst)
        md.addCons(carga <= float(resid[i]) * (y[i] if i in y else 1))
        if i in y:
            for _, v in lst:
                md.addCons(v <= y[i])
    md.setObjective(sum(float(prob.fixed[i]) * v for i, v in y.items())
                    + sum(float(prob.cost[i, j]) * v for (i, j), v in x.items()))
    sol = md.createSol()
    for (i, j), v in x.items():
        md.setSolVal(sol, v, 1.0 if a[j] == i else 0.0)
    abertos_livres = {int(a[j]) for j in livres}
    for i, v in y.items():
        md.setSolVal(sol, v, 1.0 if i in abertos_livres else 0.0)
    md.addSol(sol, free=True)
    restante = prazo - time.perf_counter()
    if restante <= 0:
        return a.copy()
    md.setParam("limits/time", restante)
    md.optimize()
    if md.getNSols() == 0:
        return None
    best = md.getBestSol()
    out = a.copy()
    for (i, j), v in x.items():
        if md.getSolVal(best, v) > 0.5:
            out[j] = i
    return out


def lns_clusters(prob: Problema, tempo: float, tamanho: int = 30, t_sub: float = 3.0,
                 seed: int = 0, movimentos: tuple[str, ...] = ("interior", "fronteira",
                                                                "liberacao"),
                 inicial: np.ndarray | None = None) -> SaidaDecomp:
    t0 = time.perf_counter()
    prazo = t0 + tempo
    rng = np.random.default_rng(seed)
    a = construtivo(prob) if inicial is None else np.asarray(inicial, dtype=np.int64).copy()
    obj = validar(prob, a).objetivo
    traj = [(time.perf_counter() - t0, obj)]
    gs = grupos(prob, tamanho, seed)
    viz = _vizinhos_grupos(prob, gs) if len(gs) > 1 else None
    cont = {mv: [0, 0] for mv in movimentos}
    tsub: list[float] = []
    k = 0
    falhas_seguidas = 0
    while True:
        restante = prazo - time.perf_counter()
        if restante <= 0.05:  # não inicia subproblema sem orçamento
            break
        mv = movimentos[k % len(movimentos)]
        k += 1
        if mv == "interior":
            livres = gs[int(rng.integers(len(gs)))]
        elif mv == "fronteira" and viz is not None:
            g = int(rng.integers(len(gs)))
            h = int(np.argsort(viz[g])[int(rng.integers(min(2, len(gs) - 1)))])
            livres = np.union1d(gs[g], gs[h])
        elif mv == "liberacao":
            abertos = np.unique(a)
            carga = np.bincount(a, weights=prob.demand, minlength=prob.m)[abertos]
            # custo fixo por unidade atendida: candidatos caros a fechar (sorteio ponderado)
            peso = prob.fixed[abertos] / np.maximum(carga, _EPS)
            alvo = rng.choice(abertos, size=min(2, abertos.size), replace=False,
                              p=(peso / peso.sum() if np.all(peso > 0) else None))
            unit = prob.cost / prob.demand[None, :]
            perfil = np.abs(unit[abertos][:, None, :] - unit[alvo][None, :, :]).sum(axis=2)
            proximos = abertos[np.argsort(perfil.min(axis=1))[: min(4, abertos.size)]]
            livres = np.flatnonzero(np.isin(a, np.union1d(alvo, proximos)))
            limite = int(1.5 * tamanho)
            if livres.size > 2 * limite:  # mantém os clientes dos alvos e amostra o resto
                dos_alvos = np.flatnonzero(np.isin(a, alvo))
                resto = np.setdiff1d(livres, dos_alvos)
                livres = np.union1d(dos_alvos, rng.choice(resto, size=min(resto.size, limite),
                                                          replace=False))
        else:
            continue
        cont[mv][0] += 1
        ts = time.perf_counter()
        orc = min(t_sub * (1 + 0.5 * min(falhas_seguidas, 4)), prazo - ts - 0.02)
        if orc <= 0.02:
            break
        cand = reotimizar(prob, a, livres, orc)
        tsub.append(time.perf_counter() - ts)
        if cand is not None:
            v = validar(prob, cand)
            if v.valida and v.objetivo < obj - _EPS:
                a, obj = cand, v.objetivo
                cont[mv][1] += 1
                traj.append((time.perf_counter() - t0, obj))
                falhas_seguidas = 0
                continue
        falhas_seguidas += 1
    v = validar(prob, a)
    t_total = time.perf_counter() - t0
    return SaidaDecomp(a, v.objetivo, v.valida, t_total, traj, cont, tsub, len(gs),
                       max(0.0, t_total - tempo))


@dataclass
class SaidaIngenua:
    soma_subproblemas: float  # o que a decomposição "acredita" custar
    custo_uniao: float  # custo real da união (fixo cobrado uma vez), sem checar capacidade
    viavel_uniao: bool
    centros_compartilhados: int  # centros abertos por mais de um grupo
    sobrecarga: float  # demanda acima da capacidade na união
    atribuicao_reparada: np.ndarray | None
    custo_reparado: float
    t_total: float


def decomposicao_ingenua(prob: Problema, tempo: float, tamanho: int = 30,
                         seed: int = 0) -> SaidaIngenua:
    """Cada grupo resolvido isoladamente (todos os centros, capacidade CHEIA, custo fixo
    próprio) e as soluções unidas. Mostra por que a coordenação global é necessária (ponto 3 da
    revisão): a união pode violar capacidade e a soma dos subproblemas cobra o fixo em dobro.
    Reparo: realoca clientes dos centros sobrecarregados, do maior para o menor, ao centro mais
    barato com capacidade residual (abrindo se preciso)."""
    t0 = time.perf_counter()
    gs = grupos(prob, tamanho, seed)
    a = np.full(prob.n, -1, dtype=np.int64)
    soma = 0.0
    abertos_por_grupo = []
    orc = tempo / max(len(gs), 1)
    for g in gs:
        sub = Problema(nome="g", familia="g", fixed=prob.fixed, capacity=prob.capacity,
                       demand=prob.demand[g], cost=prob.cost[:, g])
        ini = construtivo(sub)
        livres = np.arange(g.size)
        # subproblema isolado = reotimizar todos os clientes do grupo, sem clientes fixos
        sol = reotimizar(sub, ini, livres, orc, k_cand=prob.m) if orc > 0.05 else ini
        sol = ini if sol is None else sol
        a[g] = sol
        soma += validar(sub, sol).objetivo
        abertos_por_grupo.append(set(np.unique(sol).tolist()))
    cont = np.zeros(prob.m, dtype=int)
    for s in abertos_por_grupo:
        for i in s:
            cont[i] += 1
    carga = np.bincount(a, weights=prob.demand, minlength=prob.m)
    sobre = float(np.maximum(carga - prob.capacity, 0).sum())
    v = validar(prob, a)
    custo_uniao = v.objetivo
    rep = a.copy()
    resid = prob.capacity - carga
    ok = True
    for i in np.flatnonzero(resid < -_EPS):
        clientes = np.flatnonzero(rep == i)
        for j in clientes[np.argsort(-prob.demand[clientes])]:
            if resid[i] >= -_EPS:
                break
            aberto = np.bincount(rep, minlength=prob.m) > 0
            custo = prob.cost[:, j] + np.where(aberto, 0.0, prob.fixed)
            custo[resid < prob.demand[j] - _EPS] = np.inf
            custo[i] = np.inf
            k = int(np.argmin(custo))
            if not np.isfinite(custo[k]):
                ok = False
                break
            rep[j] = k
            resid[k] -= prob.demand[j]
            resid[i] += prob.demand[j]
    vr = validar(prob, rep)
    return SaidaIngenua(soma, custo_uniao, v.valida, int((cont > 1).sum()), sobre,
                        rep if ok and vr.valida else None,
                        vr.objetivo if ok and vr.valida else np.inf, time.perf_counter() - t0)
