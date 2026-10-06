"""CLNS — LNS orientado por clusters com seleção de subproblemas, para o SSCFLP estrito.

Núcleo do artigo "Learning to Delegate in Capacitated Facility Location". A cada iteração:

  1. gera um conjunto de CANDIDATOS a subproblema a partir do estado atual:
       interior   clientes de um grupo (k-means sobre perfis de custo);
       fronteira  união de dois grupos vizinhos (transferências entre grupos);
       liberacao  clientes de um centro aberto caro e dos centros de perfil parecido (permite
                  fechar centros cujos clientes estariam presos no interior de grupos);
       regret     clientes vizinhos do cliente de maior "regret" dual = custo reduzido, no PL
                  forte, da atribuição atual x_{a(j)j} (inclui duais de atendimento, capacidade e
                  ligação; >= 0 no ótimo do PL). Obs.: c_ij - u_j com u só da linha de
                  atendimento NÃO serve: u embute o rateio do custo fixo e c - u < 0 sempre;
  2. um SELETOR escolhe um candidato;
  3. o subproblema é reotimizado por SCIP com o restante FIXO e capacidade RESIDUAL
     (decomposicao.reotimizar): a união é sempre viável e o custo fixo é cobrado uma única vez;
  4. aceita só melhoria estrita.

Seletores:
  rotacao    percorre os tipos em ciclo, candidato aleatório dentro do tipo;
  aleatorio  candidato uniforme;
  alns       roleta por tipo com pesos adaptativos (Ropke & Pisinger, 2006): baseline clássico
             obrigatório de qualquer seletor aprendido;
  dual       maior regret dual por cliente livre, com tabu dos últimos tentados;
  aprendido  modelo prevê o ganho por segundo de cada candidato (estilo Learning to Delegate,
             Li, Yan & Wu 2021); escolhe o máximo com exploração epsilon.

Relógio único a partir de t0 (inclui PL, agrupamento, construção, subproblemas e validação);
nenhum subproblema começa sem orçamento.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from hashlib import blake2b

import numpy as np
from scipy.spatial.distance import cdist

from alocacao_capacitada.pesquisa.decomposicao import (
    _vizinhos_grupos,
    construtivo,
    grupos,
    reotimizar,
)
from alocacao_capacitada.pesquisa.exato import relaxacao_linear
from alocacao_capacitada.pesquisa.problema import Problema, validar

_EPS = 1e-6
TIPOS = ("interior", "fronteira", "liberacao", "regret")
N_ATRIB = 16


@dataclass
class Candidato:
    tipo: str
    chave: str
    livres: np.ndarray


@dataclass
class SaidaClns:
    atribuicao: np.ndarray
    objetivo: float
    valida: bool
    t_total: float
    trajetoria: list[tuple[float, float]] = field(default_factory=list)
    tentados: dict[str, int] = field(default_factory=dict)
    aceitos: dict[str, int] = field(default_factory=dict)
    t_inicial: float = 0.0
    t_pl: float = 0.0
    t_selecao: float = 0.0
    t_sub: float = 0.0
    iteracoes: int = 0
    estouro: float = 0.0
    repetidos: int = 0  # subproblemas escolhidos que já tinham falhado no MESMO estado local
    pulados: int = 0  # candidatos descartados pela memória antes da seleção


# ------------------------------------------------------------------ solução inicial
def inicial_pl(prob: Problema, x_lp: np.ndarray) -> np.ndarray:
    """Arredondamento guiado pelo PL: clientes em ordem decrescente de demanda vão ao centro de
    maior x_ij do PL que ainda comporta (desempate pelo custo). Fallback: construtivo."""
    resid = prob.capacity.astype(float).copy()
    a = np.full(prob.n, -1, dtype=np.int64)
    for j in np.argsort(-prob.demand, kind="stable"):
        pref = np.lexsort((prob.cost[:, j], -x_lp[:, j]))
        for i in pref:
            if resid[i] >= prob.demand[j] - _EPS:
                a[j] = i
                resid[i] -= prob.demand[j]
                break
        if a[j] < 0:
            return construtivo(prob)
    return a


# ------------------------------------------------------------------ candidatos e atributos
class Estado:
    """Dados fixos da instância + estatísticas de cada candidato ao longo da busca."""

    def __init__(self, prob: Problema, tamanho: int, seed: int, rc: np.ndarray | None,
                 gnn: np.ndarray | None = None, agrupamento: str = "perfil") -> None:
        self.prob = prob
        self.gnn = gnn  # probabilidade (GNN) de cada centro estar numa boa solução, ou None
        self.unit = prob.cost / prob.demand[None, :]
        self.gs = grupos(prob, tamanho, seed, agrupamento)
        self.viz = _vizinhos_grupos(prob, self.gs) if len(self.gs) > 1 else None
        self.rc = rc  # custos reduzidos de x_ij no PL forte, shape (m, n) (ou None)
        self.tamanho = tamanho
        self.ordem_cli = np.argsort(cdist(self.unit.T, self.unit.T, metric="cityblock"),
                                   axis=1) if prob.n <= 600 else None
        self.hist: dict[str, list[float]] = {}  # chave -> [tentativas, ganho total, última it]
        self.escala = max(float(np.median(prob.cost)), _EPS)

    def regret(self, a: np.ndarray) -> np.ndarray:
        idx = np.arange(self.prob.n)
        if self.rc is None:  # sem PL: excesso sobre o centro mais barato
            return self.prob.cost[a, idx] - self.prob.cost.min(axis=0)
        return self.rc[a, idx] - self.rc.min(axis=0)

    def vizinhos_cliente(self, j: int, k: int) -> np.ndarray:
        if self.ordem_cli is not None:
            return self.ordem_cli[j, :k]
        d = np.abs(self.unit - self.unit[:, [j]]).sum(axis=0)
        return np.argsort(d)[:k]

    def candidatos(self, a: np.ndarray, rng: np.random.Generator) -> list[Candidato]:
        out: list[Candidato] = []
        for g, membros in enumerate(self.gs):
            out.append(Candidato("interior", f"i{g}", membros))
        if self.viz is not None:
            for g in range(len(self.gs)):
                h = int(np.argmin(self.viz[g]))
                if g < h or int(np.argmin(self.viz[h])) != g:
                    out.append(Candidato("fronteira", f"f{min(g, h)}-{max(g, h)}",
                                         np.union1d(self.gs[g], self.gs[h])))
        abertos = np.unique(a)
        carga = np.bincount(a, weights=self.prob.demand, minlength=self.prob.m)
        custo_un = self.prob.fixed[abertos] / np.maximum(carga[abertos], _EPS)
        perfil = cdist(self.unit[abertos], self.unit[abertos], metric="cityblock")
        for k in np.argsort(-custo_un)[: min(6, abertos.size)]:
            i = abertos[k]
            prox = abertos[np.argsort(perfil[k])[: min(3, abertos.size)]]
            livres = np.flatnonzero(np.isin(a, prox))
            if livres.size > 3 * self.tamanho:
                dos_alvo = np.flatnonzero(a == i)
                resto = np.setdiff1d(livres, dos_alvo)
                livres = np.union1d(dos_alvo, rng.choice(resto, size=min(resto.size,
                                                                          2 * self.tamanho),
                                                         replace=False))
            out.append(Candidato("liberacao", f"l{i}", livres))
        reg = self.regret(a)
        for j in np.argsort(-reg)[:4]:
            out.append(Candidato("regret", f"r{j}", self.vizinhos_cliente(int(j), self.tamanho)))
        if self.gnn is not None:
            # abertura: centros FECHADOS que a GNN acha que deveriam abrir; libera os clientes
            # mais baratos para eles (só existe quando há ranking da GNN)
            fechados = np.setdiff1d(np.arange(self.prob.m), abertos)
            for i in fechados[np.argsort(-self.gnn[fechados])][:4]:
                out.append(Candidato("abertura", f"a{i}",
                                     np.argsort(self.prob.cost[i])[: self.tamanho]))
        return out

    def discordancia(self, c: Candidato, a: np.ndarray) -> float:
        """Quanto a configuração local discorda da GNN: centros abertos que ela considera
        improváveis + centros fechados próximos que ela considera prováveis, por cliente livre."""
        assert self.gnn is not None
        abertos_loc = np.unique(a[c.livres])
        perto = np.unique(np.argsort(self.prob.cost[:, c.livres], axis=0)[:3].ravel())
        fechados_loc = np.setdiff1d(perto, np.unique(a))
        return float(((1 - self.gnn[abertos_loc]).sum() + self.gnn[fechados_loc].sum())
                     / max(len(abertos_loc) + len(fechados_loc), 1))

    def atributos(self, c: Candidato, a: np.ndarray, it: int, reg: np.ndarray) -> np.ndarray:
        p = self.prob
        L = c.livres
        fac = np.unique(a[L])
        carga = np.bincount(a, weights=p.demand, minlength=p.m)
        folga = (p.capacity[fac] - carga[fac]).sum() / max(p.demand[L].sum(), _EPS)
        h = self.hist.get(c.chave, [0.0, 0.0, -1.0])
        tipo = np.zeros(len(TIPOS))
        if c.tipo in TIPOS:  # 'abertura' (só com GNN) fica fora do vocabulário do seletor treinado
            tipo[TIPOS.index(c.tipo)] = 1
        return np.concatenate([[
            L.size / p.n,
            p.demand[L].sum() / p.demand.sum(),
            p.cost[a[L], L].sum() / (self.escala * max(L.size, 1)),
            np.maximum(reg[L], 0).sum() / (self.escala * max(L.size, 1)),
            np.maximum(reg[L], 0).max() / self.escala if L.size else 0.0,
            fac.size,
            folga,
            (p.fixed[fac] / np.maximum(carga[fac], _EPS)).mean() /
            max(p.fixed.mean() / max(p.capacity.mean(), _EPS), _EPS),
            h[0],
            h[1] / (self.escala * p.n),
            (it - h[2]) / 10.0 if h[2] >= 0 else 10.0,
            p.razao_capacidade,
        ], tipo]).astype(np.float32)


# ------------------------------------------------------------------ seletores
class Seletor:
    nome = "base"

    def escolher(self, cands: list[Candidato], est: Estado, a: np.ndarray, it: int,
                 rng: np.random.Generator) -> int:
        raise NotImplementedError

    def retorno(self, c: Candidato, ganho: float, tempo: float) -> None:
        return None


class Rotacao(Seletor):
    nome = "rotacao"

    def __init__(self) -> None:
        self.k = 0

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        for _ in range(len(TIPOS)):
            tipo = TIPOS[self.k % len(TIPOS)]
            self.k += 1
            idx = [i for i, c in enumerate(cands) if c.tipo == tipo]
            if idx:
                return int(rng.choice(idx))
        return int(rng.integers(len(cands)))


class Aleatorio(Seletor):
    nome = "aleatorio"

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        return int(rng.integers(len(cands)))


class Alns(Seletor):
    """Roleta por tipo; peso_t <- (1 - r) peso_t + r * recompensa média do segmento
    (Ropke & Pisinger, 2006), recompensa = 1 se melhorou, 0 caso contrário, segmentos de 10."""

    nome = "alns"

    def __init__(self, r: float = 0.2, segmento: int = 10) -> None:
        self.w = {t: 1.0 for t in (*TIPOS, "abertura")}
        self.acum = {t: [0.0, 0] for t in (*TIPOS, "abertura")}
        self.r, self.seg, self.n = r, segmento, 0

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        disp = sorted({c.tipo for c in cands})
        p = np.array([self.w[t] for t in disp])
        tipo = disp[int(rng.choice(len(disp), p=p / p.sum()))]
        idx = [i for i, c in enumerate(cands) if c.tipo == tipo]
        return int(rng.choice(idx))

    def retorno(self, c, ganho, tempo):  # type: ignore[no-untyped-def]
        s = self.acum[c.tipo]
        s[0] += 1.0 if ganho > _EPS else 0.0
        s[1] += 1
        self.n += 1
        if self.n % self.seg == 0:
            for t in self.acum:
                tot, cnt = self.acum[t]
                if cnt:
                    self.w[t] = max(0.05, (1 - self.r) * self.w[t] + self.r * tot / cnt)
                self.acum[t] = [0.0, 0]


class Dual(Seletor):
    """Maior regret dual positivo por cliente livre, com decaimento por fracasso: o score de
    um candidato é dividido por 2^(tentativas sem ganho desde a última melhora). Sem isso o
    seletor insiste nos mesmos grupos (observado no piloto: 6% contra 2,2% do ALNS)."""

    nome = "dual"

    def __init__(self) -> None:
        self.falhas: dict[str, int] = {}

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        reg = np.maximum(est.regret(a), 0)
        sc = np.array([reg[c.livres].mean() / 2.0 ** self.falhas.get(c.chave, 0)
                       for c in cands])
        sc += 1e-9 * rng.random(len(cands))  # desempate aleatório
        return int(np.argmax(sc))

    def retorno(self, c, ganho, tempo):  # type: ignore[no-untyped-def]
        if ganho > _EPS:
            self.falhas.clear()  # o estado mudou: todos voltam a ser elegíveis
        else:
            self.falhas[c.chave] = self.falhas.get(c.chave, 0) + 1


class Aprendido(Seletor):
    """Escolhe o candidato de maior ganho-por-segundo previsto; epsilon de exploração."""

    nome = "aprendido"

    def __init__(self, modelo: object, eps: float = 0.05) -> None:
        self.modelo, self.eps = modelo, eps
        self.t_inf = 0.0

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        if rng.random() < self.eps:
            return int(rng.integers(len(cands)))
        t0 = time.perf_counter()
        reg = est.regret(a)
        X = np.stack([est.atributos(c, a, it, reg) for c in cands])
        k = int(np.argmax(self.modelo.predict(X)))  # type: ignore[attr-defined]
        self.t_inf += time.perf_counter() - t0
        return k


class GuiadoGnn(Seletor):
    """Ataca primeiro onde a solução atual mais discorda do ranking da GNN (centros abertos
    improváveis, centros fechados prováveis por perto), com o mesmo decaimento por fracasso do
    seletor dual. Sem treino adicional: reaproveita a GNN da Etapa 1."""

    nome = "gnn"

    def __init__(self) -> None:
        self.falhas: dict[str, int] = {}

    def escolher(self, cands, est, a, it, rng):  # type: ignore[no-untyped-def]
        sc = np.array([est.discordancia(c, a) / 2.0 ** self.falhas.get(c.chave, 0)
                       for c in cands]) + 1e-9 * rng.random(len(cands))
        return int(np.argmax(sc))

    def retorno(self, c, ganho, tempo):  # type: ignore[no-untyped-def]
        if ganho > _EPS:
            self.falhas.clear()
        else:
            self.falhas[c.chave] = self.falhas.get(c.chave, 0) + 1


SELETORES = {"rotacao": Rotacao, "aleatorio": Aleatorio, "alns": Alns, "dual": Dual,
             "gnn": GuiadoGnn}


# ------------------------------------------------------------------ memória de subproblemas
def chave_subproblema(prob: Problema, a: np.ndarray, livres: np.ndarray) -> bytes:
    """Identidade do subproblema que `reotimizar` monta: ele depende só dos clientes livres,
    dos centros atuais deles (entram nos candidatos e na dica) e da carga fixa por centro (define
    capacidade residual e custo fixo já pago). Mesma chave => mesmo modelo; se já falhou com
    orçamento igual ou maior, repetir só gasta tempo (tabela de transposição da busca)."""
    liv = np.sort(np.asarray(livres, dtype=np.int64))
    fixos = np.ones(prob.n, dtype=bool)
    fixos[liv] = False
    carga = np.bincount(a[fixos], weights=prob.demand[fixos], minlength=prob.m)
    h = blake2b(digest_size=12)
    h.update(liv.tobytes())
    h.update(np.asarray(a[liv], dtype=np.int64).tobytes())
    h.update(np.round(carga, 6).tobytes())
    return h.digest()


# ------------------------------------------------------------------ laço principal
def clns(prob: Problema, tempo: float, seletor: Seletor, tamanho: int = 15, t_sub: float = 0.5,
         seed: int = 0, usar_pl: bool = True, inicial: str = "pl",
         atrib_inicial: np.ndarray | None = None,
         lp_pronto: tuple[np.ndarray, np.ndarray] | None = None,
         gnn: np.ndarray | None = None, t_offset: float = 0.0,
         memoria: bool = False, agrupamento: str = "perfil", k_cand: int = 10) -> SaidaClns:
    """`atrib_inicial`: partida externa (ex.: expansão adaptativa); `lp_pronto` = (rc_x, x) já
    calculados (não recalcula nem cobra de novo); `gnn` = probabilidades por centro;
    `t_offset` = tempo já gasto antes desta chamada, somado à trajetória (relógio único).
    `memoria`: descarta, antes da seleção, candidatos cujo subproblema (ver `chave_subproblema`)
    já falhou com orçamento >= ao atual. Desligada, a repetição é apenas contada."""
    t0 = time.perf_counter()
    prazo = t0 + tempo
    rng = np.random.default_rng(seed)
    rc = x_lp = None
    t_pl = 0.0
    if lp_pronto is not None:
        rc, x_lp = lp_pronto
    elif usar_pl:
        try:
            lp = relaxacao_linear(prob, tempo_s=max(0.25 * tempo, 0.1))
            rc, x_lp = lp.rc_x, lp.x
        except (TimeoutError, RuntimeError):
            rc = x_lp = None
        t_pl = time.perf_counter() - t0
    ti = time.perf_counter()
    if atrib_inicial is not None and validar(prob, atrib_inicial).valida:
        a = np.asarray(atrib_inicial, dtype=np.int64).copy()
    elif inicial == "pl" and x_lp is not None:
        a = inicial_pl(prob, x_lp)
    else:
        a = construtivo(prob)
    obj = validar(prob, a).objetivo
    t_ini = time.perf_counter() - ti
    traj = [(t_offset + time.perf_counter() - t0, obj)]
    est = Estado(prob, tamanho, seed, rc, gnn, agrupamento)
    tent = {t: 0 for t in (*TIPOS, "abertura")}
    acc = {t: 0 for t in (*TIPOS, "abertura")}
    t_sel = t_subs = 0.0
    it = 0
    falhas = 0
    esgotados: dict[bytes, float] = {}  # chave do subproblema -> maior orçamento que já falhou
    repetidos = pulados = 0
    while prazo - time.perf_counter() > 0.05:
        it += 1
        ts = time.perf_counter()
        cands = est.candidatos(a, rng)
        orc_nominal = t_sub * (1 + 0.5 * min(falhas, 4))
        if memoria:
            novos = [c for c in cands
                     if esgotados.get(chave_subproblema(prob, a, c.livres), -1.0) < orc_nominal]
            pulados += len(cands) - len(novos)
            cands = novos or cands  # tudo esgotado: segue com a lista inteira (ótimo local)
        k = seletor.escolher(cands, est, a, it, rng)
        c = cands[k]
        chave = chave_subproblema(prob, a, c.livres)
        repetidos += int(esgotados.get(chave, -1.0) >= orc_nominal)
        t_sel += time.perf_counter() - ts
        orc = min(orc_nominal, prazo - time.perf_counter() - 0.02)
        if orc <= 0.02:
            break
        tr = time.perf_counter()
        cand = reotimizar(prob, a, c.livres, orc, k_cand=k_cand)
        dt = time.perf_counter() - tr
        t_subs += dt
        ganho = 0.0
        if cand is not None:
            v = validar(prob, cand)
            if v.valida and v.objetivo < obj - _EPS:
                ganho = obj - v.objetivo
                a, obj = cand, v.objetivo
                traj.append((t_offset + time.perf_counter() - t0, obj))
        tent[c.tipo] += 1
        acc[c.tipo] += int(ganho > 0)
        h = est.hist.setdefault(c.chave, [0.0, 0.0, -1.0])
        h[0] += 1
        h[1] += ganho
        h[2] = it
        seletor.retorno(c, ganho, dt)
        if ganho <= 0:
            esgotados[chave] = max(esgotados.get(chave, -1.0), orc)
        falhas = 0 if ganho > 0 else falhas + 1
    v = validar(prob, a)
    t_total = time.perf_counter() - t0
    return SaidaClns(a, v.objetivo, v.valida, t_total, traj, tent, acc, t_ini, t_pl, t_sel,
                     t_subs, it, max(0.0, t_total - tempo), repetidos, pulados)


# ------------------------------------------------------------------ coleta de rótulos
def coletar(prob: Problema, tempo: float, k_amostra: int = 8, tamanho: int = 15,
            t_sub: float = 0.5, seed: int = 0) -> list[tuple[np.ndarray, float, float]]:
    """Rótulos fora da política: em cada estado visitado, reotimiza `k_amostra` candidatos
    sorteados a partir do MESMO estado e registra (atributos, ganho, tempo). Segue pelo melhor
    deles (ou mantém o estado). O estado evolui como uma busca razoável, então os rótulos cobrem
    estados do início ao fim de uma execução real."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lp = relaxacao_linear(prob)
    a = inicial_pl(prob, lp.x)
    obj = validar(prob, a).objetivo
    est = Estado(prob, tamanho, seed, lp.rc_x)
    out: list[tuple[np.ndarray, float, float]] = []
    it = 0
    while time.perf_counter() - t0 < tempo:
        it += 1
        cands = est.candidatos(a, rng)
        reg = est.regret(a)
        amostra = rng.choice(len(cands), size=min(k_amostra, len(cands)), replace=False)
        melhor = (0.0, None, None)
        for k in amostra:
            c = cands[int(k)]
            x = est.atributos(c, a, it, reg)
            tr = time.perf_counter()
            cand = reotimizar(prob, a, c.livres, t_sub)
            dt = time.perf_counter() - tr
            ganho = 0.0
            if cand is not None:
                v = validar(prob, cand)
                if v.valida:
                    ganho = max(0.0, obj - v.objetivo)
            out.append((x, ganho / est.escala, dt))
            h = est.hist.setdefault(c.chave, [0.0, 0.0, -1.0])
            h[0] += 1
            h[1] += ganho
            h[2] = it
            if ganho > melhor[0]:
                melhor = (ganho, cand, c)
        if melhor[1] is not None:
            a = melhor[1]
            obj = validar(prob, a).objetivo
    return out
