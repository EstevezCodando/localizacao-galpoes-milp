"""Políticas que usam um score de centros para orientar o solver (doc 08).

Todas devolvem uma solução VALIDADA no problema original e o tempo online total, que inclui
atributos, inferência, PL (quando usado), reparo, todas as resoluções e o fallback.

  topk          mantém a fração rho mais bem ranqueada (+ reparo de viabilidade); heurística
  adaptativa    20% -> 40% -> 60% -> 100% com expansão guiada; reaproveita o incumbente
  kernel_search núcleo = centros com y > 0 no PL; baldes dos demais por custo reduzido
                (Guastaroba & Speranza, 2014). Baseline clássico, sem aprendizado
  warm          resolve o reduzido rápido e passa a solução ao completo (domínio intacto)
  prioridade    completo com prioridade de branching pelos scores (orientação sem poda)
  certificada   reduzido + certificado de custo reduzido do PL ORIGINAL: se L + rc_i >= U para
                todo centro podado i, nenhum centro podado entra numa solução melhor que U.
                Com o reduzido resolvido à otimalidade, isso PROVA o ótimo global.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from alocacao_capacitada.pesquisa.exato import Relaxacao, Resultado, resolver
from alocacao_capacitada.pesquisa.geradores import _ffd
from alocacao_capacitada.pesquisa.problema import Problema, validar
from alocacao_capacitada.pesquisa.tempos import cronometrar

Score = Callable[[Problema], tuple[np.ndarray, float]]  # (score maior = manter, tempo gasto)


@dataclass
class Saida:
    politica: str
    objetivo: float
    valida: bool
    t_total: float
    t_score: float
    estagios: int
    fracao_final: float
    fallback: bool
    certificado: bool
    trajetoria: list[tuple[float, float]] = field(default_factory=list)  # (t online, obj)
    nos: int = 0
    atribuicao: np.ndarray | None = None


@cronometrar("t_reparo")
def reparar(prob: Problema, ordem: np.ndarray, k: int, viz: int = 5) -> np.ndarray:
    """Primeiros k da ordem + os seguintes até o FFD achar uma atribuição de fonte única
    (condição suficiente; sem ela o reduzido pode ser inviável) e até cada cliente ter ao
    menos 1 centro mantido entre seus `viz` mais baratos (cobertura local)."""
    keep = list(ordem[:k])
    rest = list(ordem[k:])
    while rest and not _ffd(prob.demand, prob.capacity[keep]):
        keep.append(rest.pop(0))
    near = np.argsort(prob.cost, axis=0)[: min(viz, prob.m)]
    kept = np.zeros(prob.m, dtype=bool)
    kept[keep] = True
    pos = np.empty(prob.m, dtype=int)
    pos[ordem] = np.arange(prob.m)
    for j in np.flatnonzero(~kept[near].any(axis=0)):
        if not kept[near[:, j]].any():  # pode ter sido coberto por inclusão anterior
            i = int(near[np.argmin(pos[near[:, j]]), j])  # o vizinho mais bem ranqueado
            kept[i] = True
            keep.append(i)
    return np.array(sorted(set(int(i) for i in keep)))


def _deslocar(traj: list[tuple[float, float]], t0: float) -> list[tuple[float, float]]:
    return [(t + t0, o) for t, o in traj]


def topk(prob: Problema, score: np.ndarray, t_score: float, rho: float, tempo: float,
         seed: int = 0) -> Saida:
    t0 = time.perf_counter()
    ordem = np.argsort(-score, kind="stable")
    keep = reparar(prob, ordem, max(1, int(round(rho * prob.m))))
    ts = t_score + time.perf_counter() - t0
    r = resolver(prob, tempo - ts, centros=keep, seed=seed)
    t = t_score + time.perf_counter() - t0
    return Saida(f"topk{int(rho * 100)}", r.objetivo, r.valida, t, t_score, 1,
                 keep.size / prob.m, False, False, _deslocar(r.trajetoria, ts),
                 r.nos, r.atribuicao)


def certificado_poda(lp: Relaxacao, upper: float, podados: np.ndarray) -> np.ndarray:
    """Centros podados que o certificado de custo reduzido NÃO consegue excluir."""
    return podados[lp.bound + lp.rc_y[podados] < upper - 1e-6]


def adaptativa(
    prob: Problema, score: np.ndarray, t_score: float, tempo: float, lp: Relaxacao | None,
    fracoes: tuple[float, ...] = (0.2, 0.4, 0.6, 1.0), orc_estagio: float = 0.25,
    seed: int = 0,
) -> Saida:
    """Expansão 20->40->60->100%. Cada estágio recebe `orc_estagio` do tempo restante (o último
    recebe todo o resto). Para cedo se o certificado (requer `lp`) provar o ótimo global."""
    t0 = time.perf_counter()
    ordem = np.argsort(-score, kind="stable")
    inc: np.ndarray | None = None
    best = np.inf
    traj: list[tuple[float, float]] = []
    nos = 0
    keep = np.array([], dtype=int)
    cert = False
    k_est = 0
    fallback = False
    for rho in fracoes:
        restante = tempo - t_score - (time.perf_counter() - t0)
        if restante <= 0.05:
            break
        if rho >= 1.0:
            keep_new = np.arange(prob.m)
        else:
            keep_new = reparar(prob, ordem, max(1, int(round(rho * prob.m))))
            if inc is not None:
                keep_new = np.union1d(keep_new, np.unique(inc))
            if lp is not None and np.isfinite(best):
                # expansão guiada: inclui todo podado que o certificado não exclui, se couber
                podados = np.setdiff1d(np.arange(prob.m), keep_new)
                faltam = certificado_poda(lp, best, podados)
                if keep_new.size + faltam.size <= max(rho, 0.0) * prob.m * 1.5:
                    keep_new = np.union1d(keep_new, faltam)
        keep = keep_new
        ultimo = rho >= 1.0
        ts = t_score + time.perf_counter() - t0
        restante = tempo - ts
        if restante <= 0:
            break
        orc = restante if ultimo else restante * orc_estagio
        k_est += 1
        fallback = ultimo
        r = resolver(prob, orc, centros=None if ultimo else keep,
                     dica=inc if inc is not None and np.isin(inc, keep).all() else None, seed=seed)
        nos += r.nos
        traj += _deslocar(r.trajetoria, ts)
        if r.valida and r.objetivo < best:
            best, inc = r.objetivo, r.atribuicao
        if ultimo:
            cert = r.status == "optimal"
            break
        if lp is not None and r.status == "optimal" and np.isfinite(best):
            podados = np.setdiff1d(np.arange(prob.m), keep)
            if certificado_poda(lp, best, podados).size == 0:
                cert = True
                break
    v = validar(prob, inc) if inc is not None else None
    t = t_score + time.perf_counter() - t0
    return Saida("adaptativa", v.objetivo if v else np.inf, bool(v and v.valida), t, t_score,
                 k_est, keep.size / prob.m if keep.size else 0.0, fallback, cert,
                 traj, nos, inc)


def kernel_search(prob: Problema, lp: Relaxacao, t_pre: float, tempo: float,
                  orc_nucleo: float = 0.25, seed: int = 0) -> Saida:
    """Kernel Search (Guastaroba & Speranza, 2014) sobre os centros, sem aprendizado.

    Núcleo inicial: centros com y_i > 0 no PL forte (completado por `reparar` até existir uma
    atribuição de fonte única). Os demais centros são ordenados por custo reduzido crescente e
    cortados em baldes do tamanho do núcleo. Resolve o núcleo com `orc_nucleo` do tempo restante
    e depois núcleo + balde, um balde por vez, com o tempo restante dividido igualmente entre os
    baldes que faltam; os centros do balde abertos na nova solução entram no núcleo. O incumbente
    é sempre passado como dica, então nenhum subproblema devolve algo pior.

    Simplificação em relação ao artigo: não impomos a restrição "abrir ao menos um centro do
    balde" nem o corte de objetivo; a dica cumpre o papel do corte. `t_pre` = tempo do PL."""
    t0 = time.perf_counter()
    positivos = np.flatnonzero(lp.y > 1e-6)
    demais = np.setdiff1d(np.arange(prob.m), positivos)
    ordem = np.concatenate([positivos[np.argsort(-lp.y[positivos], kind="stable")],
                            demais[np.argsort(lp.rc_y[demais], kind="stable")]])
    nucleo = reparar(prob, ordem, max(1, positivos.size))
    fora = ordem[~np.isin(ordem, nucleo)]
    tam = max(1, nucleo.size)
    baldes = [fora[k:k + tam] for k in range(0, fora.size, tam)]
    inc: np.ndarray | None = None
    best = np.inf
    traj: list[tuple[float, float]] = []
    nos = 0
    resolvidos = 0
    for b in range(-1, len(baldes)):
        ts = t_pre + time.perf_counter() - t0
        restante = tempo - ts
        if restante <= 0.05:
            break
        keep = nucleo if b < 0 else np.union1d(nucleo, baldes[b])
        if b < 0 and baldes:
            orc = restante * orc_nucleo
        else:
            orc = restante / max(len(baldes) - b, 1)
        r = resolver(prob, orc, centros=keep,
                     dica=inc if inc is not None and np.isin(inc, keep).all() else None, seed=seed)
        resolvidos += 1
        nos += r.nos
        traj += _deslocar(r.trajetoria, ts)
        if r.valida and r.objetivo < best:
            best, inc = r.objetivo, r.atribuicao
            if b >= 0:
                nucleo = np.union1d(nucleo, np.intersect1d(np.unique(inc), baldes[b]))
    v = validar(prob, inc) if inc is not None else None
    t = t_pre + time.perf_counter() - t0
    return Saida("kernel", v.objetivo if v else np.inf, bool(v and v.valida), t, t_pre,
                 resolvidos, nucleo.size / prob.m, False, False, traj, nos, inc)


def warm(prob: Problema, score: np.ndarray, t_score: float, tempo: float, rho: float = 0.3,
         orc_reduzido: float = 0.15, seed: int = 0) -> Saida:
    t0 = time.perf_counter()
    ordem = np.argsort(-score, kind="stable")
    keep = reparar(prob, ordem, max(1, int(round(rho * prob.m))))
    ts = t_score + time.perf_counter() - t0
    r1 = resolver(prob, orc_reduzido * max(tempo - ts, 0.0), centros=keep, seed=seed)
    traj = _deslocar(r1.trajetoria, ts)
    ts = t_score + time.perf_counter() - t0
    r2 = resolver(prob, tempo - ts, dica=r1.atribuicao if r1.valida else None, seed=seed)
    traj += _deslocar(r2.trajetoria, ts)
    melhor = min([r for r in (r1, r2) if r.valida], key=lambda r: r.objetivo, default=r2)
    t = t_score + time.perf_counter() - t0
    return Saida("warm", melhor.objetivo, melhor.valida, t, t_score, 2, 1.0, False,
                 r2.status == "optimal", traj, r1.nos + r2.nos, melhor.atribuicao)


def prioridade(prob: Problema, score: np.ndarray, t_score: float, tempo: float,
               seed: int = 0) -> Saida:
    t0 = time.perf_counter()
    pr = np.argsort(np.argsort(score))  # maior score -> maior prioridade
    ts = t_score + time.perf_counter() - t0
    r = resolver(prob, tempo - ts, prioridade=pr, seed=seed)
    t = t_score + time.perf_counter() - t0
    return Saida("prioridade", r.objetivo, r.valida, t, t_score, 1, 1.0, False,
                 r.status == "optimal", _deslocar(r.trajetoria, ts), r.nos,
                 r.atribuicao)


def completo(prob: Problema, tempo: float, seed: int = 0) -> tuple[Saida, Resultado]:
    r = resolver(prob, tempo, seed=seed)
    return (Saida("completo", r.objetivo, r.valida, r.t_total, 0.0, 1, 1.0, False,
                  r.status == "optimal", r.trajetoria, r.nos,
                  r.atribuicao), r)


def reducao_classica(prob: Problema, tempo: float, lp: Relaxacao,
                     upper: tuple[float, np.ndarray] | None, t_pre: float, seed: int = 0) -> Saida:
    """Redução SEGURA por custo reduzido (fixação clássica): com bound L do PL original e uma
    solução U, toda variável com L + rc > U vale 0 em qualquer solução melhor que U. Remove
    centros e pares sem perder o ótimo; o solver roda no domínio restante com U como dica.
    `t_pre` = tempo já gasto (PL + heurística que gerou U) e entra no total."""
    t0 = time.perf_counter()
    if upper is None:
        r = resolver(prob, tempo - t_pre, seed=seed)
        return Saida("reducao_classica", r.objetivo, r.valida, t_pre + r.t_total, t_pre, 1, 1.0,
                     False, r.status == "optimal", _deslocar(r.trajetoria, t_pre), r.nos,
                     r.atribuicao)
    u, a = upper
    keep = np.flatnonzero(lp.bound + lp.rc_y < u - 1e-6)
    keep = np.union1d(keep, np.unique(a))  # nunca remove o que a solução U usa
    pares = (lp.bound + lp.rc_x < u - 1e-6)
    pares[a, np.arange(prob.n)] = True
    ts = t_pre + time.perf_counter() - t0
    r = resolver(prob, tempo - ts, centros=keep, pares_permitidos=pares, dica=a, seed=seed)
    t = t_pre + time.perf_counter() - t0
    traj = [(t_pre, u)] + _deslocar(r.trajetoria, ts)
    melhor_obj, melhor_a = (r.objetivo, r.atribuicao) if r.valida and r.objetivo <= u else (u, a)
    # a redução é exata: ótimo do reduzido = ótimo original (ou U já era ótimo)
    return Saida("reducao_classica", melhor_obj, True, t, t_pre, 1, keep.size / prob.m, False,
                 r.status == "optimal", traj, r.nos, melhor_a)


def tempo_ate_alvo(traj: list[tuple[float, float]], alvo: float) -> float | None:
    """Primeiro instante em que o incumbente fica <= alvo; None = censurado (não atingiu)."""
    for t, o in traj:
        if o <= alvo + 1e-9:
            return t
    return None
