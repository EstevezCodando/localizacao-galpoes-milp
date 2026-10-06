"""Diagnóstico do parecer (3.2): um subproblema que falhou por limite de tempo estava esgotado?

Em instâncias de VALIDAÇÃO, roda o CLNS com rotação por 30 s (sem memória), a partir de dois
contextos: "pl" (arredondamento do PL) e "expansao" (solução da expansão adaptativa ordenada pelo
PL, 30 s). Guarda os estados em que o subproblema escolhido era REPETIÇÃO de um que já tinha
falhado com limite igual ou maior. Depois reexecuta uma amostra desses subproblemas, do mesmo
estado, com limites de 5 s e 30 s, e registra se melhorou e com que status o solver terminou.

Leitura registrada antes de rodar (docs/pesquisa/12-preregistro-diagnosticos.md):
  status "optimal" sem melhoria  -> o subproblema estava de fato esgotado (limite da vizinhança);
  melhoria com mais tempo        -> a falha era falta de tempo;
  "timelimit" sem melhoria       -> indeterminado.

uv run python scripts/diag_repeticao.py --workers 3
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa import clns as C
from alocacao_capacitada.pesquisa import hibrido as H
from alocacao_capacitada.pesquisa import medicao
from alocacao_capacitada.pesquisa.dados import RAIZ, ler
from alocacao_capacitada.pesquisa.decomposicao import reotimizar
from alocacao_capacitada.pesquisa.exato import relaxacao_linear
from alocacao_capacitada.pesquisa.problema import validar

OUT = Path("results/pesquisa/diagnosticos")
T_BUSCA, T_SUB, AMOSTRA, LIMITES = 30.0, 0.5, 8, (5.0, 30.0)


def instancias(limite: int) -> list[Path]:
    """Mesma seleção intercalada por (família, razão) usada nos pilotos."""
    celulas: dict[tuple, list[Path]] = {}
    for a in sorted((RAIZ / "validacao").glob("*.pkl")):
        pr = ler(a).prob
        celulas.setdefault((pr.familia, pr.meta.get("razao")), []).append(a)
    arqs = [g[k] for k in range(max(map(len, celulas.values()))) for g in celulas.values() if k < len(g)]
    return arqs[:limite]


def rodar(args: tuple[str, str]) -> list[dict]:
    caminho, contexto = args
    p = ler(Path(caminho)).prob
    rng = np.random.default_rng(0)
    lp = relaxacao_linear(p)
    if contexto == "expansao":
        sc = lp.y + 1e-3 * (-lp.rc_y / max(np.abs(lp.rc_y).max(), 1e-9))
        s1 = H.adaptativa(p, sc, 0.0, T_BUSCA, lp, seed=0)
        a = np.asarray(s1.atribuicao, dtype=np.int64).copy()
    else:
        a = C.inicial_pl(p, lp.x)
    obj = validar(p, a).objetivo
    est = C.Estado(p, 15, 0, lp.rc_x)
    sel = C.Rotacao()
    esgotados: dict[bytes, float] = {}
    repetidos: dict[bytes, tuple[np.ndarray, np.ndarray, float, str]] = {}
    it = falhas = n_rep = n_falha = 0
    prazo = time.perf_counter() + T_BUSCA
    while prazo - time.perf_counter() > 0.05:
        it += 1
        cands = est.candidatos(a, rng)
        c = cands[sel.escolher(cands, est, a, it, rng)]
        orc_nominal = T_SUB * (1 + 0.5 * min(falhas, 4))
        chave = C.chave_subproblema(p, a, c.livres)
        rep = esgotados.get(chave, -1.0) >= orc_nominal
        orc = min(orc_nominal, prazo - time.perf_counter() - 0.02)
        if orc <= 0.02:
            break
        cand = reotimizar(p, a, c.livres, orc)
        ganho = 0.0
        if cand is not None:
            v = validar(p, cand)
            if v.valida and v.objetivo < obj - 1e-6:
                ganho = obj - v.objetivo
                a, obj = cand, v.objetivo
        if ganho <= 0:
            n_falha += 1
            if rep:
                n_rep += 1
                repetidos.setdefault(chave, (a.copy(), np.asarray(c.livres).copy(), orc_nominal, c.tipo))
            esgotados[chave] = max(esgotados.get(chave, -1.0), orc)
        falhas = 0 if ganho > 0 else falhas + 1
    linhas = []
    chaves = list(repetidos)
    for k in rng.choice(len(chaves), size=min(AMOSTRA, len(chaves)), replace=False) if chaves else []:
        a0, livres, orc0, tipo = repetidos[chaves[int(k)]]
        base = validar(p, a0).objetivo
        for lim in LIMITES:
            info: dict = {}
            novo = reotimizar(p, a0, livres, lim, info=info)
            depois = validar(p, novo).objetivo if novo is not None else base
            linhas.append({"instancia": p.nome, "contexto": contexto, "tipo": tipo, "n_livres": int(len(livres)),
                           "limite_original_s": orc0, "limite_s": lim, "objetivo_antes": base,
                           "objetivo_depois": depois, "melhorou": bool(depois < base - 1e-6),
                           "ganho_rel": float((base - depois) / base), "iteracoes_busca": it,
                           "falhas_busca": n_falha, "repeticoes_busca": n_rep,
                           "chaves_repetidas": len(chaves), **info})
    if not linhas:
        linhas.append({"instancia": p.nome, "contexto": contexto, "iteracoes_busca": it,
                       "falhas_busca": n_falha, "repeticoes_busca": n_rep, "chaves_repetidas": 0})
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limite", type=int, default=16)
    a = ap.parse_args()
    jobs = [(str(x), c) for x in instancias(a.limite) for c in ("pl", "expansao")]
    fila = medicao.fila_de_nucleos(a.workers)
    linhas: list[dict] = []
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=medicao.inicializar_worker, initargs=(fila,)) as ex:
        for res in ex.map(rodar, jobs):
            linhas += res
            pd.DataFrame(linhas).to_csv(OUT / "repeticao.csv", index=False)
            print(res[0]["instancia"], res[0]["contexto"], len(res), flush=True)
    d = pd.DataFrame(linhas)
    d = d[d["limite_s"].notna()] if "limite_s" in d else d
    if len(d):
        d["classe"] = np.where(d["melhorou"], "melhorou",
                               np.where(d["status"] == "optimal", "esgotado (ótimo provado)", "indeterminado"))
        print(d.groupby(["contexto", "limite_s"])["classe"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
