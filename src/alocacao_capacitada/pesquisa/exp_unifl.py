"""Experimento da frente P06 (UniFL): MP vs busca local vs MPNN não supervisionada vs ótimo.

Treino: 200 instâncias (n = 100 e 200, uniforme/clusters), 30 épocas, 2 sementes, com e sem o
raio de MP como atributo. Limiar de decodificação escolhido em 20 instâncias de VALIDAÇÃO.
Teste: n = 100, 200 (ótimo provado), 500 (SCIP 300 s, razão certificada pelo bound) e
1000 (sem exato: desvio contra o melhor método).

uv run python -m alocacao_capacitada.pesquisa.exp_unifl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa.unifl import (
    busca_local,
    custo,
    decodificar,
    exato,
    gerar_unifl,
    mettu_plaxton,
    treinar_mpnn,
)

OUT = Path("results/pesquisa/unifl")


def main() -> None:
    import sys

    estavel = "--estavel" in sys.argv
    suf = "_estavel" if estavel else ""
    OUT.mkdir(parents=True, exist_ok=True)
    modelos, custo_off = {}, {}
    for com_raio in (False, True):
        for seed in (0, 1):
            nome = f"mpnn{suf}{'_raio' if com_raio else ''}_s{seed}"
            modelos[nome], custo_off[nome] = treinar_mpnn(com_raio=com_raio, seed=seed,
                                                          estavel=estavel)
            print(nome, f"{custo_off[nome]:.0f}s", flush=True)
    # limiar na validação
    val = [gerar_unifl(150, ("uniforme", "clusters")[s % 2], 20_000 + s) for s in range(20)]
    limiar = {}
    for nome, mdl in modelos.items():
        cr = "_raio" in nome
        best = min((0.2, 0.3, 0.4, 0.5, 0.6, 0.7), key=lambda t: np.mean([
            custo(v, decodificar(mdl, v, v.dist(), cr, limiar=t)) for v in val]))
        limiar[nome] = best
    (OUT / f"config{suf}.json").write_text(json.dumps({"limiar": limiar, "treino_s": custo_off}, indent=2))
    rows = []
    plano = [(100, 20), (200, 20), (500, 10), (1000, 6)]
    for n, k in plano:
        for s in range(k):
            fam = ("uniforme", "clusters")[s % 2]
            inst = gerar_unifl(n, fam, 30_000 + 1000 * n + s)
            t0 = time.perf_counter()
            d = inst.dist()
            t_d = time.perf_counter() - t0
            t0 = time.perf_counter()
            if n <= 500:
                z, lb, _, st = exato(inst, 120 if n <= 200 else 300)
            else:  # 10^6 variáveis: referência = melhor método (calculada na análise)
                z, lb, st = np.nan, np.nan, "sem_exato"
            t_ex = time.perf_counter() - t0
            base = {"n": n, "familia": fam, "seed": s, "otimo": z, "bound": lb, "status": st,
                    "t_exato": t_ex}

            def reg(metodo: str, S: np.ndarray, t: float, base: dict = base,
                    d: np.ndarray = d) -> None:
                c = custo(inst, S, d)
                rows.append({**base, "metodo": metodo, "custo": c, "razao_vs_ref": c / base["otimo"],
                             "razao_cert": c / base["bound"], "abertos": S.size,
                             "tempo": t + t_d})

            t0 = time.perf_counter()
            mp = mettu_plaxton(inst, d)
            t_mp = time.perf_counter() - t0
            reg("mettu_plaxton", mp, t_mp)
            t0 = time.perf_counter()
            reg("mp+busca_local", busca_local(inst, mp, d), t_mp + time.perf_counter() - t0)
            for nome, mdl in modelos.items():
                cr = "_raio" in nome
                t0 = time.perf_counter()
                S = decodificar(mdl, inst, d, cr, limiar=limiar[nome])
                t_g = time.perf_counter() - t0
                reg(nome, S, t_g)
                t0 = time.perf_counter()
                reg(nome + "+busca_local", busca_local(inst, S, d), t_g + time.perf_counter() - t0)
            pd.DataFrame(rows).to_csv(OUT / f"resultados{suf}.csv", index=False)
            print(n, s, st, f"{t_ex:.1f}s", flush=True)


if __name__ == "__main__":
    main()
