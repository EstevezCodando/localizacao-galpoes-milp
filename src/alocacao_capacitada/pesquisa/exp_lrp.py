"""Experimento da frente P07 (Etapa 2): FLP->VRP vs aproximação contínua vs NEO (Deep Sets
embutido em SCIP), sempre avaliados pelas rotas reais.

Rótulos: subconjuntos de instâncias de treino (sementes 1-3), CVRP OR-Tools 1 s.
Teste: n in {20, 50, 100} clientes, m in {5, 10} depósitos, uniforme/clusters, sementes 9000+.
Referência: melhor custo final entre todos os métodos + enumeração de conjuntos de depósitos
(m = 5: 31 conjuntos, cada um com atribuição pelo MIP de aproximação contínua e rotas reais).

uv run python -m alocacao_capacitada.pesquisa.exp_lrp rotulos --workers 3
uv run python -m alocacao_capacitada.pesquisa.exp_lrp testar
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa import lrp as L

OUT = Path("results/pesquisa/lrp/v2")
DADOS = Path("data/processed/pesquisa/lrp")


def _lote(args: tuple[int, int]) -> list[tuple[np.ndarray, float]]:
    n_inst, seed = args
    return L.amostrar_rotulos(n_inst, seed, tempo=1.0)


def rotulos(workers: int) -> None:
    DADOS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        lotes = list(ex.map(_lote, [(40, s) for s in range(100, 112)]))  # 12 x 40 instâncias
    tr = [a for lt in lotes[:10] for a in lt]
    va = [a for lt in lotes[10:] for a in lt]
    with open(DADOS / "rotulos.pkl", "wb") as fh:
        pickle.dump({"treino": tr, "validacao": va, "t_s": time.perf_counter() - t0}, fh)
    print(len(tr), len(va), f"{time.perf_counter() - t0:.0f}s")


def _enumerar(inst: L.Clrp, tempo_rotas: float) -> float:
    melhor = np.inf
    c = L.custo_aprox_continua(inst)
    for k in range(1, inst.m + 1):
        for S in itertools.combinations(range(inst.m), k):
            if inst.cap_dep[list(S)].sum() < inst.demand.sum():
                continue
            fixo = inst.open_cost.copy()
            proib = np.ones(inst.m, dtype=bool)
            proib[list(S)] = False
            cc = c.copy()
            cc[proib] = 1e9
            try:
                a, _ = L.mip_linear(inst, cc, 20)
            except Exception:  # noqa: BLE001
                continue
            if proib[np.unique(a)].any():
                continue
            melhor = min(melhor, L.avaliar(inst, a, tempo_total=tempo_rotas)["custo"])
            del fixo
    return melhor


def testar(tempo_mip: float, tempo_rotas: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(DADOS / "rotulos.pkl", "rb") as fh:
        d = pickle.load(fh)
    t0 = time.perf_counter()
    model, hist = L.treinar_deepsets(d["treino"], d["validacao"], seed=0)
    t_tr = time.perf_counter() - t0
    (OUT / "custo_offline.json").write_text(json.dumps({
        "n_rotulos_treino": len(d["treino"]), "n_rotulos_val": len(d["validacao"]),
        "t_rotulos_s": d["t_s"], "t_treino_s": t_tr, "mape_val": float(min(hist))}, indent=2))
    rows = []
    for n, m, k in ((20, 5, 6), (50, 5, 6), (50, 10, 4), (100, 10, 4)):
        for s in range(k):
            fam = ("uniforme", "clusters")[s % 2]
            inst = L.gerar_clrp(n, m, 9000 + 100 * n + 10 * m + s, fam)
            res = {}
            t1 = time.perf_counter()
            a_flp, _ = L.mip_linear(inst, L.custo_flp(inst), tempo_mip)
            res["flp_vrp"] = (a_flp, time.perf_counter() - t1)
            t1 = time.perf_counter()
            a_ca, _ = L.mip_linear(inst, L.custo_aprox_continua(inst), tempo_mip)
            res["aprox_cont"] = (a_ca, time.perf_counter() - t1)
            t1 = time.perf_counter()
            ini, _ = L.mip_linear(inst, L.custo_aprox_continua(inst), tempo_mip * 0.2)
            restante = tempo_mip - (time.perf_counter() - t1)
            try:
                a_neo, _, nbin = L.mip_neo(inst, model, restante, ini=ini)
            except (TimeoutError, RuntimeError):
                a_neo, nbin = ini, 0
            res["neo_ds"] = (a_neo, time.perf_counter() - t1)
            for metodo, (a, t_mip) in res.items():
                ev = L.avaliar(inst, a, tempo_total=tempo_rotas)
                rows.append({"instancia": inst.nome, "n": n, "m": m, "familia": fam,
                             "metodo": metodo, **ev, "t_mip": t_mip,
                             "t_total": t_mip + ev["t_rotas"],
                             "orcamento_total": tempo_mip + tempo_rotas,
                             "estouro_s": max(0., t_mip + ev["t_rotas"] - tempo_mip - tempo_rotas),
                             "custo_previsto_neo": L.prever(model, inst, a),
                             "binarias_neurais": nbin if metodo == "neo_ds" else 0})
            ref = min(r["custo"] for r in rows if r["instancia"] == inst.nome)
            if m == 5:
                ref = min(ref, _enumerar(inst, tempo_rotas))
            for r in rows:
                if r["instancia"] == inst.nome:
                    r["referencia"] = ref
            pd.DataFrame(rows).to_csv(OUT / "resultados.csv", index=False)
            print(inst.nome, {r["metodo"]: round(r["custo"]) for r in rows
                              if r["instancia"] == inst.nome}, round(ref), flush=True)


def suplementar(tempo_mip: float, tempo_rotas: float) -> None:
    """Métodos acrescentados após o diagnóstico do MIP neural (desvio documentado):
    neo_multi = MIP neural partindo da melhor solução (pelo surrogate) entre CA e FLP;
    neo_busca = busca local de realocação avaliada pelo surrogate (Caminho B, doc 09).
    Mesmo modelo (semente 0, mesmos rótulos) e mesmas instâncias do teste principal."""
    with open(DADOS / "rotulos.pkl", "rb") as fh:
        d = pickle.load(fh)
    model, _ = L.treinar_deepsets(d["treino"], d["validacao"], seed=0)
    base = pd.read_csv(OUT / "resultados.csv")
    rows = []
    for nome, g in base.groupby("instancia", sort=False):
        n, m, fam = int(g["n"].iloc[0]), int(g["m"].iloc[0]), g["familia"].iloc[0]
        seed = int(nome.split("_s")[-1])
        inst = L.gerar_clrp(n, m, seed, fam)
        assert inst.nome == nome
        t1 = time.perf_counter()
        a_ca, _ = L.mip_linear(inst, L.custo_aprox_continua(inst), tempo_mip * 0.1)
        a_f, _ = L.mip_linear(inst, L.custo_flp(inst), tempo_mip * 0.1)
        ini = min((a_ca, a_f), key=lambda a: L.prever(model, inst, a))
        t_ini = time.perf_counter() - t1
        t1 = time.perf_counter()
        a_b, _, it = L.busca_surrogate(inst, model, ini, max(0., tempo_mip - t_ini))
        t_b = time.perf_counter() - t1 + t_ini
        t1 = time.perf_counter()
        try:
            a_m, _, nbin = L.mip_neo(inst, model, tempo_mip - t_ini, ini=ini)
        except (TimeoutError, RuntimeError):
            a_m, nbin = ini, 0
        t_m = time.perf_counter() - t1 + t_ini
        for metodo, a, t in (("neo_multi", a_m, t_m), ("neo_busca", a_b, t_b)):
            ev = L.avaliar(inst, a, tempo_total=tempo_rotas)
            rows.append({"instancia": nome, "n": n, "m": m, "familia": fam, "metodo": metodo,
                         **ev, "t_mip": t, "t_total": t + ev["t_rotas"],
                         "orcamento_total": tempo_mip + tempo_rotas,
                         "estouro_s": max(0., t + ev["t_rotas"] - tempo_mip - tempo_rotas),
                         "custo_previsto_neo": L.prever(model, inst, a),
                         "binarias_neurais": nbin, "referencia": g["referencia"].iloc[0]})
        pd.DataFrame(rows).to_csv(OUT / "resultados_suplementar.csv", index=False)
        print(nome, {r["metodo"]: round(r["custo"]) for r in rows if r["instancia"] == nome},
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["rotulos", "testar", "suplementar"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tempo-mip", type=float, default=60)
    ap.add_argument("--tempo-rotas", type=float, default=2.0)
    a = ap.parse_args()
    if a.fase == "rotulos":
        rotulos(a.workers)
    elif a.fase == "testar":
        testar(a.tempo_mip, a.tempo_rotas)
    else:
        suplementar(a.tempo_mip, a.tempo_rotas)


if __name__ == "__main__":
    main()
