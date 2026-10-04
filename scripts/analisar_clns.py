"""Análise de uma execução do executor CLNS (results/pesquisa/clns/<run_id>/resultados.csv).

Agrega as sementes por (instância, método) ANTES da estatística (sementes não são instâncias
novas), calcula integral primal com mínimo acumulado, desvios contra BKS (ou contra o ótimo
publicado com --ref-externa) e Wilcoxon pareado com Holm contra:
  * o melhor seletor NÃO aprendido (hipótese C3) e
  * o melhor baseline não-CLNS (hipótese C1).

uv run python scripts/analisar_clns.py results/pesquisa/clns/<run_id> --horizonte 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa.analise import bootstrap_ic, holm, preparar
from scipy.stats import wilcoxon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--horizonte", type=float, required=True)
    ap.add_argument("--ref-externa", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.run / "resultados.csv")
    df["valida"] = df["valida"].astype(bool)
    if "certificado" not in df:
        df["certificado"] = False
    df["certificado"] = df["certificado"].fillna(False)
    d = preparar(df, a.horizonte, ref_externa=a.ref_externa)
    d["sucesso_1pct"] = d["t_alvo_0.01"] <= a.horizonte
    # agregação por instância (média entre sementes)
    agg = d.groupby(["instancia", "metodo"]).agg(
        integral=("integral_primal", "mean"), desvio=("desvio", "mean"),
        sucesso=("sucesso_1pct", "mean"), validas=("valida", "mean"),
        t_total=("t_total", "mean"), n_seeds=("seed_solver", "nunique")).reset_index()
    linhas = []
    for m, g in agg.groupby("metodo"):
        lo, hi = bootstrap_ic(g["integral"].to_numpy())
        linhas.append({"metodo": m, "instancias": len(g), "sementes": int(g["n_seeds"].max()),
                       "integral_media": g["integral"].mean(), "integral_ic95": f"[{lo:.4f}, {hi:.4f}]",
                       "desvio_mediano": g["desvio"].median(), "desvio_medio": g["desvio"].mean(),
                       "p90_desvio": g["desvio"].quantile(0.9), "ate_1pct": g["sucesso"].mean(),
                       "validas": g["validas"].mean(), "t_total": g["t_total"].mean()})
    res = pd.DataFrame(linhas).sort_values("integral_media")
    piv = agg.pivot(index="instancia", columns="metodo", values="integral")
    clns = [m for m in piv if m.startswith("clns:")]
    nao_aprendidos = [m for m in clns if m != "clns:aprendido"]
    baselines = [m for m in piv if not m.startswith("clns:")]
    testes = []
    for alvo, rivais, hip in (("clns:aprendido", nao_aprendidos, "C3"),):
        if alvo in piv and rivais:
            melhor = min(rivais, key=lambda m: piv[m].mean())
            par = piv[[alvo, melhor]].dropna()
            p = wilcoxon(par[alvo], par[melhor]).pvalue if (par[alvo] != par[melhor]).sum() >= 5 else np.nan
            testes.append({"hipotese": hip, "metodo": alvo, "contra": melhor,
                           "dif_mediana": float((par[alvo] - par[melhor]).median()),
                           "vence": float((par[alvo] < par[melhor]).mean()), "p": p})
    if clns and baselines:
        melhor_b = min(baselines, key=lambda m: piv[m].mean())
        pv = {}
        for m in clns:
            par = piv[[m, melhor_b]].dropna()
            pv[m] = wilcoxon(par[m], par[melhor_b]).pvalue if (par[m] != par[melhor_b]).sum() >= 5 else np.nan
            testes.append({"hipotese": "C1", "metodo": m, "contra": melhor_b,
                           "dif_mediana": float((par[m] - par[melhor_b]).median()),
                           "vence": float((par[m] < par[melhor_b]).mean()), "p": pv[m]})
        adj = holm({k: v for k, v in pv.items() if np.isfinite(v)})
        for t in testes:
            if t["hipotese"] == "C1":
                t["p_holm"] = adj.get(t["metodo"], np.nan)
    tst = pd.DataFrame(testes)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    print()
    print(tst.round(4).to_string(index=False))
    res.to_csv(a.run / "resumo_clns.csv", index=False)
    tst.to_csv(a.run / "testes_clns.csv", index=False)
    (a.run / "analise_config.json").write_text(json.dumps(vars(a), default=str, indent=2))


if __name__ == "__main__":
    main()
