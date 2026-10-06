"""Análise dos diagnósticos de validação (docs/pesquisa/12-preregistro-diagnosticos.md).

uv run python scripts/analisar_diagnosticos.py results/pesquisa/diagnosticos <run_d2_d3>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from alocacao_capacitada.pesquisa.analise import bootstrap_ic, integral_primal

COMPARACOES = (("D2", "clns:rotacao", "clns:rotacao+alea"), ("D2", "hibridopl:rotacao", "hibridopl:rotacao+alea"),
               ("D3", "clns:rotacao", "clns:rotacao+k30"), ("D3", "hibridopl:rotacao", "hibridopl:rotacao+k20"),
               ("D3", "hibridopl:rotacao", "hibridopl:rotacao+k30"))


def holm(ps: list[float]) -> list[float]:
    ordem = np.argsort(ps)
    adj = np.empty(len(ps))
    atual = 0.0
    for k, i in enumerate(ordem):
        atual = max(atual, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, atual)
    return [float(a) for a in adj]


def main() -> None:
    dest, run = Path(sys.argv[1]), Path(sys.argv[2])
    out: dict[str, object] = {}
    md = ["# Diagnósticos de validação — resultados\n"]

    # ---------------------------------------------------------------- D1
    r = pd.read_csv(dest / "repeticao.csv")
    busca = r.groupby(["instancia", "contexto"])[["iteracoes_busca", "falhas_busca", "repeticoes_busca",
                                                   "chaves_repetidas"]].first()
    busca["taxa_rep"] = busca["repeticoes_busca"] / busca["iteracoes_busca"].clip(lower=1)
    d = r[r["limite_s"].notna()].copy() if "limite_s" in r else r.iloc[0:0]
    md += ["## D1 — subproblemas repetidos reexecutados com limite maior\n",
           "| Contexto | Execuções | Iterações (mediana) | Taxa de repetição (mediana) | Subproblemas reexecutados |",
           "|---|---:|---:|---:|---:|"]
    for c, g in busca.groupby("contexto"):
        md.append(f"| {c} | {len(g)} | {g['iteracoes_busca'].median():.0f} | {100 * g['taxa_rep'].median():.1f}% | "
                  f"{int((d['contexto'] == c).sum() / max(d['limite_s'].nunique(), 1)) if len(d) else 0} |")
    if len(d):
        d["classe"] = np.where(d["melhorou"], "falta de tempo (melhorou)",
                               np.where(d["status"] == "optimal", "esgotado (ótimo provado)", "indeterminado"))
        t = d.groupby(["contexto", "limite_s"])["classe"].value_counts().unstack(fill_value=0)
        out["D1"] = {f"{c}|{lim}": {k: int(v) for k, v in linha.items()} for (c, lim), linha in t.iterrows()}
        md += ["", "| Contexto | Limite | " + " | ".join(t.columns) + " | Tempo mediano até terminar | Variáveis (mediana) |",
               "|---|---:|" + "---:|" * (len(t.columns) + 2)]
        for (c, lim), linha in t.iterrows():
            g = d[(d["contexto"] == c) & (d["limite_s"] == lim)]
            md.append(f"| {c} | {lim:.0f} s | " + " | ".join(str(int(v)) for v in linha) +
                      f" | {g['tempo'].median():.2f} s | {g['variaveis'].median():.0f} |")
        m = d[d["melhorou"]]
        out["D1_ganho_quando_melhora_pct"] = float(100 * m["ganho_rel"].median()) if len(m) else None
        md += ["", f"Quando houve melhoria, o ganho mediano foi de {100 * m['ganho_rel'].median():.3f}% do custo."
               if len(m) else "Nenhuma reexecução melhorou a solução.", ""]
        por_tipo = d[d["limite_s"] == d["limite_s"].max()].groupby("tipo")["classe"].value_counts().unstack(fill_value=0)
        md += ["Por tipo de subproblema, no limite maior:\n", "| Tipo | " + " | ".join(por_tipo.columns) + " |",
               "|---|" + "---:|" * len(por_tipo.columns)]
        md += [f"| {tp} | " + " | ".join(str(int(v)) for v in linha) + " |" for tp, linha in por_tipo.iterrows()]

    # ---------------------------------------------------------------- D2 e D3
    x = pd.read_csv(run / "resultados.csv")
    x["traj"] = x["trajetoria"].map(json.loads)
    bks = pd.concat([x.groupby("instancia")["objetivo"].min(), x.groupby("instancia")["rotulo_melhor"].first()],
                    axis=1).min(axis=1)
    x["bks"] = x["instancia"].map(bks)
    x["obj_T"] = [min((o for t, o in tr if t <= h + 1e-9), default=np.inf) for tr, h in zip(x["traj"], x["orcamento"])]
    x["desvio"] = np.minimum(1.0, x["obj_T"] / x["bks"] - 1)
    x["integral"] = [integral_primal(t, b, h) for t, b, h in zip(x["traj"], x["bks"], x["orcamento"])]
    pi = x.groupby(["instancia", "metodo"]).agg(integral=("integral", "mean"), desvio=("desvio", "mean"),
                                                iteracoes=("iteracoes", "mean")).reset_index()
    md += ["", "## D2 e D3 — agrupamento e número de centros candidatos\n",
           "| Método | Integral primal | Desvio final mediano | Subproblemas por execução |", "|---|---:|---:|---:|"]
    for m, g in pi.groupby("metodo"):
        md.append(f"| `{m}` | {g['integral'].mean():.4f} | {100 * g['desvio'].median():.2f}% | {g['iteracoes'].mean():.0f} |")
    linhas = []
    for campo, k in (("integral", 1.0), ("desvio", 100.0)):
        piv = pi.pivot(index="instancia", columns="metodo", values=campo)
        bloco = []
        for diag, a, b in COMPARACOES:
            dif = (piv[b] - piv[a]).dropna()  # variante − referência
            nz = dif[dif.abs() > 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
            lo, hi = bootstrap_ic(dif.to_numpy(), est="mean")
            bloco.append({"diag": diag, "referencia": a, "variante": b, "metrica": campo, "n": int(len(dif)),
                          "dif": float(k * dif.mean()), "lo": float(k * lo), "hi": float(k * hi),
                          "variante_melhor": int((dif < 0).sum()), "referencia_melhor": int((dif > 0).sum()), "p": p})
        for r_, pa in zip(bloco, holm([b["p"] for b in bloco])):
            r_["p_holm"] = pa
        linhas += bloco
    out["D2_D3"] = linhas
    md += ["", "Diferença média **variante − referência** (negativo favorece a variante); desvio em pontos percentuais.\n",
           "| Diag. | Referência | Variante | Métrica | Diferença (IC 95%) | Variante melhor / referência melhor | p | p (Holm) |",
           "|---|---|---|---|---|---|---:|---:|"]
    for r_ in linhas:
        md.append(f"| {r_['diag']} | `{r_['referencia']}` | `{r_['variante']}` | {r_['metrica']} | "
                  f"{r_['dif']:+.4f} ({r_['lo']:+.4f} a {r_['hi']:+.4f}) | {r_['variante_melhor']} / "
                  f"{r_['referencia_melhor']} | {r_['p']:.3f} | {r_['p_holm']:.3f} |")
    out["medicao"] = {"execucoes": int(len(x)), "razao_mediana": float(x["razao_cpu"].median()),
                      "marcadas": int((x["razao_cpu"] < 0.9).sum())}
    (dest / "analise.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "analise.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
