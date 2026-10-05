"""Análise do teste fechado conforme docs/pesquisa/10-preregistro-teste-fechado.md.

Entrada: pares conjunto=diretório de execução (results/pesquisa/clns/<run_id>). O horizonte de
cada conjunto vem da coluna `orcamento`. Sementes são agregadas por instância antes de qualquer
estatística. O desvio final é g(T), reconstruído da trajetória; execução sem solução até T conta
como desvio 1 (emenda de 05/10/2026, seção 7 do pré-registro). Saída: results/pesquisa/fechado/analise.json e tabelas.md.

uv run python scripts/analisar_fechado.py teste=<run> gen_corredor=<run> gen_escala=<run> \
    holmberg=<run> olist=<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from alocacao_capacitada.pesquisa.analise import bootstrap_ic, integral_primal

OUT = Path("results/pesquisa/fechado")
AGREGADO = ("teste", "gen_corredor", "gen_escala")  # conjunto das hipóteses (pré-registro, seção 4)
HIPOTESES = (("H1", "adaptativa:pl", "completo"), ("H2", "adaptativa:gnn", "adaptativa:pl"),
             ("H3", "hibridopl:rotacao", "adaptativa:pl"), ("H4", "adaptativa:pl", "kernel"))
TOL = 1e-4


def carregar(conjunto: str, run: Path) -> pd.DataFrame:
    d = pd.read_csv(run / "resultados.csv")
    d["conjunto"] = conjunto
    d["valida"] = d["valida"].astype(bool)
    d["traj"] = d["trajetoria"].map(json.loads)
    bks = d[d["valida"]].groupby("instancia")["objetivo"].min()
    bks = pd.concat([bks, d.groupby("instancia")["rotulo_melhor"].first()], axis=1).min(axis=1)
    d["bks"] = d["instancia"].map(bks)
    # Desvio final = g(T): melhor incumbente com instante <= T (não o objetivo devolvido, que pode
    # ter melhorado durante um estouro). Sem solução até T => g(T) = 1, como na integral; a
    # execução permanece no denominador. O desvio pelo objetivo de retorno fica em outra coluna.
    d["obj_T"] = [min((o for t, o in tr if t <= h + 1e-9), default=np.inf)
                  for tr, h in zip(d["traj"], d["orcamento"])]
    d["desvio"] = np.minimum(1.0, d["obj_T"] / d["bks"] - 1)
    d["desvio_retorno"] = np.where(d["valida"], d["objetivo"] / d["bks"] - 1, np.nan)
    d["mudou_apos_T"] = d["valida"] & (d["objetivo"] < d["obj_T"] - 1e-9)
    d["integral"] = [integral_primal(t, b, h) for t, b, h in zip(d["traj"], d["bks"], d["orcamento"])]
    return d


def por_instancia(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["conjunto", "instancia", "metodo"]).agg(
        integral=("integral", "mean"), desvio=("desvio", "mean")).reset_index()


def tabela(pi: pd.DataFrame, base: str = "completo") -> list[dict[str, object]]:
    piv = pi.pivot(index="instancia", columns="metodo", values="desvio")
    linhas = []
    for m, g in pi.groupby("metodo"):
        lo, hi = bootstrap_ic(g["integral"].to_numpy(), est="mean")
        r: dict[str, object] = {
            "metodo": m, "n": int(len(g)), "integral": float(g["integral"].mean()),
            "integral_lo": float(lo), "integral_hi": float(hi),
            "desvio_mediano": float(g["desvio"].median()), "desvio_medio": float(g["desvio"].mean()),
            "ate_1pct": float((g["desvio"] <= 0.01).mean()),
            "ate_01pct": float((g["desvio"] <= 0.001).mean())}
        if base in piv and m != base:
            dif = (piv[m] - piv[base]).dropna()
            r |= {"vitorias": int((dif < -TOL).sum()), "empates": int((dif.abs() <= TOL).sum()),
                  "derrotas": int((dif > TOL).sum())}
        linhas.append(r)
    return sorted(linhas, key=lambda r: r["integral"])  # type: ignore[arg-type,return-value]


def holm(ps: list[float]) -> list[float]:
    ordem = np.argsort(ps)
    adj = np.empty(len(ps))
    atual = 0.0
    for k, i in enumerate(ordem):
        atual = max(atual, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, atual)
    return [float(a) for a in adj]


def hipoteses(pi: pd.DataFrame, campo: str) -> list[dict[str, object]]:
    piv = pi.pivot(index=["conjunto", "instancia"], columns="metodo", values=campo)
    linhas = []
    for nome, a, b in HIPOTESES:
        dif = (piv[a] - piv[b]).dropna()
        nz = dif[dif.abs() > 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
        lo, hi = bootstrap_ic(dif.to_numpy(), est="mean")
        linhas.append({"hipotese": nome, "a": a, "b": b, "n": int(len(dif)),
                       "dif_media": float(dif.mean()), "dif_lo": float(lo), "dif_hi": float(hi),
                       "a_melhor": int((dif < -(TOL if campo == "desvio" else 0)).sum()),
                       "b_melhor": int((dif > (TOL if campo == "desvio" else 0)).sum()), "p": p})
    for r, pa in zip(linhas, holm([r["p"] for r in linhas])):  # type: ignore[misc]
        r["p_holm"] = pa
    return linhas


def medicao(d: pd.DataFrame) -> dict[str, object]:
    return {"execucoes": int(len(d)), "razao_mediana": float(d["razao_cpu"].median()),
            "razao_p05": float(d["razao_cpu"].quantile(0.05)),
            "marcadas": int((d["razao_cpu"] < 0.9).sum()),
            "estouro_p95_s": float(d["estouro_s"].quantile(0.95)),
            "estouro_max_s": float(d["estouro_s"].max()), "invalidas": int((~d["valida"]).sum()),
            "sem_solucao_ate_T": int((~np.isfinite(d["obj_T"])).sum()),
            "melhoraram_apos_T": int(d["mudou_apos_T"].sum())}


def fmt_tabela(linhas: list[dict[str, object]]) -> str:
    out = ["| Método | n | Integral primal (IC 95%) | Desvio mediano | Desvio médio | Até 1% | Até 0,1% | V/E/D contra SCIP |",
           "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in linhas:
        ved = f"{r['vitorias']}/{r['empates']}/{r['derrotas']}" if "vitorias" in r else "–"
        out.append(f"| `{r['metodo']}` | {r['n']} | {r['integral']:.4f} ({r['integral_lo']:.4f}–"
                   f"{r['integral_hi']:.4f}) | {100 * r['desvio_mediano']:.2f}% | "  # type: ignore[operator]
                   f"{100 * r['desvio_medio']:.2f}% | {100 * r['ate_1pct']:.1f}% | "  # type: ignore[operator]
                   f"{100 * r['ate_01pct']:.1f}% | {ved} |")  # type: ignore[operator]
    return "\n".join(out)


def fmt_hip(linhas: list[dict[str, object]], k: float) -> str:
    out = ["| Hipótese | A | B | n | A − B (IC 95%) | A melhor / B melhor | p | p (Holm) |",
           "|---|---|---|---:|---|---|---:|---:|"]
    for r in linhas:
        out.append(f"| {r['hipotese']} | `{r['a']}` | `{r['b']}` | {r['n']} | "
                   f"{k * r['dif_media']:+.4f} ({k * r['dif_lo']:+.4f} a {k * r['dif_hi']:+.4f}) | "  # type: ignore[operator]
                   f"{r['a_melhor']} / {r['b_melhor']} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    return "\n".join(out)


def main() -> None:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    global OUT
    OUT = Path(runs.pop("saida", str(OUT)))  # ex.: saida=results/pesquisa/fechado_nuvem
    dados = {c: carregar(c, Path(r)) for c, r in runs.items()}
    res: dict[str, object] = {"runs": runs}
    md = ["# Teste fechado — tabelas\n"]
    todos = pd.concat(dados.values())
    for c, d in dados.items():
        pi = por_instancia(d)
        t = tabela(pi)
        res[c] = {"tabela": t, "medicao": medicao(d)}
        md += [f"## {c} ({d['instancia'].nunique()} instâncias, {d['orcamento'].iloc[0]:.0f} s)\n",
               fmt_tabela(t), ""]
        cert = d[d["metodo"].str.startswith("adaptativa")].groupby("metodo")["certificado"].sum()
        res[c]["certificados"] = {k: int(v) for k, v in cert.items()}  # type: ignore[index]
        md += ["Certificado de ótimo antes do último estágio: "
               + ", ".join(f"`{k}` {int(v)}" for k, v in cert.items()) + "\n"]
        if c == "holmberg":
            otimo = d.groupby("instancia")["rotulo_melhor"].first()
            d2 = d.assign(no_otimo=d["objetivo"] <= d["instancia"].map(otimo) * (1 + 1e-6))
            ating = d2.groupby(["metodo", "instancia"])["no_otimo"].mean().groupby("metodo").agg(
                todas=lambda s: int((s == 1).sum()), alguma=lambda s: int((s > 0).sum()))
            res[c]["otimo_atingido"] = ating.reset_index().to_dict("records")  # type: ignore[index]
            md += ["Instâncias em que o ótimo publicado foi atingido (em todas as sementes / em "
                   "alguma): " + ", ".join(f"`{m}` {r.todas}/{r.alguma}" for m, r in ating.iterrows())
                   + "\n"]
    usados = [c for c in AGREGADO if c in dados]
    if usados:
        ag = pd.concat([dados[c] for c in usados])
        pi = por_instancia(ag)
        res["agregado"] = {"conjuntos": usados, "tabela": tabela(pi),
                           "hipoteses_integral": hipoteses(pi, "integral"),
                           "hipoteses_desvio": hipoteses(pi, "desvio")}
        limpo = por_instancia(ag[ag["razao_cpu"] >= 0.9])
        res["agregado"]["sensibilidade_sem_marcadas"] = {  # type: ignore[index]
            "hipoteses_integral": hipoteses(limpo, "integral")}
        md += [f"## Agregado ({' + '.join(usados)}, {pi['instancia'].nunique()} instâncias)\n",
               fmt_tabela(res["agregado"]["tabela"]), "",  # type: ignore[index]
               "### Hipóteses — integral primal (A − B < 0 favorece A)\n",
               fmt_hip(res["agregado"]["hipoteses_integral"], 1.0), "",  # type: ignore[index]
               "### Hipóteses — desvio final, em pontos percentuais\n",
               fmt_hip(res["agregado"]["hipoteses_desvio"], 100.0), "",  # type: ignore[index]
               "### Sensibilidade: integral primal sem as execuções marcadas por contenção\n",
               fmt_hip(res["agregado"]["sensibilidade_sem_marcadas"]["hipoteses_integral"], 1.0), ""]  # type: ignore[index]
    res["medicao_total"] = medicao(todos)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analise.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "tabelas.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
