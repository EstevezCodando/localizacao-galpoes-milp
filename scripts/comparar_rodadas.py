"""Compara as duas rodadas do teste fechado segundo a emenda 9 do pré-registro.

Regra fixada antes da segunda rodada: uma hipótese é CONFIRMADA somente se for significativa
(Holm, 5%) na rodada da nuvem E tiver o mesmo sinal na rodada local; caso contrário, NÃO REPLICADA.
Tempos e integrais absolutos não são comparáveis entre rodadas (hardware diferente).

uv run python scripts/comparar_rodadas.py results/pesquisa/fechado results/pesquisa/fechado_nuvem
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    local = json.loads((Path(sys.argv[1]) / "analise.json").read_text(encoding="utf-8"))
    nuvem = json.loads((Path(sys.argv[2]) / "analise.json").read_text(encoding="utf-8"))
    out: dict[str, object] = {"regra": "confirmada = significativa (Holm 5%) na nuvem e mesmo sinal na local"}
    md = ["# Teste fechado — as duas rodadas lado a lado\n"]
    for chave, nome, k in (("hipoteses_integral", "Integral primal", 1.0),
                           ("hipoteses_desvio", "Desvio final (p.p.)", 100.0)):
        linhas = []
        md += [f"## {nome}\n",
               "| Hipótese | A − B local (IC 95%) | p Holm local | A − B nuvem (IC 95%) | p Holm nuvem | Veredito |",
               "|---|---|---:|---|---:|---|"]
        for a, b in zip(local["agregado"][chave], nuvem["agregado"][chave]):
            mesmo = (a["dif_media"] > 0) == (b["dif_media"] > 0)
            sig = b["p_holm"] < 0.05
            ver = "confirmada" if (sig and mesmo) else "não replicada" if (a["p_holm"] < 0.05 or sig) else "sem diferença nas duas"
            linhas.append({"hipotese": a["hipotese"], "a": a["a"], "b": a["b"], "local": a, "nuvem": b,
                           "mesmo_sinal": mesmo, "veredito": ver})
            md.append(f"| {a['hipotese']} `{a['a']}` − `{a['b']}` | {k * a['dif_media']:+.4f} "
                      f"({k * a['dif_lo']:+.4f} a {k * a['dif_hi']:+.4f}) | {a['p_holm']:.4f} | "
                      f"{k * b['dif_media']:+.4f} ({k * b['dif_lo']:+.4f} a {k * b['dif_hi']:+.4f}) | "
                      f"{b['p_holm']:.4f} | {ver} |")
        out[chave] = linhas
        md.append("")
    md += ["## Tabela do conjunto agregado (64 instâncias sintéticas)\n",
           "| Método | Integral local | Integral nuvem | Desvio mediano local | Desvio mediano nuvem |",
           "|---|---:|---:|---:|---:|"]
    n = {r["metodo"]: r for r in nuvem["agregado"]["tabela"]}
    for r in local["agregado"]["tabela"]:
        q = n[r["metodo"]]
        md.append(f"| `{r['metodo']}` | {r['integral']:.4f} | {q['integral']:.4f} | "
                  f"{100 * r['desvio_mediano']:.2f}% | {100 * q['desvio_mediano']:.2f}% |")
    md += ["", "## Medição\n", "| Rodada | Execuções | Mediana CPU/parede | Marcadas (< 0,9) |", "|---|---:|---:|---:|"]
    for rot, d in (("local", local), ("nuvem", nuvem)):
        m = d["medicao_total"]
        md.append(f"| {rot} | {m['execucoes']} | {m['razao_mediana']:.3f} | {m['marcadas']} |")
    dest = Path(sys.argv[2])
    (dest / "comparacao.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "comparacao.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
