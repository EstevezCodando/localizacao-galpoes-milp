"""Agregados de uma execução do executor CLNS para os gráficos do documento.

Sempre: sementes agregadas por instância ANTES de qualquer estatística entre instâncias.
BKS por instância = melhor custo validado entre todos os métodos e o rótulo.

Saída: <run>/agregados.json com
  metodos      integral média + IC95, desvio final mediano, % até 1% e 0,1%, % vitórias
  pareado      vitórias/empates/derrotas de cada método contra `--base` (desvio final, tol 0,01%)
  por_celula   desvio final mediano por (família, razão) e método
  anytime      gap mediano (entre instâncias) do melhor incumbente em t = 0..T s
  tempo        frações de tempo por fase nos métodos que as registram
  medicao      distribuição da razão CPU/parede e estouros

uv run python scripts/agregados_clns.py results/pesquisa/clns/<run_id> --horizonte 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa.analise import bootstrap_ic, integral_primal


def gap_em(traj: list[tuple[float, float]], bks: float, t: float) -> float:
    melhor = np.inf
    for ti, o in sorted(traj):
        if ti > t:
            break
        melhor = min(melhor, o)
    return min(1.0, (melhor - bks) / bks) if np.isfinite(melhor) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--horizonte", type=float, required=True)
    ap.add_argument("--base", default="completo")
    a = ap.parse_args()
    df = pd.read_csv(a.run / "resultados.csv")
    df["valida"] = df["valida"].astype(bool)
    df["traj"] = df["trajetoria"].map(json.loads)
    bks = df[df["valida"]].groupby("instancia")["objetivo"].min()
    bks = pd.concat([bks, df.groupby("instancia")["rotulo_melhor"].first()], axis=1).min(axis=1)
    df["bks"] = df["instancia"].map(bks)
    df["desvio"] = np.where(df["valida"], df["objetivo"] / df["bks"] - 1, np.nan)
    df["integral"] = [integral_primal(t, b, a.horizonte) for t, b in zip(df["traj"], df["bks"])]
    grade = np.arange(0, a.horizonte + 1e-9, 1.0)
    df["curva"] = [[gap_em(t, b, g) for g in grade] for t, b in zip(df["traj"], df["bks"])]
    # agregação por instância
    por_inst = df.groupby(["instancia", "metodo"]).agg(
        integral=("integral", "mean"), desvio=("desvio", "mean"),
        familia=("familia", "first"), razao=("razao", "first"),
        n_seeds=("seed_solver", "nunique")).reset_index()
    curvas = df.groupby(["instancia", "metodo"])["curva"].apply(
        lambda s: np.mean(np.stack(s.to_list()), axis=0)).reset_index()
    out: dict[str, object] = {"horizonte": a.horizonte, "n_instancias": int(df["instancia"].nunique())}
    melhor_por_inst = por_inst.groupby("instancia")["desvio"].transform("min")
    por_inst["vence"] = por_inst["desvio"] <= melhor_por_inst + 1e-4
    linhas = []
    for m, g in por_inst.groupby("metodo"):
        lo, hi = bootstrap_ic(g["integral"].to_numpy(), est="mean")
        linhas.append({"metodo": m, "integral": float(g["integral"].mean()), "integral_lo": lo,
                       "integral_hi": hi, "desvio_mediano": float(g["desvio"].median()),
                       "ate_1pct": float((g["desvio"] <= 0.01).mean()),
                       "ate_01pct": float((g["desvio"] <= 0.001).mean()),
                       "melhor_em": float(g["vence"].mean()), "sementes": int(g["n_seeds"].max())})
    out["metodos"] = sorted(linhas, key=lambda r: r["integral"])
    piv = por_inst.pivot(index="instancia", columns="metodo", values="desvio")
    par = []
    if a.base in piv:
        for m in piv:
            if m == a.base:
                continue
            d = (piv[m] - piv[a.base]).dropna()
            par.append({"metodo": m, "vitorias": int((d < -1e-4).sum()),
                        "empates": int((d.abs() <= 1e-4).sum()), "derrotas": int((d > 1e-4).sum())})
    out["pareado"] = {"base": a.base, "linhas": par}
    out["por_celula"] = [
        {"familia": f, "razao": float(r), "metodo": m, "desvio_mediano": float(g["desvio"].median())}
        for (f, r, m), g in por_inst.groupby(["familia", "razao", "metodo"])]
    out["anytime"] = {"grade": grade.tolist(), "metodos": {
        m: np.median(np.stack(g["curva"].to_list()), axis=0).round(5).tolist()
        for m, g in curvas.groupby("metodo")}}
    fases = [c for c in ("t_pl", "t_inicial", "t_selecao", "t_sub", "t_adaptativa", "t_inferencia")
             if c in df]
    out["tempo"] = {m: {c: float(g[c].mean()) for c in fases if g[c].notna().any()}
                    | {"t_parede": float(g["t_parede_s"].mean())}
                    for m, g in df.groupby("metodo") if "t_parede_s" in df}
    if "razao_cpu" in df:
        out["medicao"] = {
            "razao_cpu_quantis": df["razao_cpu"].quantile([0.01, 0.05, 0.5, 0.95]).round(4).to_dict(),
            "contencao_suspeita": int(df["contencao_suspeita"].sum()),
            "estouro_p95_s": float(df["estouro_s"].quantile(0.95)),
            "estouro_max_s": float(df["estouro_s"].max()), "execucoes": int(len(df)),
            "histograma_razao": np.histogram(df["razao_cpu"].clip(0.8, 1.02),
                                             bins=np.arange(0.8, 1.025, 0.01))[0].tolist()}
    (a.run / "agregados.json").write_text(json.dumps(out, indent=1, default=float))
    print(pd.DataFrame(out["metodos"]).round(4).to_string(index=False))
    print(pd.DataFrame(par).to_string(index=False))
    print(json.dumps(out.get("medicao", {}), indent=1))


if __name__ == "__main__":
    main()
