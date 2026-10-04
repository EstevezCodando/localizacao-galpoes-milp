"""Análise estatística (doc 10): comparação pareada por instância, bootstrap por instância,
Wilcoxon com correção de Holm, tempo até alvo (censurado) e integral primal (Berthold, 2013).

Convenções: desvio = (U - BKS) / BKS, BKS = melhor custo VALIDADO conhecido para a instância
(mínimo entre todos os métodos e o rótulo), ou o ótimo publicado/provado quando existe.
Timeout não é tempo de conclusão: tempo até alvo não atingido fica censurado no orçamento.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ic(x: np.ndarray, est: str = "median", n: int = 4000, seed: int = 0
                 ) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    f = np.median if est == "median" else np.mean
    b = f(rng.choice(x, (n, x.size)), axis=1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def media_geom_deslocada(x: np.ndarray, s: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x + s))) - s)


def integral_primal(traj: list[tuple[float, float]], bks: float, horizonte: float) -> float:
    """Integral do gap primal normalizado em [0, horizonte], dividida pelo horizonte (0 = ótimo
    desde t = 0; 1 = sem solução o tempo todo). gap(t) = min(1, (inc - bks) / bks)."""
    if not np.isfinite(bks) or bks < 0 or horizonte <= 0:
        raise ValueError("BKS finito não negativo e horizonte positivo são obrigatórios")
    t_prev, g_prev, area = 0.0, 1.0, 0.0
    for t, o in sorted(traj):
        if not np.isfinite(t) or t < 0 or not np.isfinite(o):
            continue
        if t > horizonte:
            break
        area += g_prev * (t - t_prev)
        t_prev = t
        gap = float(o > 1e-9) if bks == 0 else max(0.0, (o - bks) / bks)
        g_prev = min(g_prev, gap)
    area += g_prev * (horizonte - t_prev)
    return area / horizonte


def tempo_alvo(traj: list[tuple[float, float]], alvo: float,
               horizonte: float = np.inf) -> float:
    for t, o in sorted(traj):
        if 0 <= t <= horizonte and np.isfinite(o) and o <= alvo * (1 + 1e-9):
            return t
    return np.inf


def holm(pvals: dict[str, float]) -> dict[str, float]:
    itens = sorted(pvals.items(), key=lambda kv: kv[1])
    k = len(itens)
    out, acum = {}, 0.0
    for r, (nome, p) in enumerate(itens):
        acum = max(acum, min(1.0, (k - r) * p))
        out[nome] = acum
    return out


def preparar(df: pd.DataFrame, horizonte: float, ref_externa: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["traj"] = df["trajetoria"].map(json.loads)
    df["traj"] = [t if ok else [] for t, ok in zip(df["traj"], df["valida"], strict=True)]
    val = df[df["valida"]]
    bks = val.groupby("instancia")["objetivo"].min()
    if ref_externa:  # ótimo publicado: é a referência, e nenhum método pode ficar abaixo
        bks = df.groupby("instancia")["rotulo_melhor"].first()
        assert (val["objetivo"] >= val["instancia"].map(bks) * (1 - 1e-6)).all(), \
            "custo abaixo do ótimo publicado: erro de validação ou de transcrição"
    else:
        lab = df.groupby("instancia")["rotulo_melhor"].first()
        bks = pd.concat([bks, lab], axis=1).min(axis=1)
    df["bks"] = df["instancia"].map(bks)
    # O custo final após o prazo continua no CSV, mas não vence uma comparação a T fixo.
    df["objetivo_no_prazo"] = [
        min((o for t, o in traj if 0 <= t <= horizonte and np.isfinite(o)), default=np.inf)
        for traj in df["traj"]
    ]
    df["sucesso_no_prazo"] = np.isfinite(df["objetivo_no_prazo"]) & df["valida"]
    df["desvio"] = np.where(
        df["sucesso_no_prazo"],
        (df["objetivo_no_prazo"] - df["bks"]) / df["bks"].clip(lower=1e-9), np.nan)
    df["integral_primal"] = [integral_primal(t, b, horizonte)
                              for t, b in zip(df["traj"], df["bks"], strict=True)]
    for eps in (0.001, 0.01):
        df[f"t_alvo_{eps}"] = [tempo_alvo(t, b * (1 + eps), horizonte)
                               for t, b in zip(df["traj"], df["bks"], strict=True)]
    return df


def resumo(df: pd.DataFrame, horizonte: float, base: str = "completo") -> pd.DataFrame:
    linhas = []
    piv = df.pivot_table(index="instancia", columns="metodo", values="desvio")
    for metodo, g in df.groupby("metodo"):
        d = g["desvio"].to_numpy()
        # Repetições de sementes não são instâncias independentes.
        por_instancia = g.groupby("instancia")["desvio"].mean().to_numpy()
        lo, hi = bootstrap_ic(por_instancia)
        lin = {
            "metodo": metodo, "n": len(g), "validas": float(g["valida"].mean()),
            "n_instancias": g["instancia"].nunique(),
            "desvios_condicionais_ao_sucesso": True,
            "desvio_mediano": float(np.nanmedian(por_instancia)), "ic95_lo": lo, "ic95_hi": hi,
            "desvio_medio": float(np.nanmean(por_instancia)),
            "p90": float(np.nanquantile(por_instancia, 0.9)),
            "ate_0.1%": float(np.mean(d <= 0.001)), "ate_1%": float(np.mean(d <= 0.01)),
            "integral_primal": float(g["integral_primal"].mean()),
            "t_total_med": float(g["t_total"].median()),
            "atinge_1%": float((g["t_alvo_0.01"] <= horizonte).mean()),
            "t_alvo_1%_med": float(np.median(np.minimum(g["t_alvo_0.01"], horizonte))),
            "certificado": float(g["certificado"].mean()),
        }
        if metodo != base and base in piv:
            par = piv[[metodo, base]].dropna()
            lin["vence_base"] = float((par[metodo] < par[base] - 1e-9).mean())
            lin["perde_base"] = float((par[metodo] > par[base] + 1e-9).mean())
        linhas.append(lin)
    return pd.DataFrame(linhas).sort_values("desvio_mediano")


def testes_pareados(df: pd.DataFrame, contra: str, metodos: list[str]) -> pd.DataFrame:
    piv = df.pivot_table(index="instancia", columns="metodo", values="integral_primal")
    pv, efeito = {}, {}
    for m in metodos:
        if m not in piv or contra not in piv or m == contra:
            continue
        par = piv[[m, contra]].dropna()
        dif = par[m] - par[contra]
        if (dif != 0).sum() < 5:
            continue
        pv[m] = float(wilcoxon(par[m], par[contra]).pvalue)
        efeito[m] = float(np.median(dif))
    adj = holm(pv)
    return pd.DataFrame({"metodo": list(pv), "contra": contra, "dif_mediana_integral":
                         [efeito[m] for m in pv], "p": list(pv.values()),
                         "p_holm": [adj[m] for m in pv]})
