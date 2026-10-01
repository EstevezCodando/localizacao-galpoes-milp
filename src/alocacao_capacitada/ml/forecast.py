"""Previsão de população por município, com backtest de origem móvel e intervalos calibrados.

Dados: estimativas anuais oficiais do IBGE (tabela 6579). Ressalvas que devem acompanhar qualquer
número daqui:
  * as estimativas são modeladas pelo IBGE, não contagens; o backtest mede concordância com a
    série oficial, não com a população "verdadeira";
  * existem duas safras: 2001–2021 (projeção antes do Censo 2022) e 2024–2025 (refeita após o
    Censo). Comparar níveis entre safras mistura revisão metodológica com crescimento. Por isso o
    backtest usa só a safra 2001–2021 e a projeção parte do nível mais recente (2025);
  * 2007 e 2010 não constam da tabela: são interpoladas em log (e marcadas como tais).

Família de modelos (todas escolhem a TAXA de crescimento; o nível parte da última observação):
  taxa efetiva = w · taxa do município + (1 − w) · taxa de referência do grupo (UF × faixa de
  população), com amortecimento opcional `phi` ao longo do horizonte. w, janela e phi são
  escolhidos em origens ANTIGAS e avaliados em uma origem mais nova (disciplina temporal).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

YEARS_PRE = list(range(2001, 2022))
INTERPOLATED = (2007, 2010)


def pop_series(municipios: pd.DataFrame) -> pd.DataFrame:
    """Matriz município × ano (2001–2021) com 2007 e 2010 interpolados em log."""
    cols = {y: f"est_{y}" for y in range(2001, 2022) if f"est_{y}" in municipios}
    frame = municipios.set_index("cod")[list(cols.values())]
    frame.columns = list(cols.keys())
    frame = frame.dropna(how="any")
    log = frame.apply(np.log)
    for year in INTERPOLATED:
        log[year] = 0.5 * (log[year - 1] + log[year + 1])
    out: pd.DataFrame = log[sorted(log.columns)].apply(np.exp)
    return out


@dataclass(frozen=True)
class Config:
    window: int
    w: float
    phi: float

    def label(self) -> str:
        return f"janela{self.window}_w{self.w:g}_phi{self.phi:g}"


def size_band(pop: pd.Series) -> pd.Series:
    bins = [0, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000, np.inf]
    return pd.cut(pop, bins, labels=False, include_lowest=True).astype(int)


def predict(
    series: pd.DataFrame, uf: pd.Series, origin: int, horizon: int, config: Config
) -> pd.Series:
    """População prevista em origin+horizon usando só dados até `origin`."""
    last = series[origin]
    past = series[origin - config.window]
    g_muni = (last / past) ** (1.0 / config.window) - 1.0
    group = pd.DataFrame({"uf": uf.reindex(series.index), "band": size_band(last), "g": g_muni})
    g_ref = group.groupby(["uf", "band"])["g"].transform("median")
    g_eff = config.w * g_muni + (1.0 - config.w) * g_ref
    steps = sum(config.phi**i for i in range(1, horizon + 1))
    return last * np.exp(np.log1p(g_eff) * steps)


def backtest(
    series: pd.DataFrame,
    uf: pd.Series,
    origins: list[int],
    horizons: list[int],
    configs: list[Config],
) -> pd.DataFrame:
    rows = []
    for origin, h, cfg in product(origins, horizons, configs):
        target = origin + h
        if target not in series.columns or origin - cfg.window < int(min(series.columns)):
            continue
        pred = predict(series, uf, origin, h, cfg)
        actual = series[target]
        ape = (pred / actual - 1).abs()
        big = actual >= 50_000
        rows.append(
            {
                "origem": origin,
                "horizonte": h,
                "config": cfg.label(),
                "janela": cfg.window,
                "w": cfg.w,
                "phi": cfg.phi,
                "mediana_ape": float(ape.median()),
                "ape_medio_pop50k": float(ape[big].mean()),
                "vies_total": float(pred.sum() / actual.sum() - 1),
                "p90_ape": float(ape.quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def default_configs() -> list[Config]:
    return [
        Config(k, w, phi)
        for k, w, phi in product((1, 3, 5, 8), (0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.9))
    ]


def naive_config() -> Config:
    """Nível constante: taxa zero (w=0 e referência 0 só vale com g_ref=0; usa-se à parte)."""
    return Config(window=1, w=0.0, phi=0.0)


def log_error_quantiles(
    series: pd.DataFrame, uf: pd.Series, origins: list[int], horizon: int, config: Config
) -> pd.DataFrame:
    """Quantis 5%/50%/95% do erro em log (previsto/real) por faixa de população, a um horizonte."""
    parts = []
    for origin in origins:
        pred = predict(series, uf, origin, horizon, config)
        actual = series[origin + horizon]
        parts.append(pd.DataFrame({"band": size_band(series[origin]), "e": np.log(pred / actual)}))
    e = pd.concat(parts)
    quantiles = {"q05": 0.05, "q50": 0.5, "q95": 0.95}
    by_band = e.groupby("band")["e"]
    return pd.DataFrame({name: by_band.quantile(q) for name, q in quantiles.items()})


# ---------------------------------------------------------------------------------------------
# Teste com contagens REAIS: Censo 2000 e 2010 -> previsão do Censo 2022
# ---------------------------------------------------------------------------------------------


def census_predict(m: pd.DataFrame, w: float, phi: float, years: int = 12) -> pd.Series:
    """População de 2022 prevista só com os Censos de 2000 e 2010 (sem usar o de 2022)."""
    x = m.set_index("cod")[["uf", "censo_2000", "censo_2010"]].dropna()
    g = (x["censo_2010"] / x["censo_2000"]) ** (1 / 10) - 1
    ref = pd.DataFrame({"uf": x["uf"], "band": size_band(x["censo_2010"]), "g": g})
    g_ref = ref.groupby(["uf", "band"])["g"].transform("median")
    g_eff = w * g + (1 - w) * g_ref
    steps = sum(phi**i for i in range(1, years + 1))
    return x["censo_2010"] * np.exp(np.log1p(g_eff) * steps)


def census_backtest(m: pd.DataFrame, ws: list[float], phis: list[float]) -> pd.DataFrame:
    """Erro contra o Censo 2022 real, por (w, phi): mediana do erro absoluto, viés e quantis."""
    actual = m.set_index("cod")["censo_2022"]
    rows = []
    for w, phi in product(ws, phis):
        pred = census_predict(m, w, phi)
        a = actual.reindex(pred.index)
        ok = a.notna()
        err = pd.Series(np.log((pred[ok] / a[ok]).to_numpy()), index=pred[ok].index)
        big = a[ok] >= 50_000
        ape = (err.abs()).apply(np.expm1)
        rows.append(
            {
                "w": w,
                "phi": phi,
                "n": int(ok.sum()),
                "mediana_ape": float(ape.median()),
                "ape_medio_pop50k": float(ape[big].mean()),
                "vies_total": float(pred[ok].sum() / a[ok].sum() - 1),
                "erro_log_q05": float(err.quantile(0.05)),
                "erro_log_q95": float(err.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)
