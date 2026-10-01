"""Backtest de previsão de população, calibração de intervalos e projeção 2026–2030.

Uso:  uv run python -m alocacao_capacitada.ml.run_forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.ml.forecast import (
    Config,
    backtest,
    default_configs,
    log_error_quantiles,
    pop_series,
    predict,
)

SELECTION_ORIGINS = [2012, 2014, 2016]  # escolha de w, janela e phi (verdade até 2020)
TEST_ORIGIN = [2017]  # avaliação (verdade até 2021)
HORIZONS = [1, 2, 3, 4]


def run_census_and_projection(m: pd.DataFrame, out: Path) -> None:
    """Teste contra o Censo 2022 real, escolha de (w, phi) por validação espacial e projeção."""
    from sklearn.model_selection import GroupKFold

    from alocacao_capacitada.ml.forecast import census_backtest, census_predict, size_band

    ws, phis = [0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.97, 0.94, 0.9, 0.85]
    grid = census_backtest(m, ws, phis)
    grid.to_csv(out / "ml_previsao_censo.csv", index=False)

    # Escolha de (w, phi) com validação espacial: cada UF é avaliada com a escolha feita nas outras.
    actual = m.set_index("cod")["censo_2022"]
    ufs = m.set_index("cod")["uf"]
    preds = {(w, phi): census_predict(m, w, phi) for w in ws for phi in phis}
    idx = preds[(1.0, 1.0)].index
    a = actual.reindex(idx)
    groups = ufs.reindex(idx)
    cv_rows = []
    for tr, te in GroupKFold(n_splits=5).split(idx, groups=groups):

        def score(ix: np.ndarray, key: tuple[float, float]) -> float:
            err = np.abs(np.expm1(np.log(preds[key].iloc[ix].to_numpy() / a.iloc[ix].to_numpy())))
            return float(err[a.iloc[ix].to_numpy() >= 50_000].mean())

        best_key = min(preds, key=lambda k: score(tr, k))
        cv_rows.append(
            {
                "escolha": f"w={best_key[0]:g}, phi={best_key[1]:g}",
                "ape_pop50k_dobra_teste": score(te, best_key),
                "ape_pop50k_sem_decaimento": score(te, (1.0, 1.0)),
            }
        )
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "ml_previsao_censo_cv_espacial.csv", index=False)

    # Projeção 2026-2030 a partir da estimativa oficial mais recente (2025).
    # Escolha mais frequente entre as dobras espaciais (mais estável que o melhor ponto da grade).
    mode = cv["escolha"].mode().iloc[0]
    w = float(mode.split(",")[0].split("=")[1])
    phi = float(mode.split(",")[1].split("=")[1])
    base = m.set_index("cod")
    ok = base["est_2024"].notna() & base["est_2025"].notna()
    x = base[ok]
    g = x["est_2025"] / x["est_2024"] - 1
    ref = pd.DataFrame({"uf": x["uf"], "band": size_band(x["est_2025"]), "g": g})
    g_ref = ref.groupby(["uf", "band"])["g"].transform("median")
    # Trava de plausibilidade: taxa efetiva limitada ao intervalo 1%–99% observado entre os
    # municípios no Censo 2010–2022 (evita extrapolar saltos de revisão, como um +10% em um ano).
    g_eff = (w * g + (1 - w) * g_ref).clip(lower=-0.0205, upper=0.0385)
    # Erro em log por faixa de população medido contra o Censo real (12 anos), por ano.
    pred12 = census_predict(m, w, phi)
    a12 = actual.reindex(pred12.index)
    band12 = size_band(m.set_index("cod").loc[pred12.index, "censo_2010"])
    e = pd.DataFrame({"band": band12, "e": np.log((pred12 / a12).to_numpy())}).dropna()
    q = e.groupby("band")["e"].quantile
    rate = pd.DataFrame({"q05": q(0.05) / 12, "q95": q(0.95) / 12})
    out_frame = pd.DataFrame({"pop_2025": x["est_2025"]}, index=x.index)
    band = size_band(x["est_2025"])
    for h in range(1, 6):
        steps = sum(phi**i for i in range(1, h + 1))
        central = x["est_2025"] * np.exp(np.log1p(g_eff) * steps)
        out_frame[f"pop_{2025 + h}"] = central
        out_frame[f"pop_{2025 + h}_lo"] = central * np.exp(-band.map(rate["q95"]) * h)
        out_frame[f"pop_{2025 + h}_hi"] = central * np.exp(-band.map(rate["q05"]) * h)
    out_frame.reset_index().to_parquet(out / "ml_populacao_projecao.parquet", index=False)
    pd.DataFrame(
        {
            "w": [w],
            "phi": [phi],
            "pop_2025_milhoes": [out_frame["pop_2025"].sum() / 1e6],
            "pop_2030_milhoes": [out_frame["pop_2030"].sum() / 1e6],
        }
    ).to_csv(out / "ml_populacao_projecao_resumo.csv", index=False)
    print("CENSO 2022 (real): melhores configurações")
    print(grid.sort_values("ape_medio_pop50k").head(5).round(4).to_string(index=False))
    print("sem decaimento nem encolhimento (w=1, phi=1):")
    print(grid[(grid.w == 1.0) & (grid.phi == 1.0)].round(4).to_string(index=False))
    print("validação espacial da escolha:")
    print(cv.round(4).to_string(index=False))
    total25, total30 = out_frame["pop_2025"].sum() / 1e6, out_frame["pop_2030"].sum() / 1e6
    print(f"projeção nacional (milhões): 2025 = {total25:.1f}; 2030 = {total30:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--municipios", type=Path, default=Path("data/reference/municipios.parquet")
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    m = pd.read_parquet(args.municipios)
    series = pop_series(m)
    uf = m.set_index("cod")["uf"]

    configs = default_configs()
    sel = backtest(series, uf, SELECTION_ORIGINS, HORIZONS, configs)
    table = sel.groupby(["config", "janela", "w", "phi"], as_index=False)[
        ["ape_medio_pop50k", "mediana_ape", "vies_total"]
    ].mean()
    best = table.sort_values("ape_medio_pop50k").iloc[0]
    chosen = Config(int(best["janela"]), float(best["w"]), float(best["phi"]))

    test = backtest(series, uf, TEST_ORIGIN, HORIZONS, configs)
    naive = test[(test["janela"] == 1) & (test["w"] == 0.0) & (test["phi"] == 1.0)]  # referência
    chosen_test = test[test["config"] == chosen.label()]
    # Baseline sem crescimento: previsão = última observação.
    last = series[TEST_ORIGIN[0]]
    flat_rows = []
    for h in HORIZONS:
        actual = series[TEST_ORIGIN[0] + h]
        ape = (last / actual - 1).abs()
        flat_rows.append(
            {
                "horizonte": h,
                "config": "nivel_constante",
                "mediana_ape": float(ape.median()),
                "ape_medio_pop50k": float(ape[actual >= 50_000].mean()),
                "vies_total": float(last.sum() / actual.sum() - 1),
            }
        )
    flat = pd.DataFrame(flat_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    table.sort_values("ape_medio_pop50k").to_csv(args.out / "ml_previsao_selecao.csv", index=False)
    pd.concat(
        [
            chosen_test.assign(papel="escolhida"),
            naive.assign(papel="cagr_1ano"),
            flat.assign(papel="nivel_constante"),
        ]
    ).to_csv(args.out / "ml_previsao_teste.csv", index=False)

    q = log_error_quantiles(series, uf, SELECTION_ORIGINS, 1, chosen)
    # Cobertura do intervalo 90% (quantis do erro por faixa) na origem de teste, horizonte 4.
    q4 = log_error_quantiles(series, uf, SELECTION_ORIGINS, 4, chosen)
    pred4 = predict(series, uf, TEST_ORIGIN[0], 4, chosen)
    actual4 = series[TEST_ORIGIN[0] + 4]
    from alocacao_capacitada.ml.forecast import size_band

    band = size_band(series[TEST_ORIGIN[0]])
    lo = pred4 * np.exp(-band.map(q4["q95"]))
    hi = pred4 * np.exp(-band.map(q4["q05"]))
    coverage = float(((actual4 >= lo) & (actual4 <= hi)).mean())
    pd.DataFrame(
        {"horizonte": [4], "cobertura_nominal": [0.90], "cobertura_observada": [coverage]}
    ).to_csv(args.out / "ml_previsao_cobertura.csv", index=False)
    q4.to_csv(args.out / "ml_previsao_quantis_h4.csv")

    run_census_and_projection(m, args.out)
    pd.set_option("display.width", 200)
    print("escolhida:", chosen.label())
    print(table.sort_values("ape_medio_pop50k").head(6).round(4).to_string(index=False))
    print("\nTESTE origem 2017 (verdade até 2021):")
    print(
        pd.concat(
            [
                chosen_test.assign(papel="escolhida"),
                naive.assign(papel="cagr_1ano"),
                flat.assign(papel="nivel_constante"),
            ]
        )[["papel", "horizonte", "mediana_ape", "ape_medio_pop50k", "vies_total"]]
        .round(4)
        .to_string(index=False)
    )
    print("\ncobertura 90% h=4:", round(coverage, 3))
    _ = q


if __name__ == "__main__":
    main()
