import numpy as np
import pandas as pd
import pytest

from alocacao_capacitada.ml.forecast import (
    Config,
    census_predict,
    log_error_quantiles,
    pop_series,
    predict,
    size_band,
)


def exponential_frame(growth: float, base: float = 10_000.0) -> pd.DataFrame:
    years = list(range(2001, 2022))
    values = [base * (1 + growth) ** (y - 2001) for y in years]
    cols = {f"est_{y}": [v] for y, v in zip(years, values, strict=True) if y not in (2007, 2010)}
    return pd.DataFrame({"cod": ["1"], **cols})


def test_serie_interpola_2007_e_2010_em_log() -> None:
    series = pop_series(exponential_frame(0.02))
    assert series.loc["1", 2007] == pytest.approx(10_000 * 1.02**6, rel=1e-9)
    assert series.loc["1", 2010] == pytest.approx(10_000 * 1.02**9, rel=1e-9)


def test_crescimento_constante_e_recuperado_sem_amortecimento() -> None:
    series = pop_series(exponential_frame(0.02))
    uf = pd.Series({"1": "SP"})
    pred = predict(series, uf, origin=2015, horizon=3, config=Config(window=1, w=1.0, phi=1.0))
    assert pred["1"] == pytest.approx(10_000 * 1.02**17, rel=1e-9)


def test_amortecimento_reduz_o_crescimento_previsto() -> None:
    series = pop_series(exponential_frame(0.02))
    uf = pd.Series({"1": "SP"})
    free = predict(series, uf, 2015, 4, Config(1, 1.0, 1.0))["1"]
    damped = predict(series, uf, 2015, 4, Config(1, 1.0, 0.9))["1"]
    assert 10_000 * 1.02**14 < damped < free


def test_faixas_de_populacao_sao_monotonicas() -> None:
    bands = size_band(pd.Series([1_000, 7_000, 15_000, 30_000, 70_000, 200_000, 2_000_000]))
    assert list(bands) == [0, 1, 2, 3, 4, 5, 6]


def test_previsao_censitaria_recupera_crescimento_geometrico() -> None:
    m = pd.DataFrame(
        {
            "cod": ["1", "2"],
            "uf": ["SP", "SP"],
            "censo_2000": [10_000.0, 50_000.0],
            "censo_2010": [12_000.0, 50_000.0],
        }
    )
    pred = census_predict(m, w=1.0, phi=1.0, years=12)
    assert pred["1"] == pytest.approx(12_000 * (1.2 ** (12 / 10)), rel=1e-9)
    assert pred["2"] == pytest.approx(50_000.0)


def test_quantis_do_erro_por_faixa() -> None:
    series = pop_series(exponential_frame(0.02))
    uf = pd.Series({"1": "SP"})
    q = log_error_quantiles(series, uf, [2012, 2014], 1, Config(1, 1.0, 1.0))
    assert np.allclose(q.to_numpy(), 0.0, atol=1e-9)  # sem ruído, o erro é nulo
