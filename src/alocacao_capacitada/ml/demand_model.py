"""Modelos supervisionados de propensão de demanda por município, com validação espacial.

Alvo: pedidos do Olist por município na janela 2017-01..2018-08. Modelamos a TAXA
(pedidos por habitante) com peso = população, o que equivale a uma regressão de Poisson com
exposição. A previsão em pedidos é taxa × população.

Validação: nunca sorteamos municípios ao acaso (vizinhos se parecem e inflariam o resultado).
Cada município é previsto por um modelo treinado SEM o seu bloco geográfico: por UF (GroupKFold
com 5 dobras de UFs) e por macrorregião (deixa uma região de fora por vez).

Interpretação: o Olist é um marketplace de 2016–2018, concentrado no Sudeste. O modelo estima a
propensão RELATIVA de comprar nele, não a demanda nacional absoluta.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alocacao_capacitada.ml.features import FEATURES, OSM_FEATURES

MIN_RATE = 1e-9


@dataclass(frozen=True)
class ModelSpec:
    name: str
    columns: list[str]
    make: Callable[[], BaseEstimator] | None  # None = baseline proporcional à população


def model_specs(with_osm: bool = False) -> list[ModelSpec]:
    """`with_osm` inclui os modelos com variáveis do OpenStreetMap (exigem a tabela com OSM)."""
    base = _base_specs()
    if not with_osm:
        return base
    cols = [*FEATURES, *OSM_FEATURES]
    return [
        *base,
        ModelSpec(
            "glm_osm",
            cols,
            lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
        ),
        ModelSpec(
            "gbm_osm",
            cols,
            lambda: HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=250,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ),
        ),
    ]


def _base_specs() -> list[ModelSpec]:
    return [
        ModelSpec("pop_proporcional", [], None),
        ModelSpec(
            "glm_pop_dist",
            ["log_pop", "log_dist_polo"],
            lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
        ),
        ModelSpec(
            "glm_completo",
            FEATURES,
            lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
        ),
        ModelSpec(
            "gbm_completo",
            FEATURES,
            lambda: HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=250,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=0,
            ),
        ),
    ]


def folds(table: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Índices (treino, teste) por esquema espacial: 'uf' (5 dobras) ou 'regiao' (leave-one-out)."""
    groups = table["uf"] if scheme == "uf" else table["regiao"]
    splitter = GroupKFold(n_splits=5 if scheme == "uf" else groups.nunique())
    return [(tr, te) for tr, te in splitter.split(table, groups=groups)]


def cross_predict(table: pd.DataFrame, scheme: str, with_osm: bool = False) -> pd.DataFrame:
    """Previsão de pedidos de cada município por cada modelo, sempre fora do seu bloco."""
    data = table[table["completo"]].reset_index(drop=True)
    out = data[["cod", "uf", "regiao", "pop", "pedidos"]].copy()
    for spec in model_specs(with_osm):
        pred = np.zeros(len(data))
        for train_idx, test_idx in folds(data, scheme):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            if spec.make is None:
                rate = train["pedidos"].sum() / train["pop"].sum()
                pred[test_idx] = rate * test["pop"].to_numpy()
                continue
            model = spec.make()
            y = (train["pedidos"] / train["pop"]).to_numpy()
            fit_kwargs = {
                f"{model.steps[-1][0]}__sample_weight"
                if hasattr(model, "steps")
                else "sample_weight": train["pop"].to_numpy()
            }
            model.fit(train[spec.columns], y, **fit_kwargs)
            rate_hat = np.clip(model.predict(test[spec.columns]), MIN_RATE, None)
            pred[test_idx] = rate_hat * test["pop"].to_numpy()
        out[spec.name] = pred
    return out


def evaluate_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Métricas por modelo (e pelos dois recortes: todos os municípios e os de ≥ 50 mil hab.)."""
    rows = []
    actual = pred["pedidos"].to_numpy()
    big = (pred["pop"] >= 50_000).to_numpy()
    top_actual = float(np.sort(actual)[::-1][:300].sum() / actual.sum())
    for name in [s.name for s in model_specs(True) if s.name in pred.columns]:
        yhat = np.clip(pred[name].to_numpy(), MIN_RATE, None)
        order = np.argsort(-yhat)[:300]
        rows.append(
            {
                "modelo": name,
                "deviance_poisson": mean_poisson_deviance(actual, yhat),
                "deviance_poisson_pop50k": mean_poisson_deviance(actual[big], yhat[big]),
                "spearman_pop50k": float(spearmanr(actual[big], yhat[big]).statistic),
                "captura_top300": float(actual[order].sum() / actual.sum()),
                "captura_top300_ideal": top_actual,
                "razao_soma_prev_real": float(yhat.sum() / actual.sum()),
            }
        )
    return pd.DataFrame(rows)


def decile_calibration(pred: pd.DataFrame, model: str, bins: int = 10) -> pd.DataFrame:
    """Previsto × observado por decil de previsão (municípios de ≥ 20 mil habitantes)."""
    d = pred[pred["pop"] >= 20_000].copy()
    d["decil"] = pd.qcut(d[model].rank(method="first"), bins, labels=False) + 1
    g = d.groupby("decil").agg(
        previsto=(model, "sum"), observado=("pedidos", "sum"), n=("cod", "size")
    )
    g["razao"] = g["previsto"] / g["observado"]
    return g.reset_index()


def permutation_importance(
    table: pd.DataFrame,
    scheme: str = "uf",
    repeats: int = 5,
    seed: int = 0,
    model_name: str = "gbm_completo",
) -> pd.DataFrame:
    """Aumento da deviance de Poisson (fora do bloco) ao embaralhar cada variável do GBM.

    Quanto maior o aumento, mais o modelo depende da variável para prever bem municípios que
    ele não viu. Variáveis correlacionadas dividem importância: leia como ordem de grandeza.
    """
    data = table[table["completo"]].reset_index(drop=True)
    spec = next(s for s in model_specs(True) if s.name == model_name)
    assert spec.make is not None
    rng = np.random.default_rng(seed)
    increases: dict[str, list[float]] = {c: [] for c in spec.columns}
    for train_idx, test_idx in folds(data, scheme):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        model = spec.make()
        weight = train["pop"].to_numpy()
        model.fit(
            train[spec.columns], (train["pedidos"] / train["pop"]).to_numpy(), sample_weight=weight
        )
        pop_test = test["pop"].to_numpy()
        actual = test["pedidos"].to_numpy()

        def deviance(
            frame: pd.DataFrame,
            model: BaseEstimator = model,
            actual: np.ndarray = actual,
            pop_test: np.ndarray = pop_test,
        ) -> float:
            rate = np.clip(model.predict(frame[spec.columns]), MIN_RATE, None)
            return float(mean_poisson_deviance(actual, rate * pop_test))

        base = deviance(test)
        for col in spec.columns:
            for _ in range(repeats):
                shuffled = test.copy()
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                increases[col].append(deviance(shuffled) - base)
    rows = [
        {"variavel": c, "aumento_deviance": float(np.mean(v)), "desvio": float(np.std(v))}
        for c, v in increases.items()
    ]
    return (
        pd.DataFrame(rows).sort_values("aumento_deviance", ascending=False).reset_index(drop=True)
    )
