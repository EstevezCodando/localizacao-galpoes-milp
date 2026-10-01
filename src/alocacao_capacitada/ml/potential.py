"""Demanda potencial por município: propensão (modelo supervisionado) × população × mercado.

    pedidos_j(t) = N(t) · share_j(t),    share_j(t) = taxa_j · pop_j(t) / Σ_k taxa_k · pop_k(t)

`taxa_j` é a propensão de comprar estimada pelo modelo (pedidos do Olist por habitante, que só
vale como medida RELATIVA entre municípios). O volume nacional N(t) é uma premissa ancorada em
relatórios setoriais, nunca extraída do Olist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alocacao_capacitada.ml.demand_model import MIN_RATE, model_specs

# Volume nacional de pedidos de e-commerce (ABComm): 414,9 milhões em 2024 e projeção de
# 435,6 milhões em 2025. Fonte setorial (associação de comércio eletrônico), não auditada.
NATIONAL_ORDERS_2025 = 435.6e6
ORDERS_GROWTH_SCENARIOS = {"baixo": 0.03, "base": 0.06, "alto": 0.09}  # a.a.; premissa


def fit_propensity(table: pd.DataFrame, model: str = "gbm_completo") -> pd.Series:
    """Taxa (pedidos por habitante, janela do Olist) para TODOS os municípios completos."""
    spec = next(s for s in model_specs(True) if s.name == model)
    if spec.make is None:
        raise ValueError("o baseline não tem propensão por município")
    data = table[table["completo"]]
    estimator = spec.make()
    y = (data["pedidos"] / data["pop"]).to_numpy()
    kwargs = (
        {f"{estimator.steps[-1][0]}__sample_weight": data["pop"].to_numpy()}
        if hasattr(estimator, "steps")
        else {"sample_weight": data["pop"].to_numpy()}
    )
    estimator.fit(data[spec.columns], y, **kwargs)
    rate = np.clip(estimator.predict(data[spec.columns]), MIN_RATE, None)
    return pd.Series(rate, index=data["cod"].to_numpy(), name="taxa")


def national_orders(year: int, growth: float) -> float:
    """Pedidos nacionais por ano, a partir de 2025 e do crescimento anual do cenário."""
    return float(NATIONAL_ORDERS_2025 * (1.0 + growth) ** (year - 2025))


def monthly_demand(
    rate: pd.Series, population: pd.Series, year: int, growth: float, company_share: float
) -> pd.Series:
    """Pedidos por mês de UMA operadora com participação `company_share` do mercado."""
    common = rate.index.intersection(population.index)
    weight = rate[common] * population[common]
    share = weight / weight.sum()
    return (national_orders(year, growth) * company_share / 12.0 * share).rename("pedidos_mes")
