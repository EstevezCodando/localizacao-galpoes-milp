"""Aluguel de galpão por estado (R$/m²/mês): silver e gold a partir do levantamento coletado.

Fonte (asking prices, 8.700+ anúncios, 1–5/fev/2026, 18 cidades): galpaodasmaquinas.com.br.
É uma fonte SECUNDÁRIA de preços PEDIDOS em portais (não contratos fechados) e cobre só 9
estados; os demais recebem a média das 18 cidades, marcada como imputada.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

BAND = "medio_1001_3000"
_BANDS = {
    "Pequeno · até 1.000 m²": "pequeno_ate_1000",
    "Médio · 1.001–3.000 m²": BAND,
    "Grande · 3.001–5.000 m²": "grande_3001_5000",
}


def parse_rent_table(html: bytes) -> pd.DataFrame:
    """Tabela por cidade: cidade, uf, e uma coluna numérica por faixa de tamanho."""
    tables = pd.read_html(io.StringIO(html.decode("utf-8", "replace")))
    raw = next((t for t in tables if "Cidade" in t.columns), None)
    if raw is None:
        raise ValueError("tabela de aluguel por cidade não encontrada na página")
    missing = set(_BANDS) - set(raw.columns)
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")

    out = pd.DataFrame()
    parts = raw["Cidade"].str.extract(r"^(?P<cidade>.+?)\s+(?P<uf>[A-Z]{2})$")
    if parts.isna().any().any():
        raise ValueError("não foi possível separar cidade e UF")
    out["cidade"], out["uf"] = parts["cidade"], parts["uf"]
    for label, name in _BANDS.items():
        out[name] = raw[label].map(_money)
    if ((out[list(_BANDS.values())] < 5) | (out[list(_BANDS.values())] > 100)).any().any():
        raise ValueError("preço fora da faixa plausível (R$ 5–100/m²)")
    return out


def _money(text: object) -> float:
    return float(re.sub(r"[^\d,]", "", str(text)).replace(",", "."))


def rent_by_state(cities: pd.DataFrame) -> pd.DataFrame:
    """Média da faixa média por UF; `imputado` marca estados sem observação (usam a média geral)."""
    by_state = cities.groupby("uf")[BAND].agg(["mean", "count"]).reset_index()
    by_state.columns = pd.Index(["uf", "rs_m2_mes", "n_cidades"])
    by_state["imputado"] = False
    return by_state


def rent_lookup(cities: pd.DataFrame, states: list[str]) -> dict[str, tuple[float, bool]]:
    """{uf: (R$/m²/mês, imputado)} para todos os estados pedidos."""
    observed = rent_by_state(cities).set_index("uf")["rs_m2_mes"].to_dict()
    fallback = float(cities[BAND].mean())
    return {uf: (observed[uf], False) if uf in observed else (fallback, True) for uf in states}


def load_rent_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
