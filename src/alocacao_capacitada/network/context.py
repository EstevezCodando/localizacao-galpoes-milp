"""Carrega e prepara tudo o que a rede nacional precisa (dados, propensão, projeção, custos)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.lake.freight import FreightModel, load_freight_model
from alocacao_capacitada.lake.osrm_matrix import RoadMatrix, road_matrix
from alocacao_capacitada.lake.rent import load_rent_table
from alocacao_capacitada.ml.features import load_modeling_table
from alocacao_capacitada.ml.potential import fit_propensity


@dataclass(frozen=True)
class Context:
    municipios: pd.DataFrame
    table: pd.DataFrame
    rate: pd.Series
    projection: pd.DataFrame
    freight_by_ovv: dict[float, FreightModel]
    rent_cities: pd.DataFrame
    root: Path
    coverage: float = 1.0  # fração do mercado nacional dentro da área de atuação

    def population(self, year: int, bound: str = "central") -> pd.Series:
        """População por município em `year`: 2025 oficial; 2026–2030 projetada (central/lo/hi)."""
        proj = self.projection.set_index("cod")
        if year == 2025:
            return proj["pop_2025"]
        suffix = "" if bound == "central" else f"_{bound}"
        return proj[f"pop_{year}{suffix}"]


def load_context(
    root: Path,
    orders_per_vehicle: tuple[float, ...] = (200.0,),
    focus: Sequence[str] | None = None,
    rate_model: str = "gbm_completo",
) -> Context:
    """`focus` (prefixos de UF) restringe a rede às regiões de atuação: os demais municípios saem
    do modelo de demanda, dos nós e dos candidatos. O volume dessas UFs é a fração do mercado
    nacional que o modelo nacional lhes atribui (`coverage`), e a distribuição entre municípios
    vem de um modelo ajustado só com elas, com as variáveis do OpenStreetMap."""
    ref, res = root / "data" / "reference", root / "results"
    municipios = pd.read_parquet(ref / "municipios.parquet")
    panel = ref / "olist_painel_municipal.parquet"
    projection = pd.read_parquet(res / "ml_populacao_projecao.parquet")
    coverage = 1.0
    if focus is None:
        table = load_modeling_table(ref / "municipios.parquet", panel)
        rate = fit_propensity(table)
    else:
        national = fit_propensity(load_modeling_table(ref / "municipios.parquet", panel))
        pop = projection.set_index("cod")["pop_2025"]
        weight = national * pop.reindex(national.index)
        inside = weight.index.str[:2].isin(focus)
        coverage = float(weight[inside].sum() / weight.sum())
        table = load_modeling_table(ref / "municipios.parquet", panel, focus=focus, osm_dir=ref)
        rate = fit_propensity(table, rate_model)
        municipios = municipios[municipios["cod"].str[:2].isin(focus)].reset_index(drop=True)
        projection = projection[projection["cod"].str[:2].isin(focus)].reset_index(drop=True)
    freight = {
        v: load_freight_model(ref / "antt_tabela_a.csv", axles=2, orders_per_vehicle=v)
        for v in orders_per_vehicle
    }
    return Context(
        municipios,
        table,
        rate,
        projection,
        freight,
        load_rent_table(ref / "aluguel_galpao_cidades.csv"),
        root,
        coverage,
    )


def top_by_demand(demand: pd.Series, n: int) -> list[str]:
    return list(demand.nlargest(n).index)


def matrix_for(
    ctx: Context, candidates: list[str], nodes: list[str], name: str = "matriz"
) -> RoadMatrix:
    m = ctx.municipios.set_index("cod")
    o = m.loc[candidates, ["lat", "lon"]].to_numpy()
    d = m.loc[nodes, ["lat", "lon"]].to_numpy()
    return road_matrix(
        np.asarray(o), np.asarray(d), ctx.root / "data" / "bronze" / "osrm_matriz" / name
    )
