"""Tabela de modelagem: um município por linha, com demografia, renda e pedidos do Olist.

Cuidados de datação (para não vazar informação do futuro):
  * população de referência = estimativa de 2017 (meio da janela de pedidos);
  * faixa etária vem do Censo 2022 e IDHM/renda/Gini do Censo 2010: são características
    estruturais, mas defasadas ou posteriores à janela. Isso é declarado, não escondido;
  * o PIB per capita usa PIB e população de 2017.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.geo import haversine_matrix

PERIOD = ("2017-01", "2018-08")  # janela com atividade regular do Olist (20 meses)
AGE_YOUNG = ("15_19", "20_24", "25_29")
AGE_MID = ("30_34", "35_39", "40_44")
AGE_OLD = ("60_64", "65_69", "70_74", "75_79", "80_84", "85_89", "90_94", "95_99", "100_mais")
AGE_ALL = [
    "0_4", "5_9", "10_14", "15_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49",
    "50_54", "55_59", *AGE_OLD,
]  # fmt: skip

FEATURES = [
    "log_pop",
    "cagr_10_22",
    "share_15_29",
    "share_30_44",
    "share_60_mais",
    "idhm",
    "idhm_r",
    "idhm_e",
    "log_rdpc",
    "gini",
    "log_pib_pc",
    "log_dist_polo",
]

# Vias e infraestrutura industrial do OpenStreetMap (só existem para as UFs de S, SE e CO).
OSM_FEATURES = [
    "log_dens_rodovia",
    "log_km_troncal",
    "log_dens_arruamento",
    "log_ha_galpao",
    "log_ha_industrial",
]


def add_osm_features(t: pd.DataFrame, osm_dir: Path) -> pd.DataFrame:
    """Junta densidade viária e área de galpões/zonas industriais (OSM, extratos regionais)."""
    roads = pd.read_csv(osm_dir / "osm_vias_municipio.csv", dtype={"cod": str})
    shed = pd.read_csv(osm_dir / "osm_industrial_municipio.csv", dtype={"cod": str})
    t = t.merge(roads, on="cod", how="left").merge(shed, on="cod", how="left", suffixes=("", "_i"))
    t["log_dens_rodovia"] = np.log1p(100.0 * t["dens_rodovia_km_km2"])
    t["log_km_troncal"] = np.log1p(t["km_rodovia_troncal"])
    t["log_dens_arruamento"] = np.log1p(t["dens_arruamento_km_km2"])
    # Sem polígono mapeado = zero (ausência de registro), não dado faltante.
    t["log_ha_galpao"] = np.log1p(100.0 * t["km2_galpao"].fillna(0.0))
    t["log_ha_industrial"] = np.log1p(100.0 * t["km2_industrial"].fillna(0.0))
    return t


def load_modeling_table(
    municipios: Path,
    panel: Path,
    polos_csv: Path | None = None,
    focus: Sequence[str] | None = None,
    osm_dir: Path | None = None,
) -> pd.DataFrame:
    """`focus` (prefixos de código de UF) descarta os demais municípios; `osm_dir` acrescenta as
    variáveis do OpenStreetMap (e só deixa municípios que as têm)."""
    m = pd.read_parquet(municipios)
    if focus is not None:
        m = m[m["cod"].str[:2].isin(focus)].reset_index(drop=True)
    p = pd.read_parquet(panel)
    p = p[(p["mes"] >= PERIOD[0]) & (p["mes"] <= PERIOD[1])]
    orders = p.groupby("cod")["pedidos"].sum().rename("pedidos")
    t = m.merge(orders, on="cod", how="left")
    t["pedidos"] = t["pedidos"].fillna(0.0)

    t["pop"] = t["est_2017"]
    age_cols = [f"idade_{g}" for g in AGE_ALL]
    total_age = t[age_cols].sum(axis=1)
    t["log_pop"] = np.log(t["pop"])
    t["cagr_10_22"] = (t["censo_2022"] / t["censo_2010"]) ** (1 / 12) - 1
    t["share_15_29"] = t[[f"idade_{g}" for g in AGE_YOUNG]].sum(axis=1) / total_age
    t["share_30_44"] = t[[f"idade_{g}" for g in AGE_MID]].sum(axis=1) / total_age
    t["share_60_mais"] = t[[f"idade_{g}" for g in AGE_OLD]].sum(axis=1) / total_age
    t["idhm"] = t["adh_idhm_2010"]
    t["idhm_r"] = t["adh_idhm_r_2010"]
    t["idhm_e"] = t["adh_idhm_e_2010"]
    t["log_rdpc"] = np.log(t["adh_rdpc_2010"])
    t["gini"] = t["adh_gini_2010"]
    t["log_pib_pc"] = np.log(t["pib_2017"] * 1000.0 / t["est_2017"])
    polos = pd.read_csv(polos_csv or panel.parent / "olist_polos_vendedores.csv")
    dist = haversine_matrix(
        t["lat"].to_numpy(), t["lon"].to_numpy(), polos["lat"].to_numpy(), polos["lon"].to_numpy()
    )
    t["log_dist_polo"] = np.log1p(dist.min(axis=1))

    if osm_dir is not None:
        t = add_osm_features(t, osm_dir)
    needed = ["pop", *FEATURES, *(OSM_FEATURES if osm_dir is not None else [])]
    complete = t[needed].notna().all(axis=1) & np.isfinite(t[needed]).all(axis=1)
    t["completo"] = complete
    return t
