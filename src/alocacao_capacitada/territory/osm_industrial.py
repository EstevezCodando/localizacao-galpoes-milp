"""Galpões e áreas industriais do OpenStreetMap por município, a partir dos extratos regionais.

Mesma fonte de `osm_roads` (Geofabrik). Conta polígonos com `building=warehouse|industrial`,
`landuse=industrial` ou `industrial=warehouse|logistics` e soma suas áreas (km²) por município,
atribuindo cada polígono pelo ponto representativo. É indício de infraestrutura logística, não
medida: depende da completude do mapeamento voluntário, que varia entre cidades.

Uso:  uv run python -m alocacao_capacitada.territory.osm_industrial
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from alocacao_capacitada.lake.api import ApiCollector
from alocacao_capacitada.territory.geometry import load_polygons, locate
from alocacao_capacitada.territory.osm_roads import CRS_METRICO, REGIOES, UF_FOCO

SQL = (
    "SELECT osm_id, building, landuse, other_tags FROM multipolygons "
    "WHERE building IN ('warehouse','industrial') OR landuse = 'industrial' "
    'OR other_tags LIKE \'%"industrial"=>"warehouse"%\' '
    'OR other_tags LIKE \'%"industrial"=>"logistics"%\''
)


def classify(frame: pd.DataFrame) -> pd.Series:
    """`galpao` (prédio de depósito/industrial) ou `industrial` (zona de uso do solo)."""
    tags = frame["other_tags"].fillna("")
    shed = frame["building"].isin(["warehouse", "industrial"]) | tags.str.contains(
        '"industrial"=>"(?:warehouse|logistics)"', regex=True
    )
    return pd.Series(np.where(shed, "galpao", "industrial"), index=frame.index)


def aggregate(frame: gpd.GeoDataFrame, polygons: pd.DataFrame) -> pd.DataFrame:
    metric = frame.to_crs(CRS_METRICO)
    area_km2 = metric.geometry.area.to_numpy() / 1e6
    point = frame.geometry.representative_point()
    cod = locate(polygons, point.y.to_numpy(), point.x.to_numpy())
    table = pd.DataFrame({"cod": cod, "classe": classify(frame).to_numpy(), "km2": area_km2})
    table = table[table["cod"] != ""]
    count = table.pivot_table(index="cod", columns="classe", values="km2", aggfunc="count")
    area = table.pivot_table(index="cod", columns="classe", values="km2", aggfunc="sum")
    out = count.add_prefix("n_").join(area.add_prefix("km2_")).fillna(0.0)
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    bronze = args.root / "data" / "bronze"
    polygons = load_polygons(ApiCollector(bronze / "ibge"))
    parts = []
    for region, date in REGIOES.items():
        pbf = bronze / "osm_regional" / f"{region}-{date}.osm.pbf"
        frame = pyogrio.read_dataframe(pbf, sql=SQL, sql_dialect="OGRSQL")
        parts.append(aggregate(frame, polygons))
        print(region, len(frame), "polígonos", flush=True)
    both = pd.concat(parts)
    total = both.groupby("cod", as_index=False).sum()
    total = total[total["cod"].str[:2].isin(UF_FOCO)]
    for col in ("n_galpao", "n_industrial", "km2_galpao", "km2_industrial"):
        if col not in total:
            total[col] = 0.0
    out = args.root / "data" / "reference" / "osm_industrial_municipio.csv"
    total.round(6).to_csv(out, index=False)
    print("municípios com indício:", len(total), "->", out)


if __name__ == "__main__":
    main()
