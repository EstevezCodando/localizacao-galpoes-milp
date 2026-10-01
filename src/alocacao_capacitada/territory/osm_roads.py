"""Vias do OpenStreetMap (extratos regionais do Geofabrik): rodovias e arruamentos por município.

Lê o `.osm.pbf` de uma região, separa `highway=*` em rodovias (motorway, trunk, primary,
secondary e seus acessos) e arruamentos (tertiary, unclassified, residential, living_street,
service), grava um shapefile de cada e agrega o comprimento por município.

Ficam de fora trilhas, caminhos, calçadas, ciclovias, escadas e vias em construção ou
propostas: não servem à logística de carga. A atribuição ao município usa o ponto médio de cada
trecho; trechos que cruzam divisas são contados inteiros no município do ponto médio.

Uso:  uv run python -m alocacao_capacitada.territory.osm_roads
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely import line_interpolate_point

from alocacao_capacitada.lake.api import ApiCollector
from alocacao_capacitada.territory.geometry import load_polygons, locate

REGIOES = {"centro-oeste": "260930", "sul": "260930", "sudeste": "260930"}
SOURCE = "https://download.geofabrik.de/south-america/brazil/{regiao}-{data}.osm.pbf"
# Sul (41-43), Sudeste (31-35) e Centro-Oeste (50-53): onde o Olist tem cobertura.
UF_FOCO = ("31", "32", "33", "35", "41", "42", "43", "50", "51", "52", "53")
CRS_METRICO = "EPSG:5880"  # SIRGAS 2000 / Brazil Polyconic: comprimentos e áreas em metros

ROD_TRONCAL = {"motorway", "motorway_link", "trunk", "trunk_link"}
ROD_REGIONAL = {"primary", "primary_link", "secondary", "secondary_link"}
ARRUAMENTO = {
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
}

TAGS = ("ref", "maxspeed", "lanes", "surface", "oneway")
_TAG = {t: re.compile(rf'"{t}"=>"([^"]*)"') for t in TAGS}


def parse_tag(other_tags: pd.Series, key: str) -> pd.Series:
    """Valor de `key` no campo hstore `other_tags` do GDAL (vazio se ausente)."""
    return other_tags.fillna("").str.extract(_TAG[key], expand=False)


def classify(highway: pd.Series) -> pd.Series:
    out = pd.Series("", index=highway.index, dtype=object)
    out[highway.isin(ROD_TRONCAL)] = "rodovia_troncal"
    out[highway.isin(ROD_REGIONAL)] = "rodovia_regional"
    out[highway.isin(ARRUAMENTO)] = "arruamento"
    return out


def read_roads(pbf: Path) -> gpd.GeoDataFrame:
    wanted = sorted(ROD_TRONCAL | ROD_REGIONAL | ARRUAMENTO)
    quoted = ",".join(f"'{h}'" for h in wanted)
    sql = f"SELECT osm_id, name, highway, other_tags FROM lines WHERE highway IN ({quoted})"
    frame = pyogrio.read_dataframe(pbf, sql=sql, sql_dialect="OGRSQL")
    frame["classe"] = classify(frame["highway"])
    for tag in TAGS:
        frame[tag] = parse_tag(frame["other_tags"], tag)
    return frame.drop(columns="other_tags")


def write_shapefiles(roads: gpd.GeoDataFrame, out_dir: Path, region: str) -> dict[str, int]:
    """Um shapefile de rodovias e outro de arruamentos (nomes de campo com até 10 letras)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for label, mask in (
        ("rodovias", roads["classe"].str.startswith("rodovia")),
        ("arruamentos", roads["classe"] == "arruamento"),
    ):
        part = roads.loc[mask, ["osm_id", "name", "highway", "classe", *TAGS, "geometry"]]
        part.to_file(out_dir / f"{region}_{label}.shp", encoding="utf-8")
        counts[label] = len(part)
    return counts


def aggregate(roads: gpd.GeoDataFrame, polygons: pd.DataFrame) -> pd.DataFrame:
    """Quilômetros por classe em cada município, a partir do ponto médio de cada trecho."""
    metric = roads.to_crs(CRS_METRICO)
    km = metric.geometry.length.to_numpy() / 1000.0
    mid = line_interpolate_point(roads.geometry.to_numpy(), 0.5, normalized=True)
    cod = locate(polygons, np.array([p.y for p in mid]), np.array([p.x for p in mid]))
    table = pd.DataFrame({"cod": cod, "classe": roads["classe"].to_numpy(), "km": km})
    table = table[table["cod"] != ""]
    wide = table.pivot_table(index="cod", columns="classe", values="km", aggfunc="sum").fillna(0.0)
    return wide.add_prefix("km_").reset_index()


def provenance(pbf: Path, region: str, date: str) -> dict[str, str | int]:
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with pbf.open("rb") as fh:
        while block := fh.read(1 << 22):
            digest.update(block)
            md5.update(block)
    return {
        "fonte": SOURCE.format(regiao=region, data=date),
        "licenca": "ODbL (OpenStreetMap contributors)",
        "extrato": date,
        "bytes": pbf.stat().st_size,
        "sha256": digest.hexdigest(),
        "md5": md5.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    bronze = args.root / "data" / "bronze"
    raw, shp = bronze / "osm_regional", bronze / "osm_regional" / "shp"
    polygons = load_polygons(ApiCollector(bronze / "ibge"))
    area = (
        gpd.GeoSeries(
            gpd.GeoSeries(list(polygons["geometry"]), crs="EPSG:4326").to_crs(CRS_METRICO)
        ).area.to_numpy()
        / 1e6
    )
    parts = []
    for region, date in REGIOES.items():
        pbf = raw / f"{region}-{date}.osm.pbf"
        roads = read_roads(pbf)
        counts = write_shapefiles(roads, shp, region)
        agg = aggregate(roads, polygons)
        agg["regiao_extrato"] = region
        parts.append(agg)
        meta = {**provenance(pbf, region, date), "trechos": counts}
        (raw / f"{region}-{date}.provenance.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(region, len(roads), "trechos", counts, flush=True)
    # Um município na divisa de extratos pode aparecer em dois; soma-se (as vias não se repetem).
    km_cols = [c for c in pd.concat(parts).columns if c.startswith("km_")]
    total = pd.concat(parts).groupby("cod", as_index=False)[km_cols].sum()
    # Os extratos vazam para estados vizinhos; fora do foco a cobertura é parcial.
    total = total[total["cod"].str[:2].isin(UF_FOCO)]
    total = total.merge(pd.DataFrame({"cod": polygons["cod"], "area_km2": area}), on="cod")
    total["km_rodovia"] = total.get("km_rodovia_troncal", 0.0) + total.get(
        "km_rodovia_regional", 0.0
    )
    total["dens_rodovia_km_km2"] = total["km_rodovia"] / total["area_km2"]
    total["dens_arruamento_km_km2"] = total["km_arruamento"] / total["area_km2"]
    out = args.root / "data" / "reference" / "osm_vias_municipio.csv"
    total.round(4).to_csv(out, index=False)
    print("municípios com vias:", len(total), "->", out)


if __name__ == "__main__":
    main()
