"""Malha municipal do IBGE: centroides e localização de pontos (ponto-em-polígono)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import STRtree, points
from shapely.geometry import shape

from alocacao_capacitada.lake.api import ApiCollector

MALHA = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}"
    "?formato=application/vnd.geo%2Bjson&qualidade=minima&intrarregiao=municipio"
)
UF_CODES = (
    11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35,
    41, 42, 43, 50, 51, 52, 53,
)  # fmt: skip


def load_polygons(api: ApiCollector) -> pd.DataFrame:
    """Uma linha por município: `cod`, `geometry` (shapely) e centroide (`lat`, `lon`)."""
    rows = []
    for uf in UF_CODES:
        data = api.get_json(f"malha_{uf}", MALHA.format(uf=uf))
        for feature in data["features"]:
            geometry = shape(feature["geometry"])
            centroid = geometry.centroid
            rows.append(
                {
                    "cod": str(feature["properties"]["codarea"]),
                    "geometry": geometry,
                    "lat": centroid.y,
                    "lon": centroid.x,
                }
            )
    out = pd.DataFrame(rows)
    if out["cod"].duplicated().any():
        raise ValueError("município repetido na malha")
    return out


def locate(polygons: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Código do município que contém cada ponto; vazio ('') se nenhum (mar, fronteira)."""
    tree = STRtree(list(polygons["geometry"]))
    pts = points(lon, lat)
    # `query` devolve pares (ponto, polígono) cujos envelopes se cruzam; confirma-se por `covers`.
    hit_pt, hit_poly = tree.query(pts, predicate="intersects")  # ponto toca o polígono
    result = np.full(len(lat), "", dtype=object)
    codes = polygons["cod"].to_numpy()
    for p_idx, poly_idx in zip(hit_pt, hit_poly, strict=True):
        if result[p_idx] == "":
            result[p_idx] = codes[poly_idx]
    return result
