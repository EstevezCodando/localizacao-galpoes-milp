"""Sinais de infraestrutura logística no OpenStreetMap (Overpass): zonas industriais e galpões.

Por município (relação OSM com a tag IBGE:GEOCODIGO) contamos polígonos mapeados como
`landuse=industrial` e `building=warehouse`. É um proxy de infraestrutura, afetado pela
completude do mapeamento voluntário: municípios maiores e mais ativos no OSM tendem a ter mais
polígonos. Falhas do servidor viram NaN (nunca zero): "sem dado" não é "sem galpão".
"""

from __future__ import annotations

import urllib.error
import urllib.parse

import numpy as np
import pandas as pd

from alocacao_capacitada.lake.api import ApiCollector

OVERPASS = "https://overpass-api.de/api/interpreter"
FILTERS = {
    "osm_industrial": '["landuse"="industrial"]',
    "osm_galpao": '["building"="warehouse"]',
}


def _query(code: str, tag_filter: str) -> str:
    ql = (
        f'[out:json][timeout:90];rel["IBGE:GEOCODIGO"="{code}"];map_to_area->.a;'
        f"way(area.a){tag_filter};out count;"
    )
    return f"{OVERPASS}?data={urllib.parse.quote(ql)}"


def logistics_proxies(api: ApiCollector, codes: list[str]) -> pd.DataFrame:
    rows = []
    for code in codes:
        row: dict[str, object] = {"cod": code}
        for name, tag_filter in FILTERS.items():
            try:
                data = api.get_json(f"osm_{name}_{code}", _query(code, tag_filter))
                row[name] = float(data["elements"][0]["tags"]["ways"])
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError):
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
