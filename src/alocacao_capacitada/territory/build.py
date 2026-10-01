"""Coleta e monta a tabela territorial (gold): um município por linha, com demografia e renda.

Uso:  uv run python -m alocacao_capacitada.territory.build
Saída: data/reference/municipios.parquet e data/reference/municipios_cobertura.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alocacao_capacitada.lake.api import ApiCollector
from alocacao_capacitada.territory import ibge
from alocacao_capacitada.territory.geometry import load_polygons

IPEA_SERIES = ("ADH_IDHM", "ADH_IDHM_E", "ADH_IDHM_L", "ADH_IDHM_R", "ADH_RDPC", "ADH_GINI")


def build(bronze: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    api = ApiCollector(bronze / "ibge")
    base = ibge.municipalities(api)
    parts = {
        "censo_2022": ibge.population_census_2022(api),
        "censo_2000": ibge.population_census_2000(api),
        "censo_2010": ibge.population_census_2010(api),
        "estimativas": ibge.population_estimates(api),
        "idade_2022": ibge.age_groups_2022(api),
        "pib": ibge.gdp(api),
    }
    for code in IPEA_SERIES:
        parts[code] = ibge.ipeadata_series(api, code)
    polygons = load_polygons(api)
    centroids = polygons[["cod", "lat", "lon"]]

    table = base.merge(centroids, on="cod", how="left")
    coverage = [("municipios (lista oficial)", len(base), len(base))]
    coverage.append(("com centroide (malha)", int(table["lat"].notna().sum()), len(base)))
    for name, frame in parts.items():
        table = table.merge(frame, on="cod", how="left")
        key = frame.columns[1]
        coverage.append((f"com dado: {name}", int(table[key].notna().sum()), len(base)))
    cov = pd.DataFrame(coverage, columns=["etapa", "n", "de"])
    cov["pct"] = 100 * cov["n"] / cov["de"]
    return table, cov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze", type=Path, default=Path("data/bronze"))
    parser.add_argument("--out", type=Path, default=Path("data/reference"))
    args = parser.parse_args()
    table, cov = build(args.bronze)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "municipios.parquet", index=False)
    cov.to_csv(args.out / "municipios_cobertura.csv", index=False)
    print(cov.round(1).to_string(index=False))
    print(table.shape)


if __name__ == "__main__":
    main()
