"""Gera o painel município × mês do Olist e o relatório de cobertura.

Uso:  uv run python -m alocacao_capacitada.territory.panel_build
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alocacao_capacitada.lake.api import ApiCollector
from alocacao_capacitada.territory import ibge
from alocacao_capacitada.territory.geometry import load_polygons
from alocacao_capacitada.territory.olist_panel import build_panel, seller_hubs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--bronze", type=Path, default=Path("data/bronze"))
    parser.add_argument("--out", type=Path, default=Path("data/reference"))
    args = parser.parse_args()

    api = ApiCollector(args.bronze / "ibge")
    municipalities = ibge.municipalities(api)
    polygons = load_polygons(api)
    panel, coverage = build_panel(args.raw_dir, polygons, municipalities)
    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out / "olist_painel_municipal.parquet", index=False)
    coverage.to_csv(args.out / "olist_cobertura.csv", index=False)
    hubs = seller_hubs(args.raw_dir, polygons)
    hubs.to_csv(args.out / "olist_polos_vendedores.csv", index=False)
    months = panel.groupby("mes")["pedidos"].sum()
    print(coverage.round(2).to_string(index=False))
    print("municípios com pedidos:", panel["cod"].nunique())
    print(months.to_string())


if __name__ == "__main__":
    main()
