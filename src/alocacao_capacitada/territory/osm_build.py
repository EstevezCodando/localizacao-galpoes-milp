"""Coleta os sinais do OSM para os candidatos a centro (pool de 120 municípios).

Uso:  uv run python -m alocacao_capacitada.territory.osm_build
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alocacao_capacitada.lake.api import ApiCollector
from alocacao_capacitada.network.context import load_context, top_by_demand
from alocacao_capacitada.network.experiments import BASE_2025, scenario_demand
from alocacao_capacitada.territory.osm import logistics_proxies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--pool", type=int, default=120)
    args = parser.parse_args()
    ctx = load_context(args.root)
    pool = top_by_demand(scenario_demand(ctx, BASE_2025, 0.10), args.pool)
    api = ApiCollector(args.root / "data" / "bronze" / "osm", min_interval_s=8.0)
    table = logistics_proxies(api, pool)
    out = args.root / "data" / "reference" / "osm_logistica_candidatos.csv"
    table.to_csv(out, index=False)
    print(table.describe().round(1).to_string())
    print("sem dado:", int(table.isna().any(axis=1).sum()), "de", len(table))


if __name__ == "__main__":
    main()
