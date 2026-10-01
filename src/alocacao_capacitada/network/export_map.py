"""Exporta, para o mapa da página, a rede nacional de um projeto sob uma demanda de cenário.

Lê os módulos abertos de `results/rede_projetos.csv`, reotimiza só a atribuição das zonas aos
módulos abertos (o mesmo recurso usado na avaliação) e grava um JSON com nós, centros e ligações.

Uso:  uv run python -m alocacao_capacitada.network.export_map
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.network.builder import Network, NetworkConfig
from alocacao_capacitada.network.experiments import (
    BASE_2025,
    BASE_2030,
    Scenario,
    scenario_demand,
    solve_design,
    subset,
)
from alocacao_capacitada.network.run_future import Setup
from alocacao_capacitada.territory.osm_roads import UF_FOCO

VIEWS = [
    ("Projeto para 2025, demanda 2025", "2025 base", BASE_2025),
    ("Projeto para 2025, demanda 2030", "2025 base", BASE_2030),
    ("Projeto robusto (média dos 3), demanda 2030", "2030 média dos 3", BASE_2030),
]


def design_indices(net: Network, projects: pd.DataFrame, name: str) -> np.ndarray:
    wanted = projects[projects["projeto"] == name]
    keys = {f"{c}:{int(m)}" for c, m in zip(wanted["cod"], wanted["modulo"], strict=True)}
    ids = list(net.instance.facility_ids)
    # Um mesmo município pode ter dois módulos iguais; os ids são únicos por (cod, tamanho).
    return np.array([i for i, fid in enumerate(ids) if fid in keys])


def view_payload(
    s: Setup, projects: pd.DataFrame, project: str, scenario: Scenario, iterations: int
) -> dict[str, object]:
    demand = scenario_demand(s.ctx, scenario, s.cfg.company_share)
    net = s.network(demand)
    idx = design_indices(net, projects, project)
    sub = subset(net.instance, idx)
    solution = solve_design(sub, iterations)
    ev = evaluate(sub, solution)
    assignment = solution.assignment
    hubs = []
    for k, i in enumerate(idx):
        site = int(net.facility_site[i])
        served = np.flatnonzero(assignment == k)
        hubs.append(
            {
                "cod": str(net.candidates["cod"].iloc[site]),
                "nome": str(net.candidates["nome"].iloc[site]),
                "uf": str(net.candidates["uf"].iloc[site]),
                "lat": float(net.candidates["lat"].iloc[site]),
                "lon": float(net.candidates["lon"].iloc[site]),
                "modulo": int(net.facility_size[i]),
                "carga": float(sub.demand[served].sum()),
                "capacidade": float(sub.capacity[k]),
                "k": k,
            }
        )
    nodes = [
        {
            "nome": str(r["nome"]),
            "uf": str(r["uf"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "pedidos": float(r["pedidos_mes"]),
            "hub": int(assignment[j]),
        }
        for j, r in enumerate(net.nodes.to_dict("records"))
    ]
    fixed = float(net.instance.fixed_cost[idx].sum())
    return {
        "nodes": nodes,
        "hubs": hubs,
        "custo_total": fixed + ev.transport_cost + ev.penalty_cost,
        "custo_fixo": fixed,
        "frete": ev.transport_cost,
        "terceirizado": ev.penalty_cost,
        "pct_terceirizado": 100.0 * ev.unserved_demand / net.instance.total_demand,
        "demanda_total": net.instance.total_demand,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--foco", action="store_true", help="só Sul, Sudeste e Centro-Oeste")
    parser.add_argument("--modelo-taxa", default="gbm_completo")
    args = parser.parse_args()
    folder = args.root / "results" / ("foco" if args.foco else "")
    args.out = args.out or folder / "rede_mapa.json"
    s = Setup(args.root, NetworkConfig(), UF_FOCO if args.foco else None, args.modelo_taxa)
    projects = pd.read_csv(folder / "rede_projetos.csv", dtype={"cod": str})
    payload = {
        "vistas": [
            {"nome": label, **view_payload(s, projects, project, scenario, args.iterations)}
            for label, project, scenario in VIEWS
        ]
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for v in payload["vistas"]:
        cost = float(v["custo_total"]) / 1e6  # type: ignore[arg-type]
        print(v["nome"], "->", len(v["hubs"]), "módulos;", round(cost, 2), "mi/mês")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
