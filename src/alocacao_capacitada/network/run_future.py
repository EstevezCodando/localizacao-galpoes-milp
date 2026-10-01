"""Experimentos da rede nacional com demanda potencial e visão de futuro.

  b  cenários 2025/2030 (população e mercado): projeto para cada cenário avaliado em todos
  c  previsão → decisão: projetos feitos com cada previsão de população de 2022, avaliados no
     Censo real
  e  distância viária × linha reta: o erro de projetar com a reta

Uso:  uv run python -m alocacao_capacitada.network.run_future --only b c e --iterations 100
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.geo import haversine_matrix
from alocacao_capacitada.ml.forecast import census_predict
from alocacao_capacitada.ml.potential import monthly_demand
from alocacao_capacitada.network.builder import Network, NetworkConfig
from alocacao_capacitada.network.context import Context, load_context, matrix_for, top_by_demand
from alocacao_capacitada.network.experiments import (
    BASE_2025,
    BASE_2030,
    HIGH_2030,
    LOW_2030,
    evaluate_design,
    make_network,
    open_set,
    scenario_demand,
    solve_design,
)
from alocacao_capacitada.territory.osm_roads import UF_FOCO

POOL = 120


class Setup:
    def __init__(
        self,
        root: Path,
        cfg: NetworkConfig,
        focus: Sequence[str] | None = None,
        rate_model: str = "gbm_completo",
    ) -> None:
        self.ctx: Context = load_context(root, focus=focus, rate_model=rate_model)
        self.cfg = cfg
        self.tag = "foco_" if focus is not None else ""
        base = scenario_demand(self.ctx, BASE_2025, cfg.company_share)
        self.nodes = top_by_demand(base, cfg.n_nodes)
        self.pool = top_by_demand(base, POOL)
        self.km_pool = matrix_for(
            self.ctx, self.pool, self.nodes, f"{self.tag}pool120_x_top400"
        ).distance_km
        self.zone_ref = scenario_demand(self.ctx, HIGH_2030, cfg.company_share)

    def candidates(self, k: int | None = None) -> list[str]:
        return self.pool[: k or self.cfg.n_candidates]

    def km(self, candidates: list[str]) -> np.ndarray:
        idx = [self.pool.index(c) for c in candidates]
        return self.km_pool[idx]

    def network(
        self,
        demand: pd.Series,
        candidates: list[str] | None = None,
        km: np.ndarray | None = None,
    ) -> Network:
        cands = candidates or self.candidates()
        distances = self.km(cands) if km is None else km
        return make_network(self.ctx, demand, self.nodes, cands, distances, self.cfg, self.zone_ref)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def experiment_scenarios(s: Setup, iterations: int, out: Path) -> None:
    scenarios = [BASE_2025, BASE_2030, LOW_2030, HIGH_2030]
    nets = {sc.name: s.network(scenario_demand(s.ctx, sc, s.cfg.company_share)) for sc in scenarios}
    mean_demand = pd.concat(
        [
            scenario_demand(s.ctx, sc, s.cfg.company_share)
            for sc in (BASE_2030, LOW_2030, HIGH_2030)
        ],
        axis=1,
    ).mean(axis=1)
    nets["2030 média dos 3"] = s.network(mean_demand)
    designs = {}
    for name, net in nets.items():
        t = time.time()
        designs[name] = open_set(solve_design(net.instance, iterations))
        log(f"projeto '{name}': {len(designs[name])} módulos ({time.time() - t:.0f}s)")
    rows = []
    for dname, design in designs.items():
        for sc in scenarios:
            res = evaluate_design(nets[sc.name].instance, design)
            rows.append({"projeto": dname, "cenario": sc.name, **res})
            log(f"  {dname} em {sc.name}: {res['custo_total'] / 1e6:.3f} mi/mês")
    table = pd.DataFrame(rows)
    best = table.groupby("cenario")["custo_total"].transform("min")
    table["arrependimento_pct"] = 100 * (table["custo_total"] / best - 1)
    table.to_csv(out / "rede_cenarios.csv", index=False)
    site_rows = []
    net = nets["2025 base"]
    for dname, design in designs.items():
        for i in design:
            site = net.facility_site[i]
            site_rows.append(
                {
                    "projeto": dname,
                    "cod": net.candidates["cod"].iloc[site],
                    "nome": net.candidates["nome"].iloc[site],
                    "uf": net.candidates["uf"].iloc[site],
                    "modulo": int(net.facility_size[i]),
                }
            )
    pd.DataFrame(site_rows).to_csv(out / "rede_projetos.csv", index=False)


def experiment_forecast_to_decision(s: Setup, iterations: int, out: Path) -> None:
    m = s.ctx.municipios
    ref22 = m.set_index("cod")
    actual = ref22["censo_2022"].dropna()
    populations = {
        "nível constante (Censo 2010)": ref22["censo_2010"].dropna(),
        "extrapolação simples (w=1, phi=1)": census_predict(m, 1.0, 1.0),
        "extrapolação amortecida (w=0,75, phi=0,9)": census_predict(m, 0.75, 0.9),
        "projeção oficial IBGE para 2021": ref22["est_2021"].dropna(),
        "Censo 2022 (conhecido depois)": actual,
    }
    # Municípios sem alguma das séries (poucos, pequenos) usam a estimativa de 2017 em TODOS os
    # métodos, para que a comparação entre previsões não dependa deles.
    fallback = ref22["est_2017"]
    keep = sorted(fallback.dropna().index)
    nets = {}
    for name, pop in populations.items():
        filled = pop.reindex(keep).fillna(fallback)
        demand = monthly_demand(
            s.ctx.rate, filled, 2025, 0.06, s.cfg.company_share * s.ctx.coverage
        )
        nets[name] = s.network(demand)
    truth = nets["Censo 2022 (conhecido depois)"]
    rows = []
    for name, net in nets.items():
        t = time.time()
        design = open_set(solve_design(net.instance, iterations))
        res = evaluate_design(truth.instance, design)
        rows.append({"previsao_usada": name, "modulos": len(design), **res})
        log(
            f"previsão '{name}': {len(design)} módulos, custo real "
            f"{res['custo_total'] / 1e6:.3f} mi ({time.time() - t:.0f}s)"
        )
    table = pd.DataFrame(rows)
    best = float(table[table["previsao_usada"].str.startswith("Censo")]["custo_total"].iloc[0])
    table["arrependimento_pct"] = 100 * (table["custo_total"] / best - 1)
    table.to_csv(out / "rede_previsao_decisao.csv", index=False)


def experiment_road_vs_straight(s: Setup, iterations: int, out: Path) -> None:
    demand = scenario_demand(s.ctx, BASE_2025, s.cfg.company_share)
    cands = s.candidates()
    road = s.km(cands)
    m = s.ctx.municipios.set_index("cod")
    straight = haversine_matrix(
        m.loc[cands, "lat"].to_numpy(),
        m.loc[cands, "lon"].to_numpy(),
        m.loc[s.nodes, "lat"].to_numpy(),
        m.loc[s.nodes, "lon"].to_numpy(),
    )
    net_road = s.network(demand, cands, road)
    net_straight = s.network(demand, cands, straight)
    d_road = open_set(solve_design(net_road.instance, iterations))
    d_straight = open_set(solve_design(net_straight.instance, iterations))
    rows = []
    for name, design in (
        ("projetado com distância viária", d_road),
        ("projetado com linha reta", d_straight),
    ):
        res = evaluate_design(net_road.instance, design)  # ambos avaliados na estrada
        rows.append({"projeto": name, "modulos": len(design), **res})
        log(f"{name}: {res['custo_total'] / 1e6:.3f} mi/mês (na estrada)")
    table = pd.DataFrame(rows)
    table["arrependimento_pct"] = 100 * (table["custo_total"] / table["custo_total"].min() - 1)
    union = set(d_road) | set(d_straight)
    table["jaccard_modulos"] = len(set(d_road) & set(d_straight)) / max(len(union), 1)
    table.to_csv(out / "rede_reta_vs_estrada.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=["b", "c", "e"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--foco", action="store_true", help="só Sul, Sudeste e Centro-Oeste")
    parser.add_argument("--modelo-taxa", default="gbm_completo")
    args = parser.parse_args()
    args.out = args.out or Path("results") / ("foco" if args.foco else "")
    args.out.mkdir(parents=True, exist_ok=True)
    s = Setup(args.root, NetworkConfig(), UF_FOCO if args.foco else None, args.modelo_taxa)
    log(f"setup: {len(s.nodes)} nós, pool de {len(s.pool)} candidatos")
    if "b" in args.only:
        experiment_scenarios(s, args.iterations, args.out)
    if "c" in args.only:
        experiment_forecast_to_decision(s, args.iterations, args.out)
    if "e" in args.only:
        experiment_road_vs_straight(s, args.iterations, args.out)


if __name__ == "__main__":
    main()
