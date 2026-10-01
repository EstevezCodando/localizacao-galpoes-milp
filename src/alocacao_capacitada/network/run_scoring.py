"""Score de candidatos: treina com soluções do otimizador e testa como pré-filtro de decisão.

Uso:  uv run python -m alocacao_capacitada.network.run_scoring --iterations 100
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.lake.rent import rent_lookup
from alocacao_capacitada.ml.scoring import (
    OSM_ALL_FEATURES,
    auc,
    candidate_table,
    out_of_fold_scores,
    precision_at_k,
)
from alocacao_capacitada.network.builder import NetworkConfig
from alocacao_capacitada.network.experiments import (
    BASE_2025,
    evaluate_design,
    open_set,
    scenario_demand,
    solve_design,
)
from alocacao_capacitada.network.run_future import POOL, Setup, log
from alocacao_capacitada.territory.osm_roads import UF_FOCO

# (participação da operadora, custo de terceirizar R$/pedido, densidade pedidos/m²/mês)
SCENARIOS = [
    (0.10, 100.0, 20.0),
    (0.05, 100.0, 20.0),
    (0.20, 100.0, 20.0),
    (0.10, 60.0, 20.0),
    (0.10, 100.0, 10.0),
]
TOP_K = 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--foco", action="store_true", help="só Sul, Sudeste e Centro-Oeste")
    parser.add_argument("--modelo-taxa", default="gbm_completo")
    args = parser.parse_args()
    args.out = args.out or Path("results") / ("foco" if args.foco else "")
    args.out.mkdir(parents=True, exist_ok=True)

    s = Setup(args.root, NetworkConfig(), UF_FOCO if args.foco else None, args.modelo_taxa)
    ctx = s.ctx
    ufs = ctx.municipios.set_index("cod").loc[s.pool, "uf"].astype(str).tolist()
    rent = {uf: v[0] for uf, v in rent_lookup(ctx.rent_cities, sorted(set(ufs))).items()}

    frames = []
    for share, outsourcing, density in SCENARIOS:
        cfg = replace(
            s.cfg,
            company_share=share,
            outsourcing_cost=outsourcing,
            density_orders_m2_month=density,
        )
        demand = scenario_demand(ctx, BASE_2025, share)
        s.cfg = cfg
        net = s.network(demand, s.pool)
        t = time.time()
        design = open_set(solve_design(net.instance, args.iterations))
        sites = {int(net.facility_site[i]) for i in design}
        log(
            f"cenário share={share} terc={outsourcing} dens={density}: "
            f"{len(sites)} sítios ({time.time() - t:.0f}s)"
        )
        node_demand = demand.reindex(s.nodes).to_numpy()
        table = candidate_table(
            s.pool, demand, ctx.municipios, ctx.table, rent, s.km_pool, node_demand
        )
        table["aberto"] = [int(k in sites) for k in range(POOL)]
        table["share"], table["terceirizacao"], table["densidade"] = share, outsourcing, density
        table["cenario"] = f"share={share}, terc={outsourcing}, dens={density}"
        frames.append(table)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(args.out / "rede_score_dados.parquet", index=False)

    data["score_logit"] = out_of_fold_scores(data, "logit")
    data["score_gbm"] = out_of_fold_scores(data, "gbm")
    data["score_demanda"] = data["log_demanda_local"]
    names = ["score_demanda", "score_logit", "score_gbm"]
    if args.foco:  # mesmo modelo, com as variáveis do OpenStreetMap
        data["score_logit_osm"] = out_of_fold_scores(data, "logit", features=OSM_ALL_FEATURES)
        data["score_gbm_osm"] = out_of_fold_scores(data, "gbm", features=OSM_ALL_FEATURES)
        names += ["score_logit_osm", "score_gbm_osm"]
    rows = []
    for name in names:
        per_scenario_auc, per_scenario_p = [], []
        for _, g in data.groupby("cenario"):
            per_scenario_auc.append(auc(g["aberto"].to_numpy(), g[name].to_numpy()))
            per_scenario_p.append(precision_at_k(g["aberto"].to_numpy(), g[name].to_numpy(), TOP_K))
        rows.append(
            {
                "score": name,
                "auc_medio": float(np.nanmean(per_scenario_auc)),
                f"precisao_top{TOP_K}": float(np.mean(per_scenario_p)),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "rede_score_metricas.csv", index=False)
    log("métricas de ranking:\n" + metrics.round(3).to_string(index=False))

    # Decisão: pré-filtrar os candidatos do cenário base pelo score (fora da amostra).
    s.cfg = replace(s.cfg, company_share=0.10, outsourcing_cost=100.0, density_orders_m2_month=20.0)
    base_name = "share=0.1, terc=100.0, dens=20.0"
    base = data[data["cenario"] == base_name].reset_index(drop=True)
    demand = scenario_demand(ctx, BASE_2025, 0.10)
    full_net = s.network(demand, s.pool)
    t = time.time()
    full_design = open_set(solve_design(full_net.instance, args.iterations))
    full_time = time.time() - t
    full_cost = evaluate_design(full_net.instance, full_design)["custo_total"]
    results = [
        {
            "filtro": f"todos os {POOL}",
            "k": POOL,
            "custo_total": full_cost,
            "arrependimento_pct": 0.0,
            "tempo_s": full_time,
        }
    ]
    for k in (30, 45, 60):
        for name in names:
            top = base.nlargest(k, name)["cod"].tolist()
            net = s.network(demand, top)
            t = time.time()
            design = open_set(solve_design(net.instance, args.iterations))
            elapsed = time.time() - t
            cost = evaluate_design(net.instance, design)["custo_total"]
            results.append(
                {
                    "filtro": name,
                    "k": k,
                    "custo_total": cost,
                    "arrependimento_pct": 100 * (cost / full_cost - 1),
                    "tempo_s": elapsed,
                }
            )
            log(f"{name} top-{k}: {cost / 1e6:.3f} mi/mês ({100 * (cost / full_cost - 1):+.2f}%)")
    pd.DataFrame(results).to_csv(args.out / "rede_score_decisao.csv", index=False)


if __name__ == "__main__":
    main()
