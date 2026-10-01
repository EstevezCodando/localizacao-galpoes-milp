"""Repete a decisão com pré-filtro de candidatos em várias sementes, para separar efeito de ruído.

A heurística de busca (LNS) tem aleatoriedade e, no problema completo de 120 candidatos, converge
menos que nos problemas filtrados; por isso o "arrependimento" contra o problema completo pode
sair negativo. Aqui o custo de cada filtro é comparado ao MELHOR custo conhecido entre todas as
execuções, e reportado com média e desvio entre sementes.

Uso:  uv run python -m alocacao_capacitada.network.run_scoring_seeds --seeds 1 2 3
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from alocacao_capacitada.ml.scoring import OSM_ALL_FEATURES, out_of_fold_scores
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--ks", nargs="+", type=int, default=[45, 60])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--foco", action="store_true", help="só Sul, Sudeste e Centro-Oeste")
    parser.add_argument("--modelo-taxa", default="gbm_completo")
    args = parser.parse_args()
    args.out = args.out or Path("results") / ("foco" if args.foco else "")

    data = pd.read_parquet(args.out / "rede_score_dados.parquet")
    data["score_logit"] = out_of_fold_scores(data, "logit")
    data["score_gbm"] = out_of_fold_scores(data, "gbm")
    data["score_demanda"] = data["log_demanda_local"]
    names = ["score_demanda", "score_logit", "score_gbm"]
    if args.foco:
        data["score_logit_osm"] = out_of_fold_scores(data, "logit", features=OSM_ALL_FEATURES)
        data["score_gbm_osm"] = out_of_fold_scores(data, "gbm", features=OSM_ALL_FEATURES)
        names += ["score_logit_osm", "score_gbm_osm"]
    base = data[data["cenario"] == "share=0.1, terc=100.0, dens=20.0"].reset_index(drop=True)

    cfg = replace(
        NetworkConfig(), company_share=0.10, outsourcing_cost=100.0, density_orders_m2_month=20.0
    )
    s = Setup(args.root, cfg, UF_FOCO if args.foco else None, args.modelo_taxa)
    demand = scenario_demand(s.ctx, BASE_2025, 0.10)
    rows = []

    def run(label: str, k: int, cands: list[str], seed: int) -> None:
        net = s.network(demand, cands)
        design = open_set(solve_design(net.instance, args.iterations, seed=seed))
        cost = evaluate_design(net.instance, design)["custo_total"]
        rows.append({"filtro": label, "k": k, "semente": seed, "custo_total": cost})
        pd.DataFrame(rows).to_csv(args.out / "rede_score_decisao_sementes.csv", index=False)
        log(f"{label} k={k} semente={seed}: {cost / 1e6:.3f} mi/mês")

    for seed in args.seeds:
        run("todos", POOL, s.pool, seed)
        for k in args.ks:
            for name in names:
                run(name, k, base.nlargest(k, name)["cod"].tolist(), seed)


if __name__ == "__main__":
    main()
