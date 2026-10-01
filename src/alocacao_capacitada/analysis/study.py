"""Estudo integrado, offline e reproduzível: dados, ablação, otimização e paisagem exata.

Execute: python -m alocacao_capacitada.analysis.study --out results/estudo_integrado
Figuras são geradas separadamente por scripts/render_study.py, com matplotlib.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import Solution
from alocacao_capacitada.lake.freight import load_freight_model
from alocacao_capacitada.ml.features import FEATURES, load_modeling_table
from alocacao_capacitada.solvers.base import Solver
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver


def tiny_instance() -> Instance:
    """Mesmo exemplo didático do projeto, com 4**5 atribuições possíveis."""
    return Instance(
        ("A", "B", "C"),
        ("R1", "R2", "R3", "R4", "R5"),
        np.array([4.0, 8.0, 8.0, 4.0, 2.0]),
        np.array([14.0, 14.0, 14.0]),
        np.array([33.0, 35.0, 38.0]),
        np.array([[6.0, 6.0, 8.0, 8.0, 8.0], [7.0, 6.0, 8.0, 1.0, 1.0], [7.0, 4.0, 6.0, 4.0, 8.0]]),
        50.0,
    )


def enumerate_tiny(instance: Instance) -> pd.DataFrame:
    """Enumeração completa; nenhuma interpolação ou redução antes de otimizar."""
    rows = []
    for values in itertools.product(range(-1, instance.n_facilities), repeat=instance.n_demand):
        solution = Solution(np.array(values, dtype=np.int64))
        ev = evaluate(instance, solution)
        if not ev.feasible:
            continue
        loads = [float(instance.demand[solution.assignment == i].sum()) for i in range(3)]
        rows.append(
            {
                "assignment": " ".join(map(str, values)),
                "fixo": ev.fixed_cost,
                "frete": ev.transport_cost,
                "penalidade": ev.penalty_cost,
                "custo": ev.total_cost,
                "nao_atendido": ev.unserved_demand,
                "centros": ev.n_open,
                "carga_A": loads[0],
                "carga_B": loads[1],
                "carga_C": loads[2],
            }
        )
    return pd.DataFrame(rows)


def controlled_instance(
    base: Instance,
    demand: float = 1.0,
    capacity: float = 1.0,
    fixed: float = 1.0,
    transport: float = 1.0,
    penalty: float = 1.0,
) -> Instance:
    """Cada multiplicador altera exclusivamente o seu fator; geografia não muda."""
    return replace(
        base,
        demand=base.demand * demand,
        capacity=base.capacity * capacity,
        fixed_cost=base.fixed_cost * fixed,
        unit_cost=base.unit_cost * transport,
        unserved_penalty=base.unserved_penalty * penalty,
    )


def save_instance(instance: Instance, path: Path) -> None:
    np.savez_compressed(
        path,
        demand=instance.demand,
        capacity=instance.capacity,
        fixed_cost=instance.fixed_cost,
        unit_cost=instance.unit_cost,
        facility_ids=instance.facility_ids,
        demand_ids=instance.demand_ids,
        penalty=instance.unserved_penalty,
    )


def measure(
    instance: Instance, solver: Solver, budget: float, label: str, out: Path
) -> dict[str, Any]:
    result = solver.solve(instance, budget)
    ev = evaluate(instance, result.solution)
    if not ev.feasible or not np.isfinite(ev.total_cost):
        raise ValueError(f"Solução inválida: {label}")
    np.save(out / f"{label}.npy", result.solution.assignment)
    lb = result.lower_bound
    if lb is not None and lb > ev.total_cost + max(1e-5, 1e-7 * ev.total_cost):
        raise ValueError("Limite inferior excede incumbente")
    return {
        "id": label,
        "custo": ev.total_cost,
        "fixo": ev.fixed_cost,
        "frete": ev.transport_cost,
        "penalidade": ev.penalty_cost,
        "servico_pct": 100 * (1 - ev.unserved_demand / instance.total_demand),
        "centros": ev.n_open,
        "tempo_s": result.runtime_s,
        "status": result.status,
        "bound_solver": lb,
        "orcamento_s": budget,
    }


def audit_data(root: Path, out: Path) -> pd.DataFrame:
    raw, ref = root / "data/raw", root / "data/reference"
    orders = pd.read_csv(raw / "olist_orders_dataset.csv")
    delivered = orders[orders.order_status == "delivered"].copy()
    delivered["mes"] = delivered.order_purchase_timestamp.str[:7]
    delivered.groupby("mes").size().rename("pedidos").to_csv(out / "pedidos_mensais.csv")
    table = load_modeling_table(ref / "municipios.parquet", ref / "olist_painel_municipal.parquet")
    table.to_csv(out / "dados_municipais.csv", index=False)
    missing = table[["pedidos", *FEATURES]].isna().sum().rename("ausentes").to_frame()
    missing["pct"] = 100 * missing.ausentes / len(table)
    missing.to_csv(out / "ausentes.csv")
    corr = table[["pedidos", *FEATURES]].corr(method="spearman")
    corr.to_csv(out / "correlacao_spearman.csv")
    panel = pd.read_parquet(ref / "olist_painel_municipal.parquet")
    panel = panel.merge(table[["cod", "uf"]], on="cod", validate="many_to_one")
    panel.groupby(["uf", "mes"])["pedidos"].sum().unstack(fill_value=0).to_csv(out / "uf_mes.csv")
    quality = {
        "pedidos_brutos": len(orders),
        "entregues": len(delivered),
        "order_id_duplicados": int(orders.order_id.duplicated().sum()),
        "municipios": len(table),
        "municipios_completos": int(table.completo.sum()),
        "municipios_sem_pedidos": int((table.pedidos == 0).sum()),
        "inicio": delivered.order_purchase_timestamp.min(),
        "fim": delivered.order_purchase_timestamp.max(),
    }
    (out / "qualidade.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    provenance = []
    for path in sorted((root / "data/bronze").rglob("*.provenance.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        filename = info.get("file")
        target = path.parent / filename if filename else None
        valid = None
        if target is not None and target.is_file() and info.get("sha256"):
            valid = hashlib.sha256(target.read_bytes()).hexdigest() == info["sha256"]
        provenance.append(
            {
                "arquivo": str(path.relative_to(root)),
                "hash_confere": valid,
                "url": info.get("url"),
                "status_http": info.get("status"),
                "verificavel": valid is not None,
            }
        )
    pd.DataFrame(provenance).to_csv(out / "proveniencia.csv", index=False)
    return table


def ablation(table: pd.DataFrame, out: Path) -> None:
    """Mesmos municípios/folds; exclui atributos posteriores e polos derivados do alvo.

    É validação espacial retrospectiva, não backtest temporal de demanda futura.
    """
    historical = ["log_pop", "idhm", "idhm_r", "idhm_e", "log_rdpc", "gini", "log_pib_pc"]
    groups = {"populacao": [], "historico": historical, "retrospectivo_completo": FEATURES}
    data = table[table.completo].reset_index(drop=True)
    predictions, metrics = [], []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(data, groups=data.uf)):
        train, test = data.iloc[tr], data.iloc[te]
        y = (train.pedidos / train["pop"]).to_numpy()
        for name, features in groups.items():
            if not features:
                pred = np.full(len(test), train.pedidos.sum() / train["pop"].sum())
            else:
                estimator = HistGradientBoostingRegressor(
                    loss="poisson",
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    random_state=0,
                )
                estimator.fit(train[features], y, sample_weight=train["pop"].to_numpy())
                pred = estimator.predict(test[features])
            predicted = np.clip(pred, 1e-9, None) * test["pop"].to_numpy()
            metrics.append(
                {
                    "fold": fold,
                    "modelo": name,
                    "n": len(test),
                    "deviance": mean_poisson_deviance(test.pedidos, predicted),
                    "mae": float(np.abs(test.pedidos - predicted).mean()),
                    "razao_total": float(predicted.sum() / test.pedidos.sum()),
                }
            )
            predictions.append(
                test[["cod", "uf", "pedidos"]].assign(fold=fold, modelo=name, previsto=predicted)
            )
        print(f"ablação: dobra {fold + 1}/5", flush=True)
    pd.DataFrame(metrics).to_csv(out / "ablacao_metricas.csv", index=False)
    pd.concat(predictions).to_csv(out / "ablacao_previsoes.csv", index=False)


def experiments(root: Path, out: Path, budget: float, sweep_budget: float) -> None:
    assignments = out / "solucoes"
    assignments.mkdir(exist_ok=True)
    tiny = tiny_instance()
    space = enumerate_tiny(tiny)
    space.to_csv(out / "espaco_exato.csv", index=False)
    save_instance(tiny, out / "instancia_didatica.npz")
    tiny_rows = []
    for name, solver in [
        ("Guloso", NearestFeasibleGreedy()),
        ("Busca local", LocalSearch(NearestFeasibleGreedy())),
        ("MILP", MilpSolver()),
    ]:
        row = measure(tiny, solver, 5.0, f"tiny_{name.replace(' ', '_')}", assignments)
        tiny_rows.append({**row, "metodo": name})
    pd.DataFrame(tiny_rows).to_csv(out / "didatico_metodos.csv", index=False)
    assert abs(tiny_rows[-1]["custo"] - space.custo.min()) < 1e-6
    nodes = load_geo_nodes(root / "data/raw", period=("2017-01-01", "2018-08-31"))
    freight = load_freight_model(root / "data/reference/antt_tabela_a.csv", 2, 50)
    rows, history = [], []
    for n, m in [(50, 15), (100, 30)]:
        inst = build_instance(nodes, InstanceConfig(n_demand=n, n_facilities=m, freight=freight))
        save_instance(inst, out / f"instancia_{n}x{m}.npz")
        nodes.demand.head(n).to_csv(out / f"regioes_{n}.csv")
        nodes.supply.head(m).to_csv(out / f"centros_{m}.csv")
        lp = solve_lp_relaxation(inst).lower_bound
        methods: list[tuple[str, int, Solver]] = [
            ("Guloso", -1, NearestFeasibleGreedy()),
            ("Busca local", -1, LocalSearch(NearestFeasibleGreedy())),
            ("MILP", -1, MilpSolver()),
            ("MILP aquecido", -1, MilpSolver(warm_start=LocalSearch(NearestFeasibleGreedy()))),
        ]
        for seed in range(3):

            def progress(
                it: int,
                seconds: float,
                cost: float,
                seed: int = seed,
                label: str = f"{n}x{m}",
                lower_bound: float = lp,
            ) -> None:
                history.append(
                    {
                        "instancia": label,
                        "seed": seed,
                        "iteracao": it,
                        "tempo_s": seconds,
                        "custo": cost,
                        "lb_pl": lower_bound,
                    }
                )

            methods.append(
                (
                    "LNS",
                    seed,
                    LnsSolver(
                        LocalSearch(NearestFeasibleGreedy()),
                        seed=seed,
                        free_size=25,
                        on_progress=progress,
                    ),
                )
            )
        for name, seed, solver in methods:
            label = f"{n}x{m}_{name.replace(' ', '_')}_{seed}"
            row = measure(inst, solver, budget, label, assignments)
            rows.append(
                {
                    **row,
                    "instancia": f"{n}x{m}",
                    "metodo": name,
                    "seed": seed,
                    "lb_pl": lp,
                    "gap_ub_pl_pct": 100 * (row["custo"] - lp) / row["custo"],
                    "cobertura_pct": 100 * inst.total_demand / nodes.demand.orders.sum(),
                }
            )
            pd.DataFrame(rows).to_csv(out / "comparacao.csv", index=False)
            print(label, round(row["custo"]), row["status"], flush=True)
    pd.DataFrame(history).to_csv(out / "trajetoria_lns.csv", index=False)
    z = np.load(out / "instancia_50x15.npz")
    base = Instance(
        tuple(z["facility_ids"]),
        tuple(z["demand_ids"]),
        z["demand"],
        z["capacity"],
        z["fixed_cost"],
        z["unit_cost"],
        float(z["penalty"]),
    )
    sensitivity = []
    # Cache por conteúdo, orçamento fixo nesta execução; não mistura tempos ou solver configs.
    cache = {}

    def one(inst: Instance, label: str) -> dict[str, Any]:
        key = b"".join(
            a.tobytes()
            for a in [
                inst.demand,
                inst.capacity,
                inst.fixed_cost,
                inst.unit_cost,
                np.array([inst.unserved_penalty]),
            ]
        )
        if key not in cache:
            row = measure(inst, MilpSolver(), sweep_budget, label, assignments)
            cache[key] = row
        row = cache[key]
        return {
            **row,
            "demanda": inst.total_demand,
            "capacidade": float(inst.capacity.sum()),
            "gap_solver_pct": 100 * (row["custo"] - row["bound_solver"]) / row["custo"],
        }

    for field, levels in {
        "demand": [0.6, 0.8, 1.0, 1.2, 1.4],
        "capacity": [0.6, 0.8, 1.0, 1.2, 1.4],
        "fixed": [0.25, 0.5, 1.0, 2.0, 4.0],
        "transport": [0.5, 0.75, 1.0, 1.25, 1.5],
        "penalty": [0.1, 0.25, 0.5, 1.0, 2.0],
    }.items():
        for level in levels:
            inst = controlled_instance(base, **{field: level})
            row = one(inst, f"sens_{field}_{level}")
            sensitivity.append({**row, "fator": field, "nivel": level})
        pd.DataFrame(sensitivity).to_csv(out / "sensibilidade_controlada.csv", index=False)
        print("sensibilidade", field, flush=True)
    grid = []
    for demand, cap in itertools.product([0.6, 0.8, 1.0, 1.2, 1.4], repeat=2):
        row = one(controlled_instance(base, demand=demand, capacity=cap), f"grid_{demand}_{cap}")
        grid.append({**row, "escala_demanda": demand, "escala_capacidade": cap})
        pd.DataFrame(grid).to_csv(out / "interacao_demanda_capacidade.csv", index=False)
    print("grade fatorial concluída", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/estudo_integrado"))
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--sweep-budget", type=float, default=3.0)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("Use uma pasta nova para preservar execuções anteriores")
    out.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    table = audit_data(root, out)
    ablation(table, out)
    experiments(root, out, args.budget, args.sweep_budget)
    inputs = [
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "data/raw").glob("*.csv")),
        *sorted((root / "data/reference").glob("*.parquet")),
        root / "data/reference/antt_tabela_a.csv",
    ]
    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "ortools", "scikit-learn"]},
        "budgets": {"comparison": args.budget, "sweep": args.sweep_budget},
        "lns": {"free_size": 25, "seeds": [0, 1, 2], "sub_time_limit_s": 2},
        "ablation": {"folds": 5, "seed": 0, "iterations": 150},
        "period": ["2017-01-01", "2018-08-31"],
        "scope": "offline; dados locais; resultados novos separados dos históricos",
        "hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Estudo concluído:", out, flush=True)


if __name__ == "__main__":
    main()
