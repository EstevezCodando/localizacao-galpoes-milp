"""Sensibilidade um-parâmetro-por-vez, com cada fator realmente isolado.

Para cada nível de cada parâmetro resolve-se a instância com LNS e mede-se: custo, nº de
centros, demanda não atendida, estabilidade da decisão (Jaccard dos centros abertos contra o
caso-base), volume realocado e utilização. O gap contra o limite do PL dá a confiança na solução.

Cuidados de isolamento (auditoria de 30/09/2026):
  * "demanda, capacidade congelada": só o vetor de demanda muda; capacidade, custos e candidatos
    ficam idênticos aos do caso-base (testa a rede instalada sob estresse);
  * "demanda e capacidade juntas": expansão conjunta, mantida como experimento separado;
  * a penalidade de não atendimento é FIXA em todos os níveis (antes variava junto com os
    pedidos por veículo, porque ambos dependiam do mesmo modelo de frete).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import GeoNodes
from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.lake.freight import FreightModel
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation

BASE_LEVELS = {
    "demanda, capacidade congelada (x)": "1.0",
    "demanda e capacidade juntas (x)": "1.0",
    "capacidade (folga)": "1.5",
    "custo fixo (pedidos x1000km)": "40.0",
    "pedidos por veiculo": "50.0",
}


def _with_orders_per_vehicle(base: FreightModel, value: float) -> FreightModel:
    return replace(base, orders_per_vehicle=value)


def sweeps(nodes: GeoNodes, base: InstanceConfig) -> dict[str, list[tuple[str, Instance]]]:
    """Instâncias de cada nível; o nível de `BASE_LEVELS` de cada lista é o caso-base."""
    freight = base.freight
    if freight is None:
        raise ValueError("a sensibilidade exige o modelo de frete da ANTT")
    penalty = float(freight.cost_per_order(np.array([base.penalty_km]))[0])
    fixed_penalty = replace(base, penalty_per_order=penalty)
    reference = build_instance(nodes, fixed_penalty)

    def make(config: InstanceConfig) -> Instance:
        return build_instance(nodes, replace(config, penalty_per_order=penalty))

    return {
        "demanda, capacidade congelada (x)": [
            (f"{v}", replace(reference, demand=reference.demand * v))
            for v in (0.6, 0.8, 1.0, 1.2, 1.4)
        ],
        "demanda e capacidade juntas (x)": [
            (f"{v}", make(replace(base, demand_scale=v))) for v in (0.6, 0.8, 1.0, 1.2, 1.4)
        ],
        "capacidade (folga)": [
            (f"{v}", make(replace(base, capacity_slack=v))) for v in (1.1, 1.3, 1.5, 2.0, 3.0)
        ],
        "custo fixo (pedidos x1000km)": [
            (f"{v}", make(replace(base, fixed_cost_per_order=v)))
            for v in (10.0, 20.0, 40.0, 80.0, 160.0)
        ],
        "pedidos por veiculo": [
            (f"{v}", make(replace(base, freight=_with_orders_per_vehicle(freight, v))))
            for v in (10.0, 25.0, 50.0, 100.0, 200.0)
        ],
    }


def run_sensitivity(
    nodes: GeoNodes, base: InstanceConfig, time_limit_s: float = 60.0, iterations: int = 120
) -> pd.DataFrame:
    rows = []
    for parameter, levels in sweeps(nodes, base).items():
        base_assignment: np.ndarray | None = None
        base_open: set[str] = set()
        solved = []
        for level, instance in levels:
            solution = _solve_cached(instance, time_limit_s, iterations)
            solved.append((level, instance, solution))
            if level == BASE_LEVELS[parameter]:
                base_assignment = solution.assignment
                base_open = {instance.facility_ids[i] for i in solution.open_facilities()}
        for level, instance, solution in solved:
            ev = evaluate(instance, solution)
            opened = {instance.facility_ids[i] for i in solution.open_facilities()}
            lp = solve_lp_relaxation(instance)
            moved = np.nan
            if base_assignment is not None:
                both = (solution.assignment != UNASSIGNED) & (base_assignment != UNASSIGNED)
                changed = both & (solution.assignment != base_assignment)
                moved = float(
                    instance.demand[changed].sum() / max(instance.demand[both].sum(), 1e-9)
                )
            rows.append(
                {
                    "parametro": parameter,
                    "nivel": level,
                    "custo_total": ev.total_cost,
                    "custo_fixo": ev.fixed_cost,
                    "custo_transporte": ev.transport_cost,
                    "demanda_nao_atendida_pct": 100 * ev.unserved_demand / instance.total_demand,
                    "centros_abertos": ev.n_open,
                    "utilizacao_max": ev.max_utilization,
                    "capacidade_total": float(instance.capacity.sum()),
                    "demanda_total": instance.total_demand,
                    "gap_ub_vs_pl": (ev.total_cost - lp.lower_bound) / ev.total_cost,
                    "volume_realocado_pct": 100 * moved,
                    "jaccard_vs_base": _jaccard(opened, base_open),
                }
            )
    return pd.DataFrame(rows)


_CACHE: dict[bytes, Solution] = {}


def _solve_cached(instance: Instance, time_limit_s: float, iterations: int) -> Solution:
    """LNS determinístico (orçamento em iterações) e sem resolver duas vezes a mesma instância.

    O orçamento por iterações elimina a dependência da carga da máquina: com a mesma semente, o
    resultado é o mesmo. O tempo fica como trava de segurança.
    """
    key = (
        b"".join(
            np.ascontiguousarray(a).tobytes()
            for a in (instance.demand, instance.capacity, instance.fixed_cost, instance.unit_cost)
        )
        + np.float64(instance.unserved_penalty).tobytes()
    )
    if key not in _CACHE:
        solver = LnsSolver(LocalSearch(NearestFeasibleGreedy()), seed=0, max_iterations=iterations)
        _CACHE[key] = solver.solve(instance, time_limit_s).solution
    return _CACHE[key]


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0
