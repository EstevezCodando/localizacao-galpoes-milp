"""Blocos reutilizáveis dos experimentos de rede: cenários, projeto da rede e avaliação de projetos.

Avaliar um PROJETO (conjunto de módulos abertos) sob uma demanda diferente da que o originou
exige fixar a capacidade instalada e deixar apenas a atribuição se reajustar (recurso). Os custos
fixos do projeto não mudam; o frete e a terceirização, sim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import Solution
from alocacao_capacitada.ml.potential import ORDERS_GROWTH_SCENARIOS, monthly_demand
from alocacao_capacitada.network.builder import Network, NetworkConfig, build_network
from alocacao_capacitada.network.context import Context
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.local_search import LocalSearch


@dataclass(frozen=True)
class Scenario:
    name: str
    year: int
    bound: str  # central | lo | hi (intervalo da projeção de população)
    growth_label: str  # baixo | base | alto (crescimento dos pedidos)

    @property
    def growth(self) -> float:
        return ORDERS_GROWTH_SCENARIOS[self.growth_label]


BASE_2025 = Scenario("2025 base", 2025, "central", "base")
BASE_2030 = Scenario("2030 base", 2030, "central", "base")
LOW_2030 = Scenario("2030 baixo", 2030, "lo", "baixo")
HIGH_2030 = Scenario("2030 alto", 2030, "hi", "alto")


def scenario_demand(ctx: Context, sc: Scenario, share: float) -> pd.Series:
    return monthly_demand(
        ctx.rate, ctx.population(sc.year, sc.bound), sc.year, sc.growth, share * ctx.coverage
    )


def make_network(
    ctx: Context,
    demand: pd.Series,
    nodes: list[str],
    candidates: list[str],
    km: np.ndarray,
    config: NetworkConfig,
    zone_reference: pd.Series,
    orders_per_vehicle: float = 200.0,
) -> Network:
    cand_frame = ctx.municipios.set_index("cod").loc[candidates].reset_index()
    return build_network(
        demand,
        ctx.municipios,
        cand_frame,
        km,
        ctx.freight_by_ovv[orders_per_vehicle],
        ctx.rent_cities,
        config,
        nodes,
        zone_reference=zone_reference,
    )


def solve_design(
    instance: Instance, iterations: int, seed: int = 0, free_size: int = 80
) -> Solution:
    solver = LnsSolver(
        LocalSearch(NearestFeasibleGreedy()),
        seed=seed,
        free_size=free_size,
        max_iterations=iterations,
    )
    return solver.solve(instance, 3600.0).solution


def open_set(solution: Solution) -> np.ndarray:
    return solution.open_facilities()


def subset(instance: Instance, idx: np.ndarray, fixed_to_zero: bool = True) -> Instance:
    """Instância só com as instalações `idx` (projeto fixado)."""
    return replace(
        instance,
        facility_ids=tuple(instance.facility_ids[i] for i in idx),
        capacity=instance.capacity[idx],
        fixed_cost=np.zeros(len(idx)) if fixed_to_zero else instance.fixed_cost[idx],
        unit_cost=instance.unit_cost[idx, :],
    )


def evaluate_design(
    instance: Instance, design: np.ndarray, iterations: int = 40, seed: int = 0
) -> dict[str, float]:
    """Custo mensal de operar o projeto `design` na demanda de `instance` (recurso otimizado)."""
    fixed = float(instance.fixed_cost[design].sum())
    sub = subset(instance, design)
    solution = solve_design(sub, iterations, seed)
    ev = evaluate(sub, solution)
    total = fixed + ev.transport_cost + ev.penalty_cost
    return {
        "custo_total": total,
        "custo_fixo": fixed,
        "frete": ev.transport_cost,
        "terceirizado": ev.penalty_cost,
        "pct_terceirizado": 100.0 * ev.unserved_demand / instance.total_demand,
        "modulos": float(len(design)),
    }
