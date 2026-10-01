"""Testes sobre uma instância minúscula com ótimo conhecido (verificável à mão)."""

from __future__ import annotations

import numpy as np
import pytest

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import UNASSIGNED, Solution
from alocacao_capacitada.geo import haversine_matrix
from alocacao_capacitada.solvers.base import gap
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.local_search import LocalSearch
from alocacao_capacitada.solvers.milp import MilpSolver


@pytest.fixture
def tiny() -> Instance:
    """2 centros, 3 nós. Centro 0 é barato de abrir e cabe tudo; centro 1 é caro.

    Ótimo: abrir só o centro 0 (ver OPTIMAL_TINY).
    """
    return Instance(
        facility_ids=("A", "B"),
        demand_ids=("n1", "n2", "n3"),
        demand=np.array([1.0, 1.0, 1.0]),
        capacity=np.array([3.0, 3.0]),
        fixed_cost=np.array([10.0, 100.0]),
        unit_cost=np.array([[1.0, 1.0, 5.0], [2.0, 2.0, 1.0]]),
        unserved_penalty=1000.0,
    )


OPTIMAL_TINY = 10.0 + 1.0 + 1.0 + 5.0  # só A aberto: fixo 10 + transportes 1, 1 e 5


def test_evaluate_known_cost(tiny: Instance) -> None:
    ev = evaluate(tiny, Solution(np.array([0, 0, 0])))
    assert ev.total_cost == pytest.approx(OPTIMAL_TINY)
    assert ev.feasible and ev.n_open == 1


def test_evaluate_detects_capacity_violation(tiny: Instance) -> None:
    small = Instance(**{**tiny.__dict__, "capacity": np.array([2.0, 3.0])})
    ev = evaluate(small, Solution(np.array([0, 0, 0])))
    assert not ev.feasible


def test_evaluate_penalizes_unserved(tiny: Instance) -> None:
    ev = evaluate(tiny, Solution(np.array([0, UNASSIGNED, UNASSIGNED])))
    assert ev.unserved_demand == 2.0
    assert ev.penalty_cost == pytest.approx(2000.0)


def test_milp_finds_optimum_and_proves_it(tiny: Instance) -> None:
    result = MilpSolver().solve(tiny, time_limit_s=10)
    ev = evaluate(tiny, result.solution)
    assert result.status == "OTIMO"
    assert ev.total_cost == pytest.approx(OPTIMAL_TINY)
    assert result.lower_bound == pytest.approx(OPTIMAL_TINY)


def test_heuristics_are_feasible_and_not_better_than_milp(tiny: Instance) -> None:
    optimum = evaluate(tiny, MilpSolver().solve(tiny, 10).solution).total_cost
    for solver in (NearestFeasibleGreedy(), LocalSearch(NearestFeasibleGreedy())):
        ev = evaluate(tiny, solver.solve(tiny, 5).solution)
        assert ev.feasible
        assert ev.total_cost >= optimum - 1e-6


def test_local_search_never_worsens_baseline(tiny: Instance) -> None:
    base = evaluate(tiny, NearestFeasibleGreedy().solve(tiny).solution).total_cost
    improved = evaluate(tiny, LocalSearch(NearestFeasibleGreedy()).solve(tiny, 5).solution)
    assert improved.total_cost <= base + 1e-9


def test_instance_rejects_inconsistent_shapes(tiny: Instance) -> None:
    with pytest.raises(ValueError):
        Instance(**{**tiny.__dict__, "demand": np.array([1.0])})


def test_gap_definition() -> None:
    assert gap(100.0, 90.0) == pytest.approx(0.1)
    assert gap(100.0, None) is None


def test_haversine_sao_paulo_rio() -> None:
    d = haversine_matrix(
        np.array([-23.55]), np.array([-46.63]), np.array([-22.91]), np.array([-43.17])
    )
    assert d[0, 0] == pytest.approx(357, abs=10)


def test_gap_ub_e_excesso_lb_sao_convencoes_distintas() -> None:
    from alocacao_capacitada.solvers.base import excess_lb, gap_ub

    assert gap_ub(100.0, 90.0) == pytest.approx(10.0 / 100.0)  # 10%
    assert excess_lb(100.0, 90.0) == pytest.approx(10.0 / 90.0)  # 11,111%
    assert gap_ub(100.0, None) is None and excess_lb(100.0, None) is None
    assert gap_ub(0.0, 0.0) is None and excess_lb(5.0, 0.0) is None


def test_limite_acima_da_solucao_e_erro_e_nao_gap_zero() -> None:
    from alocacao_capacitada.solvers.base import excess_lb, gap_ub

    with pytest.raises(ValueError, match="limite inferior"):
        gap_ub(100.0, 120.0)
    with pytest.raises(ValueError, match="limite inferior"):
        excess_lb(100.0, 120.0)
    assert gap_ub(100.0, 100.0 + 1e-9) == 0.0  # dentro da tolerância numérica
