"""LNS, relaxação linear e warm start."""

import numpy as np
import pytest

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.lp_relaxation import solve_lp_relaxation
from alocacao_capacitada.solvers.milp import MilpSolver


@pytest.fixture
def medium() -> Instance:
    rng = np.random.default_rng(7)
    m, n = 6, 25
    demand = rng.integers(1, 10, n).astype(float)
    return Instance(
        facility_ids=tuple(f"F{i}" for i in range(m)),
        demand_ids=tuple(f"N{j}" for j in range(n)),
        demand=demand,
        capacity=np.full(m, demand.sum() / 3),
        fixed_cost=np.full(m, 60.0),
        unit_cost=rng.uniform(1, 20, (m, n)),
        unserved_penalty=500.0,
    )


def test_lns_nunca_piora_e_e_viavel(medium: Instance) -> None:
    base = evaluate(medium, NearestFeasibleGreedy().solve(medium).solution)
    result = LnsSolver(NearestFeasibleGreedy(), seed=1, free_size=10).solve(medium, 3)
    ev = evaluate(medium, result.solution)
    assert ev.feasible
    assert ev.total_cost <= base.total_cost + 1e-9


def test_lns_chega_perto_do_otimo(medium: Instance) -> None:
    optimum = evaluate(medium, MilpSolver().solve(medium, 30).solution).total_cost
    lns = evaluate(
        medium, LnsSolver(NearestFeasibleGreedy(), seed=1, free_size=12).solve(medium, 5).solution
    )
    assert optimum - 1e-6 <= lns.total_cost <= optimum * 1.05


def test_lp_e_limite_inferior_valido(medium: Instance) -> None:
    milp = MilpSolver().solve(medium, 30)
    optimum = evaluate(medium, milp.solution).total_cost
    lp = solve_lp_relaxation(medium)
    assert lp.lower_bound <= optimum + 1e-6
    assert lp.capacity_dual.shape == (medium.n_facilities,)
    assert ((lp.y_fraction >= -1e-9) & (lp.y_fraction <= 1 + 1e-9)).all()


def test_warm_start_preserva_o_otimo(medium: Instance) -> None:
    cold = evaluate(medium, MilpSolver().solve(medium, 30).solution).total_cost
    warm = evaluate(
        medium, MilpSolver(warm_start=NearestFeasibleGreedy()).solve(medium, 30).solution
    ).total_cost
    assert warm == pytest.approx(cold, rel=1e-6)


def test_sensibilidade_a_capacidade_e_dual_vezes_y_e_nao_o_dual(medium: Instance) -> None:
    """Teorema do envelope: d(custo)/dQ_i = dual_i * y_i, conferido por diferença finita."""
    from dataclasses import replace

    base = solve_lp_relaxation(medium)
    for i in range(medium.n_facilities):
        cap = medium.capacity.copy()
        step = 1e-3
        cap[i] += step
        bumped = solve_lp_relaxation(replace(medium, capacity=cap))
        finite = (bumped.lower_bound - base.lower_bound) / step
        assert finite == pytest.approx(base.capacity_marginal_value[i], abs=5e-3)


def test_lns_com_orcamento_em_iteracoes_e_deterministico(medium: Instance) -> None:
    a = LnsSolver(NearestFeasibleGreedy(), seed=3, free_size=10, max_iterations=15).solve(
        medium, 60
    )
    b = LnsSolver(NearestFeasibleGreedy(), seed=3, free_size=10, max_iterations=15).solve(
        medium, 60
    )
    assert a.status == "LIMITE_ITERACOES"
    assert np.array_equal(a.solution.assignment, b.solution.assignment)
