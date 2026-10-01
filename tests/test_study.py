import numpy as np
import pytest

from alocacao_capacitada.analysis.study import (
    controlled_instance,
    enumerate_tiny,
    tiny_instance,
)
from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.solvers.greedy import NearestFeasibleGreedy
from alocacao_capacitada.solvers.lns import LnsSolver
from alocacao_capacitada.solvers.milp import MilpSolver


def test_enumeration_proves_reduced_optimum():
    inst = tiny_instance()
    space = enumerate_tiny(inst)
    assert space.custo.min() == pytest.approx(203.0)
    result = MilpSolver().solve(inst, 3.0)
    assert result.status == "OTIMO"
    assert evaluate(inst, result.solution).total_cost == pytest.approx(space.custo.min())


def test_demand_stress_does_not_expand_capacity_or_change_penalty():
    base = tiny_instance()
    changed = controlled_instance(base, demand=1.4)
    np.testing.assert_array_equal(changed.capacity, base.capacity)
    np.testing.assert_array_equal(changed.fixed_cost, base.fixed_cost)
    np.testing.assert_array_equal(changed.unit_cost, base.unit_cost)
    assert changed.unserved_penalty == base.unserved_penalty
    np.testing.assert_allclose(changed.demand, base.demand * 1.4)


def test_trace_matches_incumbent_and_is_monotonic():
    events = []
    inst = tiny_instance()
    solver = LnsSolver(
        NearestFeasibleGreedy(),
        free_size=5,
        max_iterations=3,
        on_progress=lambda i, t, c: events.append((i, t, c)),
    )
    result = solver.solve(inst, 3.0)
    assert events[0][0] == 0
    assert all(
        a[1] <= b[1] and a[2] >= b[2] - 1e-7 for a, b in zip(events, events[1:], strict=False)
    )
    assert events[-1][2] == pytest.approx(evaluate(inst, result.solution).total_cost)
