"""Contratos de entrada: NaN, infinito, aliasing, IDs duplicados e soluções malformadas."""

import numpy as np
import pytest

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import Solution


def make(**over: object) -> Instance:
    base: dict[str, object] = {
        "facility_ids": ("A",),
        "demand_ids": ("R",),
        "demand": np.array([4.0]),
        "capacity": np.array([5.0]),
        "fixed_cost": np.array([1.0]),
        "unit_cost": np.array([[1.0]]),
        "unserved_penalty": 10.0,
    }
    base.update(over)
    return Instance(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["demand", "capacity", "fixed_cost"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejeita_nan_e_infinito(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match="NaN ou infinito"):
        make(**{field: np.array([bad])})


def test_rejeita_custo_unitario_invalido() -> None:
    with pytest.raises(ValueError):
        make(unit_cost=np.array([[np.nan]]))
    with pytest.raises(ValueError, match="não negativos"):
        make(unit_cost=np.array([[-1.0]]))


def test_rejeita_penalidade_nao_finita() -> None:
    with pytest.raises(ValueError):
        make(unserved_penalty=float("nan"))


def test_rejeita_ids_duplicados() -> None:
    with pytest.raises(ValueError, match="únicos"):
        make(
            facility_ids=("A", "A"),
            capacity=np.array([5.0, 5.0]),
            fixed_cost=np.ones(2),
            unit_cost=np.ones((2, 1)),
        )


def test_instancia_nao_muda_por_alias_externo() -> None:
    demand = np.array([4.0])
    inst = make(demand=demand)
    demand[0] = 99.0
    assert inst.demand[0] == 4.0
    with pytest.raises(ValueError):  # somente leitura
        inst.demand[0] = 1.0


def test_solucao_exige_inteiros_e_nao_muda() -> None:
    with pytest.raises(ValueError):
        Solution(np.array([0.5]))
    with pytest.raises(ValueError):
        Solution(np.array([[0]]))
    raw = np.array([0])
    sol = Solution(raw)
    raw[0] = 7
    assert sol.assignment[0] == 0


def test_avaliador_nunca_certifica_custo_nao_finito() -> None:
    inst = make()
    ev = evaluate(inst, Solution(np.array([0])))
    assert np.isfinite(ev.total_cost) and ev.feasible
