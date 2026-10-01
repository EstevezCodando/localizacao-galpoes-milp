import numpy as np
import pandas as pd
import pytest

from alocacao_capacitada.analysis.sensitivity import sweeps
from alocacao_capacitada.data.instance_builder import InstanceConfig
from alocacao_capacitada.data.olist import GeoNodes
from alocacao_capacitada.lake.freight import FreightModel


@pytest.fixture
def nodes() -> GeoNodes:
    rng = np.random.default_rng(1)
    n = 12
    idx = pd.Index([f"{i:03d}" for i in range(n)], name="prefix")
    frame = pd.DataFrame(
        {
            "orders": rng.integers(50, 400, n),
            "lat": rng.uniform(-30, -10, n),
            "lon": rng.uniform(-55, -40, n),
            "state": ["SP"] * n,
        },
        index=idx,
    )
    return GeoNodes(demand=frame, supply=frame)


def test_demanda_com_capacidade_congelada_nao_muda_a_capacidade(nodes: GeoNodes) -> None:
    base = InstanceConfig(n_demand=10, n_facilities=5, freight=FreightModel(4.0, 450.0, 50.0))
    levels = dict(sweeps(nodes, base)["demanda, capacidade congelada (x)"])
    low, high = levels["0.6"], levels["1.4"]
    assert np.array_equal(low.capacity, high.capacity)  # igualdade exata, como na auditoria
    assert np.array_equal(low.fixed_cost, high.fixed_cost)
    assert np.array_equal(low.unit_cost, high.unit_cost)
    assert high.total_demand == pytest.approx(low.total_demand * 1.4 / 0.6)


def test_expansao_conjunta_muda_a_capacidade_junto_com_a_demanda(nodes: GeoNodes) -> None:
    base = InstanceConfig(n_demand=10, n_facilities=5, freight=FreightModel(4.0, 450.0, 50.0))
    levels = dict(sweeps(nodes, base)["demanda e capacidade juntas (x)"])
    assert levels["1.4"].capacity.sum() > levels["0.6"].capacity.sum()


def test_penalidade_e_fixa_ao_variar_pedidos_por_veiculo(nodes: GeoNodes) -> None:
    base = InstanceConfig(n_demand=10, n_facilities=5, freight=FreightModel(4.0, 450.0, 50.0))
    levels = sweeps(nodes, base)["pedidos por veiculo"]
    penalties = {inst.unserved_penalty for _, inst in levels}
    assert len(penalties) == 1
    assert levels[0][1].unit_cost.sum() > levels[-1][1].unit_cost.sum()  # o frete, sim, muda
