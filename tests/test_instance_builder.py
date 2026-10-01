import numpy as np
import pandas as pd
import pytest

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import GeoNodes


@pytest.fixture
def nodes() -> GeoNodes:
    demand = pd.DataFrame(
        {"orders": [100, 50], "lat": [-23.5, -22.9], "lon": [-46.6, -43.2], "state": ["SP", "RJ"]},
        index=pd.Index(["010", "200"], name="prefix"),
    )
    supply = pd.DataFrame(
        {"orders": [300, 100], "lat": [-23.0, -22.0], "lon": [-47.0, -43.0], "state": ["SP", "BA"]},
        index=pd.Index(["130", "400"], name="prefix"),
    )
    return GeoNodes(demand=demand, supply=supply)


@pytest.fixture
def cities() -> pd.DataFrame:
    return pd.DataFrame({"cidade": ["A", "B"], "uf": ["SP", "SP"], "medio_1001_3000": [20.0, 30.0]})


def test_custo_fixo_proporcional_a_capacidade_e_ao_aluguel_do_estado(
    nodes: GeoNodes, cities: pd.DataFrame
) -> None:
    config = InstanceConfig(
        n_demand=2,
        n_facilities=2,
        rent_cities=cities,
        orders_per_m2_month=10.0,
        horizon_months=20.0,
        capacity_slack=1.0,
    )
    inst = build_instance(nodes, config)
    rent_sp, rent_ba = 25.0, 25.0  # BA sem dado: média das cidades (20 e 30)
    # área = capacidade / (densidade x meses); custo = aluguel x área x meses = aluguel x cap / dens
    expected = np.array([rent_sp, rent_ba]) * inst.capacity / 10.0
    assert inst.fixed_cost == pytest.approx(expected)


def test_area_fixa_ignora_capacidade(nodes: GeoNodes, cities: pd.DataFrame) -> None:
    config = InstanceConfig(
        n_demand=2, n_facilities=2, rent_cities=cities, hub_area_m2=1000.0, horizon_months=10.0
    )
    inst = build_instance(nodes, config)
    assert inst.fixed_cost == pytest.approx(np.array([25.0, 25.0]) * 1000.0 * 10.0)


def test_sem_aluguel_mantem_a_premissa_antiga(nodes: GeoNodes) -> None:
    inst = build_instance(nodes, InstanceConfig(n_demand=2, n_facilities=2))
    assert inst.fixed_cost == pytest.approx(np.full(2, 40.0 * 1000 * 0.05))
