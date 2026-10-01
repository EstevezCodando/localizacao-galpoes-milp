"""Constrói uma `Instance` a partir de nós geográficos reais e parâmetros explícitos.

Geografia e volumes são REAIS (Olist). Custos e capacidades NÃO existem no
dataset: são premissas de modelagem, declaradas em `InstanceConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alocacao_capacitada.data.olist import GeoNodes
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.geo import haversine_matrix
from alocacao_capacitada.lake.freight import FreightModel
from alocacao_capacitada.lake.rent import rent_lookup


@dataclass(frozen=True)
class InstanceConfig:
    """Premissas de modelagem (todas documentadas e variáveis em sensibilidade).

    Attributes:
        n_demand: nº de regiões de demanda (as de maior volume).
        n_facilities: nº de centros candidatos (regiões com mais vendas).
        cost_per_order_km: custo de transporte por pedido por km.
        fixed_cost_per_order: custo fixo de abrir um centro, em múltiplos do
            custo de transporte de um pedido a 1000 km (escala o dilema abrir × transportar).
        capacity_slack: capacidade total = slack × demanda total, repartida
            proporcionalmente ao volume histórico de cada região (mín. 1 pedido).
        penalty_km: pedido não atendido custa o equivalente a transportá-lo por estes km.
        demand_scale: multiplica a demanda (teste de estresse/sensibilidade).
        rent_cities: se informado, o custo fixo de cada centro = aluguel/m² do estado da região ×
            `hub_area_m2` × `horizon_months` (estados sem dado usam a média das cidades).
        hub_area_m2: área fixa por centro (premissa). Ignora o volume: em escala Olist (~5 mil
            pedidos/mês) uma área realista de galpão torna cada centro caro demais para a demanda.
        orders_per_m2_month: se informado, a área de cada centro é proporcional à sua capacidade
            (área = capacidade / (densidade × meses)) e substitui `hub_area_m2`. Premissa.
        horizon_months: meses cobertos pela demanda (Olist: jan/2017 a ago/2018 ≈ 20).
        penalty_per_order: penalidade absoluta por pedido não atendido. Se None, deriva do frete
            (custo de transportar por `penalty_km`), o que acopla penalidade e ocupação do veículo;
            fixe um valor para variar um sem o outro.
        freight: se informado, o transporte usa o piso da ANTT (por viagem, diluído por
            pedido). A penalidade de não atendimento equivale a transportar por `penalty_km`.
    """

    n_demand: int = 100
    n_facilities: int = 30
    cost_per_order_km: float = 0.05
    fixed_cost_per_order: float = 40.0
    capacity_slack: float = 1.5
    penalty_km: float = 6000.0
    demand_scale: float = 1.0
    freight: FreightModel | None = None
    rent_cities: pd.DataFrame | None = None
    hub_area_m2: float = 5000.0
    orders_per_m2_month: float | None = None
    horizon_months: float = 20.0
    penalty_per_order: float | None = None


def build_instance(nodes: GeoNodes, config: InstanceConfig) -> Instance:
    demand_df = nodes.demand.head(config.n_demand)
    supply_df = nodes.supply.head(config.n_facilities)

    demand = demand_df["orders"].to_numpy(dtype=float) * config.demand_scale
    dist_km = haversine_matrix(
        supply_df["lat"].to_numpy(),
        supply_df["lon"].to_numpy(),
        demand_df["lat"].to_numpy(),
        demand_df["lon"].to_numpy(),
    )

    share = supply_df["orders"].to_numpy(dtype=float)
    share /= share.sum()
    capacity = np.maximum(np.ceil(config.capacity_slack * demand.sum() * share), 1.0)

    if config.rent_cities is not None:
        rents = rent_lookup(config.rent_cities, supply_df["state"].astype(str).tolist())
        monthly = np.array([rents[uf][0] for uf in supply_df["state"].astype(str)])
        if config.orders_per_m2_month is not None:
            area = capacity / (config.orders_per_m2_month * config.horizon_months)
        else:
            area = np.full(len(supply_df), config.hub_area_m2)
        fixed_cost = monthly * area * config.horizon_months
    else:
        # Abrir um centro custa o mesmo que transportar `fixed_cost_per_order` pedidos por 1000 km.
        fixed_cost = np.full(
            len(supply_df), config.fixed_cost_per_order * 1000 * config.cost_per_order_km
        )
    if config.freight is not None:
        unit_cost = config.freight.cost_per_order(dist_km)
        penalty = float(config.freight.cost_per_order(np.array([config.penalty_km]))[0])
    else:
        unit_cost = dist_km * config.cost_per_order_km
        penalty = config.penalty_km * config.cost_per_order_km
    if config.penalty_per_order is not None:
        penalty = config.penalty_per_order
    return Instance(
        facility_ids=tuple(supply_df.index.astype(str)),
        demand_ids=tuple(demand_df.index.astype(str)),
        demand=demand,
        capacity=capacity,
        fixed_cost=fixed_cost,
        unit_cost=unit_cost,
        unserved_penalty=penalty,
    )
