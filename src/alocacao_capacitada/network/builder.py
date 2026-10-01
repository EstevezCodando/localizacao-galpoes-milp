"""Instância da rede nacional a partir de municípios: demanda potencial, módulos e frete viário.

Premissas explícitas (todas variáveis nas análises):
  * a operadora hipotética tem `company_share` do mercado nacional de pedidos;
  * cada candidato oferece módulos de capacidade discreta (pedidos/mês); abrir dois módulos no
    mesmo município equivale a expandir; não há economia de escala;
  * custo fixo mensal do módulo = aluguel do estado (R$/m²/mês) × área, com área = capacidade ÷
    densidade; NÃO inclui mão de obra, equipamentos nem impostos (limitação declarada);
  * frete = piso da ANTT (viagem) diluído por `orders_per_vehicle`, sobre distância viária;
  * pedido não atendido por rede própria = terceirizado a `outsourcing_cost` por pedido;
  * fonte única exige que cada nó caiba em um módulo: municípios com demanda acima de
    `max_node_fraction` × o maior módulo são divididos em ZONAS iguais no mesmo ponto (grandes
    cidades têm várias zonas de entrega). Sem isso, São Paulo ficaria sempre sem atendimento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.lake.freight import FreightModel
from alocacao_capacitada.lake.rent import rent_lookup


@dataclass(frozen=True)
class NetworkConfig:
    n_nodes: int = 400
    n_candidates: int = 60
    company_share: float = 0.10
    module_sizes: tuple[int, ...] = (100_000, 250_000)
    density_orders_m2_month: float = 20.0
    outsourcing_cost: float = 100.0
    max_node_fraction: float = 0.4  # nó > fração × maior módulo é dividido em zonas
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Network:
    instance: Instance
    nodes: pd.DataFrame  # cod, nome, uf, lat, lon, pedidos_mes
    candidates: pd.DataFrame  # cod, nome, uf, lat, lon
    facility_site: np.ndarray  # índice do candidato de cada instalação (módulo)
    facility_size: np.ndarray  # tamanho do módulo (pedidos/mês)
    coverage: float  # fração da demanda nacional potencial coberta pelos nós


def build_network(
    demand_month: pd.Series,
    municipios: pd.DataFrame,
    candidates: pd.DataFrame,
    km: np.ndarray,
    freight: FreightModel,
    rent_cities: pd.DataFrame,
    config: NetworkConfig,
    node_cods: list[str],
    zone_reference: pd.Series | None = None,
) -> Network:
    """`km` é a matriz (candidatos × nós) de distância viária, na ordem dos DataFrames dados."""
    m = municipios.set_index("cod")
    nodes = m.loc[node_cods, ["nome", "uf", "lat", "lon"]].assign(
        pedidos_mes=demand_month.reindex(node_cods).to_numpy()
    )
    reference = zone_reference.reindex(node_cods).to_numpy() if zone_reference is not None else None
    nodes = _split_large_nodes(nodes.reset_index(), config, reference)
    cands = candidates.reset_index(drop=True)
    if km.shape != (len(cands), len(node_cods)):
        raise ValueError(f"matriz {km.shape} incompatível com {len(cands)}×{len(node_cods)}")
    # Cada zona herda a coluna de distância do município de origem.
    column = {cod: j for j, cod in enumerate(node_cods)}
    km = km[:, [column[c] for c in nodes["cod_municipio"]]]

    rents = rent_lookup(rent_cities, cands["uf"].astype(str).tolist())
    rent_month = np.array([rents[uf][0] for uf in cands["uf"].astype(str)])

    site, size, fixed, cap = [], [], [], []
    for i in range(len(cands)):
        for s in config.module_sizes:
            site.append(i)
            size.append(s)
            cap.append(float(s))
            fixed.append(float(rent_month[i] * s / config.density_orders_m2_month))
    site_arr = np.array(site)
    unit = freight.cost_per_order(km)[site_arr, :]  # (instalações × nós), por pedido
    instance = Instance(
        facility_ids=tuple(f"{cands['cod'].iloc[i]}:{s}" for i, s in zip(site, size, strict=True)),
        demand_ids=tuple(nodes["cod"]),
        demand=nodes["pedidos_mes"].to_numpy(dtype=float),
        capacity=np.array(cap),
        fixed_cost=np.array(fixed),
        unit_cost=unit,
        unserved_penalty=config.outsourcing_cost,
    )
    total = float(demand_month.sum())
    coverage = float(nodes["pedidos_mes"].sum() / total) if total > 0 else 0.0
    return Network(instance, nodes, cands, site_arr, np.array(size), coverage)


def _split_large_nodes(
    nodes: pd.DataFrame, config: NetworkConfig, reference: np.ndarray | None = None
) -> pd.DataFrame:
    """Zonas por município. Com `reference`, o número de zonas vem dela (mesma estrutura entre
    cenários, que permite comparar a mesma rede sob demandas diferentes)."""
    limit = config.max_node_fraction * max(config.module_sizes)
    rows = []
    for k, r in enumerate(nodes.to_dict("records")):
        demand = float(r["pedidos_mes"])
        basis = float(reference[k]) if reference is not None else demand
        zones = max(1, int(np.ceil(basis / limit)))
        for z in range(zones):
            rows.append(
                {
                    "cod": r["cod"] if zones == 1 else f"{r['cod']}#{z + 1}",
                    "cod_municipio": r["cod"],
                    "nome": r["nome"],
                    "uf": r["uf"],
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "pedidos_mes": demand / zones,
                }
            )
    return pd.DataFrame(rows)
