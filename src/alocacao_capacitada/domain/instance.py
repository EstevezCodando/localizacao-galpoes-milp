"""Instância do problema de alocação capacitada com abertura de centros (SSCFLP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _frozen(values: np.ndarray, name: str) -> np.ndarray:
    """Cópia float64 somente leitura; rejeita NaN e infinito."""
    array = np.array(values, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contém NaN ou infinito")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Instance:
    """Dados de uma instância, validados e somente leitura.

    Decisões: quais centros abrir e a qual centro cada nó de demanda é atribuído
    (*single-source*: a demanda de um nó não pode ser dividida).

    Attributes:
        facility_ids: identificadores dos centros candidatos (tamanho m).
        demand_ids: identificadores dos nós de demanda (tamanho n).
        demand: volume de cada nó (pedidos), shape (n,).
        capacity: capacidade de cada centro, shape (m,).
        fixed_cost: custo de abrir cada centro, shape (m,).
        unit_cost: custo por unidade de demanda, shape (m, n).
        unserved_penalty: custo por unidade de demanda não atendida. Torna
            a instância sempre viável e explicita o custo de "adiar" pedidos.
    """

    facility_ids: tuple[str, ...]
    demand_ids: tuple[str, ...]
    demand: np.ndarray
    capacity: np.ndarray
    fixed_cost: np.ndarray
    unit_cost: np.ndarray
    unserved_penalty: float

    def __post_init__(self) -> None:
        m, n = len(self.facility_ids), len(self.demand_ids)
        if len(set(self.facility_ids)) != m or len(set(self.demand_ids)) != n:
            raise ValueError("identificadores de centros e de regiões devem ser únicos")
        # Cópia defensiva e somente leitura: a instância não muda por alias externo.
        for name in ("demand", "capacity", "fixed_cost", "unit_cost"):
            object.__setattr__(self, name, _frozen(getattr(self, name), name))
        if self.demand.shape != (n,):
            raise ValueError(f"demand deve ter shape ({n},), recebido {self.demand.shape}")
        if self.capacity.shape != (m,) or self.fixed_cost.shape != (m,):
            raise ValueError(f"capacity e fixed_cost devem ter shape ({m},)")
        if self.unit_cost.shape != (m, n):
            raise ValueError(f"unit_cost deve ter shape ({m}, {n})")
        if (self.demand < 0).any() or (self.capacity < 0).any() or (self.fixed_cost < 0).any():
            raise ValueError("demanda, capacidade e custo fixo devem ser não negativos")
        if (self.unit_cost < 0).any():
            raise ValueError("custos unitários devem ser não negativos")
        if not np.isfinite(self.unserved_penalty) or self.unserved_penalty < 0:
            raise ValueError("unserved_penalty deve ser finita e não negativa")

    @property
    def n_facilities(self) -> int:
        return len(self.facility_ids)

    @property
    def n_demand(self) -> int:
        return len(self.demand_ids)

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())
