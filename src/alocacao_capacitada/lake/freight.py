"""Custo de frete a partir do piso mínimo da ANTT (Tabela A, carga geral).

    piso(km) = km x CCD + CC                      (CCD em R$/km, CC em R$ por viagem)
    retorno vazio: + 0,92 x km x CCD

Os coeficientes em `data/reference/` são gerados pelo pipeline bronze→silver→gold a partir
da resolução oficial coletada (ver colunas de proveniência do CSV).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EMPTY_RETURN_FACTOR = 0.92


@dataclass(frozen=True)
class FreightModel:
    """Custo de uma viagem de um veículo e sua diluição por pedido."""

    ccd_rs_km: float
    cc_rs: float
    orders_per_vehicle: float
    empty_return: bool = True

    def __post_init__(self) -> None:
        if self.orders_per_vehicle <= 0:
            raise ValueError("orders_per_vehicle deve ser positivo")

    def cost_per_order(self, km: np.ndarray) -> np.ndarray:
        """Custo por pedido de uma viagem de `km` quilômetros."""
        factor = 1 + (EMPTY_RETURN_FACTOR if self.empty_return else 0.0)
        vehicle: np.ndarray = km * self.ccd_rs_km * factor + self.cc_rs
        return vehicle / self.orders_per_vehicle


def load_freight_model(
    path: Path,
    axles: int,
    orders_per_vehicle: float,
    empty_return: bool = True,
    cargo: str = "carga_geral",
) -> FreightModel:
    table = pd.read_csv(path)
    row = table.loc[(table["eixos"] == axles) & (table["tipo_carga"] == cargo)]
    if row.empty:
        raise ValueError(f"sem coeficientes para {axles} eixos / {cargo} em {path.name}")
    r = row.iloc[0]
    return FreightModel(float(r["ccd_rs_km"]), float(r["cc_rs"]), orders_per_vehicle, empty_return)
