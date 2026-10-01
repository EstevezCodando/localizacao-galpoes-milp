"""Benchmark SSCFLP de Holmberg et al. (1999): 71 instâncias com ótimo publicado (fonte única).

Formato: "m n"; m linhas "capacidade custo_fixo"; linha(s) com as n demandas;
m linhas com n custos de atribuição (custo TOTAL de atender o cliente inteiro).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.domain.instance import Instance

_PENALTY = 1e4  # alta o bastante para nunca compensar não atender (conferido em cada resultado)


def load_holmberg_instance(path: Path) -> Instance:
    # Alguns arquivos originais terminam com bytes NUL de preenchimento: descartá-los.
    tokens = path.read_text(encoding="latin-1").replace(chr(0), " ").split()
    m, n = int(tokens[0]), int(tokens[1])
    expected = 2 * m + n + m * n
    if len(tokens) - 2 < expected:
        raise ValueError(f"{path.name}: {len(tokens) - 2} valores, esperados {expected}")
    # Alguns arquivos originais trazem cabeçalhos de e-mail de 1995 após os dados: ignorá-los.
    values = np.array([float(tok) for tok in tokens[2 : 2 + expected]])
    capacity_fixed = values[: 2 * m].reshape(m, 2)
    demand = values[2 * m : 2 * m + n]
    total_cost = values[2 * m + n :].reshape(m, n)
    return Instance(
        facility_ids=tuple(f"J{j}" for j in range(m)),
        demand_ids=tuple(f"I{i}" for i in range(n)),
        demand=demand,
        capacity=capacity_fixed[:, 0].copy(),
        fixed_cost=capacity_fixed[:, 1].copy(),
        unit_cost=total_cost / demand[None, :],
        unserved_penalty=_PENALTY,
    )


def load_holmberg_optima(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
