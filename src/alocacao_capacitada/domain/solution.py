"""Representação de uma solução."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNASSIGNED = -1


@dataclass(frozen=True)
class Solution:
    """Atribuição de cada nó de demanda a um centro.

    `assignment[j]` é o índice do centro que atende o nó j, ou `UNASSIGNED`.
    Os centros abertos são derivados: abre-se um centro se e somente se ele
    atende algum nó (abrir um centro vazio só aumentaria o custo).
    """

    assignment: np.ndarray

    def __post_init__(self) -> None:
        array = np.asarray(self.assignment)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("assignment deve ser um vetor 1-D de inteiros")
        frozen = array.astype(np.int64, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "assignment", frozen)

    def open_facilities(self) -> np.ndarray:
        used = self.assignment[self.assignment != UNASSIGNED]
        return np.unique(used)
