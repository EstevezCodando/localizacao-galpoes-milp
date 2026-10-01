"""Contrato comum dos solvers (princípios D e O do SOLID: dependemos da abstração)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.domain.solution import Solution


@dataclass(frozen=True)
class SolveResult:
    solution: Solution
    runtime_s: float
    status: str
    lower_bound: float | None = None


class Solver(Protocol):
    name: str

    def solve(self, instance: Instance, time_limit_s: float) -> SolveResult: ...


_BOUND_TOL = 1e-6


def _check_bound(upper_bound: float, lower_bound: float) -> None:
    if lower_bound > upper_bound * (1 + _BOUND_TOL) + _BOUND_TOL:
        raise ValueError(
            f"limite inferior ({lower_bound}) acima da solução ({upper_bound}): "
            "indica erro de instância, modelo ou avaliação"
        )


def gap_ub(upper_bound: float, lower_bound: float | None) -> float | None:
    """Gap relativo à solução: (UB - LB) / UB. É a convenção da maioria dos solvers.

    Devolve None sem limite inferior ou com UB não positivo. Levanta ValueError se LB > UB
    além da tolerância (isso é sinal de erro, não um gap zero).
    """
    if lower_bound is None or upper_bound <= 0:
        return None
    _check_bound(upper_bound, lower_bound)
    return max(0.0, (upper_bound - lower_bound) / upper_bound)


def excess_lb(upper_bound: float, lower_bound: float | None) -> float | None:
    """Excesso relativo ao limite: (UB - LB) / LB. Responde "quanto acima do piso estamos"."""
    if lower_bound is None or lower_bound <= 0:
        return None
    _check_bound(upper_bound, lower_bound)
    return max(0.0, (upper_bound - lower_bound) / lower_bound)


gap = gap_ub  # compatibilidade com código anterior
