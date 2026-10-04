"""Problema SSCFLP estrito e validador independente.

min  sum_i f_i y_i + sum_ij c_ij x_ij
s.a. sum_i x_ij = 1           para todo j  (fonte única, atendimento obrigatório)
     sum_j d_j x_ij <= Q_i y_i para todo i
     x_ij <= y_i,  x, y binários

`c_ij` é o custo TOTAL de atender o cliente j inteiro pelo centro i (contrato explícito,
doc 05: multiplicar de novo por d_j distorceria o benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TOL = 1e-6


def _ro(values: np.ndarray, name: str) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contém NaN ou infinito")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Problema:
    """Instância imutável. `xy_fac`/`xy_cli` são atributos espaciais opcionais (não custos)."""

    nome: str
    familia: str
    fixed: np.ndarray  # (m,)
    capacity: np.ndarray  # (m,)
    demand: np.ndarray  # (n,)
    cost: np.ndarray  # (m, n) custo total do par
    xy_fac: np.ndarray | None = None
    xy_cli: np.ndarray | None = None
    meta: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("fixed", "capacity", "demand", "cost"):
            object.__setattr__(self, name, _ro(getattr(self, name), name))
        if self.cost.ndim != 2 or 0 in self.cost.shape:
            raise ValueError("cost deve ser uma matriz não vazia")
        m, n = self.cost.shape
        if self.fixed.shape != (m,) or self.capacity.shape != (m,) or self.demand.shape != (n,):
            raise ValueError("dimensões incompatíveis")
        if (self.demand <= 0).any():
            raise ValueError("demanda deve ser positiva no cenário básico")
        if (self.capacity < 0).any() or (self.fixed < 0).any() or (self.cost < 0).any():
            raise ValueError("capacidade, custo fixo e custos devem ser não negativos")

    @property
    def m(self) -> int:
        return int(self.cost.shape[0])

    @property
    def n(self) -> int:
        return int(self.cost.shape[1])

    @property
    def razao_capacidade(self) -> float:
        return float(self.capacity.sum() / self.demand.sum())

    def restrito(self, centros: np.ndarray) -> Problema:
        """Subproblema com apenas os centros indicados (poda). Índices relativos mudam."""
        idx = np.asarray(centros, dtype=int)
        return Problema(
            nome=f"{self.nome}|sub{idx.size}",
            familia=self.familia,
            fixed=self.fixed[idx],
            capacity=self.capacity[idx],
            demand=self.demand,
            cost=self.cost[idx],
            xy_fac=None if self.xy_fac is None else self.xy_fac[idx],
            xy_cli=self.xy_cli,
            meta=dict(self.meta),
        )


@dataclass(frozen=True)
class Validacao:
    objetivo: float
    fixo: float
    atendimento: float
    n_abertos: int
    erros: tuple[str, ...]

    @property
    def valida(self) -> bool:
        return not self.erros


def validar(prob: Problema, atribuicao: np.ndarray) -> Validacao:
    """Recalcula o objetivo a partir dos dados ORIGINAIS e confere todas as restrições.

    Não confia em nada que o solver informe: abertura é derivada da atribuição.
    """
    a = np.asarray(atribuicao)
    erros: list[str] = []
    if a.shape != (prob.n,):
        return Validacao(np.inf, np.inf, np.inf, 0, ("atribuição com tamanho errado",))
    if not np.issubdtype(a.dtype, np.integer):
        return Validacao(np.inf, np.inf, np.inf, 0, ("atribuição não inteira",))
    if ((a < 0) | (a >= prob.m)).any():
        return Validacao(np.inf, np.inf, np.inf, 0, ("cliente sem centro ou índice inválido",))
    carga = np.bincount(a, weights=prob.demand, minlength=prob.m)
    for i in np.flatnonzero(carga > prob.capacity * (1 + _TOL) + _TOL):
        erros.append(f"capacidade excedida no centro {i}: {carga[i]:.3f} > {prob.capacity[i]:.3f}")
    abertos = np.unique(a)
    fixo = float(prob.fixed[abertos].sum())
    atend = float(prob.cost[a, np.arange(prob.n)].sum())
    return Validacao(fixo + atend, fixo, atend, int(abertos.size), tuple(erros))


def gap_certificado(upper: float, lower: float | None, eps: float = 1e-9) -> float | None:
    """g = (U - L) / max(|U|, eps), doc 05. L precisa ser bound válido do problema ORIGINAL."""
    if lower is None or not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower > upper * (1 + 1e-7) + 1e-6:
        raise ValueError(f"bound {lower} acima da solução {upper}: erro de modelo ou de escopo")
    return max(0.0, (upper - lower) / max(abs(upper), eps))
