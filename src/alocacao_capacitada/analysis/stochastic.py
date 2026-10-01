"""Demanda incerta: programação estocástica de dois estágios (SAA) e valor da solução estocástica.

1º estágio (antes de conhecer a demanda): quais centros abrir (y).
2º estágio (recurso, após observar o cenário s): atribuir cada região aos centros abertos,
com custo de transporte e penalidade de não atendimento.

Métricas clássicas:
  RP   = custo esperado da solução estocástica (aqui avaliada FORA da amostra);
  EEV  = custo esperado de usar a solução do problema determinístico (demanda média);
  WS   = custo esperado com informação perfeita (cada cenário escolhe seus centros);
  VSS  = EEV - RP   (quanto se ganha por modelar a incerteza);
  EVPI = RP - WS    (quanto valeria conhecer a demanda antes de decidir).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from ortools.linear_solver import pywraplp

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.domain.instance import Instance
from alocacao_capacitada.solvers.milp import MilpSolver


def lognormal_scenarios(
    instance: Instance, n_scenarios: int, sigma: float, seed: int, rho: float = 0.0
) -> np.ndarray:
    """Demanda por cenário, shape (S, n): média preservada, choque multiplicativo por região.

    `rho` é a parcela da variância de cada choque que é COMUM a todas as regiões (fator
    sistêmico, por exemplo um ano forte ou fraco para o mercado inteiro). Com `rho = 0` os
    choques são independentes e o risco sistêmico é subestimado.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho deve estar em [0, 1]")
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_scenarios, 1))
    own = rng.normal(size=(n_scenarios, instance.n_demand))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * own
    shocks = np.exp(sigma * z - 0.5 * sigma**2)
    scenarios: np.ndarray = instance.demand[None, :] * shocks
    return scenarios


def solve_two_stage(instance: Instance, scenarios: np.ndarray, time_limit_s: float) -> np.ndarray:
    """Resolve o SSCFLP estocástico (forma extensiva); devolve a máscara de centros abertos."""
    s_count, n = scenarios.shape
    m = instance.n_facilities
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP indisponível")
    solver.SetTimeLimit(int(time_limit_s * 1000))
    y = [solver.BoolVar(f"y{i}") for i in range(m)]
    objective = solver.Objective()
    p, weight = instance.unserved_penalty, 1.0 / s_count
    for i in range(m):
        objective.SetCoefficient(y[i], float(instance.fixed_cost[i]))
    for s in range(s_count):
        d = scenarios[s]
        x = [[solver.BoolVar("") for _ in range(n)] for _ in range(m)]
        for j in range(n):
            solver.Add(sum(x[i][j] for i in range(m)) <= 1)
        for i in range(m):
            solver.Add(
                sum(float(d[j]) * x[i][j] for j in range(n)) <= float(instance.capacity[i]) * y[i]
            )
            for j in range(n):
                solver.Add(x[i][j] <= y[i])
                coef = weight * float(d[j]) * (float(instance.unit_cost[i, j]) - p)
                objective.SetCoefficient(x[i][j], coef)
    objective.SetOffset(weight * p * float(scenarios.sum()))
    objective.SetMinimization()
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"problema estocástico sem solução (status {status})")
    return np.array([v.solution_value() > 0.5 for v in y])


def recourse(
    instance: Instance, opened: np.ndarray, demand: np.ndarray, time_limit_s: float
) -> dict[str, float | str]:
    """Custo total de um cenário com os centros `opened` fixos, com status e gap do solver."""
    fixed = float(instance.fixed_cost[opened].sum())
    frozen = replace(
        instance,
        demand=demand,
        capacity=np.where(opened, instance.capacity, 0.0),
        fixed_cost=np.zeros(instance.n_facilities),
    )
    result = MilpSolver().solve(frozen, time_limit_s)
    cost = fixed + evaluate(frozen, result.solution).total_cost
    bound = None if result.lower_bound is None else fixed + result.lower_bound
    return {
        "custo": cost,
        "status": result.status,
        "gap": 0.0 if bound is None or cost <= 0 else max(0.0, (cost - bound) / cost),
    }


def recourse_cost(
    instance: Instance, opened: np.ndarray, demand: np.ndarray, time_limit_s: float
) -> float:
    return float(recourse(instance, opened, demand, time_limit_s)["custo"])


class PolicyEvaluator:
    """Avalia políticas (máscaras de abertura) nos mesmos cenários, UMA vez por par.

    Duas políticas idênticas recebem exatamente o mesmo custo por cenário: diferenças entre
    "métodos" que abrem os mesmos centros deixam de ser ruído do solver.
    """

    def __init__(self, instance: Instance, scenarios: np.ndarray, time_limit_s: float) -> None:
        self._instance = instance
        self._scenarios = scenarios
        self._time_limit = time_limit_s
        self._cache: dict[tuple[bytes, int], dict[str, float | str]] = {}

    def per_scenario(self, opened: np.ndarray) -> np.ndarray:
        costs = []
        for s, demand in enumerate(self._scenarios):
            key = (opened.tobytes(), s)
            if key not in self._cache:
                self._cache[key] = recourse(self._instance, opened, demand, self._time_limit)
            costs.append(float(self._cache[key]["custo"]))
        return np.array(costs)

    def max_gap(self) -> float:
        return max((float(v["gap"]) for v in self._cache.values()), default=0.0)


@dataclass(frozen=True)
class StochasticReport:
    rp: float
    eev: float
    ws: float
    n_open_stochastic: int
    n_open_deterministic: int
    same_policy: bool
    max_solver_gap: float
    vss_ci95: tuple[float, float]  # IC 95% da diferença pareada EEV − RP entre cenários de teste
    per_scenario_vss: np.ndarray

    @property
    def vss(self) -> float:
        return self.eev - self.rp

    @property
    def evpi(self) -> float:
        return self.rp - self.ws


def assess(
    instance: Instance,
    sigma: float,
    n_train: int = 8,
    n_test: int = 20,
    time_limit_s: float = 30.0,
    seed: int = 0,
    rho: float = 0.0,
) -> StochasticReport:
    """Treina (SAA) com `n_train` cenários e avalia tudo em `n_test` cenários novos.

    As duas políticas passam pelo mesmo avaliador, com cache; o VSS vem com intervalo de confiança
    da diferença PAREADA por cenário. Se o intervalo contém zero, o benefício não é distinguível
    de zero neste experimento.
    """
    train = lognormal_scenarios(instance, n_train, sigma, seed, rho)
    test = lognormal_scenarios(instance, n_test, sigma, seed + 1, rho)

    y_stoch = solve_two_stage(instance, train, time_limit_s)
    det = MilpSolver().solve(instance, time_limit_s).solution
    y_det = np.zeros(instance.n_facilities, dtype=bool)
    y_det[det.open_facilities()] = True

    evaluator = PolicyEvaluator(instance, test, time_limit_s)
    rp_s = evaluator.per_scenario(y_stoch)
    eev_s = evaluator.per_scenario(y_det)
    ws_s = []
    gaps = [evaluator.max_gap()]
    for d in test:
        frozen = replace(instance, demand=d)
        result = MilpSolver().solve(frozen, time_limit_s)
        ws_s.append(evaluate(frozen, result.solution).total_cost)
        if result.lower_bound is not None and ws_s[-1] > 0:
            gaps.append(max(0.0, (ws_s[-1] - result.lower_bound) / ws_s[-1]))
    diff = eev_s - rp_s
    half = 1.96 * float(diff.std(ddof=1)) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    return StochasticReport(
        rp=float(rp_s.mean()),
        eev=float(eev_s.mean()),
        ws=float(np.mean(ws_s)),
        n_open_stochastic=int(y_stoch.sum()),
        n_open_deterministic=int(y_det.sum()),
        same_policy=bool(np.array_equal(y_stoch, y_det)),
        max_solver_gap=float(max(gaps)),
        vss_ci95=(float(diff.mean() - half), float(diff.mean() + half)),
        per_scenario_vss=diff,
    )
