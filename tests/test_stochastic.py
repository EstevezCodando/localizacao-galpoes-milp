import numpy as np
import pytest

from alocacao_capacitada.analysis.stochastic import (
    assess,
    lognormal_scenarios,
    recourse_cost,
    solve_two_stage,
)
from alocacao_capacitada.domain.instance import Instance


@pytest.fixture
def small() -> Instance:
    rng = np.random.default_rng(3)
    m, n = 4, 8
    demand = rng.integers(2, 8, n).astype(float)
    return Instance(
        facility_ids=tuple(f"F{i}" for i in range(m)),
        demand_ids=tuple(f"N{j}" for j in range(n)),
        demand=demand,
        capacity=np.full(m, demand.sum() / 2.5),
        fixed_cost=np.full(m, 40.0),
        unit_cost=rng.uniform(1, 15, (m, n)),
        unserved_penalty=300.0,
    )


def test_cenarios_preservam_a_media(small: Instance) -> None:
    sc = lognormal_scenarios(small, 4000, sigma=0.4, seed=0)
    assert sc.mean(axis=0) == pytest.approx(small.demand, rel=0.08)


def test_sem_incerteza_vss_e_evpi_sao_nulos(small: Instance) -> None:
    rep = assess(small, sigma=1e-9, n_train=2, n_test=2, time_limit_s=10)
    assert rep.vss == pytest.approx(0.0, abs=1e-3)
    assert rep.evpi == pytest.approx(0.0, abs=1e-3)


def test_informacao_perfeita_nunca_e_pior(small: Instance) -> None:
    rep = assess(small, sigma=0.6, n_train=4, n_test=6, time_limit_s=10)
    assert rep.ws <= rep.rp + 1e-6  # WS <= RP: ter informação nunca piora


def test_recurso_com_centros_fechados_so_penaliza(small: Instance) -> None:
    none_open = np.zeros(small.n_facilities, dtype=bool)
    cost = recourse_cost(small, none_open, small.demand, 10)
    assert cost == pytest.approx(small.demand.sum() * small.unserved_penalty)


def test_dois_estagios_retorna_mascara(small: Instance) -> None:
    sc = lognormal_scenarios(small, 3, 0.3, 1)
    y = solve_two_stage(small, sc, 10)
    assert y.shape == (small.n_facilities,) and y.dtype == bool


def test_politicas_identicas_tem_custos_identicos_e_vss_exatamente_zero(small: Instance) -> None:
    from alocacao_capacitada.analysis.stochastic import PolicyEvaluator

    sc = lognormal_scenarios(small, 4, 0.5, 0)
    evaluator = PolicyEvaluator(small, sc, 10)
    mask = np.array([True, True, False, False])
    assert np.array_equal(evaluator.per_scenario(mask), evaluator.per_scenario(mask.copy()))


def test_choque_comum_aumenta_a_correlacao_entre_regioes(small: Instance) -> None:
    indep = lognormal_scenarios(small, 3000, 0.5, 0, rho=0.0)
    comum = lognormal_scenarios(small, 3000, 0.5, 0, rho=0.8)
    corr_i = np.corrcoef(indep[:, 0], indep[:, 1])[0, 1]
    corr_c = np.corrcoef(comum[:, 0], comum[:, 1])[0, 1]
    assert abs(corr_i) < 0.1 and corr_c > 0.5
    with pytest.raises(ValueError):
        lognormal_scenarios(small, 2, 0.5, 0, rho=1.5)


def test_relatorio_traz_ic_pareado_e_politica_igual(small: Instance) -> None:
    rep = assess(small, sigma=0.4, n_train=3, n_test=6, time_limit_s=10)
    lo, hi = rep.vss_ci95
    assert lo <= rep.vss <= hi
    if rep.same_policy:
        assert rep.vss == 0.0 and lo == 0.0 and hi == 0.0
