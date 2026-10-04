"""Regressões de contratos e validade experimental da auditoria de outubro/2026."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alocacao_capacitada.pesquisa import hibrido as H
from alocacao_capacitada.pesquisa.analise import integral_primal, preparar, tempo_alvo
from alocacao_capacitada.pesquisa.atributos import grafo
from alocacao_capacitada.pesquisa.etapa1 import sementes_metodo
from alocacao_capacitada.pesquisa.exato import relaxacao_linear, resolver
from alocacao_capacitada.pesquisa.lrp import cvrp, validar_rotas
from alocacao_capacitada.pesquisa.problema import Problema
from alocacao_capacitada.pesquisa.registro import Registro


def problema():
    return Problema("t", "t", fixed=[0, 0], capacity=[0, 3], demand=[1, 1],
                    cost=np.zeros((2, 2)))


def test_integral_preserva_incumbente_e_censura_prazo():
    assert integral_primal([(1., 100.), (2., 150.)], 100., 10.) == pytest.approx(.1)
    assert integral_primal([(11., 100.)], 100., 10.) == 1.
    assert integral_primal([(1., 0.)], 0., 10.) == pytest.approx(.1)
    assert np.isinf(tempo_alvo([(11., 100.)], 100., 10.))


def test_analise_nao_premia_objetivo_apos_prazo():
    df = pd.DataFrame([dict(instancia="t", metodo="m", valida=True, objetivo=100.,
                            rotulo_melhor=100., trajetoria=json.dumps([(1., 150.), (11., 100.)]))])
    p = preparar(df, 10.)
    assert p.desvio.iloc[0] == pytest.approx(.5)
    assert np.isinf(p["t_alvo_0.01"].iloc[0])


def test_atributos_custos_e_capacidade_zero():
    g = grafo(problema())
    assert all(np.isfinite(a).all() for a in (g.x_fac, g.x_cli, g.x_ed))
    assert g.x_fac[0, 0] == 0  # capacidade zero não se transforma em capacidade positiva


def test_budget_zero_nao_inicia_solver_e_preserva_dica(monkeypatch):
    import alocacao_capacitada.pesquisa.exato as E

    def proibido():
        pytest.fail("Solver não deve ser construído com orçamento esgotado")

    monkeypatch.setattr(E, "Model", proibido)
    r = resolver(problema(), 0., dica=np.array([1, 1]))
    assert r.status == "budget_exhausted" and r.valida and r.objetivo == 0
    with pytest.raises(TimeoutError):
        relaxacao_linear(problema(), tempo_s=0.)


def test_completo_nao_soma_montagem_novamente(monkeypatch):
    r = SimpleNamespace(objetivo=10., valida=True, t_total=5., status="optimal",
                        trajetoria=[(3., 10.)], t_montagem=2., nos=0, atribuicao=np.array([1, 1]))
    monkeypatch.setattr(H, "resolver", lambda *a, **kw: r)
    saida, _ = H.completo(problema(), 10.)
    assert saida.trajetoria == [(3., 10.)]


def test_topk_desloca_apenas_overhead_anterior(monkeypatch):
    clock = iter([0., 2., 7.])
    monkeypatch.setattr(H.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(H, "reparar", lambda *a: np.array([1]))
    r = SimpleNamespace(objetivo=10., valida=True, trajetoria=[(3., 10.)], t_montagem=1.,
                        nos=0, atribuicao=np.array([1, 1]))
    calls = []

    def fake(prob, budget, **kwargs):
        calls.append((budget, kwargs["seed"]))
        return r

    monkeypatch.setattr(H, "resolver", fake)
    s = H.topk(problema(), np.array([0., 1.]), 1., .5, 10., seed=2)
    assert calls == [(7., 2)]
    assert s.trajetoria == [(6., 10.)]


def test_rotas_rejeitam_truncamento_e_sobrecarga():
    dep, cli, dem = np.zeros(2), np.array([[1., 0.], [2., 0.]]), np.array([1.9, 1.9])
    with pytest.raises(ValueError, match="inteiros"):
        cvrp(dep, cli, dem, 3., 100., .1)
    with pytest.raises(ValueError, match="Capacidade real"):
        validar_rotas(cli, dep, dem, 3., 100., [[0, 1]])
    with pytest.raises(ValueError, match="exatamente uma"):
        validar_rotas(cli, dep, dem, 3., 100., [[0], [0]])


def test_cvrp_retorna_rotas_validadas():
    dep, cli, dem = np.zeros(2), np.array([[1., 0.], [2., 0.]]), np.array([2., 2.])
    rotas = []
    custo, n = cvrp(dep, cli, dem, 3., 100., .1, rotas_out=rotas)
    assert n == 2 and sorted(j for r in rotas for j in r) == [0, 1]
    assert custo == validar_rotas(cli, dep, dem, 3., 100., rotas)


def test_sementes_nao_repetem_modelos_inexistentes():
    assert sementes_metodo("topk:gnn", (0, 1, 2)) == (0, 1, 2)
    assert sementes_metodo("topk:lgbm", (0, 1, 2)) == (0,)
    assert sementes_metodo("completo", (0, 1, 2)) == (-1,)
    with pytest.raises(ValueError):
        sementes_metodo("topk:gnn", (3,))


def test_checkpoint_isola_config_e_preserva_sementes(tmp_path):
    reg = Registro(tmp_path, {"modelo": "a"})
    for seed in (0, 1):
        reg.gravar(dict(instancia="t", metodo="gnn", seed_treino=seed, seed_solver=0))
    run_id = reg.run_id
    assert len(reg.exportar()) == 2
    reg.close()
    reg = Registro(tmp_path, {"modelo": "a"})
    assert len(reg.feitos()) == 2
    reg.close()
    reg = Registro(tmp_path, {"modelo": "b"})
    assert reg.run_id != run_id and not reg.feitos()
    reg.close()


def test_fases_nao_vazam_entre_execucoes():
    from alocacao_capacitada.pesquisa import tempos

    tempos.iniciar()
    tempos.registrar("t_solver", 2.)
    tempos.registrar("t_solver", 3.)
    assert tempos.finalizar() == {"t_solver": 5.}
    tempos.iniciar()
    assert tempos.finalizar() == {}
