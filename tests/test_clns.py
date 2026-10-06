"""CLNS: a coordenação global mantém viabilidade e o custo fixo é cobrado uma vez."""

from __future__ import annotations

import numpy as np
import pytest

from alocacao_capacitada.pesquisa import clns as C
from alocacao_capacitada.pesquisa.decomposicao import decomposicao_ingenua, reotimizar
from alocacao_capacitada.pesquisa.enumeracao import otimo_enumerado
from alocacao_capacitada.pesquisa.exato import relaxacao_linear
from alocacao_capacitada.pesquisa.geradores import Config, gerar
from alocacao_capacitada.pesquisa.problema import validar


@pytest.fixture(scope="module")
def prob():  # type: ignore[no-untyped-def]
    return gerar(Config("clusters", 12, 60, 1.5), 3)[0]


def test_inicial_pl_e_viavel(prob) -> None:  # type: ignore[no-untyped-def]
    a = C.inicial_pl(prob, relaxacao_linear(prob).x)
    assert validar(prob, a).valida


def test_reotimizar_nunca_piora_e_respeita_capacidade(prob) -> None:  # type: ignore[no-untyped-def]
    a = C.inicial_pl(prob, relaxacao_linear(prob).x)
    base = validar(prob, a).objetivo
    rng = np.random.default_rng(0)
    for _ in range(5):
        livres = rng.choice(prob.n, 15, replace=False)
        novo = reotimizar(prob, a, livres, 5.0)
        assert novo is not None
        v = validar(prob, novo)
        assert v.valida and v.objetivo <= base + 1e-6  # a solução atual é sempre a dica
        fixos = np.setdiff1d(np.arange(prob.n), livres)
        assert (novo[fixos] == a[fixos]).all()


def test_reotimizar_tudo_coincide_com_otimo_em_micro() -> None:
    p, _ = gerar(Config("uniforme", 4, 8, 1.4), 1)
    z, _ = otimo_enumerado(p)
    a = C.inicial_pl(p, relaxacao_linear(p).x)
    novo = reotimizar(p, a, np.arange(p.n), 10.0, k_cand=p.m)
    assert validar(p, novo).objetivo == pytest.approx(z, rel=1e-7)


@pytest.mark.parametrize("nome", ["rotacao", "aleatorio", "alns", "dual"])
def test_clns_valido_e_monotono(prob, nome) -> None:  # type: ignore[no-untyped-def]
    s = C.clns(prob, 6.0, C.SELETORES[nome](), seed=0)
    assert s.valida
    objs = [o for _, o in s.trajetoria]
    assert all(b < a for a, b in zip(objs, objs[1:]))  # só melhoria estrita
    assert s.objetivo == pytest.approx(objs[-1])
    assert s.t_total <= 6.0 + 1.0  # prazo cooperativo


def test_ingenua_registra_sobrecarga_quando_inviavel(prob) -> None:  # type: ignore[no-untyped-def]
    s = decomposicao_ingenua(prob, 5.0, tamanho=15)
    if not s.viavel_uniao:
        assert s.sobrecarga > 0
    if s.atribuicao_reparada is not None:
        assert validar(prob, s.atribuicao_reparada).valida


def test_vizinhos_equivalem_distancia_original(prob):
    est = C.Estado(prob, 15, 0, None)
    esperado = np.abs(est.unit[:, :, None] - est.unit[:, None, :]).sum(0)
    assert np.array_equal(est.ordem_cli, np.argsort(esperado, axis=1))


def test_orcamento_zero_nao_constroi_modelo(prob, monkeypatch):
    from alocacao_capacitada.pesquisa import decomposicao as decomp

    a = decomp.construtivo(prob)
    def proibido():
        pytest.fail("SCIP não deve iniciar sem orçamento")
    monkeypatch.setattr(decomp, "Model", proibido)
    assert np.array_equal(decomp.reotimizar(prob, a, np.arange(prob.n), 0), a)


def test_montagem_consumiu_orcamento_preserva_incumbente(prob, monkeypatch):
    from alocacao_capacitada.pesquisa import decomposicao as decomp

    a = decomp.construtivo(prob)
    relogio = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(decomp.time, "perf_counter", lambda: next(relogio))
    assert np.array_equal(decomp.reotimizar(prob, a, np.arange(prob.n), 1.0), a)


@pytest.mark.parametrize("livres", [np.array([0, 0]), np.array([-1]),
                                   np.array([0.5]), np.array([999])])
def test_reotimizar_rejeita_indices_invalidos(prob, livres):
    from alocacao_capacitada.pesquisa.decomposicao import construtivo

    with pytest.raises(ValueError, match="índices"):
        reotimizar(prob, construtivo(prob), livres, 1)


def test_clusters_custos_zero():
    from alocacao_capacitada.pesquisa.decomposicao import grupos
    from alocacao_capacitada.pesquisa.problema import Problema

    p = Problema("zero", "teste", np.zeros(2), np.ones(2) * 4,
                 np.ones(4), np.zeros((2, 4)))
    gs = grupos(p, 1)
    assert len(gs) == 1
    assert np.array_equal(gs[0], np.arange(4))
    est = C.Estado(p, 1, 0, None)
    a = np.zeros(4, dtype=np.int64)
    for candidato in est.candidatos(a, np.random.default_rng(0)):
        assert np.isfinite(est.atributos(candidato, a, 0, est.regret(a))).all()


def test_chave_subproblema_identifica_o_modelo(prob) -> None:  # type: ignore[no-untyped-def]
    a = C.inicial_pl(prob, relaxacao_linear(prob).x)
    livres = np.array([3, 1, 7])
    k = C.chave_subproblema(prob, a, livres)
    assert k == C.chave_subproblema(prob, a, livres[::-1])  # ordem dos livres não importa
    assert k != C.chave_subproblema(prob, a, np.array([3, 1, 8]))
    # trocar dois clientes FIXOS de mesmo centro entre si não muda carga fixa => mesma chave;
    # mover um fixo para outro centro muda a carga => outra chave
    b = a.copy()
    fixo = next(j for j in range(prob.n) if j not in livres)
    b[fixo] = (a[fixo] + 1) % prob.m
    assert k != C.chave_subproblema(prob, b, livres)


def test_memoria_nao_repete_e_mantem_validade(prob) -> None:  # type: ignore[no-untyped-def]
    sem = C.clns(prob, 6.0, C.SELETORES["rotacao"](), seed=0)
    com = C.clns(prob, 6.0, C.SELETORES["rotacao"](), seed=0, memoria=True)
    assert sem.valida and com.valida
    assert sem.pulados == 0
    assert 0 <= sem.repetidos <= sem.iteracoes
    objs = [o for _, o in com.trajetoria]
    assert all(b < a for a, b in zip(objs, objs[1:]))


def test_kernel_search_valido_e_no_prazo(prob) -> None:  # type: ignore[no-untyped-def]
    from alocacao_capacitada.pesquisa.hibrido import kernel_search

    lp = relaxacao_linear(prob)
    s = kernel_search(prob, lp, lp.tempo, 8.0, seed=0)
    assert s.valida and validar(prob, s.atribuicao).objetivo == pytest.approx(s.objetivo)
    assert s.objetivo >= lp.bound - 1e-6
    assert s.t_total <= 8.0 + 1.0
    objs = [o for _, o in s.trajetoria]
    assert min(objs) == pytest.approx(s.objetivo)


def test_grupos_aleatorios_tem_os_mesmos_tamanhos(prob) -> None:  # type: ignore[no-untyped-def]
    from alocacao_capacitada.pesquisa.decomposicao import grupos

    perfil = grupos(prob, 15, 0)
    alea = grupos(prob, 15, 0, "aleatorio")
    assert sorted(map(len, perfil)) == sorted(map(len, alea))
    assert np.array_equal(np.sort(np.concatenate(alea)), np.arange(prob.n))
    assert any(not np.array_equal(a, b) for a, b in zip(perfil, alea))


def test_reotimizar_informa_status(prob) -> None:  # type: ignore[no-untyped-def]
    a = C.inicial_pl(prob, relaxacao_linear(prob).x)
    info: dict = {}
    novo = reotimizar(prob, a, np.arange(10), 5.0, info=info)
    assert novo is not None and info["status"] in ("optimal", "timelimit")
    assert info["dual"] <= info["primal"] + 1e-6


def test_opcoes_de_metodo() -> None:
    from alocacao_capacitada.pesquisa.exp_clns import _opcoes

    assert _opcoes("clns:rotacao") == ("clns:rotacao", {"memoria": False, "agrupamento": "perfil", "k_cand": 10})
    assert _opcoes("hibridopl:rotacao+alea+k30+mem")[1] == {"memoria": True, "agrupamento": "aleatorio", "k_cand": 30}
    with pytest.raises(ValueError):
        _opcoes("clns:rotacao+xyz")


def test_clns_com_grupos_aleatorios_e_k_ampliado_e_valido(prob) -> None:  # type: ignore[no-untyped-def]
    s = C.clns(prob, 5.0, C.SELETORES["rotacao"](), seed=0, agrupamento="aleatorio", k_cand=20)
    assert s.valida
