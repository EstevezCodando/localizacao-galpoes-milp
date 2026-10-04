"""Portão G0 (doc 12): modelo, validador e bounds corretos antes de qualquer experimento."""

from __future__ import annotations

import numpy as np
import pytest

from alocacao_capacitada.pesquisa.enumeracao import otimo_enumerado
from alocacao_capacitada.pesquisa.exato import relaxacao_linear, resolver
from alocacao_capacitada.pesquisa.geradores import Config, gerar, viavel_fonte_unica
from alocacao_capacitada.pesquisa.problema import Problema, gap_certificado, validar


@pytest.mark.parametrize("familia", ["uniforme", "clusters", "corredor"])
@pytest.mark.parametrize("seed", range(4))
def test_scip_concorda_com_enumeracao(familia: str, seed: int) -> None:
    prob, _ = gerar(Config(familia, m=4, n=8, razao=1.4), seed)
    z_enum, a_enum = otimo_enumerado(prob)
    res = resolver(prob, tempo_s=30)
    assert res.status == "optimal" and res.valida
    assert res.objetivo == pytest.approx(z_enum, rel=1e-7)
    assert validar(prob, a_enum).objetivo == pytest.approx(z_enum)
    lp = relaxacao_linear(prob)
    assert lp.bound <= z_enum * (1 + 1e-9)
    assert res.bound is not None and res.bound <= z_enum * (1 + 1e-7) + 1e-6


def test_assimetrico_e_capacidade_apertada() -> None:
    prob, _ = gerar(Config("uniforme", m=3, n=9, razao=1.1, assimetrico=True), 7)
    z_enum, _ = otimo_enumerado(prob)
    assert resolver(prob, 30).objetivo == pytest.approx(z_enum, rel=1e-7)


def test_validador_detecta_violacoes() -> None:
    prob = Problema("t", "t", fixed=[10, 10], capacity=[5, 5], demand=[3, 3, 3],
                    cost=np.ones((2, 3)))
    assert not validar(prob, np.array([0, 0, 1])).valida  # 6 > 5
    assert validar(prob, np.array([0, 1, 2])).erros  # índice inválido
    ok = validar(prob, np.array([0, 1, 1]))
    assert not ok.valida  # 6 > 5 no centro 1
    prob2 = Problema("t", "t", fixed=[10, 10], capacity=[6, 6], demand=[3, 3, 3],
                     cost=np.ones((2, 3)))
    v = validar(prob2, np.array([0, 0, 1]))
    assert v.valida and v.objetivo == pytest.approx(23.0) and v.n_abertos == 2


def test_poda_marca_bound_reduzido_e_mapeia_indices() -> None:
    prob, _ = gerar(Config("clusters", m=6, n=8, razao=2.0), 1)
    z, _ = otimo_enumerado(prob)
    res = resolver(prob, 30, centros=np.array([5, 0, 3, 2]))
    assert res.bound_escopo == "reduzido"
    assert res.valida and res.objetivo >= z - 1e-6  # domínio reduzido nunca melhora o ótimo
    assert set(np.unique(res.atribuicao)) <= {0, 2, 3, 5}


def test_warm_start_e_aceito() -> None:
    prob, _ = gerar(Config("uniforme", m=10, n=30, razao=1.5), 3)
    base = resolver(prob, 30)
    assert base.atribuicao is not None
    res = resolver(prob, 30, dica=base.atribuicao)
    assert res.trajetoria and res.trajetoria[0][1] == pytest.approx(base.objetivo, rel=1e-6)


def test_gap_certificado_rejeita_bound_invalido() -> None:
    assert gap_certificado(100.0, 90.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        gap_certificado(100.0, 120.0)


def test_viabilidade_fonte_unica() -> None:
    # capacidade total sobra, mas nenhum centro comporta o cliente grande
    assert viavel_fonte_unica(np.array([10.0, 1.0]), np.array([6.0, 6.0])) is False
    # FFD falha mas existe solução (3+3 | 2+2+2): o MILP decide
    assert viavel_fonte_unica(np.array([3, 3, 2, 2, 2.0]), np.array([6.0, 6.0])) is True
