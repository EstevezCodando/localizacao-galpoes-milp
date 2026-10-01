import pandas as pd

from alocacao_capacitada.validation.classify import classify_methods


def test_uma_unidade_acima_nao_conta_como_igualdade_mas_conta_como_proximo() -> None:
    t = pd.DataFrame(
        {
            "otimo_publicado": [10486.0, 100.0, 100.0],
            "lns_custo": [10487.0, 100.0, 99.0],
            "lns_viavel": [True, True, True],
            "lns_status": ["LIMITE_TEMPO"] * 3,
        }
    )
    out = classify_methods(t, ("lns",))
    assert list(out["lns_coincide_ref"]) == [False, True, False]
    assert list(out["lns_dentro_0_01pct"]) == [True, True, False]  # p20: 1 em 10486 = 0,0095%
    assert list(out["lns_abaixo_da_referencia"]) == [False, False, True]
    assert not out["lns_certificado"].any()


def test_certificado_exige_status_otimo_e_viabilidade() -> None:
    t = pd.DataFrame(
        {
            "otimo_publicado": [10.0, 10.0],
            "milp_custo": [10.0, 10.0],
            "milp_viavel": [True, False],
            "milp_status": ["OTIMO", "OTIMO"],
        }
    )
    out = classify_methods(t, ("milp",))
    assert list(out["milp_certificado"]) == [True, False]
    assert list(out["milp_coincide_ref"]) == [True, False]
