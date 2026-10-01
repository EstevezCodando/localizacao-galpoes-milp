from pathlib import Path

import pandas as pd
import pytest

from alocacao_capacitada.lake.rent import parse_rent_table, rent_by_state, rent_lookup

BRONZE = (
    Path(__file__).parents[1] / "data" / "bronze" / "aluguel" / "galpaodasmaquinas_fev2026.html"
)


@pytest.fixture(scope="module")
def cities() -> pd.DataFrame:
    return parse_rent_table(BRONZE.read_bytes())


def test_parse_reproduz_a_tabela_publicada(cities: pd.DataFrame) -> None:
    assert len(cities) == 18
    sp = cities[cities["cidade"] == "São Paulo"].iloc[0]
    assert (sp["uf"], sp["pequeno_ate_1000"], sp["medio_1001_3000"]) == ("SP", 41.51, 36.05)
    assert cities["grande_3001_5000"].min() == pytest.approx(16.61)  # Blumenau


def test_media_por_estado_e_imputacao(cities: pd.DataFrame) -> None:
    by_state = rent_by_state(cities).set_index("uf")
    assert by_state.loc["MG", "n_cidades"] == 3
    lookup = rent_lookup(cities, ["SP", "BA"])
    assert lookup["SP"][1] is False
    assert lookup["BA"] == (pytest.approx(cities["medio_1001_3000"].mean()), True)


def test_tabela_ausente_falha_alto() -> None:
    with pytest.raises(ValueError):
        parse_rent_table(b"<html><table><tr><td>x</td></tr></table></html>")
