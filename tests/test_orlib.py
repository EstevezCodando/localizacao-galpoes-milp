"""Validação contra os ótimos publicados da OR-Library (dados em data/bronze/orlib)."""

from pathlib import Path

import pytest

from alocacao_capacitada.validation.orlib import (
    load_orlib_instance,
    load_published_optima,
    solve_splittable,
)

BRONZE = Path(__file__).parents[1] / "data" / "bronze" / "orlib"
OPTIMA = load_published_optima(BRONZE / "capopt.txt")


@pytest.mark.parametrize("name", ["cap41", "cap42", "cap43", "cap44"])
def test_modelo_reproduz_otimo_publicado(name: str) -> None:
    instance = load_orlib_instance(BRONZE / f"{name}.txt")
    assert (instance.n_facilities, instance.n_demand) == (16, 50)
    value, status = solve_splittable(instance, time_limit_s=60)
    assert status == "OTIMO"
    assert value == pytest.approx(OPTIMA[name], rel=1e-6)


def test_loader_rejeita_arquivo_truncado(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("2 1\n10 5\n10 5\n3\n1 2\n1 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_orlib_instance(bad)
