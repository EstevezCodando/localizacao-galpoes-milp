from pathlib import Path

import pytest

from alocacao_capacitada.domain.evaluation import evaluate
from alocacao_capacitada.solvers.milp import MilpSolver
from alocacao_capacitada.validation.holmberg import load_holmberg_instance, load_holmberg_optima

BRONZE = Path(__file__).parents[1] / "data" / "bronze" / "holmberg"
OPTIMA = load_holmberg_optima(BRONZE / "Holmberg_Solution_Values.txt").set_index("name")


def test_dimensoes_batem_com_a_tabela_publicada() -> None:
    for name in ("p1", "p13", "p25", "p41", "p56"):
        inst = load_holmberg_instance(BRONZE / "instances" / name)
        row = OPTIMA.loc[name]
        assert (inst.n_facilities, inst.n_demand) == (row["m"], row["n"])


@pytest.mark.parametrize("name", ["p1", "p2", "p3", "p13"])
def test_milp_reproduz_otimo_publicado_de_fonte_unica(name: str) -> None:
    inst = load_holmberg_instance(BRONZE / "instances" / name)
    result = MilpSolver().solve(inst, 60)
    ev = evaluate(inst, result.solution)
    assert result.status == "OTIMO" and ev.feasible and ev.unserved_demand == 0
    assert ev.total_cost == pytest.approx(OPTIMA.loc[name, "optimal"], rel=1e-9, abs=1e-6)


def test_lixo_apos_os_dados_e_ignorado_e_falta_de_dados_falha(tmp_path: Path) -> None:
    good = tmp_path / "ok"
    good.write_text("1 2\n10 5\n3 4\n1 2\nFrom: alguem@exemplo.org Tue Oct 24\n", encoding="utf-8")
    inst = load_holmberg_instance(good)
    assert (inst.n_facilities, inst.n_demand) == (1, 2)
    short = tmp_path / "curto"
    short.write_text("1 2\n10 5\n3 4\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_holmberg_instance(short)
