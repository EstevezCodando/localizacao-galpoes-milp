from pathlib import Path

import numpy as np
import pytest

from alocacao_capacitada.lake.freight import FreightModel, load_freight_model
from alocacao_capacitada.lake.silver import parse_antt_coefficients

ROOT = Path(__file__).parents[1]
REF = ROOT / "data" / "reference" / "antt_tabela_a.csv"
BRONZE = ROOT / "data" / "bronze" / "antt" / "res_00006084.html"


def test_piso_formula_ida_e_volta() -> None:
    model = FreightModel(ccd_rs_km=4.0, cc_rs=400.0, orders_per_vehicle=1.0, empty_return=True)
    assert model.cost_per_order(np.array([100.0]))[0] == pytest.approx(100 * 4 * 1.92 + 400)


def test_piso_sem_retorno_vazio_e_diluicao_por_pedido() -> None:
    model = FreightModel(4.0, 400.0, orders_per_vehicle=10.0, empty_return=False)
    assert model.cost_per_order(np.array([100.0]))[0] == pytest.approx((400 + 400) / 10)


def test_referencia_oficial_res_6084() -> None:
    model = load_freight_model(REF, axles=2, orders_per_vehicle=50)
    assert model.ccd_rs_km == pytest.approx(3.9826)
    assert model.cc_rs == pytest.approx(451.84)


def test_eixos_inexistentes() -> None:
    with pytest.raises(ValueError):
        load_freight_model(REF, axles=8, orders_per_vehicle=50)


def test_orders_per_vehicle_invalido() -> None:
    with pytest.raises(ValueError):
        FreightModel(1.0, 1.0, orders_per_vehicle=0)


def test_silver_le_virgula_decimal_e_milhar_ptbr() -> None:
    """Regressão: '4,0144' virava 40144 e '1.016,29' não era lido."""
    table = parse_antt_coefficients(BRONZE.read_bytes())
    row = table[(table.tabela == "A") & (table.tipo_carga == "carga_geral") & (table.eixos == 9)]
    assert row["ccd_rs_km"].iloc[0] == pytest.approx(9.2027)
    assert table["cc_rs"].max() > 1000  # valores com separador de milhar foram lidos
    assert table["ccd_rs_km"].max() < 50
