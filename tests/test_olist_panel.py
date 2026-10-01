from pathlib import Path

import pandas as pd
from shapely.geometry import box

from alocacao_capacitada.territory.olist_panel import build_panel


def test_painel_usa_poligono_depois_nome_e_reporta_o_que_sobra(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "order_status": ["delivered"] * 3 + ["canceled"],
            "order_purchase_timestamp": ["2017-01-10", "2017-01-12", "2017-02-01", "2017-02-02"],
        }
    ).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "customer_zip_code_prefix": [10000, 20000, 30000, 10000],
            "customer_city": ["Aqui", "Lugar Novo", "Nada", "Aqui"],
            "customer_state": ["SP", "RJ", "MG", "SP"],
        }
    ).to_csv(tmp_path / "olist_customers_dataset.csv", index=False)
    pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [10000],
            "geolocation_lat": [-23.5],
            "geolocation_lng": [-46.5],
        }
    ).to_csv(tmp_path / "olist_geolocation_dataset.csv", index=False)

    polygons = pd.DataFrame({"cod": ["3550308"], "geometry": [box(-47, -24, -46, -23)]})
    municipalities = pd.DataFrame(
        {"cod": ["3550308", "3304557"], "nome": ["São Paulo", "Lugar Novo"], "uf": ["SP", "RJ"]}
    )
    panel, coverage = build_panel(tmp_path, polygons, municipalities)

    assert dict(zip(coverage["etapa"], coverage["pedidos"], strict=True)) == {
        "pedidos entregues": 3,
        "localizados por polígono": 1,
        "localizados por nome da cidade": 1,
        "sem município": 1,
    }
    got = {(r.cod, r.mes): r.pedidos for r in panel.itertuples()}
    assert got == {("3550308", "2017-01"): 1, ("3304557", "2017-01"): 1}
