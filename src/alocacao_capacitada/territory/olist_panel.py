"""Painel município × mês de pedidos do Olist, com relatório de cobertura.

Cada cliente do Olist traz um prefixo de CEP de 5 dígitos; a geolocalização do Olist dá a
média de coordenadas desse prefixo. Localizamos o ponto no polígono do município (IBGE). Quando
o prefixo não tem geolocalização válida ou cai fora de todos os polígonos, tentamos casar o nome
da cidade + UF informados pelo cliente. O que sobrar é reportado, nunca descartado em silêncio.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alocacao_capacitada.territory.geometry import locate
from alocacao_capacitada.territory.ibge import normalize_name

_LAT = (-34.0, 5.5)
_LON = (-74.0, -34.0)


def build_panel(
    raw_dir: Path, polygons: pd.DataFrame, municipalities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    delivered = orders[orders["order_status"] == "delivered"]
    customers = pd.read_csv(
        raw_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    )
    merged = delivered.merge(customers, on="customer_id", how="left")
    total = len(delivered)

    geo = pd.read_csv(
        raw_dir / "olist_geolocation_dataset.csv",
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    )
    geo = geo[geo["geolocation_lat"].between(*_LAT) & geo["geolocation_lng"].between(*_LON)]
    centroids = geo.groupby("geolocation_zip_code_prefix").mean().rename_axis("zip5").reset_index()
    centroids["cod_poligono"] = locate(
        polygons, centroids["geolocation_lat"].to_numpy(), centroids["geolocation_lng"].to_numpy()
    )

    merged = merged.merge(
        centroids[["zip5", "cod_poligono"]],
        left_on="customer_zip_code_prefix",
        right_on="zip5",
        how="left",
    )
    by_polygon = merged["cod_poligono"].fillna("").ne("")

    names = municipalities.assign(chave=municipalities["nome"].map(normalize_name))
    lookup = {(r.chave, r.uf): r.cod for r in names.itertuples()}
    fallback = [
        lookup.get((normalize_name(str(c)), str(u)), "")
        for c, u in zip(merged["customer_city"], merged["customer_state"], strict=True)
    ]
    merged["cod"] = merged["cod_poligono"].fillna("")
    use_name = ~by_polygon
    merged.loc[use_name, "cod"] = pd.Series(fallback, index=merged.index)[use_name]
    by_name = use_name & merged["cod"].ne("")
    unmapped = merged["cod"].eq("")

    coverage = pd.DataFrame(
        [
            ("pedidos entregues", total, 100.0),
            ("localizados por polígono", int(by_polygon.sum()), 100 * by_polygon.mean()),
            ("localizados por nome da cidade", int(by_name.sum()), 100 * by_name.mean()),
            ("sem município", int(unmapped.sum()), 100 * unmapped.mean()),
        ],
        columns=["etapa", "pedidos", "pct"],
    )
    merged["mes"] = merged["order_purchase_timestamp"].dt.to_period("M").astype(str)
    panel = merged[~unmapped].groupby(["cod", "mes"]).size().rename("pedidos").reset_index()
    return panel, coverage


def seller_hubs(raw_dir: Path, polygons: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Municípios onde mais se vende (lado da OFERTA): usa vendedores e itens, não os clientes.

    Serve de covariável independente do alvo de demanda: a distância ao polo vendedor mede
    quão longe um município está de onde o marketplace despacha.
    """
    sellers = pd.read_csv(
        raw_dir / "olist_sellers_dataset.csv", usecols=["seller_id", "seller_zip_code_prefix"]
    )
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv", usecols=["order_id", "seller_id"]
    )
    geo = pd.read_csv(
        raw_dir / "olist_geolocation_dataset.csv",
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    )
    geo = geo[geo["geolocation_lat"].between(*_LAT) & geo["geolocation_lng"].between(*_LON)]
    centroids = geo.groupby("geolocation_zip_code_prefix").mean().rename_axis("zip5").reset_index()
    centroids["cod"] = locate(
        polygons, centroids["geolocation_lat"].to_numpy(), centroids["geolocation_lng"].to_numpy()
    )
    sold = items.merge(sellers, on="seller_id").merge(
        centroids, left_on="seller_zip_code_prefix", right_on="zip5", how="left"
    )
    sold = sold[sold["cod"].fillna("") != ""]
    top = sold.groupby("cod").size().rename("itens_vendidos").nlargest(k).reset_index()
    cent = polygons[["cod", "lat", "lon"]]
    return top.merge(cent, on="cod", how="left")
