"""Leitura e agregação do dataset público Olist (CC BY-NC-SA 4.0) em nós geográficos.

Agregamos por prefixo de 3 dígitos do CEP (~ micro-região): reduz ruído e
mantém a instância tratável. Coordenadas = média dos pontos do prefixo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Caixa aproximada do território brasileiro: descarta pontos de geolocalização inválidos.
_LAT_RANGE = (-34.0, 5.5)
_LON_RANGE = (-74.0, -34.0)


@dataclass(frozen=True)
class GeoNodes:
    """Nós geográficos: índice = prefixo de 3 dígitos; colunas lat, lon, orders."""

    demand: pd.DataFrame  # regiões de clientes
    supply: pd.DataFrame  # regiões de vendedores (candidatas a centro)


def _prefix3(series: pd.Series) -> pd.Series:
    return series.astype(str).str.zfill(5).str[:3]


def _region_centroids(raw_dir: Path) -> pd.DataFrame:
    geo = pd.read_csv(
        raw_dir / "olist_geolocation_dataset.csv",
        usecols=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_state",
            "geolocation_city",
        ],
    )
    geo = geo[
        geo["geolocation_lat"].between(*_LAT_RANGE) & geo["geolocation_lng"].between(*_LON_RANGE)
    ]
    geo["prefix"] = _prefix3(geo["geolocation_zip_code_prefix"])
    return (
        geo.groupby("prefix")
        .agg(
            lat=("geolocation_lat", "mean"),
            lon=("geolocation_lng", "mean"),
            state=("geolocation_state", lambda s: s.mode().iloc[0]),
            city=("geolocation_city", lambda s: s.mode().iloc[0]),
        )
        .rename_axis("prefix")
    )


def load_geo_nodes(raw_dir: Path, period: tuple[str, str] | None = None) -> GeoNodes:
    """Conta pedidos entregues por região do cliente e por região do vendedor.

    Args:
        raw_dir: pasta com os CSVs do Olist.
        period: (início, fim) inclusivos em 'YYYY-MM-DD' sobre a data de compra.
    """
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    orders = orders[orders["order_status"] == "delivered"]
    if period is not None:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)
        orders = orders[orders["order_purchase_timestamp"].between(start, end, inclusive="left")]

    customers = pd.read_csv(
        raw_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix"],
    )
    sellers = pd.read_csv(
        raw_dir / "olist_sellers_dataset.csv", usecols=["seller_id", "seller_zip_code_prefix"]
    )
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv", usecols=["order_id", "seller_id"]
    )

    centroids = _region_centroids(raw_dir)

    cust = orders.merge(customers, on="customer_id")
    cust["prefix"] = _prefix3(cust["customer_zip_code_prefix"])
    demand = cust.groupby("prefix").size().rename("orders").to_frame()

    # Um pedido pode ter vários vendedores: contamos pares (pedido, vendedor) distintos.
    sold = items.drop_duplicates().merge(orders[["order_id"]], on="order_id")
    sold = sold.merge(sellers, on="seller_id")
    sold["prefix"] = _prefix3(sold["seller_zip_code_prefix"])
    supply = sold.groupby("prefix").size().rename("orders").to_frame()

    return GeoNodes(
        demand=demand.join(centroids, how="inner").sort_values("orders", ascending=False),
        supply=supply.join(centroids, how="inner").sort_values("orders", ascending=False),
    )
