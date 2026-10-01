"""Camada silver: extrai e tipa os coeficientes do Anexo II (Tabelas A–D) da ANTT.

O HTML oficial usa vírgula decimal; ler sem `decimal=","` transforma 4,0144 em 40144.
"""

from __future__ import annotations

import io
import re

import pandas as pd

_TABLE_NAMES = ("A", "B", "C", "D")
_CCD, _CC = "Deslocamento (CCD)", "Carga e descarga (CC)"


def parse_antt_coefficients(html: bytes) -> pd.DataFrame:
    """Retorna (tabela, tipo_carga, eixos, ccd_rs_km, cc_rs) em formato longo.

    Raises:
        ValueError: se a estrutura não for a esperada (falha alta em vez de dado errado).
    """
    text = html.decode("latin-1")
    tables = pd.read_html(io.StringIO(text), decimal=",", thousands=None)  # type: ignore[arg-type]
    if len(tables) != len(_TABLE_NAMES):
        raise ValueError(f"esperava {len(_TABLE_NAMES)} tabelas, encontrei {len(tables)}")

    frames = []
    for name, raw in zip(_TABLE_NAMES, tables, strict=True):
        frames.append(_parse_one(name, raw))
    return pd.concat(frames, ignore_index=True)


def _parse_one(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    body = raw.iloc[:, 1:].copy()
    axles = [int(v) for v in body.iloc[1, 3:]]
    body = body.iloc[2:].reset_index(drop=True)
    body.columns = ["tipo_carga", "coeficiente", "unidade", *axles]
    body["tipo_carga"] = body["tipo_carga"].ffill()
    if not set(body["coeficiente"]) <= {_CCD, _CC}:
        raise ValueError(f"coeficientes inesperados na tabela {name}: {set(body['coeficiente'])}")

    long = body.melt(
        id_vars=["tipo_carga", "coeficiente"], value_vars=axles, var_name="eixos", value_name="v"
    )
    wide = long.pivot_table(
        index=["tipo_carga", "eixos"], columns="coeficiente", values="v", aggfunc="first"
    ).reset_index()
    out = wide.rename(columns={_CCD: "ccd_rs_km", _CC: "cc_rs"})
    out.columns.name = None
    out.insert(0, "tabela", name)
    out["tipo_carga"] = out["tipo_carga"].map(_slug)
    out["eixos"] = out["eixos"].astype(int)
    for col in ("ccd_rs_km", "cc_rs"):
        out[col] = out[col].map(_to_float)
    if ((out["ccd_rs_km"] <= 0) | (out["ccd_rs_km"] > 50) | (out["cc_rs"] <= 0)).any():
        raise ValueError(f"coeficiente fora de faixa plausível na tabela {name}")
    return out[["tabela", "tipo_carga", "eixos", "ccd_rs_km", "cc_rs"]]


def _slug(label: str) -> str:
    return re.sub(r"\W+", "_", str(label).strip().lower()).strip("_")


def _to_float(value: object) -> float:
    """Converte '1.016,29' (pt-BR) ou um float já lido em número."""
    text = str(value).strip()
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)
