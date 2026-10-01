"""Dados territoriais abertos: IBGE (SIDRA e localidades) e Ipeadata (Atlas do Desenvolvimento).

Cada função devolve um DataFrame longo ou largo com a chave `cod` (código IBGE de 7 dígitos, str).
As consultas passam por `ApiCollector` (cache, hash, intervalo); aqui só se monta a URL e se
interpreta o JSON, com validações que falham alto quando o formato muda.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

import pandas as pd

from alocacao_capacitada.lake.api import ApiCollector

SIDRA = "https://apisidra.ibge.gov.br/values"
LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado"
IPEA = "http://www.ipeadata.gov.br/api/odata4"

# Grupos quinquenais da tabela 9514 (Censo 2022, população por idade) e o rótulo usado aqui.
AGE_GROUPS: dict[str, str] = {
    "93070": "0_4", "93084": "5_9", "93085": "10_14", "93086": "15_19", "93087": "20_24",
    "93088": "25_29", "93089": "30_34", "93090": "35_39", "93091": "40_44", "93092": "45_49",
    "93093": "50_54", "93094": "55_59", "93095": "60_64", "93096": "65_69", "93097": "70_74",
    "93098": "75_79", "49108": "80_84", "49109": "85_89", "60040": "90_94", "60041": "95_99",
    "6653": "100_mais",
}  # fmt: skip
_SIDRA_MAX_VALUES = 50_000  # limite de valores por consulta do SIDRA


def _sidra_frame(
    rows: list[dict[str, Any]], value_name: str, extra: str | None = None
) -> pd.DataFrame:
    """Converte a resposta do SIDRA (1ª linha = cabeçalho) em DataFrame (cod, [extra], valor)."""
    if not rows or "V" not in rows[0]:
        raise ValueError("resposta do SIDRA em formato inesperado")
    data = rows[1:]
    if not data:
        return pd.DataFrame({"cod": pd.Series(dtype=str), value_name: pd.Series(dtype=float)})
    out = pd.DataFrame(
        {
            "cod": [r["D1C"] for r in data],
            value_name: pd.to_numeric([r["V"] for r in data], errors="coerce"),
        }
    )
    if extra is not None:
        out[extra] = [r[extra] for r in data]
    if out["cod"].str.len().ne(7).any():
        raise ValueError("código de município com tamanho inesperado")
    return out


def municipalities(api: ApiCollector) -> pd.DataFrame:
    """Lista dos municípios (código, nome, UF, região)."""
    rows = api.get_json("localidades_municipios", LOCALIDADES)
    out = pd.DataFrame(
        {
            "cod": [str(r["municipio-id"]) for r in rows],
            "nome": [r["municipio-nome"] for r in rows],
            "uf": [r["UF-sigla"] for r in rows],
            "regiao": [r["regiao-sigla"] for r in rows],
        }
    )
    if out["cod"].duplicated().any():
        raise ValueError("códigos de município duplicados")
    return out


def population_census_2022(api: ApiCollector) -> pd.DataFrame:
    rows = api.get_json("pop_2022", f"{SIDRA}/t/4709/n6/all/v/93/p/2022?formato=json")
    return _sidra_frame(rows, "censo_2022")


def population_census_2010(api: ApiCollector) -> pd.DataFrame:
    rows = api.get_json("pop_2010", f"{SIDRA}/t/202/n6/all/v/93/p/2010/c1/0/c2/0?formato=json")
    return _sidra_frame(rows, "censo_2010")


def population_census_2000(api: ApiCollector) -> pd.DataFrame:
    rows = api.get_json("pop_2000", f"{SIDRA}/t/202/n6/all/v/93/p/2000/c1/0/c2/0?formato=json")
    return _sidra_frame(rows, "censo_2000")


def population_estimates(api: ApiCollector, years: range = range(2001, 2027)) -> pd.DataFrame:
    """Estimativas anuais de população (tabela 6579), em formato largo: uma coluna por ano."""
    frames = []
    for year in years:
        rows = api.get_json(
            f"pop_estim_{year}", f"{SIDRA}/t/6579/n6/all/v/9324/p/{year}?formato=json"
        )
        frame = _sidra_frame(rows, f"est_{year}")
        if frame.empty:  # 2007 (Contagem) e 2010 (Censo) não constam desta tabela
            continue
        frames.append(frame.set_index("cod"))
    return pd.concat(frames, axis=1).reset_index()


def age_groups_2022(api: ApiCollector) -> pd.DataFrame:
    """População por grupo quinquenal de idade, Censo 2022 (em lotes: limite de 50 mil valores)."""
    codes = list(AGE_GROUPS)
    per_request = max(1, _SIDRA_MAX_VALUES // 5570)
    chunks = [codes[i : i + per_request] for i in range(0, len(codes), per_request)]
    parts = []
    for k, chunk in enumerate(chunks):
        rows = api.get_json(
            f"idade_2022_{k}",
            f"{SIDRA}/t/9514/n6/all/v/93/p/2022/c2/6794/c287/{','.join(chunk)}/c286/113635"
            "?formato=json",
        )
        data = rows[1:]
        frame = pd.DataFrame(
            {
                "cod": [r["D1C"] for r in data],
                "grupo": [AGE_GROUPS[r["D5C"]] for r in data],
                "v": pd.to_numeric([r["V"] for r in data], errors="coerce"),
            }
        )
        parts.append(frame)
    long = pd.concat(parts)
    wide = long.pivot_table(index="cod", columns="grupo", values="v", aggfunc="first")
    wide.columns = [f"idade_{c}" for c in wide.columns]
    return wide.reset_index()


def gdp(api: ApiCollector, years: range = range(2010, 2022)) -> pd.DataFrame:
    """PIB municipal a preços correntes (R$ mil), uma coluna por ano (tabela 5938)."""
    frames = []
    for year in years:
        rows = api.get_json(f"pib_{year}", f"{SIDRA}/t/5938/n6/all/v/37/p/{year}?formato=json")
        frame = _sidra_frame(rows, f"pib_{year}")
        if not frame.empty:
            frames.append(frame.set_index("cod"))
    return pd.concat(frames, axis=1).reset_index()


def ipeadata_series(api: ApiCollector, code: str) -> pd.DataFrame:
    """Série municipal do Ipeadata: colunas cod e `<serie>_<ano>` (valores decenais)."""
    quoted = urllib.parse.quote(f"'{code}'")
    data = api.get_json(
        f"ipea_{code}",
        f"{IPEA}/ValoresSerie(SERCODIGO={quoted})?$filter=NIVNOME%20eq%20'Munic%C3%ADpios'",
    )
    values = data["value"]
    if not values:
        raise ValueError(f"série {code} sem valores")
    frame = pd.DataFrame(
        {
            "cod": [str(v["TERCODIGO"]) for v in values],
            "ano": [int(v["VALDATA"][:4]) for v in values],
            "v": [v["VALVALOR"] for v in values],
        }
    )
    wide = frame.pivot_table(index="cod", columns="ano", values="v", aggfunc="first")
    wide.columns = [f"{code.lower()}_{c}" for c in wide.columns]
    return wide.reset_index()


def normalize_name(text: str) -> str:
    """Nome sem acentos, minúsculo e sem pontuação, para casar grafias diferentes."""
    import unicodedata

    base = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", base.lower()).strip()
