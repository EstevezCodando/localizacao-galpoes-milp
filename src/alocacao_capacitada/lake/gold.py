"""Camada gold: tabela de referência de frete pronta para o modelo, com proveniência."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alocacao_capacitada.lake.silver import parse_antt_coefficients


def build_freight_reference(
    bronze_html: Path, provenance_json: Path, resolution: str, out_csv: Path
) -> pd.DataFrame:
    """Gera o CSV de coeficientes (Tabela A) rastreável até o HTML oficial coletado."""
    prov = json.loads(provenance_json.read_text(encoding="utf-8"))
    table = parse_antt_coefficients(bronze_html.read_bytes())
    table = table[table["tabela"] == "A"].copy()
    table["resolucao"] = resolution
    table["fonte_url"] = prov["url"]
    table["coletado_em"] = prov["fetched_at"]
    table["sha256_bronze"] = prov["sha256"]
    table["verificado_oficial"] = True
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table
