"""Três evidências diferentes que não devem ser misturadas sob o rótulo "ótimo".

* `coincide_ref`: solução viável com custo igual ao valor publicado (tolerância relativa estrita).
* `dentro_0_01pct`: solução viável a até 0,01% da referência (inclui a anterior).
* `certificado`: o próprio solver provou otimalidade (status OTIMO) em uma solução viável.

Custo abaixo da referência além da tolerância é sinal de erro (de dado, de avaliação ou da
referência) e vira `abaixo_da_referencia`, para investigação; não entra como acerto.
"""

from __future__ import annotations

import pandas as pd

REL_TOL_EXACT = 1e-6
REL_TOL_NEAR = 1e-4


def classify_methods(table: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    out = table.copy()
    ref = out["otimo_publicado"]
    for m in methods:
        viable = out[f"{m}_viavel"].astype(bool)
        rel = (out[f"{m}_custo"] - ref) / ref
        out[f"{m}_coincide_ref"] = viable & (rel.abs() <= REL_TOL_EXACT)
        out[f"{m}_dentro_0_01pct"] = viable & (rel <= REL_TOL_NEAR) & (rel >= -REL_TOL_EXACT)
        out[f"{m}_certificado"] = viable & (out[f"{m}_status"] == "OTIMO")
        out[f"{m}_abaixo_da_referencia"] = rel < -REL_TOL_EXACT
    return out
