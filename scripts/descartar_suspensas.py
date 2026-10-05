"""Remove do checkpoint as execuções invalidadas por suspensão da máquina (pré-registro 10, seção 7).

Critério fixo, só de tempo: estouro de orçamento > 60 s E razão CPU ÷ parede < 0,1. As linhas
removidas são acrescentadas a results/pesquisa/fechado/descartadas_suspensao.csv; a retomada do
executor as refaz.

uv run python scripts/descartar_suspensas.py <conjunto> results/pesquisa/clns/<run_id>
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

SAIDA = Path("results/pesquisa/fechado/descartadas_suspensao.csv")


def main() -> None:
    conjunto, run = sys.argv[1], Path(sys.argv[2])
    db = sqlite3.connect(run / "checkpoint.sqlite")
    ruins = []
    for chave, linha in db.execute("select chave, linha from resultados").fetchall():
        r = json.loads(linha)
        if r["estouro_s"] > 60 and r["razao_cpu"] < 0.1:
            ruins.append(dict(r, chave=chave, conjunto=conjunto,
                              motivo="suspensao da maquina durante a execucao"))
    if ruins:
        novo = pd.DataFrame(ruins)
        if SAIDA.exists():
            novo = pd.concat([pd.read_csv(SAIDA), novo], ignore_index=True)
        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        novo.to_csv(SAIDA, index=False)
        db.executemany("delete from resultados where chave = ?", [(r["chave"],) for r in ruins])
        db.commit()
    print(f"{conjunto}: {len(ruins)} descartadas")


if __name__ == "__main__":
    main()
