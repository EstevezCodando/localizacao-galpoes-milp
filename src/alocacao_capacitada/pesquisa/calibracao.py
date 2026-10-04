"""Piloto de calibração (INICIO.md, passo 6): tempo, memória e taxa de solução por célula.

uv run python -m alocacao_capacitada.pesquisa.calibracao --tempo 60 --workers 3
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from alocacao_capacitada.pesquisa.exato import relaxacao_linear, resolver
from alocacao_capacitada.pesquisa.geradores import FAMILIAS, Config, gerar
from alocacao_capacitada.pesquisa.problema import gap_certificado


def _uma(args: tuple[str, int, int, float, int, float]) -> dict[str, object]:
    familia, m, n, razao, seed, tempo = args
    prob, rej = gerar(Config(familia, m, n, razao), seed)
    lp = relaxacao_linear(prob)
    res = resolver(prob, tempo)
    return {
        "familia": familia, "m": m, "n": n, "razao": razao, "seed": seed, "rejeitadas": rej,
        "status": res.status, "obj": res.objetivo, "bound": res.bound,
        "gap": gap_certificado(res.objetivo, res.bound), "lp": lp.bound,
        "gap_lp": gap_certificado(res.objetivo, lp.bound), "t_lp": lp.tempo,
        "t_total": res.t_total, "t_montagem": res.t_montagem, "nos": res.nos,
        "t_primeira": res.trajetoria[0][0] if res.trajetoria else None,
        "abertos": int(len(set(res.atribuicao.tolist()))) if res.atribuicao is not None else None,
        "valida": res.valida,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tempo", type=float, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tamanhos", default="30x150,50x200,75x300,100x400")
    ap.add_argument("--razoes", default="1.2,1.5,2.0")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/pesquisa/calibracao.csv"))
    a = ap.parse_args()
    tam = [tuple(map(int, t.split("x"))) for t in a.tamanhos.split(",")]
    raz = [float(r) for r in a.razoes.split(",")]
    jobs = [(f, m, n, r, s, a.tempo) for f, (m, n), r, s in
            itertools.product(FAMILIAS, tam, raz, range(a.seeds))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for row in ex.map(_uma, jobs):
            rows.append(row)
            print({k: row[k] for k in ("familia", "m", "n", "razao", "status", "gap", "t_total",
                                       "nos", "gap_lp")}, flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
