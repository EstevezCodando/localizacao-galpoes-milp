"""Converte as 71 instâncias de Holmberg et al. (1999) para o contrato estrito (custo TOTAL por
par, atendimento obrigatório) e grava o split `holmberg` com o ótimo PUBLICADO como referência.

Teste externo de distribuição: custos não euclidianos, sem coordenadas, m = 10-30, n = 50-200.
uv run python -m alocacao_capacitada.pesquisa.holmberg_split
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from alocacao_capacitada.pesquisa.dados import RAIZ, Rotulado
from alocacao_capacitada.pesquisa.problema import Problema
from alocacao_capacitada.validation.holmberg import load_holmberg_optima

DIR = Path("data/bronze/holmberg")


def carregar_holmberg(path: Path) -> Problema:
    tokens = path.read_text(encoding="latin-1").replace(chr(0), " ").split()
    m, n = int(tokens[0]), int(tokens[1])
    v = np.array([float(t) for t in tokens[2 : 2 + 2 * m + n + m * n]])
    cf = v[: 2 * m].reshape(m, 2)
    return Problema(nome=f"holmberg_{path.name}", familia="holmberg", fixed=cf[:, 1],
                    capacity=cf[:, 0], demand=v[2 * m : 2 * m + n],
                    cost=v[2 * m + n :].reshape(m, n),
                    meta={"razao": float(cf[:, 0].sum() / v[2 * m : 2 * m + n].sum())})


def main() -> None:
    out = RAIZ / "holmberg"
    out.mkdir(parents=True, exist_ok=True)
    opt = load_holmberg_optima(DIR / "Holmberg_Solution_Values.txt")
    for _, r in opt.iterrows():
        p = carregar_holmberg(DIR / "instances" / str(r["name"]))
        z = float(r["optimal"])
        rot = Rotulado(p, [], z, np.zeros(p.n, dtype=np.int64), z, "otimo_publicado", 0.0, 0.0, 0)
        with open(out / f"{p.nome}.pkl", "wb") as fh:
            pickle.dump(rot, fh)
    print(len(opt), "instâncias")


if __name__ == "__main__":
    main()
