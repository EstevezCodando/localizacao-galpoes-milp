"""Geração do conjunto de dados e dos rótulos (doc 07).

Divisão sem vazamento: sementes disjuntas por split (treino 0-999, validação 1000-1999,
teste 2000-2999, generalização 3000+); a família `corredor` NUNCA entra em treino/validação.

Rótulo de cada instância: pool = soluções que o SCIP guardou em `t_scip` s + solução final do
LNS em `t_lns` s. Classe de qualidade: "otimo" se o SCIP provou, senão "gap" com o gap
certificado pelo melhor bound do SCIP (bound do problema ORIGINAL).

uv run python -m alocacao_capacitada.pesquisa.dados --split treino --n 40 --workers 3
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alocacao_capacitada.pesquisa.exato import resolver
from alocacao_capacitada.pesquisa.geradores import Config, gerar
from alocacao_capacitada.pesquisa.heuristicas import lns
from alocacao_capacitada.pesquisa.problema import Problema, gap_certificado, validar

RAIZ = Path("data/processed/pesquisa")

SPLITS: dict[str, tuple[tuple[str, ...], tuple[float, ...], int, int, int]] = {
    # nome: (famílias, razões, m, n, semente inicial)
    "treino": (("uniforme", "clusters"), (1.5, 3.0), 30, 150, 0),
    "validacao": (("uniforme", "clusters"), (1.5, 3.0), 30, 150, 1000),
    "teste": (("uniforme", "clusters"), (1.5, 3.0), 30, 150, 2000),
    "gen_corredor": (("corredor",), (1.5, 3.0), 30, 150, 3000),
    "gen_escala": (("uniforme", "clusters"), (1.5, 3.0), 50, 200, 4000),
}


@dataclass
class Rotulado:
    prob: Problema
    pool: list[tuple[float, np.ndarray]]
    melhor: float
    melhor_atrib: np.ndarray
    bound: float | None
    qualidade: str
    gap: float | None
    t_rotulo: float
    rejeitadas: int


def rotular(prob: Problema, t_scip: float, t_lns: float, rejeitadas: int = 0) -> Rotulado:
    t0 = time.perf_counter()
    r = resolver(prob, t_scip, pool=True)
    pool = list(r.pool)
    cands = [(r.objetivo, r.atribuicao)] if r.valida else []
    if t_lns > 0:
        h = lns(prob, t_lns)
    best_obj, best_a = min(cands, key=lambda c: c[0]) if cands else (np.inf, None)
    if t_lns > 0 and h.valida and h.objetivo < best_obj - 1e-9 and h.atribuicao is not None:
        best_obj, best_a = h.objetivo, h.atribuicao
        y = np.zeros(prob.m, dtype=bool)
        y[np.unique(h.atribuicao)] = True
        pool.append((h.objetivo, y))
    if best_a is None:
        raise RuntimeError(f"{prob.nome}: nenhuma solução para rotular")
    assert validar(prob, best_a).valida
    status_ok = r.status == "optimal" and best_obj >= r.objetivo - 1e-6
    return Rotulado(prob, pool, best_obj, best_a, r.bound, "otimo" if status_ok else "gap",
                    gap_certificado(best_obj, r.bound), time.perf_counter() - t0, rejeitadas)


def _job(args: tuple[str, str, float, int, int, int, float, float]) -> str:
    split, familia, razao, m, n, seed, t_scip, t_lns = args
    out = RAIZ / split / f"{familia}_r{razao}_m{m}_n{n}_s{seed}.pkl"
    if out.exists():
        return f"pulado {out.name}"
    prob, rej = gerar(Config(familia, m, n, razao), seed)
    rot = rotular(prob, t_scip, t_lns, rej)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(rot, fh)
    return f"{out.name}: {rot.qualidade} gap={rot.gap:.4f} t={rot.t_rotulo:.0f}s pool={len(rot.pool)}"


class _Unpickler(pickle.Unpickler):
    """Rótulos gravados por `python -m ...dados` (workers spawn) referenciam `__mp_main__`."""

    def find_class(self, module: str, name: str) -> type:
        if module in ("__main__", "__mp_main__") and name == "Rotulado":
            return Rotulado
        return super().find_class(module, name)  # type: ignore[no-any-return]


def ler(caminho: Path) -> Rotulado:
    with open(caminho, "rb") as fh:
        return _Unpickler(fh).load()  # type: ignore[no-any-return]


def carregar(split: str) -> list[Rotulado]:
    return [ler(a) for a in sorted((RAIZ / split).glob("*.pkl"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--n", type=int, required=True, help="instâncias por (família, razão)")
    ap.add_argument("--t-scip", type=float, default=60)
    ap.add_argument("--t-lns", type=float, default=30)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    fams, razoes, m, n, s0 = SPLITS[a.split]
    jobs = [(a.split, f, r, m, n, s0 + k, a.t_scip, a.t_lns)
            for k in range(a.n) for f in fams for r in razoes]
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(_job, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
