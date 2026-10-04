"""Experimento P02/P03: imitação de strong branching em SCIP 10 (instâncias `facilities` 100x100).

coletar  30 instâncias de treino + 6 de validação, strong branching em 30% dos nós (150 s cada)
treinar  LightGBM LambdaRank por nó (baseline LambdaMART de P02); acc@1/acc@5 na validação
testar   20 instâncias de teste (sementes 500+), regras relpscost (padrão do SCIP), pscost,
         fullstrong e aprendida; limite 300 s; média geométrica deslocada (s = 1) como em P02

uv run python -m alocacao_capacitada.pesquisa.exp_l2b coletar --workers 3
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa import l2b

OUT = Path("results/pesquisa/l2b")
DADOS = Path("data/processed/pesquisa/l2b")


def _coleta(seed: int) -> tuple[int, list[tuple[np.ndarray, np.ndarray]], float]:
    t0 = time.perf_counter()
    am = l2b.coletar(l2b.gerar_facilities(100, 100, seed), 0.3, seed, tempo=150)
    return seed, am, time.perf_counter() - t0


def coletar(workers: int) -> None:
    DADOS.mkdir(parents=True, exist_ok=True)
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for seed, am, t in ex.map(_coleta, list(range(36))):
            res[seed] = (am, t)
            print(seed, len(am), f"{t:.0f}s", flush=True)
    with open(DADOS / "amostras.pkl", "wb") as fh:
        pickle.dump(res, fh)


def _grupos(amostras: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X, y, g = [], [], []
    for f, s in amostras:
        if len(s) < 2:
            continue
        r = np.argsort(np.argsort(s))  # posição no ranking do SB
        rel = np.floor(4 * r / max(len(s) - 1, 1)).astype(int)  # relevância 0..4 por quantil
        rel[np.argmax(s)] = 5  # o escolhido pelo professor
        X.append(f)
        y.append(rel)
        g.append(len(s))
    return np.vstack(X), np.concatenate(y), g


def treinar() -> None:
    import lightgbm as lgb

    OUT.mkdir(parents=True, exist_ok=True)
    with open(DADOS / "amostras.pkl", "rb") as fh:
        res = pickle.load(fh)
    tr = [a for s in range(30) for a in res[s][0]]
    va = [a for s in range(30, 36) for a in res[s][0]]
    Xtr, ytr, gtr = _grupos(tr)
    t0 = time.perf_counter()
    rk = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0,
                        verbose=-1, n_jobs=1)
    rk.fit(Xtr, ytr, group=gtr)
    t_tr = time.perf_counter() - t0
    acc1, acc5 = [], []
    for f, s in va:
        if len(s) < 2:
            continue
        p = rk.predict(f)
        top = np.argsort(-p)
        acc1.append(top[0] == np.argmax(s))
        acc5.append(np.argmax(s) in top[:5])
    met = {"amostras_treino": len(tr), "amostras_val": len(va), "acc1_val": float(np.mean(acc1)),
           "acc5_val": float(np.mean(acc5)), "t_treino_s": t_tr,
           "t_coleta_h": sum(res[s][1] for s in res) / 3600}
    (OUT / "treino.json").write_text(json.dumps(met, indent=2))
    with open(DADOS / "ranker.pkl", "wb") as fh:
        pickle.dump(rk, fh)
    print(met)


def _teste(args: tuple[int, str]) -> dict[str, object]:
    seed, regra = args
    scorer = None
    if regra == "aprendida":
        with open(DADOS / "ranker.pkl", "rb") as fh:
            scorer = pickle.load(fh)
    r = l2b.resolver(l2b.gerar_facilities(100, 100, 500 + seed), 300, regra, scorer)
    return {"seed": seed, "regra": regra, **r}


def testar(workers: int) -> None:
    jobs = [(s, r) for s in range(20) for r in ("relpscost", "pscost", "fullstrong", "aprendida")]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for row in ex.map(_teste, jobs):
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "teste.csv", index=False)
            print(row["seed"], row["regra"], f"{row['tempo']:.1f}s", row["nos"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["coletar", "treinar", "testar"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"coletar": lambda: coletar(a.workers), "treinar": treinar,
     "testar": lambda: testar(a.workers)}[a.fase]()


if __name__ == "__main__":
    main()
