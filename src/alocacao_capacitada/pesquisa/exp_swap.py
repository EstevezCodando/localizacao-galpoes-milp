"""Experimento das frentes P04/P05: busca por trocas completa vs filtrada (clássico/aprendido/
aleatório) em p-mediana sobre redes.

Treino: 30 redes n = 200 (uniforme e cidade), imitação a partir de inícios aleatórios.
Teste: n = 200 (ótimo SCIP), 500, 1000; 3 inícios aleatórios por instância (mesmos para todos
os filtros); k_in em {8, 16, 32}, k_out = 4; com e sem fallback de varredura completa.
v2: atributos estáticos em cache (a v1 reordenava a matriz de distâncias a cada iteração).

uv run python -m alocacao_capacitada.pesquisa.exp_swap
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa.swap import (
    busca_trocas,
    coletar_imitacao,
    exato_pmediana,
    gerar_rede,
    treinar_modelo,
)

OUT = Path("results/pesquisa/swap_v2")


def _p(n: int) -> int:
    return max(5, int(round(np.sqrt(n))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    amostras = []
    for s in range(30):
        inst = gerar_rede(200, _p(200), ("uniforme", "cidade")[s % 2], 50_000 + s)
        rng = np.random.default_rng(s)
        for _ in range(2):
            amostras += coletar_imitacao(inst, rng.choice(inst.n, inst.p, replace=False))
    t_col = time.perf_counter() - t0
    t0 = time.perf_counter()
    modelo = treinar_modelo(amostras)
    t_tr = time.perf_counter() - t0
    (OUT / "custo_offline.json").write_text(json.dumps(
        {"estados": len(amostras), "t_coleta_s": t_col, "t_treino_s": t_tr}, indent=2))
    print("treino", len(amostras), f"{t_col:.0f}s + {t_tr:.0f}s", flush=True)
    rows = []
    for n, k in ((200, 10), (500, 8), (1000, 5)):
        for s in range(k):
            fam = ("uniforme", "cidade")[s % 2]
            inst = gerar_rede(n, _p(n), fam, 60_000 + 1000 * n + s)
            rng = np.random.default_rng(1000 + s)
            inicios = [rng.choice(inst.n, inst.p, replace=False) for _ in range(3)]
            ref = (np.nan, np.nan, "sem_exato")
            for ini_id, S0 in enumerate(inicios):
                configs = [("completo", 0, True)] + [
                    (f, k_in, fb) for f in ("classico", "aprendido", "aleatorio")
                    for k_in in (8, 16, 32) for fb in (True, False)]
                for filtro, k_in, fb in configs:
                    r = busca_trocas(inst, S0, filtro, k_out=4, k_in=k_in or 16, modelo=modelo,
                                     tempo=600, fallback=fb)
                    rows.append({"n": n, "familia": fam, "seed": s, "inicio": ini_id,
                                 "filtro": filtro, "k_in": k_in, "fallback": fb,
                                 "custo": r.custo,
                                 "tempo": r.tempo, "iteracoes": r.iteracoes,
                                 "avaliacoes": r.avaliacoes, "fallbacks": r.fallbacks})
            if n == 200:
                best = min(row["custo"] for row in rows if row["n"] == n and row["seed"] == s)
                ref = exato_pmediana(inst, 120)
                assert ref[0] <= best + 1e-6 or ref[2] != "optimal"
            for row in rows:
                if row["n"] == n and row["seed"] == s:
                    row["otimo"], row["bound"], row["status_exato"] = ref
            pd.DataFrame(rows).to_csv(OUT / "resultados.csv", index=False)
            print(n, s, flush=True)


if __name__ == "__main__":
    main()
