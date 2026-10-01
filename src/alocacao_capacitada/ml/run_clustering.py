"""Escolhe k, ajusta as tipologias de município e salva o perfil de cada grupo.

Uso:  uv run python -m alocacao_capacitada.ml.run_clustering
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from alocacao_capacitada.ml.clustering import CLUSTER_FEATURES, choose_k, fit_clusters
from alocacao_capacitada.ml.features import load_modeling_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--municipios", type=Path, default=Path("data/reference/municipios.parquet")
    )
    parser.add_argument(
        "--painel", type=Path, default=Path("data/reference/olist_painel_municipal.parquet")
    )
    parser.add_argument("--k", type=int, default=0, help="0 = escolhe pelo critério combinado")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    table = load_modeling_table(args.municipios, args.painel)
    data = table[table["completo"]]
    x = StandardScaler().fit_transform(data[CLUSTER_FEATURES])
    scores = choose_k(np.asarray(x))
    scores.to_csv(args.out / "ml_cluster_escolha_k.csv", index=False)
    # Regra: maior estabilidade; empate (dentro de 0,05) resolve-se pela maior silhueta.
    best = scores[scores["estabilidade_ari"] >= scores["estabilidade_ari"].max() - 0.05]
    k = args.k or int(best.sort_values("silhueta", ascending=False).iloc[0]["k"])
    labels, profile = fit_clusters(table, k)
    labels.rename("grupo").reset_index().to_parquet(
        args.out / "ml_cluster_rotulos.parquet", index=False
    )
    profile.to_csv(args.out / "ml_cluster_perfil.csv", index=False)
    pd.set_option("display.width", 220)
    print(scores.round(3).to_string(index=False))
    print("k escolhido:", k)
    print(profile.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
