"""Tipologias de município por clusterização (k-means), com escolha de k por estabilidade.

Critérios para escolher o número de grupos: silhueta (separação) E estabilidade sob reamostragem
(Rand ajustado entre partições de subamostras). Um k sem estabilidade não vira tipologia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = [
    "log_pop",
    "cagr_10_22",
    "share_15_29",
    "share_60_mais",
    "idhm",
    "log_rdpc",
    "gini",
    "log_pib_pc",
    "log_dist_polo",
]


def stability(x: np.ndarray, k: int, rounds: int = 8, seed: int = 0) -> float:
    """Média do Rand ajustado entre duas partições de subamostras (80%) sobre a amostra toda."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        parts = []
        for _ in range(2):
            idx = rng.choice(len(x), size=int(0.8 * len(x)), replace=False)
            model = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1_000_000))).fit(
                x[idx]
            )
            parts.append(model.predict(x))
        scores.append(adjusted_rand_score(parts[0], parts[1]))
    return float(np.mean(scores))


def choose_k(x: np.ndarray, candidates: range = range(3, 9)) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    sample = rng.choice(len(x), size=min(2500, len(x)), replace=False)
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
        rows.append(
            {
                "k": k,
                "silhueta": float(silhouette_score(x[sample], labels[sample])),
                "estabilidade_ari": stability(x, k),
            }
        )
    return pd.DataFrame(rows)


def fit_clusters(table: pd.DataFrame, k: int) -> tuple[pd.Series, pd.DataFrame]:
    """Rótulos por município e perfil (médias) de cada grupo, ordenados por população média."""
    data = table[table["completo"]].reset_index(drop=True)
    x = StandardScaler().fit_transform(data[CLUSTER_FEATURES])
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(x)
    data = data.assign(grupo=labels)
    order = data.groupby("grupo")["pop"].median().sort_values(ascending=False).index
    rename = {old: new for new, old in enumerate(order)}
    data["grupo"] = data["grupo"].map(rename)
    profile = data.groupby("grupo").agg(
        municipios=("cod", "size"),
        pop_mediana=("pop", "median"),
        pop_total=("pop", "sum"),
        idhm=("idhm", "mean"),
        renda_pc=("adh_rdpc_2010", "mean"),
        cagr=("cagr_10_22", "mean"),
        jovens_15_29=("share_15_29", "mean"),
        pedidos=("pedidos", "sum"),
    )
    return data.set_index("cod")["grupo"], profile.reset_index()
