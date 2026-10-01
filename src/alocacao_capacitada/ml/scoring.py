"""Score de atratividade de município como centro, treinado com as soluções do otimizador.

Rótulo: o município recebeu pelo menos um módulo na rede ótima (ou quase ótima) de um cenário.
Entradas: características do município e do cenário. Validação: deixa uma macrorregião de fora por
vez (os pontos de uma mesma região se parecem), e o score usado depois é SEMPRE o fora da amostra.

O score só vale se for útil para decidir: usado como pré-filtro de candidatos, a rede restrita
precisa custar quase o mesmo que a rede com todos os candidatos (regret), em menos tempo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alocacao_capacitada.geo import haversine_matrix
from alocacao_capacitada.ml.features import OSM_FEATURES

CANDIDATE_FEATURES = [
    "log_demanda_local",
    "log_demanda_raio",
    "log_km_medio_ponderado",
    "log_pop",
    "log_rdpc",
    "idhm",
    "aluguel_uf",
    "log_dist_polo",
    "cagr_10_22",
    "rank_demanda",
]
SCENARIO_FEATURES = ["share", "terceirizacao", "densidade"]
ALL_FEATURES = [*CANDIDATE_FEATURES, *SCENARIO_FEATURES]
OSM_ALL_FEATURES = [*ALL_FEATURES, *OSM_FEATURES]


def catchment_demand(
    candidates: pd.DataFrame, demand: pd.Series, municipios: pd.DataFrame, scale_km: float = 150.0
) -> np.ndarray:
    """Soma da demanda dos municípios próximos, com decaimento exponencial pela distância (km)."""
    m = municipios.set_index("cod").loc[demand.index]
    dist = haversine_matrix(
        candidates["lat"].to_numpy(),
        candidates["lon"].to_numpy(),
        m["lat"].to_numpy(),
        m["lon"].to_numpy(),
    )
    weights = np.exp(-dist / scale_km)
    result: np.ndarray = weights @ demand.to_numpy()
    return result


def candidate_table(
    pool: list[str],
    demand: pd.Series,
    municipios: pd.DataFrame,
    table: pd.DataFrame,
    rent_by_uf: dict[str, float],
    km_to_nodes: np.ndarray,
    node_demand: np.ndarray,
) -> pd.DataFrame:
    """`km_to_nodes` (candidatos × nós) é a distância VIÁRIA (OpenStreetMap via OSRM); a média
    ponderada pela demanda mede a acessibilidade ao mercado."""
    base = table.set_index("cod").loc[pool]
    cands = municipios.set_index("cod").loc[pool].reset_index()
    out = pd.DataFrame(
        {"cod": pool, "regiao": cands["regiao"].to_numpy(), "uf": cands["uf"].to_numpy()}
    )
    local = demand.reindex(pool).to_numpy()
    out["log_demanda_local"] = np.log(local)
    out["log_demanda_raio"] = np.log(catchment_demand(cands, demand, municipios))
    out["log_km_medio_ponderado"] = np.log(
        (km_to_nodes * node_demand[None, :]).sum(axis=1) / node_demand.sum()
    )
    out["log_pop"] = base["log_pop"].to_numpy()
    out["log_rdpc"] = base["log_rdpc"].to_numpy()
    out["idhm"] = base["idhm"].to_numpy()
    out["aluguel_uf"] = [rent_by_uf[u] for u in out["uf"]]
    out["log_dist_polo"] = base["log_dist_polo"].to_numpy()
    out["cagr_10_22"] = base["cagr_10_22"].to_numpy()
    out["rank_demanda"] = pd.Series(local).rank(ascending=False).to_numpy()
    for col in OSM_FEATURES:  # só existem quando a tabela foi montada com o OpenStreetMap
        if col in base:
            out[col] = base[col].to_numpy()
    return out


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Área sob a curva ROC (Mann–Whitney), sem depender de bibliotecas extras."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score)[:k]
    return float(y[order].mean())


def out_of_fold_scores(
    data: pd.DataFrame,
    model: str = "logit",
    seed: int = 0,
    features: list[str] | None = None,
) -> np.ndarray:
    """Probabilidade de abrir, prevista sem a macrorregião do ponto (leave-one-region-out)."""
    cols = features or ALL_FEATURES
    scores = np.zeros(len(data))
    for region in data["regiao"].unique():
        test = (data["regiao"] == region).to_numpy()
        train = ~test
        if len(np.unique(data.loc[train, "aberto"])) < 2:
            scores[test] = float(data.loc[train, "aberto"].mean())
            continue
        if model == "logit":
            est = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        else:
            est = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.08, max_iter=120, random_state=seed
            )
        est.fit(data.loc[train, cols], data.loc[train, "aberto"])
        scores[test] = est.predict_proba(data.loc[test, cols])[:, 1]
    return scores
