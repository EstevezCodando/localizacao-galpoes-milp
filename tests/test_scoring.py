import numpy as np
import pandas as pd
import pytest

from alocacao_capacitada.ml.scoring import auc, out_of_fold_scores, precision_at_k


def test_auc_perfeito_aleatorio_e_invertido() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(auc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_precisao_no_topo() -> None:
    y = np.array([1, 0, 1, 0, 0])
    assert precision_at_k(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1]), 2) == 0.5


def test_score_fora_da_amostra_nao_usa_a_propria_regiao() -> None:
    rng = np.random.default_rng(0)
    n = 200
    from alocacao_capacitada.ml.scoring import ALL_FEATURES

    data = pd.DataFrame(rng.normal(size=(n, len(ALL_FEATURES))), columns=ALL_FEATURES)
    data["regiao"] = rng.choice(["N", "S", "SE"], n)
    data["aberto"] = (data["log_demanda_local"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    scores = out_of_fold_scores(data)
    assert auc(data["aberto"].to_numpy(), scores) > 0.8  # sinal real é aprendido entre regiões
    # Mudar os rótulos de UMA região não altera o score previsto para essa mesma região.
    altered = data.copy()
    altered.loc[altered["regiao"] == "N", "aberto"] = (
        1 - altered.loc[altered["regiao"] == "N", "aberto"]
    )
    scores2 = out_of_fold_scores(altered)
    mask = (data["regiao"] == "N").to_numpy()
    assert np.allclose(scores[mask], scores2[mask])
