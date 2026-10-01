import json
from pathlib import Path

import numpy as np

from alocacao_capacitada.lake.osrm_matrix import DETOUR_FALLBACK, road_matrix


def test_blocos_cache_e_pares_sem_rota(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        # 2 origens × 2 destinos; o par (1,1) não tem rota
        return json.dumps(
            {
                "code": "Ok",
                "distances": [[0.0, 100000.0], [200000.0, None]],
                "durations": [[0.0, 3600.0], [7200.0, None]],
            }
        ).encode()

    o = np.array([[-23.5, -46.6], [-22.9, -43.2]])
    d = np.array([[-23.5, -46.6], [-22.0, -43.0]])
    first = road_matrix(o, d, tmp_path, fetch=fetch, min_interval_s=0)
    again = road_matrix(o, d, tmp_path, fetch=fetch, min_interval_s=0)
    assert len(calls) == 1  # segunda chamada vem do cache
    assert first.distance_km[0, 1] == 100.0 and first.duration_h[1, 0] == 2.0
    assert first.imputed[1, 1] and not first.imputed[0, 0]
    assert first.distance_km[1, 1] > 0  # reta × desvio, nunca NaN
    assert np.array_equal(first.distance_km, again.distance_km)
    assert DETOUR_FALLBACK > 1
