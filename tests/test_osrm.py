import json
from pathlib import Path

import pytest

from alocacao_capacitada.lake.osrm import OsrmClient


def _payload(code: str = "Ok") -> bytes:
    route = {
        "distance": 433802.3,
        "duration": 20086.4,
        "geometry": {"coordinates": [[-46.6333, -23.5505], [-43.1729, -22.9068]]},
    }
    return json.dumps({"code": code, "routes": [route] if code == "Ok" else []}).encode()


def test_rota_converte_unidades_e_usa_cache(tmp_path: Path) -> None:
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return _payload()

    client = OsrmClient(tmp_path / "c.json", fetch=fetch, min_interval_s=0)
    a, b = (-23.55, -46.63), (-22.91, -43.17)
    first = client.route(a, b)
    again = client.route(a, b)
    assert first.distance_km == pytest.approx(433.8023)
    assert first.duration_h == pytest.approx(5.5796, abs=1e-3)
    assert again == first and len(urls) == 1
    assert "-46.6300,-23.5500;-43.1700,-22.9100" in urls[0]  # OSRM espera lon,lat


def test_cache_persiste_entre_instancias(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    OsrmClient(path, fetch=lambda u: _payload(), min_interval_s=0).route((0.0, 1.0), (2.0, 3.0))

    def proibido(url: str) -> bytes:
        raise AssertionError("não deveria pedir de novo")

    client = OsrmClient(path, fetch=proibido, min_interval_s=0)
    assert client.route((0.0, 1.0), (2.0, 3.0)).distance_km > 0
    assert client.requests_made == 0


def test_sem_rota_falha_alto(tmp_path: Path) -> None:
    client = OsrmClient(tmp_path / "c.json", fetch=lambda u: _payload("NoRoute"), min_interval_s=0)
    with pytest.raises(RuntimeError):
        client.route((0.0, 0.0), (1.0, 1.0))
