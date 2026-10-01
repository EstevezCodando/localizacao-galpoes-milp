import json
from pathlib import Path

import pytest

from alocacao_capacitada.lake.api import ApiCollector


def test_cache_evita_nova_requisicao_e_grava_proveniencia(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return b'{"a": 1}'

    api = ApiCollector(tmp_path, fetch=fetch, min_interval_s=0)
    assert api.get_json("x", "https://exemplo/api") == {"a": 1}
    assert api.get_json("x", "https://exemplo/api") == {"a": 1}
    assert len(calls) == 1
    prov = json.loads((tmp_path / "x.provenance.json").read_text(encoding="utf-8"))
    assert prov["url"] == "https://exemplo/api" and len(prov["sha256"]) == 64
    assert prov["robots_checked"] is False and prov["nota"]


def test_arquivo_alterado_e_detectado_pelo_hash(tmp_path: Path) -> None:
    api = ApiCollector(tmp_path, fetch=lambda u: b"[1, 2]", min_interval_s=0)
    api.get_json("y", "https://exemplo/y")
    (tmp_path / "y.json").write_bytes(b"[1, 3]")
    with pytest.raises(ValueError, match="hash"):
        api.get_json("y", "https://exemplo/y")


def test_resposta_que_nao_e_json_falha_alto(tmp_path: Path) -> None:
    api = ApiCollector(tmp_path, fetch=lambda u: b"<html>erro</html>", min_interval_s=0)
    with pytest.raises(ValueError):
        api.get_json("z", "https://exemplo/z")
    assert not (tmp_path / "z.json").exists()


def test_resposta_gzip_e_descomprimida() -> None:
    import gzip

    from alocacao_capacitada.lake import api as api_module

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return gzip.compress(b'{"ok": true}')

    original = api_module.urllib.request.urlopen
    api_module.urllib.request.urlopen = lambda *a, **k: FakeResponse()  # type: ignore[assignment]
    try:
        assert api_module.urllib_fetch("https://exemplo") == b'{"ok": true}'
    finally:
        api_module.urllib.request.urlopen = original


def test_repete_falha_de_rede_e_nao_repete_erro_4xx(tmp_path: Path) -> None:
    import urllib.error

    calls = {"n": 0}

    def flaky(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("timeout")
        return b"[1]"

    api = ApiCollector(tmp_path, fetch=flaky, min_interval_s=0)
    assert api.get_json("a", "https://exemplo/a") == [1]
    assert calls["n"] == 3

    def bad_request(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)  # type: ignore[arg-type]

    api2 = ApiCollector(tmp_path / "b", fetch=bad_request, min_interval_s=0)
    with pytest.raises(urllib.error.HTTPError):
        api2.get_json("b", "https://exemplo/b")
