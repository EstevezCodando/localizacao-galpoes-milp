"""Coletor polido: testado com fetch falso, sem rede."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from alocacao_capacitada.lake.bronze import PoliteCollector, RobotsDisallowedError


@dataclass
class FakePage:
    status: int
    body: bytes


class FakeFetch:
    def __init__(self, robots: str | None, content: bytes = b"<html>ok</html>") -> None:
        self.robots, self.content, self.calls = robots, content, []

    def __call__(self, url: str) -> FakePage:
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return (
                FakePage(404, b"") if self.robots is None else FakePage(200, self.robots.encode())
            )
        return FakePage(200, self.content)


def test_respeita_disallow(tmp_path: Path) -> None:
    fetch = FakeFetch("User-agent: *\nDisallow: /privado")
    collector = PoliteCollector(tmp_path, fetch, min_interval_s=0)
    with pytest.raises(RobotsDisallowedError):
        collector.collect("https://x.gov.br/privado/doc", "doc")
    assert fetch.calls == ["https://x.gov.br/robots.txt"]  # a página proibida nunca foi pedida


def test_coleta_grava_bruto_e_proveniencia(tmp_path: Path) -> None:
    collector = PoliteCollector(tmp_path, FakeFetch(None), min_interval_s=0)
    prov = collector.collect("https://x.gov.br/publico", "doc")
    assert (tmp_path / "doc.html").read_bytes() == b"<html>ok</html>"
    assert prov.robots_checked and prov.status == 200 and len(prov.sha256) == 64
    assert (tmp_path / "doc.provenance.json").exists()


def test_consulta_robots_uma_vez_por_host(tmp_path: Path) -> None:
    fetch = FakeFetch(None)
    collector = PoliteCollector(tmp_path, fetch, min_interval_s=0)
    collector.collect("https://x.gov.br/a", "a")
    collector.collect("https://x.gov.br/b", "b")
    assert sum(c.endswith("/robots.txt") for c in fetch.calls) == 1
