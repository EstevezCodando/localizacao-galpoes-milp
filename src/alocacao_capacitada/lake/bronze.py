"""Camada bronze: coleta respeitosa de páginas públicas com proveniência.

Regras (não configuráveis por design):
  * consulta o robots.txt do host e só baixa o que ele permitir ao nosso user-agent;
  * user-agent identificado (sem personificar navegador, sem proxy, sem referer forjado);
  * intervalo mínimo entre requisições ao mesmo host;
  * guarda o bytes originais + metadados (url, data, hash, status) ao lado.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

USER_AGENT = "alocacao-capacitada/0.1 (pesquisa academica; jean.alvarez@al.infnet.edu.br)"
MIN_INTERVAL_S = 5.0


class FetchedPage(Protocol):
    status: int
    body: bytes


class Fetch(Protocol):
    def __call__(self, url: str) -> FetchedPage: ...


class RobotsDisallowedError(PermissionError):
    """O robots.txt do host não permite baixar a URL."""


@dataclass(frozen=True)
class Provenance:
    url: str
    fetched_at: str
    status: int
    sha256: str
    bytes: int
    user_agent: str
    robots_checked: bool
    file: str


def cavuca_fetch(url: str) -> FetchedPage:
    """Busca via Cavuca (HTTP estático), sem disfarces. Dependência opcional."""
    try:
        from cavuca.fetchers import Fetcher
    except ImportError as exc:
        raise RuntimeError(
            "A coleta de páginas usa o Cavuca, que é opcional e não vem com este projeto. "
            "Instale-o à parte ou passe outra função `fetch` ao PoliteCollector."
        ) from exc

    page: Any = Fetcher.get(
        url,
        impersonate=None,
        stealthy_headers=False,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    return page  # type: ignore[no-any-return]


class PoliteCollector:
    def __init__(
        self, out_dir: Path, fetch: Fetch = cavuca_fetch, min_interval_s: float = MIN_INTERVAL_S
    ):
        self._out_dir = out_dir
        self._fetch = fetch
        self._min_interval_s = min_interval_s
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    def _robots_for(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            page = self._wait_and_fetch(origin, f"{origin}/robots.txt")
            parser = RobotFileParser()
            if page.status == 200:
                parser.parse(page.body.decode("utf-8", "replace").splitlines())
            else:  # sem robots.txt (404 etc.): convenção é permitir
                parser.parse([])
            self._robots[origin] = parser
        return self._robots[origin]

    def _wait_and_fetch(self, origin: str, url: str) -> FetchedPage:
        elapsed = time.monotonic() - self._last_request.get(origin, float("-inf"))
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)
        page = self._fetch(url)
        self._last_request[origin] = time.monotonic()
        return page

    def collect(self, url: str, name: str, suffix: str = ".html") -> Provenance:
        if not self._robots_for(url).can_fetch(USER_AGENT, url):
            raise RobotsDisallowedError(url)
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        page = self._wait_and_fetch(origin, url)
        digest = hashlib.sha256(page.body).hexdigest()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        raw = self._out_dir / f"{name}{suffix}"
        raw.write_bytes(page.body)
        prov = Provenance(
            url=url,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            status=page.status,
            sha256=digest,
            bytes=len(page.body),
            user_agent=USER_AGENT,
            robots_checked=True,
            file=raw.name,
        )
        (self._out_dir / f"{name}.provenance.json").write_text(
            json.dumps(asdict(prov), indent=2), encoding="utf-8"
        )
        return prov
