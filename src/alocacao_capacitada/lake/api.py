"""Coletor de APIs públicas de dados abertos (IBGE SIDRA, Ipeadata, malhas), com cache e origem.

Diferente do coletor de páginas (`bronze.py`), aqui os endpoints são APIs feitas para acesso
programático. Mesmo assim: user-agent identificado, intervalo mínimo entre requisições, cache em
disco (uma URL nunca é pedida duas vezes) e um registro de proveniência por arquivo. O `robots.txt`
desses hosts não responde ou não se aplica a APIs; isso fica registrado em cada proveniência, em
vez de ser omitido.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "alocacao-capacitada/0.1 (pesquisa academica; jean.alvarez@al.infnet.edu.br)"
MIN_INTERVAL_S = 1.0
ROBOTS_NOTE = (
    "API pública de dados abertos para acesso programático; o robots.txt do host não foi "
    "consultado ou não responde. Uma requisição por vez, com intervalo e cache."
)

Fetch = Callable[[str], bytes]


def urllib_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (https/http fixos)
        data: bytes = response.read()
    # Alguns servidores comprimem mesmo sem Accept-Encoding: descomprime pelo número mágico.
    return gzip.decompress(data) if data[:2] == bytes((0x1F, 0x8B)) else data


class ApiCollector:
    def __init__(
        self, out_dir: Path, fetch: Fetch = urllib_fetch, min_interval_s: float = MIN_INTERVAL_S
    ) -> None:
        self._out_dir = out_dir
        self._fetch = fetch
        self._min_interval_s = min_interval_s
        self._last = float("-inf")
        self.requests_made = 0

    def _fetch_with_retry(self, url: str, attempts: int = 4) -> bytes:
        """Repete falhas de rede (timeout, 5xx) com espera crescente; erros 4xx falham logo."""
        for attempt in range(1, attempts + 1):
            try:
                return self._fetch(url)
            except urllib.error.HTTPError as err:
                if err.code < 500 or attempt == attempts:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == attempts:
                    raise
            time.sleep(min(60.0, 5.0 * attempt**2) if self._min_interval_s > 0 else 0.0)
        raise RuntimeError("inalcançável")

    def get_json(self, name: str, url: str) -> Any:
        """JSON da URL; usa o arquivo em cache se existir e confere o hash da proveniência."""
        raw = self._out_dir / f"{name}.json"
        prov = self._out_dir / f"{name}.provenance.json"
        if raw.exists() and prov.exists():
            body = raw.read_bytes()
            expected = json.loads(prov.read_text(encoding="utf-8"))["sha256"]
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError(f"{raw.name}: hash diferente da proveniência (arquivo alterado)")
            return json.loads(body)
        wait = self._min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        body = self._fetch_with_retry(url)
        self._last = time.monotonic()
        self.requests_made += 1
        data = json.loads(body)  # falha alto se a resposta não for JSON
        self._out_dir.mkdir(parents=True, exist_ok=True)
        tmp = raw.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(raw)  # escrita atômica
        prov.write_text(
            json.dumps(
                {
                    "url": url,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "user_agent": USER_AGENT,
                    "robots_checked": False,
                    "nota": ROBOTS_NOTE,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return data
