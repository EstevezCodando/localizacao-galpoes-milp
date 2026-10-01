"""Rotas viárias reais via servidor público de demonstração do OSRM, com cache e uso polido.

O servidor de demonstração é um serviço compartilhado: limitamos a 1 requisição a cada
1,1 s, usamos user-agent identificado e guardamos tudo em cache (uma rota nunca é pedida
duas vezes). Serve para VISUALIZAR; o custo do modelo continua em linha reta.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://router.project-osrm.org"
USER_AGENT = "alocacao-capacitada/0.1 (pesquisa academica; jean.alvarez@al.infnet.edu.br)"
MIN_INTERVAL_S = 1.1

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Route:
    distance_km: float
    duration_h: float
    coords: list[list[float]]  # [[lon, lat], ...], geometria simplificada


def urllib_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (https fixo)
        data: bytes = response.read()
    return data


class OsrmClient:
    def __init__(
        self,
        cache_path: Path,
        fetch: Fetch = urllib_fetch,
        min_interval_s: float = MIN_INTERVAL_S,
        base_url: str = BASE_URL,
    ) -> None:
        self._cache_path = cache_path
        self._fetch = fetch
        self._min_interval_s = min_interval_s
        self._base_url = base_url
        self._last = float("-inf")
        self._cache: dict[str, dict[str, object]] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        )
        self.requests_made = 0

    @staticmethod
    def _key(origin: tuple[float, float], dest: tuple[float, float]) -> str:
        return f"{origin[1]:.4f},{origin[0]:.4f};{dest[1]:.4f},{dest[0]:.4f}"

    def route(self, origin: tuple[float, float], dest: tuple[float, float]) -> Route:
        """`origin` e `dest` em (lat, lon)."""
        key = self._key(origin, dest)
        if key not in self._cache:
            self._cache[key] = self._request(key)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        item = self._cache[key]
        return Route(
            float(item["distance_km"]),  # type: ignore[arg-type]
            float(item["duration_h"]),  # type: ignore[arg-type]
            item["coords"],  # type: ignore[arg-type]
        )

    def _request(self, key: str) -> dict[str, object]:
        wait = self._min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = f"{self._base_url}/route/v1/driving/{key}?overview=simplified&geometries=geojson"
        payload = json.loads(self._fetch(url))
        self._last = time.monotonic()
        self.requests_made += 1
        if payload.get("code") != "Ok" or not payload.get("routes"):
            raise RuntimeError(f"OSRM não encontrou rota para {key}: {payload.get('code')}")
        best = payload["routes"][0]
        coords = [[round(lon, 4), round(lat, 4)] for lon, lat in best["geometry"]["coordinates"]]
        return {
            "distance_km": best["distance"] / 1000.0,
            "duration_h": best["duration"] / 3600.0,
            "coords": coords,
        }
