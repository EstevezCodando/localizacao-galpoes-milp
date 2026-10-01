"""Matriz de distâncias viárias (km) e tempos (h) via serviço `table` do OSRM, em blocos.

O servidor público limita o número de coordenadas por consulta (100 funcionou nos testes); a matriz
é montada em blocos de até 50 origens × 50 destinos, com intervalo entre consultas e cache em disco.
Pares sem rota (null no OSRM, por exemplo ilhas) recebem a distância em linha reta × o desvio
mediano e ficam marcados em `imputado`.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alocacao_capacitada.geo import haversine_matrix
from alocacao_capacitada.lake.osrm import BASE_URL, MIN_INTERVAL_S, USER_AGENT

BLOCK = 50
DETOUR_FALLBACK = 1.29  # mediana viária/reta medida nas 100 rotas do mapa

Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (https fixo)
        data: bytes = response.read()
    return data


@dataclass(frozen=True)
class RoadMatrix:
    distance_km: np.ndarray  # (origens, destinos)
    duration_h: np.ndarray
    imputed: np.ndarray  # bool


def road_matrix(
    origins: np.ndarray,
    destinations: np.ndarray,
    cache_dir: Path,
    fetch: Fetch = _fetch,
    min_interval_s: float = MIN_INTERVAL_S,
    base_url: str = BASE_URL,
) -> RoadMatrix:
    """`origins` e `destinations` são arrays (n, 2) de (lat, lon)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_o, n_d = len(origins), len(destinations)
    dist = np.full((n_o, n_d), np.nan)
    dur = np.full((n_o, n_d), np.nan)
    last = float("-inf")
    for i0 in range(0, n_o, BLOCK):
        for j0 in range(0, n_d, BLOCK):
            o, d = origins[i0 : i0 + BLOCK], destinations[j0 : j0 + BLOCK]
            coords = np.vstack([o, d])
            loc = ";".join(f"{lon:.4f},{lat:.4f}" for lat, lon in coords)
            src = ";".join(map(str, range(len(o))))
            dst = ";".join(map(str, range(len(o), len(o) + len(d))))
            digest = hashlib.sha1(f"{loc}|{src}|{dst}".encode()).hexdigest()[:16]  # noqa: S324
            key = cache_dir / f"bloco_{digest}.json"
            if key.exists():
                payload = json.loads(key.read_text(encoding="utf-8"))
            else:
                wait = min_interval_s - (time.monotonic() - last)
                if wait > 0:
                    time.sleep(wait)
                url = (
                    f"{base_url}/table/v1/driving/{loc}?sources={src}&destinations={dst}"
                    "&annotations=distance,duration"
                )
                payload = json.loads(fetch(url))
                last = time.monotonic()
                if payload.get("code") != "Ok":
                    raise RuntimeError(f"OSRM table: {payload.get('code')}")
                key.write_text(json.dumps(payload), encoding="utf-8")
            dist[i0 : i0 + len(o), j0 : j0 + len(d)] = np.array(payload["distances"], dtype=float)
            dur[i0 : i0 + len(o), j0 : j0 + len(d)] = np.array(payload["durations"], dtype=float)
    missing = ~np.isfinite(dist)
    straight = haversine_matrix(
        origins[:, 0], origins[:, 1], destinations[:, 0], destinations[:, 1]
    )
    km = np.where(missing, straight * DETOUR_FALLBACK, dist / 1000.0)
    hours = np.where(missing, km / 60.0, dur / 3600.0)  # fallback: 60 km/h
    return RoadMatrix(km, hours, missing)
