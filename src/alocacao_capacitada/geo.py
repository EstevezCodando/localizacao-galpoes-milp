"""Utilitários geográficos."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(
    lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray
) -> np.ndarray:
    """Distância em km (grande círculo) entre todos os pares A×B, shape (len(A), len(B)).

    É uma aproximação: na logística real o custo depende da malha viária.
    O projeto de roteamento substituirá isto por tempos de viagem do OSRM.
    """
    phi_a, phi_b = np.radians(lat_a)[:, None], np.radians(lat_b)[None, :]
    dphi = phi_b - phi_a
    dlmb = np.radians(lon_b)[None, :] - np.radians(lon_a)[:, None]
    h = np.sin(dphi / 2) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(dlmb / 2) ** 2
    km: np.ndarray = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))
    return km
