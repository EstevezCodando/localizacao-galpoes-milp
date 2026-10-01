import numpy as np
import pandas as pd
from shapely.geometry import box

from alocacao_capacitada.territory.geometry import locate


def test_ponto_em_poligono_e_fora_de_todos() -> None:
    polygons = pd.DataFrame({"cod": ["1", "2"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]})
    lat = np.array([0.5, 0.5, 5.0])
    lon = np.array([0.5, 1.5, 0.5])
    assert list(locate(polygons, lat, lon)) == ["1", "2", ""]
