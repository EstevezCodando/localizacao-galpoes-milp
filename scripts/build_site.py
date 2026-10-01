"""Gera `docs/index.html` a partir dos CSVs em `results/` (nenhum número é digitado à mão).

Uso:  uv run python scripts/build_site.py [--with-map]
`--with-map` resolve de novo a instância 100x30 (2 x 15 s) para desenhar o mapa.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd

from alocacao_capacitada.validation.classify import classify_methods

ROOT = Path(__file__).parents[1]
RES = ROOT / "results"
REF = ROOT / "data" / "reference"


def holmberg_block() -> dict[str, object]:
    out: dict[str, object] = {}
    for key, file in (("v1", "validacao_holmberg_free12.csv"), ("v2", "validacao_holmberg.csv")):
        t = classify_methods(pd.read_csv(RES / file), ("guloso", "busca_local", "lns", "milp"))
        stats = {}
        for method in ("guloso", "busca_local", "lns", "milp"):
            gap = t[f"{method}_gap_vs_otimo"]
            stats[method] = {
                "media": float(gap.mean()),
                "mediana": float(gap.median()),
                "max": float(gap.max()),
                "coincide_ref": int(t[f"{method}_coincide_ref"].sum()),
                "dentro_0_01pct": int(t[f"{method}_dentro_0_01pct"].sum()),
                "certificado": int(t[f"{method}_certificado"].sum()),
                "abaixo_da_ref": int(t[f"{method}_abaixo_da_referencia"].sum()),
                "tempo_medio": float(t[f"{method}_tempo_s"].mean()),
            }
        out[key] = {"n": len(t), "stats": stats}
    t = pd.read_csv(RES / "validacao_holmberg.csv")
    t["grupo"] = pd.cut(
        t["n"], [0, 50, 100, 150, 200], labels=["50", "51–100", "101–150", "151–200"]
    )
    by_group = t.groupby("grupo", observed=True)[
        ["busca_local_gap_vs_otimo", "lns_gap_vs_otimo", "milp_gap_vs_otimo", "gap_pl_vs_otimo"]
    ].mean()
    out["por_tamanho"] = {
        str(k): {
            "busca_local": float(r.iloc[0]),
            "lns": float(r.iloc[1]),
            "milp": float(r.iloc[2]),
            "pl": float(r.iloc[3]),
        }
        for k, r in by_group.iterrows()
    }
    out["pontos"] = [
        {
            "i": str(r.instancia),
            "n": int(r.n),
            "lns": float(r.lns_gap_vs_otimo),
            "milp": float(r.milp_gap_vs_otimo),
        }
        for r in t.itertuples()
    ]
    return out


def _cost_of(group: pd.DataFrame, method: str) -> float:
    return float(group[group["metodo"] == method]["custo_total"].iloc[0])


def olist_block() -> list[dict[str, object]]:
    small = pd.read_csv(RES / "stage2_small.csv")
    big = pd.read_csv(RES / "stage2_200x50.csv")
    rows = []
    for table, limit in ((small, 30), (big, 60)):
        for size, g in table.groupby("instancia"):
            lns = g[g["metodo"].str.startswith("lns")]["custo_total"]
            pick = partial(_cost_of, g)
            rows.append(
                {
                    "instancia": str(size),
                    "limite_s": limit,
                    "guloso": pick("guloso_mais_proximo"),
                    "busca_local": pick("busca_local(guloso_mais_proximo)"),
                    "milp": pick("milp_scip"),
                    "milp_warm": pick("milp_scip+busca_local(guloso_mais_proximo)"),
                    "lns_medio": float(lns.mean()),
                    "lns_min": float(lns.min()),
                    "lns_max": float(lns.max()),
                    "sementes": int(len(lns)),
                    "limite_pl": float(g["limite_pl"].iloc[0]),
                }
            )
    return sorted(rows, key=lambda r: int(str(r["instancia"]).split("x")[0]))


_UF_POR_CODIGO = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}  # fmt: skip


def states_geojson() -> dict[str, object]:
    """Malha dos estados do IBGE (bronze) com sigla da UF e coordenadas em 3 casas."""
    raw = json.loads((ROOT / "data" / "bronze" / "geo" / "ibge_uf_malha.geojson").read_text())

    def rnd(coords: object) -> object:
        if isinstance(coords[0], (int, float)):  # type: ignore[index]
            return [round(coords[0], 3), round(coords[1], 3)]  # type: ignore[index]
        return [rnd(c) for c in coords]  # type: ignore[attr-defined]

    features = []
    for f in raw["features"]:
        uf = _UF_POR_CODIGO[f["properties"]["codarea"]]
        geometry = {"type": f["geometry"]["type"], "coordinates": rnd(f["geometry"]["coordinates"])}
        features.append({"type": "Feature", "properties": {"uf": uf}, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def _csv(name: str) -> list[dict[str, object]]:
    path = RES / name
    return pd.read_csv(path).to_dict("records") if path.exists() else []  # type: ignore[return-value]


def national_map() -> dict[str, object] | None:
    path = RES / "rede_mapa.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    focus = RES / "foco" / "rede_mapa.json"
    if focus.exists():  # vistas da rede restrita ao Sul, Sudeste e Centro-Oeste
        for v in json.loads(focus.read_text(encoding="utf-8"))["vistas"]:
            payload["vistas"].append({**v, "nome": f"S+SE+CO: {v['nome']}"})
    return payload


def ml_block() -> dict[str, object]:
    """Resultados de demanda, previsão, clusterização, score e rede nacional (todos de results/)."""
    proj = pd.read_parquet(RES / "ml_populacao_projecao.parquet")
    labels = pd.read_parquet(RES / "ml_cluster_rotulos.parquet")
    merged = proj.merge(labels, on="cod")
    curve = {str(y): float(proj[f"pop_{y}"].sum()) for y in range(2025, 2031)}
    curve_lo = {"2030": float(proj["pop_2030_lo"].sum())}
    curve_hi = {"2030": float(proj["pop_2030_hi"].sum())}
    by_group = (
        merged.groupby("grupo")[["pop_2025", "pop_2030"]].sum().reset_index().to_dict("records")
    )
    incert = []
    for f in sorted(RES.glob("incerteza_folga*.csv")):
        incert.extend(pd.read_csv(f).to_dict("records"))
    return {
        "demanda_metricas": _csv("ml_demanda_metricas.csv"),
        "demanda_calibracao_gbm": _csv("ml_demanda_calibracao_gbm_uf.csv"),
        "demanda_calibracao_base": _csv("ml_demanda_calibracao_base_uf.csv"),
        "demanda_importancia": _csv("ml_demanda_importancia.csv"),
        "previsao_censo": _csv("ml_previsao_censo.csv"),
        "previsao_censo_cv": _csv("ml_previsao_censo_cv_espacial.csv"),
        "previsao_teste": _csv("ml_previsao_teste.csv"),
        "previsao_cobertura": _csv("ml_previsao_cobertura.csv"),
        "projecao_curva": curve,
        "projecao_lo": curve_lo,
        "projecao_hi": curve_hi,
        "projecao_grupos": by_group,
        "cluster_k": _csv("ml_cluster_escolha_k.csv"),
        "cluster_perfil": _csv("ml_cluster_perfil.csv"),
        "score_metricas": _csv("rede_score_metricas.csv"),
        "score_decisao": _csv("rede_score_decisao.csv"),
        "score_sementes": _csv("rede_score_decisao_sementes.csv"),
        "rede_cenarios": _csv("rede_cenarios.csv"),
        "rede_previsao": _csv("rede_previsao_decisao.csv"),
        "rede_reta": _csv("rede_reta_vs_estrada.csv"),
        "incerteza": incert,
        "sens_v2": _csv("sensibilidade.csv"),
    }


def main() -> None:

    from alocacao_capacitada.data.olist import load_geo_nodes

    geo = load_geo_nodes(ROOT / "data" / "raw")
    freight = pd.read_csv(REF / "antt_tabela_a.csv")
    freight = freight[freight["tipo_carga"] == "carga_geral"]
    cities = pd.read_csv(REF / "aluguel_galpao_cidades.csv")
    rent = cities.groupby("uf")["medio_1001_3000"].agg(["mean", "count"]).reset_index()
    orlib = pd.read_csv(RES / "validacao_orlib.csv")
    data = {
        "meta": {
            "pedidos": int(geo.demand["orders"].sum()),
            "regioes_demanda": len(geo.demand),
            "regioes_vendedor": len(geo.supply),
        },
        "holmberg": holmberg_block(),
        "olist": olist_block(),
        "orlib": [
            {
                "i": r.instancia,
                "publicado": float(r.otimo_publicado_divisivel),
                "meu": float(r.meu_modelo_divisivel),
            }
            for r in orlib.itertuples()
        ],
        "frete": [
            {"eixos": int(r.eixos), "ccd": float(r.ccd_rs_km), "cc": float(r.cc_rs)}
            for r in freight.itertuples()
        ],
        "aluguel_estado": [
            {"uf": r.uf, "rs_m2": float(r.mean), "n": int(r.count)} for r in rent.itertuples()
        ],
        "aluguel_cidades": cities[["cidade", "uf", "medio_1001_3000"]].to_dict("records"),
        "sensibilidade": pd.read_csv(RES / "sensibilidade.csv").to_dict("records"),
        "aluguel_real": pd.read_csv(RES / "aluguel_real.csv").to_dict("records"),
        "area_fixa": pd.read_csv(RES / "aluguel_area_fixa.csv").to_dict("records"),
        "mapa": national_map(),
        "ml": ml_block(),
        "estados": states_geojson(),
    }
    payload = json.dumps(data, default=float, ensure_ascii=False)
    template = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    css = (ROOT / "site" / "vendor" / "maplibre-gl.css").read_text(encoding="utf-8")
    mapjs = (ROOT / "site" / "map.js").read_text(encoding="utf-8")
    mljs = (ROOT / "site" / "ml.js").read_text(encoding="utf-8")
    sections = (ROOT / "site" / "sections_ml.html").read_text(encoding="utf-8")
    page = (
        template.replace("/*__DATA__*/null", payload)
        .replace("/*__MAPLIBRE_CSS__*/", css)
        .replace("/*__MAPJS__*/", mapjs)
        .replace("/*__MLJS__*/", mljs)
        .replace("<!--__ML_SECTIONS__-->", sections)
    )
    out.write_text(page, encoding="utf-8")
    print(f"escrito {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
