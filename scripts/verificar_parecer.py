"""Verificações empíricas dos pontos do parecer acadêmico de 04/10/2026.

Só lê resultados existentes e resolve PLs nas instâncias de VALIDAÇÃO; não roda benchmarks nem
toca nas instâncias do teste fechado. Saída: results/pesquisa/parecer/verificacao.json.

uv run python scripts/verificar_parecer.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alocacao_capacitada.pesquisa import clns as C
from alocacao_capacitada.pesquisa import hibrido as H
from alocacao_capacitada.pesquisa.dados import RAIZ, ler
from alocacao_capacitada.pesquisa.exato import relaxacao_linear
from alocacao_capacitada.pesquisa.geradores import Config, gerar

RES = Path("results/pesquisa")
PILOTOS = {"piloto1": "81dbbcb06942419706f3", "piloto2": "eaa3f86d9312c0970ce9",
           "piloto3": "e1a72d506f53ea707aa1"}
out: dict[str, object] = {}


def carregar(run: str) -> pd.DataFrame:
    d = pd.read_csv(RES / "clns" / run / "resultados.csv")
    d["traj"] = d["trajetoria"].map(json.loads)
    d["valida"] = d["valida"].astype(bool)
    d["obj_T"] = [min((o for t, o in tr if t <= h + 1e-9), default=np.inf)
                  for tr, h in zip(d["traj"], d["orcamento"])]
    return d


# ---------------------------------------------------------------- 3.3 desvio final em T (pilotos)
p33 = {}
dados = {k: carregar(v) for k, v in PILOTOS.items()}
for nome, d in dados.items():
    mudou = d["valida"] & (d["objetivo"] < d["obj_T"] - 1e-9)
    bks = pd.concat([d[d["valida"]].groupby("instancia")["objetivo"].min(),
                     d.groupby("instancia")["rotulo_melhor"].first()], axis=1).min(axis=1)
    d["bks"] = d["instancia"].map(bks)
    d["dev_ret"] = np.where(d["valida"], d["objetivo"] / d["bks"] - 1, np.nan)
    d["dev_T"] = np.minimum(1.0, d["obj_T"] / d["bks"] - 1)
    a = d.groupby(["instancia", "metodo"])[["dev_ret", "dev_T"]].mean().groupby("metodo").median()
    dif = (a["dev_T"] - a["dev_ret"]).abs()
    p33[nome] = {"execucoes": int(len(d)), "melhoraram_apos_T": int(mudou.sum()),
                 "metodos_afetados": d.loc[mudou, "metodo"].value_counts().to_dict(),
                 "sem_solucao_ate_T": int((~np.isfinite(d["obj_T"])).sum()),
                 "invalidas": int((~d["valida"]).sum()),
                 "maior_mudanca_no_desvio_mediano_pp": float(100 * dif.max()),
                 "desvio_mediano_T_pct": {m: round(100 * v, 4) for m, v in a["dev_T"].items()},
                 "desvio_mediano_retorno_pct": {m: round(100 * v, 4) for m, v in a["dev_ret"].items()}}
out["3.3_desvio_final_em_T"] = p33

# ---------------------------------------------------------------- 4.11 BKS muda entre pilotos
def bks_de(d: pd.DataFrame) -> pd.Series:
    return pd.concat([d[d["valida"]].groupby("instancia")["objetivo"].min(),
                      d.groupby("instancia")["rotulo_melhor"].first()], axis=1).min(axis=1)


b = {k: bks_de(d) for k, d in dados.items()}
out["4.11_bks_entre_pilotos"] = {
    "mudou_1_para_2": int((np.abs(b["piloto1"] - b["piloto2"]) > 1e-6).sum()),
    "mudou_2_para_3": int((np.abs(b["piloto2"] - b["piloto3"]) > 1e-6).sum()),
    "instancias": int(len(b["piloto2"])),
    "queda_relativa_mediana_2_para_3_pct": float(100 * ((b["piloto2"] - b["piloto3"]) / b["piloto2"]).median())}

# ---------------------------------------------------------------- 4.9 curvas até alfa*T
ag = json.loads((RES / "clns" / PILOTOS["piloto2"] / "agregados.json").read_text())
cur = ag["anytime"]["metodos"]
out["4.9_curvas_piloto2_gap_pct"] = {
    f"t={t}s": {m: round(100 * cur[m][t], 3) for m in ("adaptativa:gnn", "hibrido:aprendido",
                                                     "hibrido:rotacao", "hibrido:gnn")}
    for t in (5, 15, 30)}

# ---------------------------------------------------------------- 3.8 relógio do clns:gnn
d2 = dados["piloto2"]
linhas = {}
for m in ("clns:gnn", "clns:rotacao", "clns:aprendido", "hibrido:gnn", "hibrido:rotacao", "adaptativa:gnn"):
    g = d2[d2["metodo"] == m]
    fim = g["traj"].map(lambda tr: max((t for t, _ in tr), default=np.nan))
    linhas[m] = {"t_total_mediano_s": float(g["t_total"].median()),
                 "estouro_mediano_s": float(g["estouro_s"].median()),
                 "estouro_max_s": float(g["estouro_s"].max()),
                 "ultimo_evento_max_s": float(fim.max())}
out["3.8_relogio_piloto2"] = linhas

# ---------------------------------------------------------------- 3.7 custos reduzidos assinados
p, _ = gerar(Config("clusters", 10, 30, 1.5), 0)
lp = relaxacao_linear(p)
diag = {"exemplo_parecer": {"min_rc_y": float(lp.rc_y.min()), "min_rc_x": float(lp.rc_x.min()),
                            "neg_y": int((lp.rc_y < -1e-9).sum()), "neg_x": int((lp.rc_x < -1e-9).sum())}}
arqs = sorted((RAIZ / "validacao").glob("*.pkl"))
neg_y = neg_x = tot_y = tot_x = 0
neg_y_so_em_y1 = True
aninh = {"instancias": 0, "K20_em_K40": 0, "K40_em_K60": 0}
aber = {"alvos": 0, "clientes_livres": 0, "alvo_entre_os_10_mais_baratos": 0, "alvos_sem_nenhum_cliente": 0}
grupos_tam: list[int] = []
cargas_inteiras = True
for arq in arqs:
    pr = ler(arq).prob
    cargas_inteiras &= bool(np.allclose(pr.demand, np.round(pr.demand, 6)))
    r = relaxacao_linear(pr)
    ny = r.rc_y < -1e-9
    neg_y += int(ny.sum())
    tot_y += pr.m
    neg_x += int((r.rc_x < -1e-9).sum())
    tot_x += pr.m * pr.n
    neg_y_so_em_y1 &= bool((r.y[ny] > 1 - 1e-6).all())
    # 4.6 aninhamento dos conjuntos após reparo (ranking do PL, sem o incumbente)
    sc = r.y + 1e-3 * (-r.rc_y / max(np.abs(r.rc_y).max(), 1e-9))
    ordem = np.argsort(-sc, kind="stable")
    ks = [set(H.reparar(pr, ordem, max(1, int(round(f * pr.m)))).tolist()) for f in (0.2, 0.4, 0.6)]
    aninh["instancias"] += 1
    aninh["K20_em_K40"] += int(ks[0] <= ks[1])
    aninh["K40_em_K60"] += int(ks[1] <= ks[2])
    # 4.8 operador abertura: o centro-alvo está entre os 10 mais baratos dos clientes liberados?
    a = C.inicial_pl(pr, r.x)
    fechados = np.setdiff1d(np.arange(pr.m), np.unique(a))
    alvo = fechados[np.argsort(-sc[fechados])][:4]  # mesmo critério do operador, com score do PL
    dez = np.argsort(pr.cost, axis=0)[:10]
    for i in alvo:
        livres = np.argsort(pr.cost[i])[:15]
        dentro = int(np.isin(i, dez[:, livres]).sum()) if False else int(sum(i in dez[:, j] for j in livres))
        aber["alvos"] += 1
        aber["clientes_livres"] += 15
        aber["alvo_entre_os_10_mais_baratos"] += dentro
        aber["alvos_sem_nenhum_cliente"] += int(dentro == 0)
    grupos_tam += [len(g) for g in C.Estado(pr, 15, 0, r.rc_x).gs]
diag["validacao_32"] = {"rc_y_negativos": neg_y, "de": tot_y, "rc_x_negativos": neg_x, "de_x": tot_x,
                        "todo_rc_y_negativo_tem_y_igual_a_1": bool(neg_y_so_em_y1)}
out["3.7_custos_reduzidos_assinados"] = diag
out["4.6_aninhamento_apos_reparo"] = aninh
out["4.8_operador_abertura"] = aber
out["3.5_cargas"] = {"demandas_representaveis_em_6_casas": bool(cargas_inteiras)}
t = np.array(grupos_tam)
out["4.8_tamanho_dos_grupos"] = {"n_grupos": int(t.size), "min": int(t.min()), "p25": float(np.percentile(t, 25)),
                                 "mediana": float(np.median(t)), "p75": float(np.percentile(t, 75)),
                                 "max": int(t.max())}

dest = RES / "parecer"
dest.mkdir(parents=True, exist_ok=True)
(dest / "verificacao.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
