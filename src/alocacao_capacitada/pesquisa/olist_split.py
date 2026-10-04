"""Estudo de caso com dados reais: instâncias Olist no contrato estrito (atendimento obrigatório).

Geografia e volumes são REAIS (pedidos entregues do Olist, regiões por prefixo de CEP de 3
dígitos); frete pelo piso da ANTT (2 eixos, 50 pedidos por veículo), como no projeto principal.
Custo total do par = demanda_j × custo unitário_ij. Capacidade e custo fixo são as premissas
declaradas em `InstanceConfig` (folga de capacidade e custo fixo por pedido).

Fonte única exige que cada região caiba inteira em algum centro. Com capacidade proporcional ao
volume histórico, uma região muito grande pode não caber: a instância é testada por
`viavel_fonte_unica` e, se inviável, a folga é aumentada em passos de 0,25 até viabilizar (o valor
final é registrado em `meta`). A referência de cada instância é o melhor custo encontrado
(SCIP + LNS), com o bound do SCIP.

uv run python -m alocacao_capacitada.pesquisa.olist_split
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from alocacao_capacitada.data.instance_builder import InstanceConfig, build_instance
from alocacao_capacitada.data.olist import load_geo_nodes
from alocacao_capacitada.lake.freight import load_freight_model
from alocacao_capacitada.pesquisa.dados import RAIZ, rotular
from alocacao_capacitada.pesquisa.geradores import viavel_fonte_unica
from alocacao_capacitada.pesquisa.problema import Problema

TAMANHOS = ((150, 30), (300, 60), (600, 100), (850, 150))


def construir(nodes: object, freight: object, n: int, m: int, folga: float) -> Problema:
    inst = build_instance(nodes, InstanceConfig(n_demand=n, n_facilities=m, freight=freight,
                                                capacity_slack=folga))
    cost = inst.unit_cost * inst.demand[None, :]
    return Problema(nome=f"olist_n{inst.n_demand}_m{inst.n_facilities}_f{folga}",
                    familia="olist", fixed=inst.fixed_cost, capacity=inst.capacity,
                    demand=inst.demand, cost=cost,
                    meta={"razao": float(inst.capacity.sum() / inst.demand.sum()),
                          "folga_config": folga, "seed": 0})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--t-scip", type=float, default=300)
    ap.add_argument("--t-lns", type=float, default=120)
    a = ap.parse_args()
    nodes = load_geo_nodes(a.raw_dir)
    freight = load_freight_model(Path("data/reference/antt_tabela_a.csv"), axles=2,
                                 orders_per_vehicle=50)
    out = RAIZ / "olist"
    out.mkdir(parents=True, exist_ok=True)
    for n, m in TAMANHOS:
        folga = 1.5
        while True:
            p = construir(nodes, freight, n, m, folga)
            ok = viavel_fonte_unica(p.demand, p.capacity, tempo_s=30)
            if ok:
                break
            folga += 0.25
            if folga > 4:
                raise RuntimeError(f"olist {n}x{m}: inviável mesmo com folga 4")
        arq = out / f"{p.nome}.pkl"
        if arq.exists():
            print("pulado", arq.name)
            continue
        rot = rotular(p, a.t_scip, a.t_lns)
        with open(arq, "wb") as fh:
            pickle.dump(rot, fh)
        print(p.nome, f"razao={p.razao_capacidade:.2f}", rot.qualidade,
              f"gap={rot.gap:.4f}", f"{rot.t_rotulo:.0f}s", flush=True)


if __name__ == "__main__":
    main()
