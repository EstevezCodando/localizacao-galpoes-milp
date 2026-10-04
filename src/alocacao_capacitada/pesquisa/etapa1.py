"""Etapa 1 — Learning the Search Space (doc 08/10): treino, seleção na validação e teste fechado.

Fases (cada uma grava em results/pesquisa/etapa1/):
  treinar   treina LGBM, MLP e GNN (3 sementes) + GNN com atributos do PL; métricas de ranking
  rho       escolhe a fração de poda por ranqueador NA VALIDAÇÃO (critério de recall, sem custo)
  testar    roda todas as políticas no split pedido com orçamento T e grava uma linha por
            (instância, método) com custo validado, tempo online e trajetória

uv run python -m alocacao_capacitada.pesquisa.etapa1 treinar
uv run python -m alocacao_capacitada.pesquisa.etapa1 rho
uv run python -m alocacao_capacitada.pesquisa.etapa1 testar --split teste --tempo 60 --workers 3
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from alocacao_capacitada.pesquisa import hibrido as H
from alocacao_capacitada.pesquisa import tempos
from alocacao_capacitada.pesquisa.aprendizado import (
    rotulo_pool,
    treinar_gnn,
    treinar_lgbm,
    treinar_mlp,
)
from alocacao_capacitada.pesquisa.atributos import grafo, razao_servico
from alocacao_capacitada.pesquisa.dados import Rotulado, carregar, ler
from alocacao_capacitada.pesquisa.exato import relaxacao_linear, resolver
from alocacao_capacitada.pesquisa.heuristicas import lns
from alocacao_capacitada.pesquisa.problema import Problema

OUT = Path("results/pesquisa/etapa1")
MODELOS = Path("data/processed/pesquisa/modelos")
GRADE_RHO = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# ---------------------------------------------------------------- ranqueadores (online)
class Ranqueadores:
    """Carrega os modelos treinados; cada score devolve (score, tempo online gasto)."""

    def __init__(self) -> None:
        with open(MODELOS / "ranqueadores.pkl", "rb") as fh:
            self.m = pickle.load(fh)
        # aquecimento único por processo (import de scipy/torch não é custo por instância)
        from alocacao_capacitada.pesquisa.geradores import Config, gerar

        p, _ = gerar(Config("uniforme", 6, 12, 2.0), 0)
        for nome in ("classico", "lgbm", "mlp", "gnn", "pl", "gnn_pl"):
            self.score(nome, p)

    def score(self, nome: str, prob: Problema, seed: int = 0,
              lp: object | None = None, tempo_s: float | None = None,
              ) -> tuple[np.ndarray, float, object | None]:
        t0 = time.perf_counter()
        if nome == "aleatorio":
            s = np.random.default_rng(seed).random(prob.m)
        elif nome == "classico":
            s = -razao_servico(prob)
        elif nome == "pl":
            lp = relaxacao_linear(prob, tempo_s=tempo_s)
            s = lp.y + 1e-3 * (-lp.rc_y / max(np.abs(lp.rc_y).max(), 1e-9))  # type: ignore[attr-defined]
        elif nome in ("lgbm", "mlp"):
            s = self.m[nome].score(grafo(prob))
        elif nome == "gnn":
            s = self.m["gnn"][seed].score(grafo(prob))
        elif nome == "gnn_pl":
            lp = relaxacao_linear(prob, tempo_s=tempo_s)
            s = self.m["gnn_pl"].score(grafo(prob, lp=lp))  # type: ignore[arg-type]
        else:
            raise ValueError(nome)
        return np.asarray(s, dtype=float), time.perf_counter() - t0, lp


# ---------------------------------------------------------------- treino
def _rotulos(rs: list[Rotulado]) -> list[np.ndarray]:
    return [rotulo_pool(r.pool, r.prob.m, tol=0.01) for r in rs]


def _metricas_ranking(scores: list[np.ndarray], rs: list[Rotulado]) -> dict[str, float]:
    aucs, rec = [], []
    for s, r in zip(scores, rs):
        best = np.zeros(r.prob.m, dtype=bool)
        best[np.unique(r.melhor_atrib)] = True
        if 0 < best.sum() < r.prob.m:
            aucs.append(roc_auc_score(best, s))
        k = int(best.sum())
        rec.append(best[np.argsort(-s)[:k]].mean())  # recall@|abertos|
    return {"auc": float(np.mean(aucs)), "recall_at_k": float(np.mean(rec))}


def treinar() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)
    tr, va = carregar("treino"), carregar("validacao")
    ytr, yva = _rotulos(tr), _rotulos(va)
    gtr = [grafo(r.prob) for r in tr]
    gva = [grafo(r.prob) for r in va]
    modelos: dict[str, object] = {}
    linhas = []
    t0 = time.perf_counter()
    modelos["lgbm"] = treinar_lgbm(gtr, ytr)
    t_lgbm = time.perf_counter() - t0
    t0 = time.perf_counter()
    modelos["mlp"] = treinar_mlp(gtr, ytr, (gva, yva))
    t_mlp = time.perf_counter() - t0
    gnns, hist_all, t_gnn = [], {}, []
    for seed in range(3):
        t0 = time.perf_counter()
        g, hist = treinar_gnn(gtr, ytr, (gva, yva), seed=seed)
        t_gnn.append(time.perf_counter() - t0)
        gnns.append(g)
        hist_all[seed] = hist
    modelos["gnn"] = gnns
    lptr = [relaxacao_linear(r.prob) for r in tr]
    lpva = [relaxacao_linear(r.prob) for r in va]
    t0 = time.perf_counter()
    modelos["gnn_pl"], _ = treinar_gnn([grafo(r.prob, lp=lp) for r, lp in zip(tr, lptr)], ytr,
                                       ([grafo(r.prob, lp=lp) for r, lp in zip(va, lpva)], yva))
    t_gnn_pl = time.perf_counter() - t0
    with open(MODELOS / "ranqueadores.pkl", "wb") as fh:
        pickle.dump(modelos, fh)
    # métricas de ranking na validação (o teste continua fechado)
    sc = {
        "aleatorio": [np.random.default_rng(0).random(r.prob.m) for r in va],
        "classico": [-razao_servico(r.prob) for r in va],
        "pl": [lp.y for lp in lpva],
        "lgbm": [modelos["lgbm"].score(g) for g in gva],  # type: ignore[attr-defined]
        "mlp": [modelos["mlp"].score(g) for g in gva],  # type: ignore[attr-defined]
        "gnn_pl": [modelos["gnn_pl"].score(grafo(r.prob, lp=lp))  # type: ignore[attr-defined]
                   for r, lp in zip(va, lpva)],
    }
    for seed, g in enumerate(gnns):
        sc[f"gnn_s{seed}"] = [g.score(x) for x in gva]
    for nome, s in sc.items():
        linhas.append({"ranqueador": nome, **_metricas_ranking(s, va)})
    pd.DataFrame(linhas).to_csv(OUT / "ranking_validacao.csv", index=False)
    custo = {"n_treino": len(tr), "t_rotulos_h": sum(r.t_rotulo for r in tr) / 3600,
             "t_lgbm_s": t_lgbm, "t_mlp_s": t_mlp, "t_gnn_s": t_gnn, "t_gnn_pl_s": t_gnn_pl,
             "epocas_gnn": {k: len(v) for k, v in hist_all.items()},
             "qualidade_rotulos": pd.Series([r.qualidade for r in tr]).value_counts().to_dict(),
             "gap_medio_rotulos": float(np.nanmean(
                 [r.gap if r.gap is not None else np.nan for r in tr]))}
    (OUT / "custo_offline.json").write_text(json.dumps(custo, indent=2))
    pd.DataFrame([(s, e, a, b) for s, h in hist_all.items() for e, a, b in h],
                 columns=["seed", "epoca", "loss_treino", "loss_val"]).to_csv(
        OUT / "curvas_gnn.csv", index=False)
    print(pd.DataFrame(linhas).to_string(), "\n", json.dumps(custo, indent=1))


# ---------------------------------------------------------------- escolha de rho
def escolher_rho(meta: float = 0.8) -> None:
    """Menor rho da grade tal que, em >= `meta` das instâncias de validação, o reparo mantém
    TODOS os centros da melhor solução conhecida (o reduzido contém essa solução)."""
    va = carregar("validacao")
    rk = Ranqueadores()
    linhas, escolha = [], {}
    for nome in ("aleatorio", "classico", "pl", "lgbm", "mlp", "gnn", "gnn_pl"):
        scores = [rk.score(nome, r.prob)[0] for r in va]
        for rho in GRADE_RHO:
            ok, frac = [], []
            for r, s in zip(va, scores, strict=True):
                keep = H.reparar(r.prob, np.argsort(-s, kind="stable"),
                                 max(1, int(round(rho * r.prob.m))))
                ok.append(np.isin(np.unique(r.melhor_atrib), keep).all())
                frac.append(keep.size / r.prob.m)
            linhas.append({"ranqueador": nome, "rho": rho, "contem_melhor": np.mean(ok),
                           "fracao_efetiva": np.mean(frac)})
        cand = [lin for lin in linhas if lin["ranqueador"] == nome and lin["contem_melhor"] >= meta]
        escolha[nome] = min(cand, key=lambda lin: lin["rho"])["rho"] if cand else 1.0
    pd.DataFrame(linhas).to_csv(OUT / "rho_validacao.csv", index=False)
    (OUT / "rho_escolhido.json").write_text(json.dumps(escolha, indent=2))
    print(pd.DataFrame(linhas).pivot(index="rho", columns="ranqueador", values="contem_melhor"))
    print(escolha)


# ---------------------------------------------------------------- teste
METODOS_COMPLETOS = (
    "completo", "warm_classico", "lns", "reducao_classica",
    "topk:aleatorio", "topk:classico", "topk:pl", "topk:lgbm", "topk:mlp", "topk:gnn",
    "topk:gnn_pl", "adaptativa:gnn", "adaptativa:classico", "warm:gnn", "prioridade:gnn",
    "topk@0.5:gnn", "topk@0.5:classico", "topk@0.5:pl",
)
METODOS_GEN = ("completo", "lns", "topk:classico", "topk:pl", "topk:gnn", "adaptativa:gnn",
               "warm:gnn")

_RK: Ranqueadores | None = None


def _rodar(args: tuple[str, str, float, str, int, int]) -> dict[str, object]:
    global _RK
    caminho, metodo, tempo, rho_json, seed_treino, seed_solver = args
    if _RK is None:
        _RK = Ranqueadores()
    rho = json.loads(rho_json)
    rot: Rotulado = ler(Path(caminho))
    p = rot.prob
    tempos.iniciar()
    t0 = time.perf_counter()
    deadline = t0 + tempo
    motivo = ""
    try:
        if metodo == "completo":
            s, _ = H.completo(p, deadline - time.perf_counter(), seed=seed_solver)
        elif metodo == "lns":
            s = lns(p, deadline - time.perf_counter(), seed=seed_solver)
        elif metodo in ("reducao_classica", "warm_classico"):
            lp = None
            if metodo == "reducao_classica":
                lp = relaxacao_linear(p, tempo_s=min(tempo * 0.2, deadline - time.perf_counter()))
            th = time.perf_counter() - t0
            h = lns(p, min(tempo * 0.1, deadline - time.perf_counter()), seed=seed_solver)
            up = (h.objetivo, h.atribuicao) if h.valida and h.atribuicao is not None else None
            pre = time.perf_counter() - t0
            if lp is not None:
                s = H.reducao_classica(p, tempo, lp, up, pre, seed=seed_solver)
            else:
                r = resolver(p, deadline - time.perf_counter(), dica=h.atribuicao,
                             seed=seed_solver)
                s = H.Saida("warm_classico", r.objetivo, r.valida,
                            time.perf_counter() - t0, pre, 2, 1., False,
                            r.status == "optimal", H._deslocar(r.trajetoria, pre),
                            r.nos, r.atribuicao)
            s.trajetoria = H._deslocar(h.trajetoria, th) + s.trajetoria
        else:
            pol, nome = metodo.split(":")
            score_seed = seed_solver if nome == "aleatorio" else max(seed_treino, 0)
            sc, t_sc, lp = _RK.score(nome, p, seed=score_seed,
                                     tempo_s=min(tempo * 0.2, deadline - time.perf_counter()))
            t_sc = time.perf_counter() - t0
            if pol == "topk":
                s = H.topk(p, sc, t_sc, rho[nome], tempo, seed=seed_solver)
            elif pol.startswith("topk@"):
                s = H.topk(p, sc, t_sc, float(pol[5:]), tempo, seed=seed_solver)
            elif pol == "adaptativa":
                if lp is None:
                    try:
                        lp = relaxacao_linear(p, tempo_s=min(tempo * 0.2,
                                                             deadline - time.perf_counter()))
                    except TimeoutError:
                        motivo = "PL sem certificado: expansão sem poda certificada"
                s = H.adaptativa(p, sc, time.perf_counter() - t0, tempo, lp, seed=seed_solver)
            elif pol == "warm":
                s = H.warm(p, sc, t_sc, tempo, rho=rho[nome], seed=seed_solver)
            elif pol == "prioridade":
                s = H.prioridade(p, sc, t_sc, tempo, seed=seed_solver)
            else:
                raise ValueError(metodo)
    except TimeoutError as exc:
        motivo = str(exc)
        pre = time.perf_counter() - t0
        r = resolver(p, deadline - time.perf_counter(), seed=seed_solver)
        s = H.Saida(metodo, r.objetivo, r.valida, time.perf_counter() - t0, pre,
                    1, 1., True, r.status == "optimal", H._deslocar(r.trajetoria, pre),
                    r.nos, r.atribuicao)
    total = time.perf_counter() - t0
    return {
        **tempos.finalizar(),
        "instancia": p.nome, "familia": p.familia, "razao": p.meta.get("razao"), "m": p.m,
        "n": p.n, "metodo": metodo, "objetivo": s.objetivo, "valida": s.valida,
        "seed_treino": seed_treino, "seed_solver": seed_solver, "orcamento": tempo,
        "t_total": total, "estouro_s": max(0., total - tempo),
        "t_score": s.t_score, "estagios": s.estagios,
        "fracao_final": s.fracao_final, "fallback": s.fallback, "certificado": s.certificado,
        "motivo": motivo, "nos": s.nos, "rotulo_melhor": rot.melhor, "rotulo_bound": rot.bound,
        "rotulo_qualidade": rot.qualidade, "trajetoria": json.dumps(s.trajetoria),
    }


def sementes_metodo(metodo: str, seeds: tuple[int, ...]) -> tuple[int, ...]:
    # Artefato legado: três GNNs; tabulares e GNN+PL têm só um treinamento.
    if metodo.endswith(":gnn"):
        if any(s not in (0, 1, 2) for s in seeds):
            raise ValueError("O artefato GNN disponível contém sementes 0, 1 e 2")
        return seeds
    if metodo.endswith((":lgbm", ":mlp", ":gnn_pl")):
        return (0,)
    return (-1,)


def testar(split: str, tempo: float, workers: int, metodos: tuple[str, ...],
           seeds: tuple[int, ...] = (0, 1, 2), solver_seeds: tuple[int, ...] = (0,),
           limite: int | None = None) -> Path:
    from alocacao_capacitada.pesquisa.dados import RAIZ
    from alocacao_capacitada.pesquisa.registro import Registro, manifesto

    if tempo <= 0 or workers < 1 or not seeds or not solver_seeds:
        raise ValueError("Orçamento, workers e listas de sementes devem ser positivos/não vazios")
    rho_file = OUT / "rho_escolhido.json"
    rho = rho_file.read_text()
    arqs = sorted((RAIZ / split).glob("*.pkl"))
    # Piloto balanceado por família/regime, sem selecionar pelo resultado.
    grupos: dict[tuple, list[Path]] = {}
    for a in arqs:
        p = ler(a).prob
        grupos.setdefault((p.familia, p.meta.get("razao")), []).append(a)
    arqs = [g[k] for k in range(max(map(len, grupos.values()), default=0))
            for g in grupos.values() if k < len(g)]
    if limite is not None:
        if limite < 1:
            raise ValueError("limite deve ser positivo")
        arqs = arqs[:limite]
    if not arqs:
        raise ValueError(f"Split vazio: {split}")
    config = dict(split=split, tempo=tempo, workers=workers, metodos=metodos,
                  seeds=seeds, solver_seeds=solver_seeds, limite=limite)
    reg = Registro(OUT / "v2", manifesto(config, [*arqs, rho_file,
                                                MODELOS / "ranqueadores.pkl"]))
    jobs = []
    feitos = reg.feitos()
    rng = np.random.default_rng(0)
    for a in arqs:
        nome = ler(a).prob.nome
        lote = [(str(a), m, tempo, rho, st, ss) for m in metodos
                for st in sementes_metodo(m, seeds) for ss in solver_seeds
                if reg.chave(nome, m, st, ss) not in feitos]
        rng.shuffle(lote)
        jobs.extend(lote)
    print(f"{len(jobs)} execuções pendentes; saída {reg.path}", flush=True)
    try:
        if workers == 1:
            for job in jobs:
                row = _rodar(job)
                reg.gravar(row)
                print(row["instancia"], row["metodo"], row["seed_treino"],
                      f"{row['objetivo']:.1f}", f"{row['t_total']:.2f}s", flush=True)
        else:
            with ProcessPoolExecutor(workers) as ex:
                for row in ex.map(_rodar, jobs):
                    reg.gravar(row)
                    print(row["instancia"], row["metodo"], flush=True)
    finally:
        reg.exportar()
        reg.close()
    return reg.path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["treinar", "rho", "testar"])
    ap.add_argument("--split", default="teste")
    ap.add_argument("--tempo", type=float, default=60)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--solver-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limite", type=int)
    ap.add_argument("--metodos", nargs="+", choices=METODOS_COMPLETOS)
    ap.add_argument("--gen", action="store_true", help="conjunto reduzido de métodos")
    a = ap.parse_args()
    if a.fase == "treinar":
        treinar()
    elif a.fase == "rho":
        escolher_rho()
    else:
        testar(a.split, a.tempo, a.workers,
               tuple(a.metodos) if a.metodos else METODOS_GEN if a.gen else METODOS_COMPLETOS,
               tuple(a.seeds), tuple(a.solver_seeds), a.limite)


if __name__ == "__main__":
    main()
