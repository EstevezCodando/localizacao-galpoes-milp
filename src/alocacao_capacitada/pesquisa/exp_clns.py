"""Experimentos do artigo CLNS (LNS orientado por clusters com seleção aprendida).

coletar   rótulos fora da política nas instâncias de TREINO (60 s cada, 8 candidatos/estado)
treinar   LightGBM de ganho-por-segundo; métricas de ranking na VALIDAÇÃO
avaliar   métodos no mesmo orçamento sobre um split, com registro versionado (manifesto + SQLite)

Métodos: completo, warm_classico, lns, adaptativa:<gnn|pl> (delegados ao executor da Etapa 1),
kernel (Kernel Search clássico), clns:<rotacao|aleatorio|alns|dual|aprendido|gnn>,
hibrido:<seletor> (expansão adaptativa por GNN + CLNS), hibridopl:<seletor> (idem com ranking
do PL, sem aprendizado) e ingenua (decomposição sem coordenação + reparo). O sufixo "+mem" em
clns/hibrido/hibridopl liga a memória de subproblemas (ex.: hibrido:rotacao+mem).

uv run python -m alocacao_capacitada.pesquisa.exp_clns coletar --workers 3
uv run python -m alocacao_capacitada.pesquisa.exp_clns treinar
uv run python -m alocacao_capacitada.pesquisa.exp_clns avaliar --split validacao --tempo 60
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

from alocacao_capacitada.pesquisa import clns as C
from alocacao_capacitada.pesquisa.dados import RAIZ, ler

OUT = Path("results/pesquisa/clns")
MODELO = Path("data/processed/pesquisa/modelos/seletor_clns.pkl")
DADOS = Path("data/processed/pesquisa/clns_rotulos.pkl")
DELEGADOS = ("completo", "warm_classico", "lns", "adaptativa:gnn", "adaptativa:pl")
# SCIP com semente fixa
DETERMINISTICOS = ("completo", "warm_classico", "adaptativa:gnn", "adaptativa:pl", "kernel")


def sementes(metodo: str, seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Métodos determinísticos rodam uma vez; repetir só gastaria CPU sem informação nova."""
    return seeds[:1] if metodo in DETERMINISTICOS else seeds


# ------------------------------------------------------------------ coleta e treino
def _coletar(caminho: str) -> tuple[str, list[tuple[np.ndarray, float, float]], float]:
    t0 = time.perf_counter()
    p = ler(Path(caminho)).prob
    seed = int(p.meta.get("seed", 0))
    return p.nome, C.coletar(p, 60.0, k_amostra=8, seed=seed), time.perf_counter() - t0


def coletar(workers: int, n: int) -> None:
    arqs = sorted((RAIZ / "treino").glob("*.pkl"))
    # balanceado: n primeiras sementes de cada (família, razão)
    sel = [a for a in arqs if int(a.stem.split("_s")[-1]) < n]
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for nome, out, t in ex.map(_coletar, map(str, sel)):
            res[nome] = (out, t)
            print(nome, len(out), sum(g > 0 for _, g, _ in out), f"{t:.0f}s", flush=True)
    DADOS.parent.mkdir(parents=True, exist_ok=True)
    with open(DADOS, "wb") as fh:
        pickle.dump(res, fh)


def _xy(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.stack([x for out, _ in res.values() for x, _, _ in out])
    g = np.array([gn for out, _ in res.values() for _, gn, _ in out])
    t = np.array([dt for out, _ in res.values() for _, _, dt in out])
    return X, g, t


def treinar() -> None:
    import lightgbm as lgb

    with open(DADOS, "rb") as fh:
        res = pickle.load(fh)
    nomes = sorted(res)
    rng = np.random.default_rng(0)
    rng.shuffle(nomes)
    corte = int(0.8 * len(nomes))  # validação por INSTÂNCIA (sem vazamento entre estados)
    tr = {k: res[k] for k in nomes[:corte]}
    va = {k: res[k] for k in nomes[corte:]}
    Xtr, gtr, ttr = _xy(tr)
    Xva, gva, tva = _xy(va)
    alvo_tr = gtr / np.maximum(ttr, 0.05)  # ganho normalizado por segundo
    t0 = time.perf_counter()
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=30, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1)
    m.fit(Xtr, alvo_tr)
    t_tr = time.perf_counter() - t0
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    pred = m.predict(Xva)
    alvo_va = gva / np.maximum(tva, 0.05)
    met = {
        "amostras_treino": int(len(gtr)), "amostras_val": int(len(gva)),
        "frac_positiva": float((gtr > 0).mean()),
        "auc_melhora_val": float(roc_auc_score(gva > 0, pred)),
        "spearman_val": float(spearmanr(pred, alvo_va).statistic),
        "t_treino_s": t_tr,
        "t_coleta_h": float(sum(t for _, t in res.values()) / 3600),
        "importancia": dict(zip([
            "frac_n", "frac_demanda", "custo_medio", "regret_medio", "regret_max", "n_centros",
            "folga", "fixo_por_carga", "tentativas", "ganho_hist", "idade", "razao",
            *[f"tipo_{t}" for t in C.TIPOS]], map(int, m.feature_importances_))),
    }
    # atributo isolado como ranqueador (o modelo precisa vencer o regret dual sozinho)
    met["auc_regret_medio_val"] = float(roc_auc_score(gva > 0, Xva[:, 3]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "treino_seletor.json").write_text(json.dumps(met, indent=2))
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELO, "wb") as fh:
        pickle.dump(m, fh)
    print(json.dumps(met, indent=1))


# ------------------------------------------------------------------ avaliação
_MODELO = None
_RK = None
ALFA = 0.5  # fração do orçamento da expansão adaptativa no híbrido (fixada a priori, sem ajuste)


def _seletor(nome: str) -> C.Seletor:
    global _MODELO
    if nome == "aprendido":
        if _MODELO is None:
            with open(MODELO, "rb") as fh:
                _MODELO = pickle.load(fh)
        return C.Aprendido(_MODELO)
    return C.SELETORES[nome]()


def _ranqueadores():  # type: ignore[no-untyped-def]
    global _RK
    if _RK is None:
        from alocacao_capacitada.pesquisa import etapa1

        _RK = etapa1.Ranqueadores()
    return _RK


def _hibrido(p, tempo: float, nome_sel: str, seed: int,  # type: ignore[no-untyped-def]
             rank: str = "gnn", memoria: bool = False) -> tuple:
    """Expansão adaptativa em ALFA do orçamento (ranking `rank`: "gnn" ou "pl"), depois CLNS no
    restante, partindo da solução adaptativa, reaproveitando o PL e (no seletor 'gnn') o ranking
    da GNN. Com rank="pl" nada é aprendido: é o controle que isola o efeito da GNN."""
    from alocacao_capacitada.pesquisa import hibrido as H
    from alocacao_capacitada.pesquisa.exato import relaxacao_linear

    if rank != "gnn" and nome_sel == "gnn":
        raise ValueError("o seletor gnn exige o ranking da GNN")
    t0 = time.perf_counter()
    sc, _, lp = _ranqueadores().score(rank, p, seed=0, tempo_s=0.2 * tempo)
    if lp is None:
        lp = relaxacao_linear(p, tempo_s=0.2 * tempo)
    t_sc = time.perf_counter() - t0
    s1 = H.adaptativa(p, sc, t_sc, ALFA * tempo, lp, seed=seed)
    gasto = time.perf_counter() - t0
    prob_gnn = 1.0 / (1.0 + np.exp(-sc))
    sel = _seletor(nome_sel)
    s2 = C.clns(p, max(tempo - gasto, 0.05), sel, seed=seed, atrib_inicial=s1.atribuicao,
                lp_pronto=(lp.rc_x, lp.x), gnn=prob_gnn if nome_sel == "gnn" else None,
                t_offset=gasto, memoria=memoria)
    traj = list(s1.trajetoria) + s2.trajetoria
    extra = {"t_adaptativa": gasto, "obj_adaptativa": s1.objetivo, "iteracoes": s2.iteracoes,
             "subproblemas": s2.iteracoes, "nos_scip_adaptativa": s1.nos,
             "tentados": json.dumps(s2.tentados), "aceitos": json.dumps(s2.aceitos),
             "t_selecao": s2.t_selecao, "t_sub": s2.t_sub,
             "t_inferencia": getattr(sel, "t_inf", 0.0),
             "repetidos": s2.repetidos, "pulados": s2.pulados}
    return s2.objetivo, traj, s2.valida, extra


def _init_worker(fila: object) -> None:
    """Fixa o núcleo e aquece imports e modelos FORA do cronômetro (custo de processo, não de
    instância): ranqueadores da Etapa 1 (inclusive o global usado por etapa1._rodar) e o
    seletor aprendido."""
    from alocacao_capacitada.pesquisa import etapa1, medicao

    medicao.inicializar_worker(fila)
    rk = _ranqueadores()
    etapa1._RK = rk
    _seletor("aprendido")


def _chave_treino(metodo: str) -> int:
    return 0 if (metodo.endswith(":gnn") and metodo in DELEGADOS) else -1


def _rodar(args: tuple[str, str, float, int]) -> dict[str, object]:
    from alocacao_capacitada.pesquisa import medicao

    caminho, metodo, tempo, seed = args
    rot = ler(Path(caminho))
    p = rot.prob
    with medicao.Cronometro() as cron:
        if metodo in DELEGADOS:
            from alocacao_capacitada.pesquisa import etapa1

            rho = (etapa1.OUT / "rho_escolhido.json").read_text()
            row = etapa1._rodar((caminho, metodo, tempo, rho, _chave_treino(metodo), seed))
            obj, traj, valida = row["objetivo"], json.loads(row["trajetoria"]), row["valida"]
            extra = {k: row[k] for k in ("nos", "certificado", "fracao_final", "estagios")
                     if k in row}
        elif metodo == "ingenua":
            from alocacao_capacitada.pesquisa.decomposicao import decomposicao_ingenua

            s = decomposicao_ingenua(p, tempo, tamanho=15, seed=seed)
            obj = s.custo_reparado
            traj = [(s.t_total, obj)] if np.isfinite(obj) else []
            extra = {"uniao_viavel": s.viavel_uniao, "sobrecarga": s.sobrecarga}
            valida = bool(np.isfinite(obj))
        elif metodo == "kernel":
            from alocacao_capacitada.pesquisa import hibrido as H
            from alocacao_capacitada.pesquisa.exato import relaxacao_linear

            t0 = time.perf_counter()
            lp = relaxacao_linear(p, tempo_s=0.2 * tempo)
            s = H.kernel_search(p, lp, time.perf_counter() - t0, tempo, seed=seed)
            obj, traj, valida = s.objetivo, s.trajetoria, s.valida
            extra = {"nos": s.nos, "estagios": s.estagios, "fracao_final": s.fracao_final,
                     "t_pl": s.t_score}
        elif metodo.startswith(("hibrido:", "hibridopl:")):
            base, _, mem = metodo.partition("+")
            pol, nome = base.split(":")
            obj, traj, valida, extra = _hibrido(p, tempo, nome, seed,
                                                rank="pl" if pol == "hibridopl" else "gnn",
                                                memoria=mem == "mem")
        else:
            base, _, mem = metodo.partition("+")
            nome = base.split(":")[1]
            gnn = None
            if nome == "gnn":
                sc, _, _ = _ranqueadores().score("gnn", p, seed=0)
                gnn = 1.0 / (1.0 + np.exp(-sc))
            sel = _seletor(nome)
            s2 = C.clns(p, tempo, sel, seed=seed, gnn=gnn, memoria=mem == "mem")
            obj, traj, valida = s2.objetivo, s2.trajetoria, s2.valida
            extra = {"iteracoes": s2.iteracoes, "subproblemas": s2.iteracoes, "t_pl": s2.t_pl,
                     "t_inicial": s2.t_inicial, "t_selecao": s2.t_selecao, "t_sub": s2.t_sub,
                     "tentados": json.dumps(s2.tentados), "aceitos": json.dumps(s2.aceitos),
                     "t_inferencia": getattr(sel, "t_inf", 0.0),
                     "repetidos": s2.repetidos, "pulados": s2.pulados}
    return {"instancia": p.nome, "familia": p.familia, "razao": p.meta.get("razao"),
            "m": p.m, "n": p.n, "metodo": metodo, "objetivo": obj, "valida": bool(valida),
            "seed_treino": _chave_treino(metodo), "seed_solver": seed, "orcamento": tempo,
            "t_total": cron.parede_s, "estouro_s": max(0.0, cron.parede_s - tempo),
            "rotulo_melhor": rot.melhor, "rotulo_bound": rot.bound,
            "rotulo_qualidade": rot.qualidade, "trajetoria": json.dumps(traj),
            **cron.registro(), **extra}


def avaliar(split: str, tempo: float, workers: int, metodos: tuple[str, ...],
            seeds: tuple[int, ...], limite: int | None) -> Path:
    from alocacao_capacitada.pesquisa import congelar, medicao
    from alocacao_capacitada.pesquisa.registro import Registro, manifesto

    hash_dados = congelar.verificar(split)  # falha se os dados divergirem da trava
    precisa_modelos = any(m.partition("+")[0].endswith(("aprendido", ":gnn"))
                          or m.startswith("hibrido:") for m in metodos)
    hash_modelos = congelar.verificar("modelos") if precisa_modelos else None
    arqs = sorted((RAIZ / split).glob("*.pkl"))
    celulas: dict[tuple, list[Path]] = {}
    for a in arqs:
        pr = ler(a).prob
        celulas.setdefault((pr.familia, pr.meta.get("razao")), []).append(a)
    arqs = [g[k] for k in range(max(map(len, celulas.values()), default=0))
            for g in celulas.values() if k < len(g)]
    if limite:
        arqs = arqs[:limite]
    config = dict(split=split, tempo=tempo, metodos=metodos, seeds=seeds, limite=limite,
                  workers=workers, alfa_hibrido=ALFA, protocolo_tempo="v3",
                  hash_dados=hash_dados, hash_modelos=hash_modelos,
                  padrao_clns=dict(tamanho=15, t_sub=0.5, inicial="pl"))
    reg = Registro(OUT, manifesto(config, list(arqs) + ([MODELO] if precisa_modelos else [])))
    feitos = reg.feitos()
    rng = np.random.default_rng(0)
    jobs = []
    for a in arqs:
        nome = ler(a).prob.nome
        lote = [(str(a), m, tempo, s) for m in metodos for s in sementes(m, seeds)
                if reg.chave(nome, m, _chave_treino(m), s) not in feitos]
        rng.shuffle(lote)
        jobs += lote
    print(f"{len(jobs)} execuções pendentes; saída {reg.path}", flush=True)
    fila = medicao.fila_de_nucleos(workers)
    try:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(fila,)) as ex:
            for row in ex.map(_rodar, jobs):
                reg.gravar(row)
                print(row["instancia"], row["metodo"], row["seed_solver"],
                      f"{row['objetivo']:.1f}", f"{row['t_total']:.1f}s",
                      f"cpu/parede={row['razao_cpu']:.2f}", f"nucleo={row['nucleo']}",
                      flush=True)
    finally:
        reg.exportar()
        reg.close()
    return reg.path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["coletar", "treinar", "avaliar"])
    ap.add_argument("--split", default="validacao")
    ap.add_argument("--tempo", type=float, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=15, help="coleta: sementes por (família, razão)")
    ap.add_argument("--metodos", nargs="+", default=[
        "completo", "warm_classico", "lns", "adaptativa:gnn", "clns:rotacao", "clns:alns",
        "clns:aprendido", "clns:gnn", "hibrido:rotacao", "hibrido:aprendido", "hibrido:gnn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    if a.fase == "coletar":
        coletar(a.workers, a.n)
    elif a.fase == "treinar":
        treinar()
    else:
        avaliar(a.split, a.tempo, a.workers, tuple(a.metodos), tuple(a.seeds), a.limite)


if __name__ == "__main__":
    main()
