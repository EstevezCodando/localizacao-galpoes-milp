"""Gera as figuras do artigo (PDF vetorial) a partir dos resultados registrados.

Cada figura lê diretamente os arquivos de results/pesquisa/: nenhum número é digitado aqui.
Estilo: dados em cinza, um único destaque (azul) no método de interesse, eixos com unidades.

uv run python paper/figuras/gerar_figuras.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "results" / "pesquisa"
OUT = Path(__file__).resolve().parent
AZUL, CINZA, CINZA_CLARO, VERDE = "#2463c7", "#6b6b6b", "#b9b9b9", "#2e8b57"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "legend.frameon": False, "pdf.fonttype": 42})

HIBRIDOS = ("hibrido:gnn", "hibrido:rotacao", "hibrido:aprendido")

NOMES = {
    "adaptativa:gnn": "Adaptive expansion (GNN)", "clns:aprendido": "CLNS, learned selector",
    "clns:rotacao": "CLNS, rotation", "clns:aleatorio": "CLNS, random", "clns:alns": "CLNS, ALNS",
    "clns:dual": "CLNS, dual regret", "clns:gnn": "CLNS, GNN-guided", "lns": "LNS",
    "warm_classico": "SCIP warm-started by LNS", "completo": "SCIP (full model)",
    "ingenua": "Naive decomposition", "hibrido:gnn": "Hybrid, GNN-guided",
    "hibrido:rotacao": "Hybrid, rotation", "hibrido:aprendido": "Hybrid, learned selector",
    "adaptativa:pl": "Adaptive expansion (LP)", "kernel": "Kernel search",
    "hibridopl:rotacao": "LP hybrid, rotation", "hibridopl:rotacao+mem": "LP hybrid + memory",
    "hibrido:rotacao+mem": "Hybrid + memory", "clns:rotacao+mem": "CLNS, rotation + memory",
}


def salvar(fig: plt.Figure, nome: str) -> None:
    fig.savefig(OUT / f"{nome}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nome}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("ok", nome)


def fig_rho() -> None:
    d = pd.read_csv(RES / "etapa1" / "rho_validacao.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    estilo = {"gnn": ("GNN", AZUL, 2.2, "-"), "pl": ("LP relaxation", CINZA, 1.6, "-"),
              "classico": ("Classical service ratio", CINZA_CLARO, 1.4, "-"),
              "aleatorio": ("Random", CINZA_CLARO, 1.2, ":")}
    for k, (rot, cor, lw, ls) in estilo.items():
        g = d[d["ranqueador"] == k].sort_values("rho")
        ax.plot(g["rho"], 100 * g["contem_melhor"], ls, color=cor, lw=lw, marker="o", ms=3,
                label=rot)
    ax.axhline(80, color="black", lw=0.8, ls="--")
    ax.text(0.205, 82.5, "pre-registered target: 80% of instances", fontsize=8)
    ax.set_xlabel(r"kept fraction $\rho$ before feasibility repair")
    ax.set_ylabel("instances containing the\nbest known solution (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, "rho")


def _barras(ag: dict, campo: str, lo: str | None, hi: str | None, rotulo: str, nome: str,
            destaque: tuple[str, ...], pct: bool = False, excluir: tuple[str, ...] = ()) -> None:
    linhas = [r for r in ag["metodos"] if r["metodo"] not in excluir]
    linhas = sorted(linhas, key=lambda r: r[campo])
    y = np.arange(len(linhas))
    v = np.array([r[campo] for r in linhas]) * (100 if pct else 1)
    cores = [AZUL if r["metodo"] in destaque else CINZA_CLARO for r in linhas]
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(linhas) + 0.8))
    xerr = None
    if lo and hi:
        xerr = np.array([[r[campo] - r[lo] for r in linhas], [r[hi] - r[campo] for r in linhas]])
    ax.barh(y, v, color=cores, xerr=xerr, error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(y, [NOMES.get(r["metodo"], r["metodo"]) for r in linhas])
    ax.invert_yaxis()
    fim = v if xerr is None else v + xerr[1]
    for yi, vi, fi in zip(y, v, fim):
        ax.text(fi, yi, f"  {vi:.1f}%" if pct else f"  {vi:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, max(fim) * 1.15)
    ax.set_xlabel(rotulo)
    salvar(fig, nome)


def fig_pilotos() -> None:
    for run, tag in (("81dbbcb06942419706f3", "piloto1"), ("eaa3f86d9312c0970ce9", "piloto2")):
        arq = RES / "clns" / run / "agregados.json"
        if not arq.exists():
            print("sem agregados:", tag)
            continue
        ag = json.loads(arq.read_text())
        dest = ("adaptativa:gnn",) if tag == "piloto1" else HIBRIDOS
        _barras(ag, "integral", "integral_lo", "integral_hi", "primal integral (lower is better)",
                f"{tag}_integral", dest, excluir=("ingenua",))
        _barras(ag, "desvio_mediano", None, None, "median final deviation from best known (%)",
                f"{tag}_desvio", dest, pct=True, excluir=("ingenua",))
        if tag == "piloto2":
            fig_v2_extras(ag)


def fig_v2_extras(ag: dict) -> None:
    # vitórias/empates/derrotas pareadas contra o SCIP completo
    par = sorted(ag["pareado"]["linhas"], key=lambda r: r["vitorias"] - r["derrotas"])
    y = np.arange(len(par))
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(par) + 0.8))
    v = np.array([r["vitorias"] for r in par])
    e = np.array([r["empates"] for r in par])
    dd = np.array([r["derrotas"] for r in par])
    ax.barh(y, v, color=VERDE, alpha=0.75, label="better than SCIP")
    ax.barh(y, e, left=v, color=CINZA_CLARO, label="tie (within 0.01%)")
    ax.barh(y, dd, left=v + e, color="#c0392b", alpha=0.7, label="worse than SCIP")
    ax.set_yticks(y, [NOMES.get(r["metodo"], r["metodo"]) for r in par])
    ax.set_xlabel("instances (final deviation, averaged over seeds)")
    ax.legend(ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.4, -0.45))
    salvar(fig, "piloto2_vitorias")
    # curva anytime
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    grade = ag["anytime"]["grade"]
    for m, curva in ag["anytime"]["metodos"].items():
        estilo = {"hibrido:aprendido": (AZUL, 2.2, "-"), "adaptativa:gnn": (VERDE, 1.8, "-"),
                  "clns:aprendido": (CINZA, 1.4, "--"), "completo": ("black", 1.4, ":"),
                  "lns": (CINZA, 1.2, ":")}.get(m)
        if estilo is None:
            ax.plot(grade, 100 * np.array(curva), color=CINZA_CLARO, lw=0.8)
        else:
            ax.plot(grade, 100 * np.array(curva), color=estilo[0], lw=estilo[1], ls=estilo[2],
                    label=NOMES.get(m, m))
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("median gap to best known (%)")
    ax.legend(fontsize=7.5)
    salvar(fig, "piloto2_anytime")
    # parcela de instâncias até 1%
    _barras(ag, "ate_1pct", None, None, "instances within 1% of best known (%)",
            "piloto2_ate1pct", HIBRIDOS, pct=True)
    # razão CPU/parede
    med = ag.get("medicao", {})
    if med:
        h = np.array(med["histograma_razao"])
        bins = np.arange(0.8, 1.025, 0.01)
        fig, ax = plt.subplots(figsize=(5.6, 2.4))
        ax.bar(bins[:-1], h, width=0.009, align="edge", color=CINZA)
        ax.axvline(0.9, color="black", ls="--", lw=0.8)
        ax.text(0.902, max(h) * 0.9, "contention flag (< 0.9)", fontsize=8)
        ax.set_xlabel("CPU time / wall-clock time per run")
        ax.set_ylabel("runs")
        salvar(fig, "piloto2_cpu_parede")


def fig_unifl() -> None:
    a = pd.read_csv(RES / "unifl" / "resultados.csv")
    b = pd.read_csv(RES / "unifl" / "resultados_estavel.csv")
    todos = pd.concat([a, b])
    melhor = todos.groupby(["n", "seed"])["custo"].min()
    d = pd.concat([a[a["metodo"].isin(["mettu_plaxton", "mp+busca_local"])],
                   b[b["metodo"].str.match(r"mpnn_estavel_s\d(\+busca_local)?$")]])
    d["ref"] = [o if np.isfinite(o) else melhor[(n, s)]
                for o, n, s in zip(d["otimo"], d["n"], d["seed"])]
    d["r"] = d["custo"] / d["ref"]
    d["grupo"] = d["metodo"].str.replace(r"_s\d", "", regex=True)
    est = {"mettu_plaxton": ("Mettu-Plaxton", CINZA, "-"),
           "mpnn_estavel": ("MPNN (stabilized)", AZUL, "-"),
           "mp+busca_local": ("Mettu-Plaxton + local search", CINZA_CLARO, "--"),
           "mpnn_estavel+busca_local": ("MPNN + local search", AZUL, "--")}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for g, (rot, cor, ls) in est.items():
        q = d[d["grupo"] == g].groupby("n")["r"]
        med, lo, hi = q.median(), q.quantile(0.1), q.quantile(0.9)
        ax.plot(med.index, med.values, ls, color=cor, marker="o", ms=3, label=rot,
                lw=2 if g == "mpnn_estavel" else 1.2)
        if "busca" not in g:
            ax.fill_between(med.index, lo.values, hi.values, color=cor, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000], ["100", "200", "500", "1000"])
    ax.axvspan(90, 220, color="0.92", zorder=0)
    ax.text(95, 1.27, "training sizes", fontsize=8)
    ax.set_xlabel("number of points $n$")
    ax.set_ylabel("cost / reference")
    ax.legend(fontsize=7.5)
    salvar(fig, "unifl")


def fig_trocas() -> None:
    d = pd.read_csv(RES / "swap_v2" / "resultados.csv")
    melhor = d.groupby(["n", "seed"])["custo"].transform("min")
    d["ref"] = np.where(d["status_exato"].eq("optimal"), d["otimo"], melhor)
    d["dev"] = d["custo"] / d["ref"] - 1
    full = d[d["filtro"] == "completo"].set_index(["n", "seed", "inicio"])["tempo"]
    d["sp"] = [full[(n, s, i)] / t for n, s, i, t in zip(d["n"], d["seed"], d["inicio"], d["tempo"])]
    g = d[d["n"] == 1000].groupby(["filtro", "fallback", "k_in"])[["sp", "dev"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    est = {"classico": ("classical add-gain filter", CINZA), "aprendido": ("learned filter", AZUL),
           "aleatorio": ("random filter", CINZA_CLARO), "completo": ("full search", "black")}
    for f, (rot, cor) in est.items():
        for fb, mk in ((False, "o"), (True, "s")):
            q = g[(g["filtro"] == f) & (g["fallback"] == fb)]
            if q.empty:
                continue
            ax.scatter(q["sp"], 100 * q["dev"], color=cor if not fb else "none", edgecolors=cor,
                       marker=mk, s=28, label=f"{rot}{' (with fallback)' if fb else ''}"
                       if f != "completo" else rot)
    ax.set_xscale("log")
    ax.set_xlabel("speedup over the full swap search (log scale)")
    ax.set_ylabel("loss vs. best solution (%)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    salvar(fig, "trocas")


def fig_piloto3() -> None:
    """Controles sem aprendizado e memória de subproblemas (piloto 3)."""
    run = RES / "clns" / "e1a72d506f53ea707aa1"
    if not (run / "agregados.json").exists():
        print("sem agregados: piloto3")
        return
    ag = json.loads((run / "agregados.json").read_text())
    aprende = ("adaptativa:gnn", "hibrido:rotacao", "hibrido:rotacao+mem")
    _barras(ag, "integral", "integral_lo", "integral_hi", "primal integral (lower is better)",
            "piloto3_integral", aprende)
    _barras(ag, "desvio_mediano", None, None, "median final deviation from best known (%)",
            "piloto3_desvio", aprende, pct=True)
    d = pd.read_csv(run / "resultados.csv")
    d = d[d["iteracoes"].notna()].copy()
    d["taxa"] = d["repetidos"] / d["iteracoes"].clip(lower=1)
    g = d.groupby(["metodo", "instancia"])[["taxa", "iteracoes"]].mean().groupby("metodo").mean()
    g = g.sort_values("taxa", ascending=False)
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(g) + 0.9))
    ax.barh(y, 100 * g["taxa"], color=[VERDE if m.endswith("+mem") else CINZA_CLARO for m in g.index])
    ax.set_yticks(y, [NOMES.get(m, m) for m in g.index])
    ax.invert_yaxis()
    for yi, (t, it) in enumerate(zip(100 * g["taxa"], g["iteracoes"])):
        ax.text(t, yi, f"  {t:.0f}%  ({it:.0f} subproblems/run)", va="center", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("selected subproblems already failed in the same local state (%)")
    salvar(fig, "piloto3_repeticao")


def fig_fechado() -> None:
    """Closed test: per-set bars and hypothesis intervals (results/pesquisa/fechado/analise.json)."""
    arq = RES / "fechado" / "analise.json"
    if not arq.exists():
        print("sem teste fechado")
        return
    an = json.loads(arq.read_text(encoding="utf-8"))
    conjuntos = [("teste", "Test (32)"), ("gen_corredor", "Corridor (16)"),
                 ("gen_escala", "Scale (16)"), ("holmberg", "Holmberg (71)")]
    metodos = ["completo", "kernel", "adaptativa:pl", "hibridopl:rotacao", "adaptativa:gnn",
               "hibrido:rotacao"]
    nomes = {"completo": "SCIP (full model)", "kernel": "Kernel search",
             "adaptativa:pl": "Adaptive expansion (LP)", "hibridopl:rotacao": "LP hybrid",
             "adaptativa:gnn": "Adaptive expansion (GNN)", "hibrido:rotacao": "Hybrid (GNN)"}
    cores = {"completo": "black", "kernel": "#d98e04", "adaptativa:pl": CINZA,
             "hibridopl:rotacao": VERDE, "adaptativa:gnn": "#8fb1ea", "hibrido:rotacao": AZUL}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    larg = 0.13
    for ax, (campo, rot, k) in zip(axs, (("integral", "mean primal integral", 1),
                                         ("desvio_medio", "mean final deviation (%)", 100))):
        for j, m in enumerate(metodos):
            linhas = [next(x for x in an[c]["tabela"] if x["metodo"] == m) for c, _ in conjuntos]
            v = [k * r[campo] for r in linhas]
            err = None
            if campo == "integral":
                err = [[r[campo] - r["integral_lo"] for r in linhas],
                       [r["integral_hi"] - r[campo] for r in linhas]]
            ax.bar(np.arange(len(conjuntos)) + (j - 2.5) * larg, v, larg * 0.92, color=cores[m],
                   label=nomes[m], yerr=err, error_kw={"lw": 0.6, "capsize": 1, "ecolor": CINZA})
        ax.set_xticks(np.arange(len(conjuntos)), [r for _, r in conjuntos], fontsize=7.5)
        ax.set_ylabel(rot)
    axs[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    salvar(fig, "fechado_conjuntos")

    ag = an["agregado"]
    rot = {"H1": "H1: LP expansion $-$ SCIP", "H2": "H2: GNN expansion $-$ LP expansion",
           "H3": "H3: LP hybrid $-$ LP expansion", "H4": "H4: LP expansion $-$ kernel search"}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.3))
    for ax, (chave, k, titulo) in zip(axs, (("hipoteses_integral", 1.0, "primal integral"),
                                           ("hipoteses_desvio", 100.0, "final deviation (p.p.)"))):
        hs = ag[chave]
        y = np.arange(len(hs))[::-1]
        for yi, h in zip(y, hs):
            sig = h["p_holm"] < 0.05
            cor = (VERDE if h["dif_media"] < 0 else "#c0392b") if sig else CINZA
            ax.plot([k * h["dif_lo"], k * h["dif_hi"]], [yi, yi], color=cor, lw=1.8)
            ax.plot(k * h["dif_media"], yi, "o", color=cor, ms=4.5)
            pv = "$p<0.001$" if h["p_holm"] < 0.001 else f"$p={h['p_holm']:.3f}$"
            ax.text(1.02, yi, pv, fontsize=7.5, color=cor, va="center",
                    transform=ax.get_yaxis_transform())
        ax.axvline(0, color="black", lw=0.7)
        ax.locator_params(axis="x", nbins=5)
        ax.set_ylim(-0.6, len(hs) - 0.4)
        ax.set_yticks(y, [rot[h["hipotese"]] for h in hs] if ax is axs[0] else [""] * len(hs),
                      fontsize=7.5)
        ax.set_xlabel(f"mean difference A $-$ B, {titulo}", fontsize=8)
    fig.tight_layout()
    salvar(fig, "fechado_hipoteses")


if __name__ == "__main__":
    fig_rho()
    fig_unifl()
    fig_trocas()
    fig_pilotos()
    fig_piloto3()
    fig_fechado()
