"""Figuras do README e do relatório da linha de pesquisa (PNG, rótulos em português).

Diagramas (caminho percorrido, acoplamento de capacidade, pipeline do híbrido) são desenhados
aqui; todos os gráficos de dados leem results/pesquisa/ — nenhum número de resultado é digitado.

uv run python scripts/figuras_pesquisa.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results" / "pesquisa"
OUT = RAIZ / "docs" / "pesquisa" / "img"
PILOTO2 = RES / "clns" / "eaa3f86d9312c0970ce9"
PILOTO3 = RES / "clns" / "e1a72d506f53ea707aa1"

AZUL, VERDE, VERMELHO, AMBAR = "#2463c7", "#2e8b57", "#c0392b", "#d98e04"
CINZA, CINZA_CLARO, TINTA = "#6b6b6b", "#c4c4c4", "#222222"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                     "figure.dpi": 100, "savefig.dpi": 180, "font.family": "DejaVu Sans"})

NOMES = {
    "completo": "SCIP (modelo completo)", "warm_classico": "SCIP com partida do LNS",
    "lns": "LNS (matheurística)", "kernel": "Kernel Search (clássico)",
    "adaptativa:gnn": "Expansão adaptativa (GNN)", "adaptativa:pl": "Expansão adaptativa (PL)",
    "clns:rotacao": "CLNS, rotação", "clns:rotacao+mem": "CLNS, rotação + memória",
    "clns:alns": "CLNS, ALNS", "clns:aprendido": "CLNS, seletor aprendido",
    "clns:gnn": "CLNS, guiado pela GNN",
    "hibrido:rotacao": "Híbrido GNN, rotação", "hibrido:rotacao+mem": "Híbrido GNN + memória",
    "hibrido:aprendido": "Híbrido GNN, seletor aprendido",
    "hibrido:gnn": "Híbrido GNN, guiado pela GNN",
    "hibridopl:rotacao": "Híbrido PL, rotação", "hibridopl:rotacao+mem": "Híbrido PL + memória",
}
APRENDE = {"adaptativa:gnn", "clns:aprendido", "clns:gnn", "hibrido:rotacao", "hibrido:rotacao+mem",
           "hibrido:aprendido", "hibrido:gnn"}


def salvar(fig: plt.Figure, nome: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nome}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("ok", nome)


def _caixa(ax, x, y, w, h, titulo, corpo, cor, fonte=9.0):  # type: ignore[no-untyped-def]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=cor, lw=1.8))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42,
                                boxstyle="round,pad=0.02,rounding_size=0.08", fc=cor, ec=cor))
    ax.text(x + w / 2, y + h - 0.21, titulo, ha="center", va="center", color="white",
            fontsize=fonte, fontweight="bold")
    ax.text(x + 0.12, y + h - 0.60, corpo, ha="left", va="top", color=TINTA, fontsize=fonte - 1,
            linespacing=1.35)


def _seta(ax, a, b, cor=CINZA):  # type: ignore[no-untyped-def]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=cor))


# ------------------------------------------------------------------ diagramas
def fig_caminho(p3: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(0.1, 7.1, "O caminho percorrido: cada etapa, o que foi medido e a decisão tomada",
            fontsize=13, fontweight="bold", color=TINTA)
    w, h = 4.1, 2.75
    xs = [0.1, 4.7, 9.3]
    c6 = ("Kernel Search, híbrido só com PL e memória de\nsubproblemas no mesmo protocolo.\n\n"
          "Resultado: ver figura dos controles.")
    if p3 is not None:
        m = {r["metodo"]: r for r in p3["metodos"]}
        if {"hibrido:rotacao", "hibridopl:rotacao", "kernel", "adaptativa:pl"} <= set(m):
            def linha(rot: str, k: str) -> str:
                return (f"{rot} {m[k]['integral']:.4f} | {100 * m[k]['desvio_mediano']:.2f}%"
                        .replace(".", ","))
            c6 = ("Integral primal | desvio final mediano:\n"
                  + linha("Expansão PL:    ", "adaptativa:pl") + "\n"
                  + linha("Híbrido PL:      ", "hibridopl:rotacao") + "\n"
                  + linha("Híbrido GNN:   ", "hibrido:rotacao") + "\n"
                  + linha("Kernel Search:", "kernel") + "\n\n"
                  "Medido: nenhuma diferença significativa a\nfavor da GNN; 54% a 58% dos "
                  "subproblemas\ndo CLNS eram repetições.\n"
                  "Decisão: o PL passa a ser a referência.")
    blocos = [
        ("1. Reconstruções (P02 a P07)",
         "Cinco frentes da literatura refeitas em\nSCIP 10 + PyTorch CPU.\n\n"
         "Medido: em todas, a heurística clássica\nde mesma função captura a maior parte\ndo ganho "
         "atribuído ao aprendizado.\n\nDecisão: aprendizado só entra se vencer\no clássico "
         "equivalente a tempo igual.", CINZA),
        ("2. Poda por ranking de centros",
         "GNN bipartida ordena os centros; resolve-se\nsó com os melhores.\n\n"
         "Medido: AUC 0,95, mas para manter a melhor\nsolução em 80% das instâncias é preciso\n"
         "ficar com ~80% dos centros. O ranking do\nPL, sem treino, é igual ou melhor.\n\n"
         "Decisão: orientar a busca, não podar.", VERMELHO),
        ("3. Decomposição por clusters (ingênua)",
         "Clusters de clientes resolvidos em separado.\n\n"
         "Medido: união inviável em todas as instâncias\nexaminadas (690 a 1.070 unidades de\n"
         "sobrecarga); após reparo, +19% a +29%.\n\n"
         "Decisão: subproblema com o restante fixo,\ncapacidade residual e custo fixo pago\numa "
         "única vez (CLNS).", VERMELHO),
        ("4. CLNS coordenado + seletor",
         "LNS por clusters; seletores rotação, ALNS,\ndual, aprendido e guiado pela GNN.\n\n"
         "Medido: acha boas soluções cedo (integral\n0,029 a 0,042 contra 0,080 do SCIP), mas\n"
         "estaciona a ~2% do melhor. O seletor\naprendido não se separa da rotação.\n\n"
         "Decisão: partir de uma solução melhor.", AMBAR),
        ("5. Híbrido: expansão + CLNS",
         "Expansão adaptativa em metade do tempo,\nCLNS na outra metade.\n\n"
         "Medido: desvio final 0,57% a 0,63% (contra\n0,81% da expansão e 2,7% do SCIP); até\n"
         "88% das instâncias a 1% do melhor.\nHipóteses H-a e H-b não confirmadas.\n\n"
         "Decisão: testar se a GNN é necessária.", VERDE),
        ("6. Controles clássicos e memória", c6, AZUL),
    ]
    for k, (t, c, cor) in enumerate(blocos):
        x = xs[k % 3]
        y = 3.95 if k < 3 else 0.55
        _caixa(ax, x, y, w, h, t, c, cor)
    for y in (3.95 + h / 2, 0.55 + h / 2):
        _seta(ax, (xs[0] + w + 0.05, y), (xs[1] - 0.05, y))
        _seta(ax, (xs[1] + w + 0.05, y), (xs[2] - 0.05, y))
    ym = (3.95 + 0.55 + h) / 2  # meio do vão entre as duas linhas de caixas
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[0] + w / 2], [3.93, ym, ym], color=CINZA, lw=1.4)
    _seta(ax, (xs[0] + w / 2, ym), (xs[0] + w / 2, 0.55 + h + 0.03))
    ax.text(0.1, 0.12, "Cores: cinza = levantamento; vermelho = caminho descartado; âmbar = "
            "resultado parcial; verde = resultado positivo; azul = etapa mais recente.",
            fontsize=8.5, color=CINZA)
    salvar(fig, "01_caminho")


def fig_acoplamento() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    paineis = (
        ("Decomposição ingênua", VERMELHO, "#fdecea",
         ["Centro compartilhado", "capacidade 100", "carga 60 + 60 = 120"],
         ("Grupo A", "envia 60 unidades", AZUL, "#eef3fb"),
         ("Grupo B", "envia 60 unidades", AZUL, "#eef3fb"),
         "Cada subproblema é viável. A união não é,\ne o custo fixo é contado duas vezes."),
        ("CLNS: restante fixo, capacidade residual", VERDE, "#eaf6ef",
         ["Centro compartilhado", "residual 100 − 60 = 40", "custo fixo já pago"],
         ("Grupo A (fixo)", "mantém 60 unidades", CINZA, "#f1f1f1"),
         ("Grupo B (livre)", "pode enviar até 40", AZUL, "#eef3fb"),
         "Toda solução aceita é viável no problema inteiro,\ne o custo fixo entra uma única vez."),
    )
    for ax, (titulo, cor, fundo, centro, ga, gb, nota) in zip(axs, paineis):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(titulo, loc="left", color=cor)
        ax.add_patch(FancyBboxPatch((2.9, 4.7), 4.2, 1.9, boxstyle="round,pad=0.03", fc=fundo,
                                    ec=cor, lw=1.8))
        ax.text(5.0, 5.65, "\n".join(centro), ha="center", va="center", color=cor,
                fontweight="bold", linespacing=1.4)
        for x, (nome, acao, c, f) in ((0.3, ga), (6.1, gb)):
            ax.add_patch(FancyBboxPatch((x, 1.7), 3.6, 1.5, boxstyle="round,pad=0.03", fc=f, ec=c,
                                        lw=1.4))
            ax.text(x + 1.8, 2.45, f"{nome}\n{acao}", ha="center", va="center", linespacing=1.4)
            _seta(ax, (x + 1.8, 3.3), (3.8 if x < 5 else 6.2, 4.6), c)
        ax.text(5.0, 0.7, nota, ha="center", va="center", fontsize=10, color=TINTA, linespacing=1.4)
    fig.tight_layout()
    salvar(fig, "02_acoplamento")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.1, 4.3, "O método híbrido: da instância à solução validada (orçamento T, relógio único)",
            fontsize=13, fontweight="bold")
    passos = [
        ("Instância", "m centros, n clientes;\ncusto fixo, capacidade,\ndemanda e custo\npor par", CINZA),
        ("PL forte (HiGHS)", "limite inferior L;\ncustos reduzidos\nde y e de x,\nreaproveitados depois", CINZA),
        ("Ranking de centros", "GNN bipartida, ou\nabertura no PL\n(controle sem\naprendizado)", AZUL),
        ("Expansão adaptativa", "20, 40, 60, 100%\ndos centros; para\nse L + r ≥ U em todo\ncentro podado", VERDE),
        ("CLNS", "subproblemas: cluster,\nfronteira, liberação,\nregret; capacidade\nresidual e memória", VERDE),
        ("Validação", "atribuição, capacidade\ne custo recalculados\ndos dados originais", CINZA),
    ]
    w, h, y = 2.05, 2.3, 1.2
    for k, (t, c, cor) in enumerate(passos):
        x = 0.1 + k * 2.25
        _caixa(ax, x, y, w, h, t, c, cor, fonte=9.0)
        if k:
            _seta(ax, (x - 0.2, y + h / 2), (x, y + h / 2))
    ax.annotate("", xy=(0.1 + 3 * 2.25, 0.85), xytext=(0.1 + 3 * 2.25 + w, 0.85),
                arrowprops={"arrowstyle": "|-|", "color": VERDE, "lw": 1.4})
    ax.text(0.1 + 3 * 2.25 + w / 2, 0.42, "α·T  (α = 0,5, fixado antes)", ha="center",
            fontsize=9, color=VERDE)
    ax.annotate("", xy=(0.1 + 4 * 2.25, 0.85), xytext=(0.1 + 4 * 2.25 + w, 0.85),
                arrowprops={"arrowstyle": "|-|", "color": VERDE, "lw": 1.4})
    ax.text(0.1 + 4 * 2.25 + w / 2, 0.42, "(1 − α)·T", ha="center", fontsize=9, color=VERDE)
    salvar(fig, "03_pipeline")


# ------------------------------------------------------------------ dados
def fig_poda() -> None:
    d = pd.read_csv(RES / "etapa1" / "rho_validacao.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    estilo = {"gnn": ("GNN (aprendida)", AZUL, 2.4, "-"),
              "pl": ("Relaxação linear (sem treino)", TINTA, 1.8, "-"),
              "classico": ("Razão de serviço clássica", CINZA, 1.4, "--"),
              "aleatorio": ("Ordem aleatória", CINZA_CLARO, 1.4, ":")}
    for k, (rot, cor, lw, ls) in estilo.items():
        g = d[d["ranqueador"] == k].sort_values("rho")
        ax.plot(g["rho"], 100 * g["contem_melhor"], ls, color=cor, lw=lw, marker="o", ms=4,
                label=rot)
    ax.axhline(80, color=VERMELHO, lw=1, ls="--")
    ax.text(0.2, 83, "meta pré-registrada: 80% das instâncias", fontsize=9, color=VERMELHO)
    ax.set_xlabel("fração de centros mantida (ρ), antes do reparo de viabilidade")
    ax.set_ylabel("instâncias em que a melhor solução\nconhecida continua alcançável (%)")
    ax.set_ylim(0, 103)
    ax.set_title("Poda por ranking: só é segura mantendo ~80% dos centros", loc="left")
    ax.legend(loc="lower right")
    salvar(fig, "04_poda")


def _cores(metodos: list[str]) -> list[str]:
    return [AZUL if m in APRENDE else CINZA_CLARO for m in metodos]


def _barras(ax, linhas, campo, lo=None, hi=None, pct=False, casas=3):  # type: ignore[no-untyped-def]
    linhas = sorted(linhas, key=lambda r: r[campo])
    y = np.arange(len(linhas))
    k = 100 if pct else 1
    v = np.array([r[campo] for r in linhas]) * k
    xerr = None
    if lo and hi:
        xerr = np.array([[r[campo] - r[lo] for r in linhas], [r[hi] - r[campo] for r in linhas]]) * k
    ax.barh(y, v, color=_cores([r["metodo"] for r in linhas]), xerr=xerr,
            error_kw={"lw": 0.9, "capsize": 2.5, "ecolor": CINZA})
    ax.set_yticks(y, [NOMES.get(r["metodo"], r["metodo"]) for r in linhas])
    ax.invert_yaxis()
    fim = v if xerr is None else v + xerr[1]
    for yi, vi, fi in zip(y, v, fim):
        ax.text(fi, yi, (f"  {vi:.{casas}f}" + ("%" if pct else "")).replace(".", ","),
                va="center", fontsize=8.5)
    ax.set_xlim(0, max(fim) * 1.18)


def _legenda_aprende(ax) -> None:  # type: ignore[no-untyped-def]
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=AZUL, label="usa componente aprendido (GNN ou seletor)"),
                       Patch(color=CINZA_CLARO, label="sem aprendizado")],
              loc="upper right", fontsize=8.5)


def fig_piloto(ag: dict, tag: str, titulo: str) -> None:
    n = ag["n_instancias"]
    linhas = [r for r in ag["metodos"] if r["metodo"] != "ingenua"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 0.36 * len(linhas) + 1.6))
    _barras(axs[0], linhas, "integral", "integral_lo", "integral_hi")
    axs[0].set_xlabel("integral primal (0 = melhor solução desde o início; menor é melhor)\n"
                      "barra = média entre instâncias; traço = IC 95% por bootstrap")
    axs[0].set_title(f"{titulo}: qualidade ao longo do tempo", loc="left")
    _legenda_aprende(axs[0])
    _barras(axs[1], linhas, "desvio_mediano", pct=True, casas=2)
    axs[1].set_xlabel("desvio final mediano em relação à melhor solução conhecida (%)\n"
                      f"{n} instâncias 30×150, 60 s por execução, sementes agregadas por instância")
    axs[1].set_title("qualidade ao fim do orçamento", loc="left")
    fig.tight_layout()
    salvar(fig, f"{tag}_barras")


def fig_anytime(ag: dict, tag: str, destaque: dict[str, tuple[str, float, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    grade = ag["anytime"]["grade"]
    for m, curva in ag["anytime"]["metodos"].items():
        if m not in destaque:
            ax.plot(grade, 100 * np.array(curva), color=CINZA_CLARO, lw=0.8, zorder=1)
    for m, (cor, lw, ls) in destaque.items():
        if m in ag["anytime"]["metodos"]:
            ax.plot(grade, 100 * np.array(ag["anytime"]["metodos"][m]), color=cor, lw=lw, ls=ls,
                    label=NOMES.get(m, m), zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 100)
    ax.set_yticks([0.5, 1, 2, 5, 10, 50, 100], ["0,5", "1", "2", "5", "10", "50", "100"])
    ax.set_xlabel("tempo de parede (s)")
    ax.set_ylabel("gap mediano à melhor solução conhecida (%)\nescala logarítmica")
    ax.set_title("Evolução do melhor custo encontrado ao longo do orçamento", loc="left")
    ax.legend(fontsize=8.5)
    salvar(fig, f"{tag}_anytime")


def fig_ate1_vitorias(ag: dict, tag: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14, 0.36 * len(ag["metodos"]) + 1.6))
    _barras(axs[0], ag["metodos"], "ate_1pct", pct=True, casas=0)
    axs[0].invert_yaxis()
    linhas = sorted(ag["metodos"], key=lambda r: -r["ate_1pct"])
    axs[0].clear()
    y = np.arange(len(linhas))
    v = np.array([100 * r["ate_1pct"] for r in linhas])
    axs[0].barh(y, v, color=_cores([r["metodo"] for r in linhas]))
    axs[0].set_yticks(y, [NOMES.get(r["metodo"], r["metodo"]) for r in linhas])
    axs[0].invert_yaxis()
    for yi, vi in zip(y, v):
        axs[0].text(vi, yi, f"  {vi:.0f}%", va="center", fontsize=8.5)
    axs[0].set_xlim(0, 112)
    axs[0].set_xlabel("instâncias que terminam a até 1% da melhor solução conhecida (%)")
    axs[0].set_title("Quantas instâncias terminam perto do melhor", loc="left")
    par = sorted(ag["pareado"]["linhas"], key=lambda r: r["derrotas"] - r["vitorias"])
    y = np.arange(len(par))
    vi_ = np.array([r["vitorias"] for r in par])
    em = np.array([r["empates"] for r in par])
    de = np.array([r["derrotas"] for r in par])
    axs[1].barh(y, vi_, color=VERDE, label="melhor que o SCIP")
    axs[1].barh(y, em, left=vi_, color=CINZA_CLARO, label="empate (0,01%)")
    axs[1].barh(y, de, left=vi_ + em, color=VERMELHO, alpha=0.8, label="pior que o SCIP")
    axs[1].set_yticks(y, [NOMES.get(r["metodo"], r["metodo"]) for r in par])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("número de instâncias (desvio final, sementes agregadas)")
    axs[1].set_title("Comparação pareada com o SCIP no modelo completo", loc="left")
    axs[1].legend(ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    salvar(fig, f"{tag}_ate1_vitorias")


def fig_tempo(ag: dict, tag: str) -> None:
    t = ag["tempo"]
    met = [m for m in t if "t_sub" in t[m]]
    partes = [("t_adaptativa", "expansão adaptativa (SCIP restrito)", VERDE),
              ("t_sub", "subproblemas do CLNS (SCIP)", AZUL),
              ("t_selecao", "geração e seleção de candidatos", AMBAR),
              ("t_pl", "relaxação linear", CINZA)]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(met) + 1.5))
    y = np.arange(len(met))
    esq = np.zeros(len(met))
    for campo, rot, cor in partes:
        v = np.array([t[m].get(campo, 0.0) for m in met])
        ax.barh(y, v, left=esq, color=cor, label=rot)
        esq += v
    ax.set_yticks(y, [NOMES.get(m, m) for m in met])
    ax.invert_yaxis()
    ax.set_xlabel("segundos por execução (média), orçamento de 60 s")
    ax.set_title("Onde o tempo é gasto", loc="left")
    ax.legend(ncol=2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    salvar(fig, f"{tag}_tempo")


def fig_medicao(ag: dict, tag: str) -> None:
    med = ag["medicao"]
    h = np.array(med["histograma_razao"])
    bins = np.arange(0.8, 1.025, 0.01)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(bins[: len(h)], h, width=0.009, align="edge", color=CINZA)
    ax.axvline(0.9, color=VERMELHO, ls="--", lw=1)
    ax.text(0.902, max(h) * 0.88, "abaixo de 0,9: suspeita de\ncontenção (execução marcada)",
            fontsize=9, color=VERMELHO)
    ax.set_xlabel("tempo de CPU ÷ tempo de parede, por execução (1,0 = nunca esperou pelo processador)")
    ax.set_ylabel("execuções")
    ax.set_title(f"Qualidade da medição de tempo: {med['execucoes']} execuções, mediana "
                 f"{med['razao_cpu_quantis']['0.5']:.3f}", loc="left")
    salvar(fig, f"{tag}_medicao")


def fig_memoria() -> None:
    arq = PILOTO3 / "resultados.csv"
    if not arq.exists():
        print("sem piloto 3")
        return
    d = pd.read_csv(arq)
    d = d[d["metodo"].str.contains("clns|hibrido") & d["iteracoes"].notna()].copy()
    d["taxa"] = d["repetidos"] / d["iteracoes"].clip(lower=1)
    d["pul"] = d["pulados"] / d["iteracoes"].clip(lower=1)
    g = d.groupby(["metodo", "instancia"])[["taxa", "pul", "iteracoes"]].mean().groupby("metodo").mean()
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.5 * len(g) + 1.6))
    g1 = g.sort_values("taxa")
    cores = [VERDE if m.endswith("+mem") else CINZA for m in g1.index]
    axs[0].barh(np.arange(len(g1)), 100 * g1["taxa"], color=cores)
    axs[0].set_yticks(np.arange(len(g1)), [NOMES.get(m, m) for m in g1.index])
    for yi, vi in enumerate(100 * g1["taxa"]):
        axs[0].text(vi, yi, f"  {vi:.1f}%", va="center", fontsize=8.5)
    axs[0].set_xlim(0, max(1.0, 100 * g1["taxa"].max()) * 1.2)
    axs[0].set_xlabel("subproblemas escolhidos que já tinham falhado no mesmo estado local\n"
                      "com orçamento igual ou maior (% das iterações)")
    axs[0].set_title("Taxa de repetição de subproblemas", loc="left")
    axs[1].barh(np.arange(len(g1)), g1["iteracoes"], color=cores)
    axs[1].set_yticks(np.arange(len(g1)), [NOMES.get(m, m) for m in g1.index])
    for yi, vi in enumerate(g1["iteracoes"]):
        axs[1].text(vi, yi, f"  {vi:.0f}", va="center", fontsize=8.5)
    axs[1].set_xlim(0, g1["iteracoes"].max() * 1.15)
    axs[1].set_xlabel("subproblemas resolvidos por execução (média)")
    axs[1].set_title("Iterações no mesmo orçamento (verde = com memória)", loc="left")
    fig.tight_layout()
    salvar(fig, "p3_memoria")


def fig_reconstrucoes() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.2))
    # (a) branching
    ax = axs[0, 0]
    d = pd.read_csv(RES / "l2b" / "teste.csv")
    sg = d.groupby("regra")["tempo"].apply(lambda s: np.exp(np.mean(np.log(s + 1))) - 1)
    rot = {"aprendida": "aprendida\n(LightGBM)", "pscost": "pscost", "relpscost": "relpscost\n(padrão)",
           "fullstrong": "fullstrong"}
    sg = sg.sort_values()
    ax.bar([rot[k] for k in sg.index], sg.values,
           color=[AZUL if k == "aprendida" else CINZA_CLARO for k in sg.index])
    for i, v in enumerate(sg.values):
        ax.text(i, v, f"{v:.1f} s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("tempo até o ótimo (s), média geométrica\ndeslocada, 20 instâncias, limite 300 s")
    ax.set_title("(a) Regra de branching aprendida (Gasse/Gupta): empata com pscost", loc="left")
    # (b) trocas
    ax = axs[0, 1]
    d = pd.read_csv(RES / "swap_v2" / "resultados.csv")
    melhor = d.groupby(["n", "seed"])["custo"].transform("min")
    d["ref"] = np.where(d["status_exato"].eq("optimal"), d["otimo"], melhor)
    d["dev"] = d["custo"] / d["ref"] - 1
    full = d[d["filtro"] == "completo"].set_index(["n", "seed", "inicio"])["tempo"]
    d["sp"] = [full[(n, s, i)] / t for n, s, i, t in zip(d["n"], d["seed"], d["inicio"], d["tempo"])]
    g = d[d["n"] == 1000].groupby(["filtro", "fallback", "k_in"])[["sp", "dev"]].median().reset_index()
    est = {"classico": ("filtro clássico (ganho de adição)", TINTA), "aprendido": ("filtro aprendido", AZUL),
           "aleatorio": ("filtro aleatório", CINZA_CLARO), "completo": ("busca completa", VERMELHO)}
    for f, (r, cor) in est.items():
        for fb, mk in ((False, "o"), (True, "s")):
            q = g[(g["filtro"] == f) & (g["fallback"] == fb)]
            if q.empty:
                continue
            ax.scatter(q["sp"], 100 * q["dev"], color=cor if not fb else "none", edgecolors=cor,
                       marker=mk, s=42, label=r + (" + busca completa ao final" if fb else "")
                       if f != "completo" else r)
    ax.set_xscale("log")
    ax.set_xlabel("aceleração sobre a busca completa de trocas (escala log; direita é melhor)")
    ax.set_ylabel("perda em relação à melhor solução (%)\n(baixo é melhor)")
    ax.set_title("(b) Filtro de trocas no p-mediana, n = 1000 (Guo/Su)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    # (c) UniFL
    ax = axs[1, 0]
    a = pd.read_csv(RES / "unifl" / "resultados.csv")
    b = pd.read_csv(RES / "unifl" / "resultados_estavel.csv")
    todos = pd.concat([a, b])
    melhor = todos.groupby(["n", "seed"])["custo"].min()
    d = pd.concat([a[a["metodo"].isin(["mettu_plaxton", "mp+busca_local"])],
                   b[b["metodo"].str.match(r"mpnn_estavel_s\d(\+busca_local)?$")]])
    d["ref"] = [o if np.isfinite(o) else melhor[(n, s)] for o, n, s in zip(d["otimo"], d["n"], d["seed"])]
    d["r"] = d["custo"] / d["ref"]
    d["grupo"] = d["metodo"].str.replace(r"_s\d", "", regex=True)
    est2 = {"mettu_plaxton": ("Mettu-Plaxton (clássico)", TINTA, "-"),
            "mpnn_estavel": ("MPNN não supervisionada", AZUL, "-"),
            "mp+busca_local": ("Mettu-Plaxton + busca local", TINTA, "--"),
            "mpnn_estavel+busca_local": ("MPNN + busca local", AZUL, "--")}
    for gk, (r, cor, ls) in est2.items():
        q = d[d["grupo"] == gk].groupby("n")["r"]
        med = q.median()
        ax.plot(med.index, med.values, ls, color=cor, marker="o", ms=4, label=r, lw=1.8)
        if "busca" not in gk:
            ax.fill_between(med.index, q.quantile(0.1).values, q.quantile(0.9).values, color=cor, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 500, 1000], ["100", "200", "500", "1000"])
    ax.axvspan(90, 220, color="0.93", zorder=0)
    ax.text(95, ax.get_ylim()[1] * 0.995, "tamanhos de treino", fontsize=8.5, va="top")
    ax.set_xlabel("número de pontos n (escala log)")
    ax.set_ylabel("custo ÷ referência (1,0 = referência)\nlinha = mediana; faixa = decis 10 a 90")
    ax.set_title("(c) Localização uniforme (Qian et al.): generaliza na escala", loc="left")
    ax.legend(fontsize=8)
    # (d) LRP
    ax = axs[1, 1]
    d = pd.concat([pd.read_csv(RES / "lrp" / "resultados.csv"),
                   pd.read_csv(RES / "lrp" / "resultados_suplementar.csv")])
    d["dev"] = 100 * (d["custo"] / d["referencia"] - 1)
    q = d.groupby("metodo")["dev"].agg(["median", "mean"]).sort_values("mean")
    rot = {"flp_vrp": "FLP → VRP (sequencial)", "aprox_cont": "aproximação contínua (clássica)",
           "neo_ds": "Deep Sets no MIP (big-M)", "neo_multi": "Deep Sets no MIP, multi-partida",
           "neo_busca": "Deep Sets + busca guiada"}
    y = np.arange(len(q))
    ax.barh(y, q["mean"], color=[AZUL if m.startswith("neo") else CINZA_CLARO for m in q.index])
    ax.plot(q["median"], y, "k|", ms=12, mew=2, label="mediana")
    ax.set_yticks(y, [rot.get(m, m) for m in q.index])
    ax.invert_yaxis()
    for yi, vi in zip(y, q["mean"]):
        ax.text(max(vi, 0), yi, f"  {vi:.2f}%", va="center", fontsize=8.5)
    ax.set_xlabel("desvio do custo em relação à melhor solução conhecida (%)\nbarra = média; traço = mediana")
    ax.set_title("(d) Localização-roteamento (Kaleem et al.)", loc="left")
    ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Reconstruções: azul = método com aprendizado; cinza/preto = equivalente clássico",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    salvar(fig, "10_reconstrucoes")


def main() -> None:
    p3 = json.loads((PILOTO3 / "agregados.json").read_text()) if (PILOTO3 / "agregados.json").exists() else None
    fig_caminho(p3)
    fig_acoplamento()
    fig_pipeline()
    fig_poda()
    p2 = json.loads((PILOTO2 / "agregados.json").read_text())
    fig_piloto(p2, "p2", "Piloto 2")
    fig_anytime(p2, "p2", {"hibrido:aprendido": (AZUL, 2.4, "-"), "adaptativa:gnn": (VERDE, 2.0, "-"),
                           "clns:aprendido": (AMBAR, 1.6, "--"), "lns": (CINZA, 1.4, ":"),
                           "completo": (TINTA, 1.6, ":")})
    fig_ate1_vitorias(p2, "p2")
    fig_tempo(p2, "p2")
    fig_medicao(p2, "p2")
    if p3 is not None:
        fig_piloto(p3, "p3", "Piloto 3 (controles)")
        fig_anytime(p3, "p3", {"hibrido:rotacao": (AZUL, 2.4, "-"), "hibridopl:rotacao": (TINTA, 2.0, "--"),
                               "adaptativa:gnn": (VERDE, 1.8, "-"), "adaptativa:pl": (VERDE, 1.6, "--"),
                               "kernel": (AMBAR, 1.6, "-"), "completo": (CINZA, 1.4, ":")})
        fig_ate1_vitorias(p3, "p3")
        fig_medicao(p3, "p3")
        fig_memoria()
    fig_reconstrucoes()


if __name__ == "__main__":
    main()
