"""Renderiza o estudo em Markdown, HTML autossuficiente e figuras PNG/SVG com Matplotlib.

Usa apenas CSV/JSON/NPZ gerados pelo estudo; não roda solvers, não busca dados na rede.
Execute com um Python que tenha matplotlib, numpy e pandas.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

COLORS = ["#007c91", "#e18a27", "#7047aa", "#287d4a", "#bb4055"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titleweight": "bold", "axes.titlesize": 13})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/estudo_integrado_20261001"))
    parser.add_argument("--out", type=Path, default=Path("docs/estudo_integrado"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, out = args.data.resolve(), args.out.resolve()
    figs = out / "figuras"
    figs.mkdir(parents=True, exist_ok=True)
    tables = {}

    def read(name):
        tables[name] = pd.read_csv(data / name)
        return tables[name]

    def save(fig, name):
        fig.savefig(figs / f"{name}.png", dpi=155, bbox_inches="tight")
        fig.savefig(figs / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    def heat(ax, frame, title, fmt=".1f", cmap="viridis", log=False):
        values = frame.to_numpy(dtype=float)
        image = ax.imshow(np.log1p(values) if log else values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(frame)), frame.index)
        ax.set_title(title)
        if values.size <= 60:
            midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
            for i, j in np.ndindex(values.shape):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        color="white" if values[i, j] < midpoint else "black", fontsize=9)
        plt.colorbar(image, ax=ax, shrink=.8)

    space = read("espaco_exato.csv")
    full = space[space.nao_atendido == 0]
    best = space.loc[space.custo.idxmin()]
    tiny = read("didatico_metodos.csv")
    # Esquema, sem geografia inventada: centros e regiões em camadas.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    optimum = list(map(int, best.assignment.split()))
    demands = [4, 8, 8, 4, 2]
    for ax, solution, title in [(axes[0], None, "Problema: qual centro atende cada região?"),
                               (axes[1], optimum, "Solução ótima: B e C; custo = 203 u.m.")]:
        for i, name in enumerate("ABC"):
            ax.scatter(0, i, marker="s", s=650, color=COLORS[i], zorder=3)
            ax.text(-.15, i, f"{name} · capacidade 14", ha="right", va="center")
        for j, demand in enumerate(demands):
            y = j * .5
            ax.scatter(2, y, s=120 + 20*demand, color="#b8c5ce", zorder=3)
            ax.text(2.15, y, f"R{j+1} · {demand}", va="center")
            for i in range(3):
                if solution is None or solution[j] == i:
                    ax.plot([0, 2], [i, y], color=COLORS[i] if solution else "#d0d6dc",
                            lw=2 if solution else .6, zorder=1)
        ax.set(xlim=(-1.05, 2.7), ylim=(-.4, 2.45), title=title)
        ax.axis("off")
    fig.suptitle("Esquema didático — não representa coordenadas reais", fontsize=11)
    save(fig, "01_problema")

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(full.carga_A, full.carga_B, full.custo, color=COLORS[0], alpha=.35, s=22)
    ax.scatter(best.carga_A, best.carga_B, best.custo, color=COLORS[1], marker="*", s=210)
    ax.text(best.carga_A, best.carga_B, best.custo-12, "Ótimo 203", fontsize=11)
    ax.set(xlabel="Carga no centro A (pedidos)", ylabel="Carga no centro B (pedidos)",
           zlabel="Custo total (u.m.)", title="Paisagem discreta: carga A × carga B × custo")
    ax.view_init(elev=23, azim=-55)
    fig.text(.08, .02, "Somente serviço completo: carga C = 26 − A − B. Pontos não formam uma superfície contínua.", fontsize=10)
    save(fig, "02_otimo_3d")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    axes[0].scatter(space.fixo, space.frete + space.penalidade, s=11, alpha=.3, color=COLORS[0])
    x = np.linspace(0, 115, 100)
    axes[0].plot(x, 203-x, "--", color=COLORS[1], label="Isocusto ótimo: F + V = 203")
    axes[0].scatter([73], [130], marker="*", s=180, color=COLORS[1])
    axes[0].set(xlabel="Custo fixo F (u.m.)", ylabel="Frete + não atendimento V (u.m.)",
                title="Todas as atribuições viáveis, incluindo rejeição")
    axes[0].legend(fontsize=8)
    bottom = np.zeros(len(tiny))
    for col, name, color in [("fixo", "Fixo", COLORS[0]), ("frete", "Frete", COLORS[1]),
                             ("penalidade", "Penalidade", COLORS[2])]:
        axes[1].bar(tiny.metodo, tiny[col], bottom=bottom, label=name, color=color)
        bottom += tiny[col]
    for i, v in enumerate(bottom):
        axes[1].text(i, v+3, f"{v:.0f}", ha="center")
    axes[1].axhline(203, ls="--", color="#334455", label="Ótimo enumerado")
    axes[1].set(ylabel="Custo (u.m.)", title="Evolução entre métodos no exemplo")
    axes[1].legend(fontsize=8)
    save(fig, "03_custos_didaticos")

    monthly = read("pedidos_mensais.csv")
    municipalities = read("dados_municipais.csv")
    q = json.loads((data / "qualidade.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    axes[0].plot(range(len(monthly)), monthly.pedidos, color=COLORS[0], marker=".")
    ticks = list(range(0, len(monthly), 4))
    axes[0].set_xticks(ticks, monthly.mes.iloc[ticks], rotation=45)
    axes[0].set(title="Pedidos entregues por mês", ylabel="Pedidos", xlabel="Mês de compra")
    positive = municipalities.pedidos[municipalities.pedidos > 0]
    axes[1].hist(np.log10(positive), bins=28, color=COLORS[0])
    axes[1].set(title="Demanda municipal: distribuição assimétrica",
                xlabel="log10(pedidos), somente municípios positivos", ylabel="Municípios")
    sorted_d = np.sort(municipalities.pedidos)[::-1]
    cumulative = 100 * np.cumsum(sorted_d) / sorted_d.sum()
    axes[2].plot(np.arange(1, len(sorted_d)+1), cumulative, color=COLORS[0])
    axes[2].set(title="Concentração espacial", xlabel="Municípios ordenados por pedidos",
                ylabel="Demanda acumulada (%)", ylim=(0, 103))
    save(fig, "04_distribuicao")

    corr = pd.read_csv(data / "correlacao_spearman.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    heat(ax, corr, "Correlação de Spearman — associação, não causalidade", cmap="RdBu_r")
    save(fig, "05_correlacao")
    uf = pd.read_csv(data / "uf_mes.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    heat(ax, uf, "Intensidade de pedidos: UF × mês · escala log(1 + pedidos)", log=True)
    ax.set(xlabel="Mês", ylabel="UF")
    save(fig, "06_calor_territorial")

    ab = read("ablacao_metricas.csv")
    preds = read("ablacao_previsoes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i, (name, group) in enumerate(ab.groupby("modelo", sort=False)):
        axes[0].scatter([i]*len(group), group.deviance, color=COLORS[i], s=55)
        axes[0].plot([i-.18, i+.18], [group.deviance.mean()]*2, color=COLORS[i], lw=3)
    axes[0].set_xticks(range(3), ["População", "Atributos históricos", "Completo retrospectivo"])
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set(ylabel="Deviance por dobra (menor é melhor)", title="Ablação espacial: 5 blocos de UFs")
    hist = preds[preds.modelo == "historico"]
    axes[1].hexbin(np.log1p(hist.pedidos), np.log1p(hist.previsto), gridsize=35, mincnt=1, cmap="Blues")
    limit = max(np.log1p(hist.pedidos).max(), np.log1p(hist.previsto).max())
    axes[1].plot([0, limit], [0, limit], "--", color=COLORS[1])
    axes[1].set(xlabel="log(1 + observado)", ylabel="log(1 + previsto fora da UF)",
                title="Distribuição dos erros: modelo histórico")
    save(fig, "07_variaveis_ml")

    bench = read("comparacao.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    names = ["Guloso", "Busca local", "MILP", "MILP aquecido", "LNS"]
    for ax, (label, group) in zip(axes, bench.groupby("instancia", sort=False)):
        for i, name in enumerate(names):
            values = group[group.metodo == name].gap_ub_pl_pct
            ax.scatter([i]*len(values), values, color=COLORS[i], s=65)
        ax.set_xticks(range(5), names, rotation=25, ha="right")
        ax.set(title=f"{label}: orçamento 10 s", ylabel="(custo − LB do PL) / custo (%)")
        ax.grid(axis="y", alpha=.2)
    save(fig, "08_comparacao")
    trace = read("trajetoria_lns.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, (label, group) in zip(axes, trace.groupby("instancia", sort=False)):
        for seed, g in group.groupby("seed"):
            ax.step(g.tempo_s, g.custo / 1000, where="post", color=COLORS[int(seed)], label=f"Semente {seed}")
        ax.axhline(group.lb_pl.iloc[0]/1000, color="#334455", ls="--", label="Limite inferior PL")
        ax.set(title=f"{label}: incumbente efetivamente registrado", xlabel="Tempo decorrido (s)", ylabel="Custo (R$ mil / horizonte)")
        ax.legend(fontsize=8)
    save(fig, "09_evolucao")

    sens = read("sensibilidade_controlada.csv")
    labels = {"demand": "Demanda", "capacity": "Capacidade", "fixed": "Custo fixo",
              "transport": "Frete unitário", "penalty": "Penalidade"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (factor, g) in zip(axes.flat, sens.groupby("fator", sort=False)):
        ax.plot(g.nivel, g.custo/1000, "o-", color=COLORS[0], label="Custo")
        lower = g.bound_solver.to_numpy()/1000
        ax.fill_between(g.nivel, lower, g.custo/1000, color=COLORS[0], alpha=.17, label="LB–UB do solver")
        ax.set(title=labels[factor], xlabel="Multiplicador; outros fatores fixos", ylabel="Custo (R$ mil)")
        ax2 = ax.twinx()
        ax2.plot(g.nivel, g.servico_pct, "s--", color=COLORS[1])
        ax2.set(ylabel="Atendimento (%)", ylim=(0, 103))
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .7, "Azul: custo, com intervalo computacional LB–UB\nLaranja: atendimento\nNão são intervalos de confiança.\n50 regiões × 15 centros; 3 s por solve.", va="top")
    save(fig, "10_sensibilidade")

    grid = read("interacao_demanda_capacidade.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, field, title in zip(axes, ["custo", "servico_pct", "gap_solver_pct"],
                               ["Custo (R$ mil)", "Atendimento (%)", "Gap do solver (%)"]):
        frame = grid.pivot(index="escala_demanda", columns="escala_capacidade", values=field)
        if field == "custo":
            frame /= 1000
        heat(ax, frame, title)
        ax.set(xlabel="Multiplicador de capacidade", ylabel="Multiplicador de demanda")
    save(fig, "11_interacao")

    # Geografia real proveniente do arquivo IBGE já coletado, sem requisição de rede.
    regions = pd.read_csv(data / "regioes_50.csv")
    centers = pd.read_csv(data / "centros_15.csv")
    selected = bench[bench.instancia == "50x15"].sort_values("custo").iloc[0]
    assigned = np.load(data / "solucoes" / f"{selected.id}.npy")
    geo = json.loads((root / "data/bronze/geo/ibge_uf_malha.geojson").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    for feature in geo["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polygons:
            ax.add_patch(Polygon(poly[0], facecolor="#f2f4f5", edgecolor="#c0c8ce", linewidth=.5))
    segments = []
    for j, i in enumerate(assigned):
        if i >= 0:
            segments.append([(centers.lon.iloc[i], centers.lat.iloc[i]),
                             (regions.lon.iloc[j], regions.lat.iloc[j])])
    ax.add_collection(LineCollection(segments, colors=COLORS[0], linewidths=.8, alpha=.35))
    ax.scatter(regions.lon, regions.lat, s=15 + 130*regions.orders/regions.orders.max(),
               color=COLORS[0], alpha=.65, label="Regiões · área proporcional a pedidos")
    opened = np.unique(assigned[assigned >= 0])
    ax.scatter(centers.lon.iloc[opened], centers.lat.iloc[opened], marker="^", s=70,
               color=COLORS[1], edgecolors="black", linewidths=.4, label="Centros abertos")
    ax.set(xlim=(-74, -33), ylim=(-34, 6), xlabel="Longitude (graus)", ylabel="Latitude (graus)",
           title=f"Rede 50×15: melhor execução desta rodada ({selected.metodo})")
    ax.set_aspect(1 / np.cos(np.deg2rad(15)))
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "12_rede")

    historical = pd.read_csv(root / "results/rede_cenarios.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    heat(ax, historical.pivot(index="projeto", columns="cenario", values="arrependimento_pct"),
         "Histórico: excesso frente ao melhor projeto testado por cenário (%)")
    save(fig, "13_cenarios_historicos")

    # Documento e figuras partilham as mesmas tabelas carregadas acima.
    sections = []

    def section(title, paragraphs, figures=(), table=None):
        sections.append((title, paragraphs, figures, table))

    def pt(value, decimals=2):
        return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")

    benchmark_summary = bench.groupby(["instancia", "metodo"], sort=False).agg(
        execucoes=("custo", "size"), custo_medio=("custo", "mean"),
        gap_medio_pct=("gap_ub_pl_pct", "mean"), tempo_medio_s=("tempo_s", "mean"),
        atendimento_pct=("servico_pct", "mean")).reset_index()
    global_metrics = []
    for name, group in preds.groupby("modelo", sort=False):
        actual, predicted = group.pedidos.to_numpy(), group.previsto.to_numpy()
        # Deviance de Poisson sem depender de sklearn no Python de renderização.
        logterm = np.zeros(len(actual))
        nz = actual > 0
        logterm[nz] = actual[nz] * np.log(actual[nz] / predicted[nz])
        global_metrics.append({"modelo": name, "deviance_global": float(2*np.mean(logterm-actual+predicted)),
                               "MAE_pedidos": float(np.abs(actual-predicted).mean()),
                               "previsto_observado": float(predicted.sum()/actual.sum())})
    gm = pd.DataFrame(global_metrics)
    gm.to_csv(data / "ablacao_resumo.csv", index=False)
    provenance = read("proveniencia.csv")
    verified = provenance[provenance.verificavel]
    mismatch = int((verified.hash_confere == False).sum())  # noqa: E712
    historical_scores = pd.read_csv(root / "results/rede_score_decisao_sementes.csv")
    section("1. Contexto", [
        "Uma rede de distribuição precisa aproximar a infraestrutura da demanda, sem abrir centros demais nem exceder sua capacidade. O projeto investiga essa decisão com geografia e pedidos do Olist, dados territoriais e cenários de custo. O objetivo deste estudo é tornar explícito o caminho entre dado, hipótese, decisão e evidência.",
        "Há três níveis distintos: (1) um exemplo didático exato; (2) experimentos novos sobre recortes históricos do Olist; (3) resultados nacionais anteriores, baseados em demanda potencial e cenários. Não somamos nem comparamos diretamente custos desses níveis: têm escalas e horizontes diferentes.",
        "O estado do projeto foi relido em 01/10/2026. A coleta territorial e os modelos de ML já existiam. Nesta rodada foram acrescentados o espaço completo de soluções, trajetórias reais do LNS, ablação temporal de atributos, análise exploratória gráfica e experimentos controlados com artefatos reproduzíveis."])
    section("2. Problema", [
        "Dados centros candidatos i e regiões j, escolher quais centros abrir e atribuir cada região a no máximo um centro. Cada centro tem capacidade Qᵢ e custo fixo fᵢ; cada região tem demanda dⱼ; transportar uma unidade custa cᵢⱼ. Demanda não atendida paga p por unidade. A decisão yᵢ é binária, assim como a atribuição xᵢⱼ.",
        "Objetivo: minimizar Σᵢ fᵢyᵢ + Σᵢⱼ dⱼcᵢⱼxᵢⱼ + p Σⱼ dⱼ(1 − Σᵢxᵢⱼ). Restrições: Σᵢxᵢⱼ ≤ 1; Σⱼdⱼxᵢⱼ ≤ Qᵢyᵢ; xᵢⱼ ≤ yᵢ; x,y ∈ {0,1}.",
        "Viabilidade matemática não significa atendimento integral: rejeitar pedidos é permitido nesta formulação. No estudo histórico, a penalidade representa demanda não atendida no horizonte; não existe backlog. Na rede nacional anterior, o mesmo termo foi interpretado como terceirização, uma hipótese distinta que exige capacidade e nível de serviço externos para aplicação operacional."],
        [("01_problema", "Cada linha é uma atribuição possível; à direita estão apenas as escolhas ótimas do exemplo.")])
    section("3. O que buscamos encontrar com as otimizações", [
        "Buscamos uma combinação de abertura, alocação e utilização que reduza o custo modelado, respeitando capacidade e explicando o atendimento. As saídas são: centros abertos, região atendida por cada centro, carga, custo fixo, frete, penalidade, atendimento e certificado ou limite de qualidade.",
        "Ganho computacional é redução de objetivo nas mesmas premissas. Ganho econômico exige validar custos e operação real. O número de centros não deve ser minimizado isoladamente: fechar um centro pode aumentar frete ou rejeição. Também não basta observar um mapa mais compacto.",
        "Para minimização, LB ≤ ótimo ≤ UB: UB é o custo de uma solução viável e LB vem da relaxação ou do solver. Este estudo usa gap = (UB − LB)/UB. O LP é um limite, não uma solução operacional e nem, necessariamente, o ótimo inteiro."])
    section("4. Visualização do ótimo em até três dimensões", [
        f"O exemplo tem 3 centros, capacidade 14 por centro, 5 regiões e demanda total 26. São 4⁵ = 1.024 atribuições possíveis, contando não atendimento. Foram enumeradas todas: {len(space)} são viáveis, das quais {len(full)} atendem tudo. O custo mínimo é {pt(best.custo,0)} unidades monetárias; MILP e enumeração concordaram.",
        "Redução exata para mostrar o objetivo: F = custo fixo e V = frete + penalidade; custo = F + V. Cada ponto é uma solução calculada, e linhas de isocusto são retas. A projeção pode sobrepor decisões diferentes, por isso a prova vem da enumeração, não do desenho.",
        "Na figura 3D os eixos são carga em A, carga em B e custo. Somente nesse gráfico filtramos atendimento completo; a carga em C é 26 − carga A − carga B. O ótimo global foi verificado também contra soluções com rejeição. Várias atribuições podem ter cargas iguais e custos diferentes; não existe uma superfície suave que autorize descida por gradiente.",
        "A solução ótima abre B e C: B atende R2, R4 e R5 (carga 14); C atende R1 e R3 (carga 12); A fecha. Custo = 73 fixo + 130 de transporte = 203. Guloso = 232; busca local = 210. Essa sequência compara métodos; a trajetória temporal real aparece mais adiante.",
        "No HTML, o simulador reenumera as 1.024 alternativas quando se muda demanda, capacidade ou custo fixo. Os controles são parâmetros do problema, não dimensões ocultas de uma superfície contínua."],
        [("02_otimo_3d", "Projeção de decisões discretas em três eixos; estrela indica ótimo enumerado."),
         ("03_custos_didaticos", "F + V permite comparar custos em duas dimensões sem inventar uma função convexa.")])
    section("5. Dados que contribuem para o problema", [
        "Olist fornece pedidos, datas, status, regiões de compradores e vendedores. IBGE/Ipeadata fornecem população, estrutura etária, renda, desenvolvimento e atividade econômica. ANTT e levantamento de aluguel informam componentes de cenários de custo; OSRM informa distâncias viárias na rede nacional anterior. Cada fonte responde a uma pergunta diferente.",
        "Demanda, custos, capacidade e penalidade entram diretamente no otimizador. Idade, renda e população entram em modelos de distribuição da demanda; não são multiplicadores mágicos da função objetivo. OSM pode apoiar triagem de infraestrutura, mas existência de um objeto no mapa não comprova imóvel disponível, capacidade operacional ou preço.",
        "Datas precisam permanecer visíveis: pedidos de 2016–2018, renda/IDHM de 2010 e atributos etários/crescimento até 2022. Usar 2022 para explicar 2017–2018 é análise retrospectiva; não é previsão disponível naquela época. Demanda nacional futura continua sendo cenário, não observação."], table=pd.DataFrame([
            {"fonte": "Olist", "papel": "Demanda histórica, sazonalidade e geografia", "limite": "Marketplace específico; não mede todo o mercado"},
            {"fonte": "IBGE / Ipeadata", "papel": "Exposição populacional e atributos territoriais", "limite": "Datas e revisões diferentes; verificar disponibilidade na decisão"},
            {"fonte": "ANTT", "papel": "Coeficientes para cenário de frete", "limite": "Piso por viagem não é tarifa observada por pedido"},
            {"fonte": "Aluguel", "papel": "Cenário de custo de área", "limite": "Preço pedido; dimensão do imóvel precisa ser compatível"},
            {"fonte": "OSRM / OSM", "papel": "Distância e infraestrutura mapeada", "limite": "Perfil viário não garante acesso de caminhão ou disponibilidade"}]))
    section("6. Coleta, ajuste e validação dos dados", [
        "Reaproveitamos os dados brutos já coletados; não houve necessidade de repetir requisições externas. O ajuste reconstruiu séries mensais, juntou o painel municipal à tabela territorial com cardinalidade many-to-one, calculou ausências e correlações e preservou IDs. Os recortes novos de otimização usam compras entre janeiro/2017 e agosto/2018, explicitamente.",
        f"No bruto há {pt(q['pedidos_brutos'],0)} pedidos, {pt(q['entregues'],0)} entregues e {q['order_id_duplicados']} IDs de pedido duplicados. A tabela territorial tem {pt(q['municipios'],0)} municípios e {pt(q['municipios_completos'],0)} linhas completas para a ablação. {pt(q['municipios_sem_pedidos'],0)} municípios não têm pedidos no painel de modelagem; isso não demonstra ausência de mercado.",
        f"A verificação de proveniência encontrou {len(provenance)} metadados, dos quais {len(verified)} permitiram confrontar arquivo e SHA-256; divergências detectadas: {mismatch}. Registros sem arquivo/hash verificável ficam marcados, não são considerados aprovados. Novas coletas OSM estavam presentes no diretório; um registro sem payload identificável não foi usado como evidência de infraestrutura validada.",
        "O relatório municipal de cobertura anterior informa 64 pedidos entregues sem município. Para CEP de três dígitos o pipeline é diferente: perdas e totais não devem ser comparados como se fossem a mesma transformação. A ablação usa a mesma amostra completa nos três modelos, evitando comparar modelos em populações diferentes.",
        "Rastreabilidade: manifest.json registra versões, commit, parâmetros e hashes; cada solve salva assignment e cada instância tem snapshot NPZ. Resultados históricos importados são identificados como históricos. Hash íntegro atesta consistência do arquivo, não correção econômica ou representatividade da fonte."])
    section("7. Processamento exploratório: distribuição, concentração e mapas de calor", [
        f"Os 300 municípios de maior volume concentram {pt(cumulative[299])}% dos pedidos do painel. A distribuição é muito assimétrica: olhar apenas média esconderia a cauda e os municípios sem observação. O histograma usa log10 para tornar a cauda legível; municípios zero são informados separadamente.",
        "A série mensal mostra a evolução do volume observado, que mistura sazonalidade, crescimento e mudança de cobertura do marketplace. A intensidade UF×mês usa log(1+pedidos), não participação percentual. Não é uma previsão.",
        "A matriz de Spearman mede associação monotônica. População e renda podem estar relacionadas com pedidos; correlação não prova que alterar renda causaria uma mudança de demanda. Variáveis territoriais correlacionadas também dificultam atribuir importância isolada."],
        [("04_distribuicao", "Série temporal, histograma e concentração espacial do dado observado."),
         ("05_correlacao", "Matriz calculada dos dados municipais; sem interpretação causal."),
         ("06_calor_territorial", "Mapa de calor permite localizar concentração e mudanças de intensidade.")])
    section("8. As variáveis realmente ajudam a modelar o problema?", [
        "Nova ablação: baseline proporcional à população; GBM com população/renda/IDHM/Gini/PIB históricos; e GBM com o conjunto retrospectivo completo. Usamos os mesmos municípios e cinco dobras agrupadas por UF. Escala e modelo são ajustados somente no treino de cada dobra. Foram mantidos 150 ciclos e a mesma configuração para os dois GBMs, sem busca de hiperparâmetros no teste.",
        "O conjunto histórico exclui crescimento 2010–2022, faixas etárias de 2022 e distância a polos derivados do período completo. Ainda é uma análise espacial contemporânea: datas de publicação do PIB/estimativas não foram reconstruídas para provar disponibilidade em tempo real. O resultado não é um backtest prospectivo de pedidos.",
        "As métricas globais abaixo agregam todas as previsões fora da UF; os pontos no gráfico são métricas por dobra. Diferenças entre dobras não são intervalos de confiança independentes. Razão previsto/observado próxima de 1 indica calibração agregada, mas não elimina erro local.",
        "Para saber se uma variável melhora a decisão, a etapa seguinte precisa alimentar a mesma instância e avaliar custos com capacidade física fixa. O experimento fatorial seguinte testa diretamente fatores de otimização; não confunde importância preditiva com valor econômico."],
        [("07_variaveis_ml", "Comparação fora do bloco geográfico e distribuição conjunta de previsão/observação.")], table=gm)
    section("9. Aplicação e comparação dos algoritmos", [
        "Executamos guloso, busca local, MILP frio, MILP aquecido e LNS em 50×15 e 100×30. O LNS usou três sementes e vizinhança 25; orçamento nominal de 10 s por execução, contando inicialização. Métodos rápidos podem encerrar antes. A relaxação linear foi calculada à parte como referência comum.",
        "O orçamento do MILP passou a descontar construção e aquecimento antes de chamar o backend. O reparo do LNS passou a descontar construção do subproblema. Ambos usam uma thread de solver; overhead final e pequenos excessos continuam registrados. Uma rodada não estabelece superioridade universal nem é teste controlado de hardware.",
        "O guloso ignora custo de abertura na escolha; busca local corrige parte dessa fraqueza. MILP combina incumbente e prova por bounds. LNS reotimiza subconjuntos e aceita melhorias; seus reparos podem encerrar sem prova de ótimo. O valor mostrado abaixo é simulado para a janela histórica inteira, não custo mensal de operação real.",
        "Três sementes dão noção inicial de variação, não uma caracterização estatística definitiva. Comparações com os resultados antigos devem considerar a nova janela explícita, orçamento e instrumentação."],
        [("08_comparacao", "Pontos individuais, sem esconder variação do LNS numa única média.")], table=benchmark_summary)
    section("10. Evolução da otimização e ponto que buscamos", [
        "O gráfico temporal registra o incumbente inicial e o custo após cada reparo efetivamente executado no LNS. Não foi criado interpolando resultados finais ou recomeçando o solver para cada tempo. Patamares indicam iterações sem melhoria; quedas indicam uma solução melhor aceita.",
        "O alvo ideal é atingir o ótimo inteiro. Na instância grande ele só é conhecido quando há prova; o limite do LP delimita o espaço de melhora ainda possível. Encostar visualmente numa linha não certifica igualdade. No exemplo didático, o alvo 203 é conhecido porque todo o espaço foi enumerado."],
        [("09_evolucao", "Trajetória real; linha tracejada é limite inferior, não ótimo declarado.")])
    section("11. Análise de comportamento e interação entre variáveis", [
        "A nova sensibilidade varia um fator por vez: demanda, capacidade, custo fixo, frete e penalidade. Demanda não recalcula capacidade; frete não recalcula penalidade. A instância-base é congelada. O custo é acompanhado de atendimento e intervalo computacional entre o bound do MILP e o incumbente.",
        "Na grade 5×5, demanda e capacidade variam independentemente, com demais fatores fixos. O mapa de atendimento identifica regimes com falta de capacidade ou rejeição economicamente escolhida. Os mapas não são superfícies em que devemos escolher demanda menor para 'otimizar': demanda é cenário externo, não variável livre da decisão.",
        f"O maior gap de solver na grade foi {pt(grid.gap_solver_pct.max())}%. As diferenças de custo menores que a incerteza computacional não sustentam uma conclusão precisa de efeito. Quantidade de centros é uma propriedade do incumbente e pode variar entre soluções quase equivalentes.",
        "Uma previsão de aumento de demanda só indica necessidade de expansão quando testada contra capacidade instalada, custos e atendimento. A cadeia de análise passa a verificar esse elo diretamente, em vez de inferir robustez de uma rede que crescia junto com a demanda."],
        [("10_sensibilidade", "Fatores isolados; faixa azul representa LB–UB, não IC estatístico."),
         ("11_interacao", "Dois parâmetros externos e uma resposta por painel: custo, serviço e qualidade computacional.")])
    section("12. Visualização geográfica da solução", [
        f"O mapa usa exatamente o assignment da melhor execução 50×15 desta rodada, método {selected.metodo}. Os limites estaduais são do arquivo IBGE coletado; pontos são centroides das regiões de CEP e candidatos, não endereços de imóveis. As ligações são retas de atribuição, não rotas viárias.",
        "A concentração dos pontos decorre também da seleção das regiões com mais pedidos. Portanto, o desenho não representa cobertura territorial nacional completa. Os círculos indicam volume; os triângulos indicam centros abertos. Avaliar acessibilidade e prazo exige matriz viária e regras do veículo adequadas ao caso de uso."],
        [("12_rede", "Geografia real do recorte; decisões e figuras compartilham o mesmo arquivo de solução.")])
    section("13. Rede nacional, previsão, cenários e scoring já existentes", [
        "Os resultados nacionais anteriores foram preservados e identificados como históricos. Eles combinam propensão estimada, população e hipóteses de participação/crescimento do mercado. O custo fixo inclui aluguel, mas ainda não todos os custos de operar um centro. A divisão de municípios grandes em zonas co-localizadas evita nós impossíveis de atender, ao preço de uma aproximação geográfica.",
        "A matriz a seguir mede excesso frente ao melhor dos projetos testados em cada cenário. Não é arrependimento em relação ao ótimo desconhecido, nem demonstra ganho realizado em 2030. As avaliações de recurso são heurísticas: diferenças pequenas podem incluir erro de otimização.",
        "O arquivo histórico reta×estrada reporta cerca de 1,27% de excesso para o projeto feito com reta quando ambos são avaliados com estrada. É evidência de sensibilidade da decisão naquele experimento, não um multiplicador universal. O scoring tem repetições por semente; comparar filtros requer controlar o orçamento e considerar o conjunto completo como referência heurística.",
        f"Há {len(historical_scores)} registros de scoring com sementes no arquivo existente. A previsão populacional tem validações próprias, mas a transferência população→pedidos é hipótese adicional. O experimento histórico previsão→decisão usa covariáveis e propensão retrospectivas: não certifica uma decisão integralmente implementável com informação disponível antes de 2022.",
        "Não refiz essas rodadas nacionais longas: já existem resultados para o processo pedido, e a prioridade foi executar os elos ausentes e tornar sua interpretação verificável. Para aplicação real, repetir com snapshots temporais, custos operacionais, SLA e bounds de recurso."],
        [("13_cenarios_historicos", "Resultados anteriores, recalculados para visualização; não reexecutados nesta rodada.")])
    section("14. Explicação geral e conclusão", [
        "O processo completo passa a ser: definir decisão e horizonte → coletar/preservar fontes → ajustar e validar → explorar distribuição e associações → testar contribuição preditiva → construir instância com unidades explícitas → otimizar → avaliar independentemente → comparar sob protocolo declarado → testar sensibilidade e cenários → comunicar solução e limites.",
        "As melhorias implementadas permitem ver por que o custo cai, quando a capacidade passa a limitar o atendimento e que parte da capacidade preditiva depende de atributos posteriores. Não é necessário que todo teste favoreça o modelo mais complexo: um baseline melhor em algum recorte é informação útil.",
        "O próximo salto de maturidade depende de três validações externas: capacidade operacional por período; custo de instalações compatíveis com a área; tarifa e nível de serviço reais. A análise atual demonstra funcionamento computacional e comportamento sob hipóteses, sem prometer uma economia operacional ainda não medida."])
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    section("15. Reprodutibilidade e arquivos de evidência", [
        f"Execução nova: {data.name}. Commit-base: {manifest['commit']}. Python de análise: {manifest['python']}. As mudanças desta rodada são identificadas também pelos hashes do código; o commit-base sozinho não representa o estado modificado.",
        "Rodar análise no ambiente do projeto: python -m alocacao_capacitada.analysis.study --out results/uma_pasta_nova. Rodar renderização com Matplotlib: python scripts/render_study.py --data results/uma_pasta_nova --out docs/estudo_integrado. O módulo impede sobrescrever uma execução existente.",
        "Arquivos principais: qualidade.json, proveniencia.csv, dados_municipais.csv, correlacao_spearman.csv, ablacao_metricas.csv, ablacao_previsoes.csv, espaco_exato.csv, comparacao.csv, trajetoria_lns.csv, sensibilidade_controlada.csv, interacao_demanda_capacidade.csv, manifest.json, snapshots NPZ e assignments NPY. Figuras têm versões PNG e SVG.",
        "Fontes primárias dos dados permanecem nos metadados de coleta do projeto. Olist: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce; IBGE: https://www.ibge.gov.br/; ANTT: https://www.gov.br/antt/; OSRM: https://project-osrm.org/. Os gráficos geográficos atribuem limites ao IBGE. Não houve nova verificação de vigência tarifária nesta rodada."])

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return pt(value)
        return str(value)

    md, body = ["# Do contexto ao ótimo: estudo integrado\n\n01/10/2026\n"], []
    for i, (title, paragraphs, figures, table) in enumerate(sections):
        md.append(f"\n## {title}\n")
        body.append(f'<section id="s{i}"><h2>{html.escape(title)}</h2>')
        for paragraph in paragraphs:
            md.append(paragraph + "\n")
            body.append(f"<p>{html.escape(paragraph)}</p>")
        if table is not None:
            columns = list(table.columns)
            md.extend(["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)])
            rows = [[cell(v) for v in r] for r in table.itertuples(index=False, name=None)]
            md.extend("| " + " | ".join(r) + " |" for r in rows)
            md.append("")
            body.append('<div class="table"><table><thead><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>")
            body.extend("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r) + "</tr>" for r in rows)
            body.append("</tbody></table></div>")
        for filename, caption in figures:
            md.append(f"![{caption}](figuras/{filename}.png)\n")
            encoded = base64.b64encode((figs / f"{filename}.png").read_bytes()).decode()
            body.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        if i == 3:
            body.append(INTERACTIVE)
        body.append("</section>")
    (out / "estudo_integrado.md").write_text("\n".join(md), encoding="utf-8")
    navigation = "".join(f'<a href="#s{i}">{html.escape(s[0])}</a>' for i, s in enumerate(sections))
    document = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Do contexto ao ótimo</title>'
    document += "<style>" + CSS + "</style><body><header><p class='eyebrow'>ESTUDO REPRODUZÍVEL · 01 OUT 2026</p><h1>Do contexto ao ótimo</h1><p>Dados, decisões e evidências na alocação capacitada</p></header>"
    document += '<details><summary>Índice do estudo</summary><nav>' + navigation + '</nav></details><main>' + "".join(body) + "</main></body></html>"
    (out / "index.html").write_text(document, encoding="utf-8")
    print(out / "index.html")


CSS = """
:root{color-scheme:light;--ink:#1c3441;--teal:#007c91}*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:17px/1.7 system-ui,sans-serif}header,main,details{max-width:1120px;margin:auto;padding:28px 32px}header{padding-top:65px;padding-bottom:30px}h1{font-size:clamp(36px,6vw,70px);line-height:1.05;letter-spacing:-.045em;margin:15px 0}h2{font-size:clamp(23px,3vw,32px);line-height:1.3}h3{font-size:21px}.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--teal)}section{padding:22px 0 38px;border-bottom:1px solid #d5dfe4}p{max-width:92ch}nav{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding-top:16px}a{color:var(--teal)}figure{margin:28px 0;background:white;padding:12px}img{max-width:100%;height:auto}figcaption{font-size:14px;padding:8px;color:#48616c}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}td,th{padding:10px 12px;border-bottom:1px solid #d8e1e5;text-align:left}th{background:#e5eff2}#sim{background:#e5eff2;padding:24px;margin-top:30px}#controls{display:flex;flex-wrap:wrap;gap:20px}label{display:grid;min-width:180px;flex:1}input{width:100%;min-height:32px}#land{width:100%;height:auto;background:#fff;margin-top:18px}#simout{font-weight:600}button{font:inherit;padding:8px 16px}summary{color:var(--teal);cursor:pointer}@media(max-width:650px){header,main,details{padding:20px 16px}nav{grid-template-columns:1fr}figure{padding:2px}#sim{padding:12px}td,th{padding:8px}body{font-size:16px}}@media print{details,#controls{display:none}body{background:white;font-size:11pt}section{break-before:auto}figure{break-inside:avoid}header{padding-top:0}}
"""

INTERACTIVE = """
<div id="sim"><h3>Explore o problema reduzido</h3><p>Enumeração exata a cada ajuste. Eixos: custo fixo e frete + penalidade, em unidades monetárias. Capacidade e custos são hipóteses do exemplo.</p>
<div id="controls"><label>Demanda × <output id="vd">1,00</output><input id="sd" type="range" min="0.5" max="1.5" step="0.05" value="1"></label><label>Capacidade × <output id="vq">1,00</output><input id="sq" type="range" min="0.5" max="1.5" step="0.05" value="1"></label><label>Custo fixo × <output id="vf">1,00</output><input id="sf" type="range" min="0.25" max="4" step="0.25" value="1"></label></div>
<svg id="land" viewBox="0 0 760 380" role="img" aria-label="Soluções viáveis no plano custo fixo por custo variável"></svg><p id="simout" aria-live="polite"></p><p id="assignment"></p></div>
<script>(()=>{const $=id=>document.getElementById(id),fmt=v=>v.toLocaleString('pt-BR',{maximumFractionDigits:2});
function update(){const ds=+$('sd').value,qs=+$('sq').value,fs=+$('sf').value;[['vd',ds],['vq',qs],['vf',fs]].forEach(([id,v])=>$(id).textContent=fmt(v));const d=[4,8,8,4,2].map(v=>v*ds),f=[33,35,38].map(v=>v*fs),c=[[6,6,8,8,8],[7,6,8,1,1],[7,4,6,4,8]],pts=[];let best=null;
for(let code=0;code<1024;code++){let k=code,a=[],load=[0,0,0],variable=0,reject=0,used=new Set();for(let j=0;j<5;j++){const i=k%4-1;k=Math.floor(k/4);a.push(i);if(i<0){variable+=50*d[j];reject+=d[j]}else{load[i]+=d[j];used.add(i);variable+=c[i][j]*d[j]}}if(load.some(v=>v>14*qs+1e-8))continue;const fixed=[...used].reduce((s,i)=>s+f[i],0),p={fixed,variable,total:fixed+variable,a,reject};pts.push(p);if(!best||p.total<best.total-1e-8)best=p;}
const xmax=Math.max(...pts.map(p=>p.fixed),1)*1.1,ymax=Math.max(...pts.map(p=>p.variable),1)*1.08,x=v=>65+640*v/xmax,y=v=>320-280*v/ymax;let s='<title>Espaço discreto e ótimo por enumeração</title><path d="M65 40 V320 H710" fill="none" stroke="#49616c"/>';
for(let i=0;i<=4;i++){const xv=xmax*i/4,yv=ymax*i/4;s+=`<text x="${x(xv)}" y="342" text-anchor="middle" font-size="13">${fmt(xv)}</text><text x="57" y="${y(yv)+4}" text-anchor="end" font-size="13">${fmt(yv)}</text>`;}
s+=pts.map(p=>`<circle cx="${x(p.fixed)}" cy="${y(p.variable)}" r="2.3" fill="#007c91" opacity=".22"/>`).join('');s+=`<circle cx="${x(best.fixed)}" cy="${y(best.variable)}" r="7" fill="#e18a27" stroke="#733b00"/><text x="390" y="371" text-anchor="middle" font-size="14">Custo fixo (u.m.)</text><text x="70" y="22" font-size="14">Frete + penalidade (u.m.)</text>`;$('land').innerHTML=s;
$('simout').textContent=`Ótimo: ${fmt(best.total)} = ${fmt(best.fixed)} fixo + ${fmt(best.variable)} variável. Atendimento: ${fmt(100*(1-best.reject/d.reduce((a,b)=>a+b,0)))}%. ${pts.length} alternativas viáveis de 1.024.`;$('assignment').textContent=best.a.map((v,j)=>`R${j+1} → ${v<0?'não atendida':'ABC'[v]}`).join(' · ');}
['sd','sq','sf'].forEach(id=>$(id).addEventListener('input',update));update();})();</script>
"""


if __name__ == "__main__":
    main()
