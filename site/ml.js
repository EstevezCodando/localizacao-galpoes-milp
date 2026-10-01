/* Demanda, previsão, tipologias e rede nacional: tabelas e gráficos a partir de D.ml. */
(function () {
  const ML = D.ml;
  if (!ML) return;
  const MODEL_NAMES = { pop_proporcional: "Proporcional à população", glm_pop_dist: "GLM: população e distância", glm_completo: "GLM completo", gbm_completo: "Gradient boosting" };
  const pc = (v, d = 1) => nf(100 * v, d) + "%";

  function barsH(el, rows, o = {}) {
    const W = o.w || 640, rh = o.rh || 28, L = o.l || 210, R = o.r || 90;
    const max = o.max || Math.max(...rows.map((r) => r.value)) * 1.05;
    const sx = (v) => L + (v / max) * (W - L - R);
    let s = "";
    rows.forEach((r, i) => {
      const y = i * rh + 4, color = r.color || "var(--accent)";
      s += `<text x="${L - 10}" y="${y + 15}" text-anchor="end" style="fill:var(--ink)">${esc(r.label)}</text><rect x="${L}" y="${y + 3}" width="${Math.max(2, sx(Math.max(r.value, 0)) - L)}" height="${rh - 9}" rx="2" fill="${color}"/><text class="v" x="${sx(Math.max(r.value, 0)) + 6}" y="${y + 15}">${esc(r.text)}</text>`;
    });
    $(el).innerHTML = svg(W, rows.length * rh + 8, s);
  }

  function linesChart(el, xs, series, o = {}) {
    const W = o.w || 560, H = o.h || 260, L = 52, B = 36, T = 14, R = 20;
    const all = series.flatMap((s) => s.ys), ymin = o.ymin ?? Math.min(...all) * 0.95, ymax = o.ymax ?? Math.max(...all) * 1.05;
    const sx = (i) => L + (xs.length === 1 ? 0 : i / (xs.length - 1)) * (W - L - R), sy = (v) => H - B - ((v - ymin) / (ymax - ymin)) * (H - B - T);
    let s = "";
    for (let k = 0; k <= 4; k++) { const v = ymin + ((ymax - ymin) * k) / 4; s += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${sy(v) + 4}" text-anchor="end">${(o.fmt || ((x) => nf(x, 1)))(v)}</text>`; }
    if (o.ref !== undefined) s += `<line x1="${L}" x2="${W - R}" y1="${sy(o.ref)}" y2="${sy(o.ref)}" stroke="var(--ink)" stroke-dasharray="4 4" stroke-opacity=".6"/>`;
    xs.forEach((x, i) => { s += `<text x="${sx(i)}" y="${H - 14}" text-anchor="middle">${x}</text>`; });
    series.forEach((se) => {
      s += `<polyline fill="none" stroke="${se.color}" stroke-width="2.2" points="${se.ys.map((v, i) => sx(i) + "," + sy(v)).join(" ")}"/>`;
      se.ys.forEach((v, i) => { s += `<circle cx="${sx(i)}" cy="${sy(v)}" r="3.4" fill="${se.color}"/>`; });
    });
    $(el).innerHTML = svg(W, H, s);
  }

  /* demanda: modelos */
  (function () {
    const rows = ["pop_proporcional", "glm_pop_dist", "glm_completo", "gbm_completo"].map((m) => {
      const uf = ML.demanda_metricas.find((r) => r.modelo === m && r.validacao === "uf");
      const rg = ML.demanda_metricas.find((r) => r.modelo === m && r.validacao === "regiao");
      return `<tr><td>${MODEL_NAMES[m]}</td><td class="num">${nf(uf.deviance_poisson, 2)}</td><td class="num">${nf(uf.spearman_pop50k, 2)}</td><td class="num">${nf(rg.deviance_poisson, 2)}</td><td class="num">${nf(rg.spearman_pop50k, 2)}</td></tr>`;
    }).join("");
    $("demModelos").innerHTML = `<table><thead><tr><th>Modelo</th><th>Deviance (UF)</th><th>Spearman (UF)</th><th>Deviance (região)</th><th>Spearman (região)</th></tr></thead><tbody>${rows}</tbody></table>`;
    legend("lgCal", [["var(--grey)", "Proporcional à população"], ["var(--accent)", "Gradient boosting"]]);
    linesChart("demCalib", ML.demanda_calibracao_gbm.map((r) => r.decil), [
      { name: "base", color: "var(--grey)", ys: ML.demanda_calibracao_base.map((r) => r.razao) },
      { name: "gbm", color: "var(--accent)", ys: ML.demanda_calibracao_gbm.map((r) => r.razao) },
    ], { ref: 1, ymin: 0.4, ymax: 2.2, fmt: (v) => nf(v, 1) });
    const labels = { log_dist_polo: "Distância ao polo vendedor", idhm: "IDHM", idhm_r: "IDHM, renda", share_30_44: "Parcela de 30 a 44 anos", share_60_mais: "Parcela de 60 anos ou mais", log_pib_pc: "PIB per capita", gini: "Gini", log_rdpc: "Renda per capita", log_pop: "População", idhm_e: "IDHM, educação", share_15_29: "Parcela de 15 a 29 anos", cagr_10_22: "Crescimento 2010–2022" };
    barsH("demImp", ML.demanda_importancia.map((r) => ({ label: labels[r.variavel] || r.variavel, value: Math.max(r.aumento_deviance, 0), text: nf(r.aumento_deviance, 2) })), { l: 220 });
  })();

  /* previsão */
  (function () {
    const grid = ML.previsao_censo, pick = (w, phi) => grid.find((r) => Math.abs(r.w - w) < 1e-9 && Math.abs(r.phi - phi) < 1e-9);
    const t = ML.previsao_teste.find((r) => r.papel === "escolhida" && r.horizonte === 4);
    barsH("futBacktests", [
      { label: "Série oficial do IBGE (modelada), 4 anos", value: t.mediana_ape, text: pc(t.mediana_ape), color: "var(--grey)" },
      { label: "Censo 2022 real, extrapolação simples", value: pick(1, 1).mediana_ape, text: pc(pick(1, 1).mediana_ape), color: "var(--amber)" },
      { label: "Censo 2022 real, com amortecimento", value: pick(1, 0.9).mediana_ape, text: pc(pick(1, 0.9).mediana_ape), color: "var(--accent)" },
    ], { l: 250, max: 0.1 });
    const rows = [[1, 1, "Extrapolação simples"], [1, 0.9, "Amortecida (φ = 0,9)"], [0.75, 0.9, "Amortecida e encolhida (w = 0,75)"]].map(([w, phi, name]) => { const r = pick(w, phi); return `<tr><td>${name}</td><td class="num">${pc(r.mediana_ape)}</td><td class="num">${pc(r.ape_medio_pop50k)}</td><td class="num">${(r.vies_total > 0 ? "+" : "") + pc(r.vies_total)}</td><td class="num">${pc(r.erro_log_q05, 0)} a +${pc(r.erro_log_q95, 0)}</td></tr>`; }).join("");
    $("futCenso").innerHTML = `<table><thead><tr><th>Método</th><th>Erro mediano</th><th>Erro (≥ 50 mil)</th><th>Viés no total</th><th>Erro em log, 90%</th></tr></thead><tbody>${rows}</tbody></table>`;
    const years = Object.keys(ML.projecao_curva), vals = years.map((y) => ML.projecao_curva[y] / 1e6);
    linesChart("futCurva", years, [{ name: "central", color: "var(--accent)", ys: vals }], { ymin: 200, ymax: 230, fmt: (v) => nf(v, 0), w: 520 });
    $("futCurvaNota").textContent = `De ${nf(vals[0], 1)} milhões em 2025 para ${nf(vals[vals.length - 1], 1)} em 2030 (${(vals[vals.length - 1] / vals[0] - 1 >= 0 ? "+" : "") + pc(vals[vals.length - 1] / vals[0] - 1)}). Soma dos limites por município em 2030: ${nf(ML.projecao_lo["2030"] / 1e6, 1)} a ${nf(ML.projecao_hi["2030"] / 1e6, 1)} milhões.`;
    const names = ["Polos urbanos desenvolvidos", "Norte e Nordeste de menor desenvolvimento", "Pequenos municípios envelhecidos"];
    $("futGrupos").innerHTML = `<table><thead><tr><th>Tipologia</th><th>2025 (mi)</th><th>2030 (mi)</th><th>Variação</th></tr></thead><tbody>${ML.projecao_grupos.map((g) => `<tr><td>${names[g.grupo]}</td><td class="num">${nf(g.pop_2025 / 1e6, 1)}</td><td class="num">${nf(g.pop_2030 / 1e6, 1)}</td><td class="num">${(g.pop_2030 / g.pop_2025 - 1 >= 0 ? "+" : "") + pc(g.pop_2030 / g.pop_2025 - 1)}</td></tr>`).join("")}</tbody></table>`;
  })();

  /* tipologias */
  (function () {
    const names = ["Polos urbanos desenvolvidos", "Norte e Nordeste de menor desenvolvimento", "Pequenos municípios envelhecidos"];
    const totalPop = ML.cluster_perfil.reduce((a, r) => a + r.pop_total, 0), totalPed = ML.cluster_perfil.reduce((a, r) => a + r.pedidos, 0);
    $("tipPerfil").innerHTML = `<table><thead><tr><th>Tipologia</th><th>Municípios</th><th>% pop.</th><th>% pedidos Olist</th><th>IDHM</th><th>Renda (R$ 2010)</th></tr></thead><tbody>${ML.cluster_perfil.map((r) => `<tr><td>${names[r.grupo]}</td><td class="num">${nf(r.municipios)}</td><td class="num">${pc(r.pop_total / totalPop)}</td><td class="num">${pc(r.pedidos / totalPed)}</td><td class="num">${nf(r.idhm, 2)}</td><td class="num">${nf(r.renda_pc, 0)}</td></tr>`).join("")}</tbody></table>`;
    legend("lgK", [["var(--accent)", "Estabilidade (ARI)"], ["var(--amber)", "Silhueta"]]);
    linesChart("tipK", ML.cluster_k.map((r) => "k=" + r.k), [
      { name: "ari", color: "var(--accent)", ys: ML.cluster_k.map((r) => r.estabilidade_ari) },
      { name: "sil", color: "var(--amber)", ys: ML.cluster_k.map((r) => r.silhueta) },
    ], { ymin: 0, ymax: 1.05, fmt: (v) => nf(v, 1), w: 480 });
  })();

  /* rede */
  (function () {
    const C = ML.rede_cenarios;
    if (C.length) {
      const designs = [...new Set(C.map((r) => r.projeto))], scs = [...new Set(C.map((r) => r.cenario))];
      const cell = (d, s) => C.find((r) => r.projeto === d && r.cenario === s);
      const head = `<tr><th>Projeto feito para</th>${scs.map((s) => `<th>${esc(s)}</th>`).join("")}</tr>`;
      const body = designs.map((d) => `<tr><td>${esc(d)}</td>${scs.map((s) => { const r = cell(d, s), a = Math.min(r.arrependimento_pct, 100); return `<td class="num" style="background:color-mix(in srgb,var(--amber) ${a * 0.7}%,transparent)">${nf(r.arrependimento_pct, 1)}%</td>`; }).join("")}</tr>`).join("");
      $("redeRegret").innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
      const only25 = C.filter((r) => r.projeto === "2025 base"), worst25 = Math.max(...only25.map((r) => r.arrependimento_pct));
      const robust = C.filter((r) => r.projeto.startsWith("2030 média")), worstR = Math.max(...robust.map((r) => r.arrependimento_pct));
      $("redeRegretNota").textContent = `Colunas: cenário em que o projeto é operado. Projetar só para 2025 chega a ${nf(worst25, 0)}% de arrependimento no pior cenário de 2030; o projeto para a média dos três cenários de 2030 limita o pior caso a ${nf(worstR, 1)}%. O custo de terceirizar (R$ 100 por pedido) pesa muito nesses números: quando falta capacidade, os pedidos vão para a terceirização.`;
    }
    const P = ML.rede_previsao;
    if (P.length) barsH("redePrev", P.map((r) => ({ label: r.previsao_usada.replace(" (conhecido depois)", ""), value: r.arrependimento_pct, text: nf(r.arrependimento_pct, 2) + "%", color: r.previsao_usada.startsWith("Censo") ? "var(--good)" : "var(--accent)" })), { l: 290, max: Math.max(1, ...P.map((r) => r.arrependimento_pct)) * 1.1 });
    const R = ML.rede_reta;
    if (R.length) {
      const viaria = R.find((r) => r.projeto.includes("viária")), reta = R.find((r) => r.projeto.includes("reta"));
      $("redeReta").innerHTML = `<div class="chips"><span class="chip">projetado com estrada <b>R$ ${nf(viaria.custo_total / 1e6, 2)} mi/mês</b></span><span class="chip">projetado com linha reta <b>R$ ${nf(reta.custo_total / 1e6, 2)} mi/mês</b></span><span class="chip">arrependimento da reta <b>${nf(reta.arrependimento_pct, 2)}%</b></span><span class="chip">módulos em comum (Jaccard) <b>${nf(viaria.jaccard_modulos, 2)}</b></span></div><p class="note">Os dois projetos foram avaliados na estrada. Projetar com linha reta custa mais, e a rede escolhida é diferente em quase um terço dos módulos.</p>`;
    }
    const S = ML.score_metricas, SD = ML.score_decisao, SS = ML.score_sementes;
    if (S.length) {
      const names = { score_demanda: "Ranking por demanda local", score_logit: "Regressão logística", score_gbm: "Gradient boosting" };
      const all = [...SD.map((r) => r.custo_total), ...SS.map((r) => r.custo_total)], best = Math.min(...all);
      const dec = (f, k) => {
        const vals = [...SD.filter((r) => r.filtro === f && r.k === k).map((r) => r.custo_total), ...SS.filter((r) => r.filtro === f && r.k === k).map((r) => r.custo_total)];
        if (!vals.length) return "–";
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        return `+${nf(100 * (mean / best - 1), 1)}% (n=${vals.length})`;
      };
      const fullVals = [...SD.filter((r) => r.filtro === "todos os 120").map((r) => r.custo_total), ...SS.filter((r) => r.filtro === "todos").map((r) => r.custo_total)];
      const full = fullVals.length ? `+${nf(100 * (fullVals.reduce((a, b) => a + b, 0) / fullVals.length / best - 1), 1)}% (n=${fullVals.length})` : "–";
      $("redeScore").innerHTML = `<table><thead><tr><th>Score</th><th>AUC fora da região</th><th>Precisão no top 40</th><th>Excesso, top 45</th><th>Excesso, top 60</th></tr></thead><tbody>${S.map((r) => `<tr><td>${names[r.score]}</td><td class="num">${nf(r.auc_medio, 3)}</td><td class="num">${nf(r.precisao_top40, 2)}</td><td class="num">${dec(r.score, 45)}</td><td class="num">${dec(r.score, 60)}</td></tr>`).join("")}<tr><td>Todos os 120 candidatos</td><td class="num">–</td><td class="num">–</td><td class="num">${full}</td><td class="num">${full}</td></tr></tbody></table>`;
      $("redeScoreNota").textContent = "O score é aprendido com as soluções do otimizador, validado deixando uma macrorregião de fora, e usado para escolher os 45 ou 60 melhores de 120 candidatos. O excesso é sobre o MENOR custo encontrado em todas as execuções (n = execuções). Em ranking (AUC), o ranking simples por demanda empata com os modelos; na decisão, a regressão logística ficou mais perto do melhor custo. O problema com os 120 candidatos converge menos na busca (iterações fixas), por isso não serve de referência.";
    }
    const U = ML.incerteza;
    if (U.length) {
      const keys = [...new Set(U.map((r) => r.folga + "|" + r.rho + "|" + r.sigma))].sort();
      const rows = keys.map((k) => { const g = U.filter((r) => r.folga + "|" + r.rho + "|" + r.sigma === k), f = g[0]; const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
        return `<tr><td>${f.folga}</td><td>${f.rho}</td><td>${f.sigma}</td><td class="num">${g.length}</td><td class="num">${nf(mean(g.map((r) => r.VSS_pct)), 3)}%</td><td class="num">${g.filter((r) => r.mesma_politica === true || r.mesma_politica === "True").length}/${g.length}</td><td class="num">${pc(Math.max(...g.map((r) => r.gap_max_solver)), 0)}</td></tr>`; }).join("");
      $("redeIncert").innerHTML = `<table><thead><tr><th>Folga</th><th>ρ</th><th>σ</th><th>Réplicas</th><th>VSS médio</th><th>Mesma política</th><th>Gap máx. do solver</th></tr></thead><tbody>${rows}</tbody></table>`;
      $("redeIncertNota").textContent = "ρ é a parcela do choque comum a todas as regiões. Quando a política estocástica e a determinística abrem os mesmos centros, o VSS é exatamente zero. Gaps do solver de até dezenas por cento no limite de tempo impedem ler o EVPI com precisão.";
    }
  })();
})();
