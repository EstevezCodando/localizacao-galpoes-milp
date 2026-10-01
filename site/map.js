/* Mapa da rede nacional (MapLibre GL). Base padrão: fronteiras dos estados (IBGE) embutidas, que
   funcionam sem rede. Base "Ruas": OpenFreeMap (tiles vetoriais do OpenStreetMap), se a rede
   permitir. Sem WebGL ou sem a biblioteca, cai para um SVG estático com as mesmas camadas. */
(function () {
  const M = D.mapa, box = $("gmap");
  if (!M || !M.vistas || !M.vistas.length) { box.innerHTML = "<p class='note' style='padding:16px'>Mapa não gerado nesta versão.</p>"; return; }
  const STREETS = "https://tiles.openfreemap.org/styles/liberty";
  const brl = (v) => "R$ " + nf(v / 1e6, 2) + " mi";
  let view = 0, base = "borders", map = null, popup = null, streetsTimer = null;

  const pal = () => {
    const s = getComputedStyle(document.documentElement), g = (n) => s.getPropertyValue(n).trim();
    return { surface: g("--surface"), land: g("--bg"), line: g("--line"), accent: g("--accent"), amber: g("--amber"), grey: g("--grey"), ink: g("--ink") };
  };
  const cur = () => M.vistas[view];

  /* hubs agrupados por município: um marcador por cidade, com a lista de módulos */
  function hubsByCity() {
    const byCity = new Map();
    cur().hubs.forEach((h) => {
      const c = byCity.get(h.cod) || { cod: h.cod, nome: h.nome, uf: h.uf, lat: h.lat, lon: h.lon, modulos: [], carga: 0, capacidade: 0 };
      c.modulos.push(h.modulo); c.carga += h.carga; c.capacidade += h.capacidade;
      byCity.set(h.cod, c);
    });
    return [...byCity.values()];
  }
  const modLabel = (mods) => { const m = {}; mods.forEach((x) => { m[x] = (m[x] || 0) + 1; }); return Object.entries(m).sort((a, b) => b[0] - a[0]).map(([s, n]) => `${n}×${nf(+s / 1000, 0)} mil`).join(" + "); };

  function collections() {
    const v = cur(), cities = hubsByCity(), byK = new Map(v.hubs.map((h) => [h.k, h]));
    const lines = { type: "FeatureCollection", features: v.nodes.filter((n) => n.hub >= 0).map((n) => { const h = byK.get(n.hub); return { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[n.lon, n.lat], [h.lon, h.lat]] } }; }) };
    const nodes = { type: "FeatureCollection", features: v.nodes.map((n, j) => ({ type: "Feature", properties: { j, nome: n.nome, uf: n.uf, pedidos: n.pedidos, terc: n.hub < 0 }, geometry: { type: "Point", coordinates: [n.lon, n.lat] } })) };
    const hubs = { type: "FeatureCollection", features: cities.map((c, i) => ({ type: "Feature", properties: { i, ...c, modulos: modLabel(c.modulos), total: c.capacidade }, geometry: { type: "Point", coordinates: [c.lon, c.lat] } })) };
    return { lines, nodes, hubs };
  }

  function renderPanel() {
    const v = cur(), cities = hubsByCity().sort((a, b) => b.carga - a.carga);
    $("mapChips").innerHTML = [
      ["módulos abertos", `${v.hubs.length} em ${cities.length} municípios`],
      ["custo mensal", brl(v.custo_total)],
      ["frete", brl(v.frete)],
      ["custo fixo", brl(v.custo_fixo)],
      ["terceirizado", `${nf(v.pct_terceirizado, 2)}% dos pedidos`],
    ].map(([k, x]) => `<span class="chip">${k} <b>${x}</b></span>`).join("");
    $("hubList").innerHTML = `<h3>Centros (${cities.length})</h3>` + cities.map((c, i) => {
      const use = Math.min(100, 100 * c.carga / c.capacidade);
      return `<button type="button" class="hub" data-i="${i}"><b>${esc(c.nome)} · ${esc(c.uf)}</b><span>${modLabel(c.modulos)} · ${nf(c.carga)} pedidos/mês</span><div class="bar" aria-hidden="true"><i style="width:${use}%"></i></div><span>${nf(use, 0)}% da capacidade</span></button>`;
    }).join("");
    $("hubList").querySelectorAll(".hub").forEach((b, i) => b.addEventListener("click", () => focusHub(cities[i])));
    $("mapNote").textContent = "Cada círculo cinza é um município (área proporcional aos pedidos mensais da operadora hipotética, com 10% do mercado; São Paulo e o Rio aparecem divididos em zonas). Linhas retas ligam cada nó ao centro que o atende; o custo de frete usa a distância viária (OSRM). Círculos azuis: centros abertos, com módulos de 100 mil e 250 mil pedidos por mês. Contornos amarelos: pedidos terceirizados.";
  }
  function hubHtml(c) {
    return `<b>${esc(c.nome)} · ${esc(c.uf)}</b><br>${modLabel(c.modulos)}<br>${nf(c.carga)} pedidos/mês (${nf(100 * c.carga / c.capacidade, 0)}% de ${nf(c.capacidade)})`;
  }

  function bordersStyle() {
    const P = pal();
    return { version: 8, sources: { est: { type: "geojson", data: D.estados, attribution: "Fronteiras: IBGE" } },
      layers: [
        { id: "bg", type: "background", paint: { "background-color": P.land } },
        { id: "est-fill", type: "fill", source: "est", paint: { "fill-color": P.surface } },
        { id: "est-line", type: "line", source: "est", paint: { "line-color": P.line, "line-width": 1 } },
      ] };
  }
  function addOverlay() {
    const P = pal(), C = collections();
    ["lines", "nodes", "hubs"].forEach((k) => { if (!map.getSource(k)) map.addSource(k, { type: "geojson", data: C[k] }); });
    const add = (l) => { if (!map.getLayer(l.id)) map.addLayer(l); };
    add({ id: "lines", type: "line", source: "lines", layout: { "line-cap": "round" }, paint: { "line-color": P.accent, "line-opacity": 0.18, "line-width": 0.9 } });
    add({ id: "nodes", type: "circle", source: "nodes", paint: { "circle-color": P.grey, "circle-opacity": 0.65, "circle-stroke-color": ["case", ["get", "terc"], P.amber, P.surface], "circle-stroke-width": ["case", ["get", "terc"], 2, 0.6], "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "pedidos"]], 5, 2.2, 160, 9] } });
    add({ id: "hubs", type: "circle", source: "hubs", paint: { "circle-color": P.accent, "circle-stroke-color": P.surface, "circle-stroke-width": 2.2, "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "total"]], 316, 6, 1000, 14] } });
  }
  function ensureOverlay() {
    try {
      if (!map || !map.getStyle() || map.getSource("lines")) return;
      addOverlay(); restyle();
    } catch (e) { /* estilo ainda carregando: o próximo evento tenta de novo */ }
  }
  function refresh() {
    if (!map || !map.getSource("lines")) return;
    const C = collections();
    Object.keys(C).forEach((k) => map.getSource(k).setData(C[k]));
  }
  function restyle() {
    if (!map || !map.isStyleLoaded()) return;
    const P = pal();
    if (base === "borders") { map.setPaintProperty("bg", "background-color", P.land); map.setPaintProperty("est-fill", "fill-color", P.surface); map.setPaintProperty("est-line", "line-color", P.line); }
    if (map.getLayer("lines")) {
      map.setPaintProperty("lines", "line-color", P.accent);
      map.setPaintProperty("nodes", "circle-color", P.grey);
      map.setPaintProperty("hubs", "circle-color", P.accent); map.setPaintProperty("hubs", "circle-stroke-color", P.surface);
    }
  }
  function showPopup(lngLat, html) { if (popup) popup.remove(); popup = new maplibregl.Popup({ closeButton: true, maxWidth: "260px" }).setLngLat(lngLat).setHTML(html).addTo(map); }
  function focusHub(c) { if (!map) return; map.flyTo({ center: [c.lon, c.lat], zoom: Math.max(map.getZoom(), 6.5), duration: 700 }); showPopup([c.lon, c.lat], hubHtml(c)); }

  function setBase(next) {
    if (next === base || !map) return;
    base = next; clearTimeout(streetsTimer);
    $("baseB").setAttribute("aria-pressed", base === "borders"); $("baseS").setAttribute("aria-pressed", base === "streets");
    if (base === "streets") {
      streetsTimer = setTimeout(() => { if (base === "streets" && !map.getLayer("lines")) fallbackToBorders("Não foi possível carregar o mapa de ruas neste ambiente."); }, 9000);
      map.setStyle(STREETS);
    } else { map.setStyle(bordersStyle()); }
  }
  function fallbackToBorders(msg) {
    base = "borders"; $("baseB").setAttribute("aria-pressed", true); $("baseS").setAttribute("aria-pressed", false);
    map.setStyle(bordersStyle()); $("mapNote").textContent = msg;
  }

  /* vistas */
  $("mapViews").innerHTML = M.vistas.map((v, i) => `<button type="button" data-v="${i}" aria-pressed="${i === 0}">${esc(v.nome)}</button>`).join("");
  function setView(i) {
    view = i;
    $("mapViews").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.v === i));
    renderPanel(); if (map) refresh(); else drawFallback();
  }
  $("mapViews").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setView(+b.dataset.v)));
  $("baseB").addEventListener("click", () => setBase("borders"));
  $("baseS").addEventListener("click", () => setBase("streets"));

  /* fallback SVG */
  const C0 = { line: "var(--line)", accent: "var(--accent)", amber: "var(--amber)", grey: "var(--grey)" };
  function drawFallback() {
    const v = cur(), cities = hubsByCity(), W = 720;
    const pts = [...v.nodes, ...v.hubs], lons = pts.map((p) => p.lon), lats = pts.map((p) => p.lat), pad = 2;
    const lo0 = Math.min(...lons) - pad, lo1 = Math.max(...lons) + pad, la0 = Math.min(...lats) - pad, la1 = Math.max(...lats) + pad;
    const k = Math.cos(((la0 + la1) / 2) * Math.PI / 180), sc = (W - 20) / ((lo1 - lo0) * k), H = Math.round((la1 - la0) * sc + 20);
    const x = (lon) => 10 + (lon - lo0) * k * sc, y = (lat) => 10 + (la1 - lat) * sc;
    const ring = (r) => "M" + r.map((q) => x(q[0]).toFixed(1) + "," + y(q[1]).toFixed(1)).join("L") + "Z";
    let s = "";
    D.estados.features.forEach((f) => { const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates; s += `<path d="${polys.map((p) => p.map(ring).join("")).join("")}" fill="var(--surface)" stroke="${C0.line}" stroke-width=".8"/>`; });
    const byK = new Map(v.hubs.map((h) => [h.k, h]));
    v.nodes.forEach((n) => { if (n.hub >= 0) { const h = byK.get(n.hub); s += `<line x1="${x(n.lon)}" y1="${y(n.lat)}" x2="${x(h.lon)}" y2="${y(h.lat)}" stroke="${C0.accent}" stroke-opacity=".15" stroke-width=".8"/>`; } });
    v.nodes.forEach((n) => { s += `<circle cx="${x(n.lon)}" cy="${y(n.lat)}" r="${Math.max(1.8, Math.sqrt(n.pedidos) * 0.07)}" fill="${C0.grey}" fill-opacity=".7"/>`; });
    cities.forEach((c) => { s += `<circle cx="${x(c.lon)}" cy="${y(c.lat)}" r="6" fill="${C0.accent}" stroke="var(--surface)" stroke-width="2"><title>${esc(c.nome)} · ${esc(c.uf)}</title></circle>`; });
    box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Mapa estático da rede" style="display:block;background:var(--bg)">${s}</svg>`;
  }

  /* inicialização */
  renderPanel();
  let webgl = false;
  try { const cv = document.createElement("canvas"); webgl = !!(window.maplibregl && (cv.getContext("webgl2") || cv.getContext("webgl"))); } catch (e) { webgl = false; }
  if (!webgl) { $("baseS").disabled = true; drawFallback(); $("mapNote").textContent += " (Mapa estático: WebGL indisponível.)"; return; }
  try {
    const all = [...cur().nodes, ...cur().hubs], lons = all.map((q) => q.lon), lats = all.map((q) => q.lat);
    map = new maplibregl.Map({ container: "gmap", style: bordersStyle(), attributionControl: { compact: true }, bounds: [[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], fitBoundsOptions: { padding: 40 } });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", ensureOverlay); map.on("styledata", ensureOverlay); map.on("idle", ensureOverlay);
    const hover = (layer, html) => {
      map.on("click", layer, (e) => showPopup(e.lngLat, html(e.features[0].properties)));
      map.on("mouseenter", layer, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", layer, () => { map.getCanvas().style.cursor = ""; });
    };
    hover("hubs", (p) => `<b>${esc(p.nome)} · ${esc(p.uf)}</b><br>${esc(p.modulos)}<br>${nf(+p.carga)} pedidos/mês (${nf(100 * p.carga / p.capacidade, 0)}% da capacidade)`);
    hover("nodes", (p) => `<b>${esc(p.nome)} · ${esc(p.uf)}</b><br>${nf(+p.pedidos)} pedidos/mês${p.terc === true || p.terc === "true" ? "<br>terceirizado" : ""}`);
    setTimeout(() => {
      if (map && !map.getLayer("lines")) { map.remove(); map = null; $("baseS").disabled = true; drawFallback(); $("mapNote").textContent += " (Mapa estático: o mapa interativo não pôde ser iniciado neste ambiente.)"; }
    }, 15000);
    new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
  } catch (err) { map = null; drawFallback(); $("mapNote").textContent += " (Mapa estático: falha ao iniciar o mapa interativo.)"; }
})();
