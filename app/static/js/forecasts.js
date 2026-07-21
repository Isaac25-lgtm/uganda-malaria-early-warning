/* Forecasts & Risk — reuses the dashboard summary/weekly/risk-table/map APIs.
 * No risk/threshold/metric is recomputed; risk_basis and values come from the
 * backend. Latest filter selection wins (generation guard + abort); the national
 * map is not refetched for district-only changes.
 */
(function () {
  "use strict";
  var RISK_COLOURS = { "Very High": "#dc2626", High: "#ea580c", Moderate: "#f59e0b", Low: "#16a34a", "No data": "#94a3b8" };
  var RISK_RANK = { "Very High": 4, High: 3, Moderate: 2, Low: 1, "No data": -1 };
  var RISK_CLASS = { "Very High": "badge-vhigh", High: "badge-high", Moderate: "badge-mod", Low: "badge-low", "No data": "badge-nodata" };
  function $(id) { return document.getElementById(id); }
  function num(v, dp) { if (v === null || v === undefined) return "—"; var n = Number(v); if (!isFinite(n)) return "—"; return n.toLocaleString("en-GB", { minimumFractionDigits: dp || 0, maximumFractionDigits: dp === undefined ? 0 : dp }); }
  function signed(v) { if (v === null || v === undefined) return "—"; var n = Number(v); return (n > 0 ? "+" : "") + num(n); }

  var S = { gen: 0, abort: null, rows: [], sort: { key: "risk_rank", dir: "desc" }, search: "", page: 1, pageSize: 25, chart: null, map: null, layer: null, feats: {}, mapKey: null, selected: null, nat: null };

  function setState(id, msg, kind) { var el = $(id); if (!el) return; el.hidden = !msg; el.textContent = msg || ""; el.className = "state-block" + (kind ? " state-" + kind : ""); }
  function params() {
    var p = {}; var h = $("fc-horizon"), d = $("fc-district"), i = $("fc-issue");
    if (h && h.value) p.horizon = h.value;
    if (d && d.value) p.district_id = d.value;
    if (i && i.value) { var s = i.value.split("-"); if (s.length === 2) { p.issue_year = s[0]; p.issue_week = s[1]; } }
    return p;
  }
  function url(path, p) { return path + "?" + Object.keys(p).map(function (k) { return k + "=" + encodeURIComponent(p[k]); }).join("&"); }
  function get(path, p, signal) {
    return fetch(url(path, p), { credentials: "same-origin", headers: { Accept: "application/json" }, signal: signal })
      .then(function (r) { var t = r.headers.get("content-type") || ""; if (t.indexOf("application/json") === -1) throw { status: r.status === 401 ? 401 : 0 }; return r.json().then(function (b) { if (!r.ok) throw { status: r.status, body: b }; return b; }); });
  }
  function guard(gen, ok, fail) {
    return { ok: function (b) { if (gen !== S.gen) return; ok(b); }, fail: function (e) { if (e && e.name === "AbortError") return; if (gen !== S.gen) return; fail(e); } };
  }
  function errText(e) { return e && e.status === 401 ? "Your session has ended. Sign in again." : (e && e.status === 503 ? "Forecast output is not available yet." : "Could not reach the server."); }

  // ---- summary ----
  function renderSummary(d) {
    setState("fc-summary-state", "", null);
    $("fc-target").textContent = (d.issue_label || "—") + " → " + (d.target_label || "—");
    $("fc-scope").textContent = d.scope === "district" ? d.district_name : "All districts";
    $("fc-count-model").textContent = d.count_model_name || "—";
    $("fc-surge-model").textContent = d.surge_model_name || "—";
    $("fc-atrisk").textContent = d.districts_at_surge_risk === null ? "—" : num(d.districts_at_surge_risk);
  }
  function summaryError(e) {
    // Never retain a previous district's or horizon's values on failure.
    ["fc-target", "fc-scope", "fc-count-model", "fc-surge-model", "fc-atrisk"].forEach(function (id) { $(id).textContent = "—"; });
    setState("fc-summary-state", errText(e) + " Summary values are unavailable.", "error");
  }

  // ---- chart (observed line + single forecast + single threshold) ----
  var labelPlugin = { id: "fcLabels", afterDatasetsDraw: function (c) {
    var ctx = c.ctx, area = c.chartArea;
    [[1, "#2563eb", "Forecast"], [2, "#dc2626", "Threshold"]].forEach(function (spec) {
      var meta = c.getDatasetMeta(spec[0]), data = c.data.datasets[spec[0]].data, el = null, val = null;
      for (var i = data.length - 1; i >= 0; i--) { if (data[i] !== null && data[i] !== undefined && meta.data[i]) { el = meta.data[i]; val = data[i]; break; } }
      if (!el) return;
      ctx.save(); ctx.font = "600 11px Inter, system-ui, sans-serif";
      var text = spec[2] + " " + num(val), w = ctx.measureText(text).width + 12, bx = el.x - w - 10; if (bx < area.left) bx = el.x + 10;
      var by = spec[0] === 1 ? el.y + 6 : el.y - 24;
      ctx.fillStyle = "#fff"; ctx.strokeStyle = spec[1]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.rect(bx, by, w, 18); ctx.fill(); ctx.stroke();
      ctx.fillStyle = spec[1]; ctx.textBaseline = "middle"; ctx.fillText(text, bx + 6, by + 9); ctx.restore();
    });
  } };

  function renderChart(d) {
    if (typeof Chart === "undefined") { setState("fc-chart-state", "Chart library unavailable.", "error"); return; }
    var rows = d.rows || [];
    var obs = rows.filter(function (r) { return r.period === "observed"; }).slice(-52).concat(rows.filter(function (r) { return r.period === "forecast"; }));
    if (!obs.length) { setState("fc-chart-state", "No data available.", "empty"); return; }
    setState("fc-chart-state", "", null);
    var labels = obs.map(function (r) { return r.week_label; });
    var datasets = [
      { label: "Observed", data: obs.map(function (r) { return r.period === "observed" ? r.actual_cases : null; }), borderColor: "#0d9488", borderWidth: 2, pointRadius: 0, tension: 0.2 },
      { label: "Forecast (" + (d.count_model_name || "count") + ")", data: obs.map(function (r) { return r.period === "forecast" ? r.predicted_cases : null; }), borderColor: "#2563eb", pointRadius: 6, pointStyle: "rectRot", showLine: false, clip: false },
      { label: "Threshold", data: obs.map(function (r) { return r.surge_threshold; }), borderColor: "#dc2626", pointRadius: 7, pointStyle: "crossRot", showLine: false, clip: false }
    ];
    if (S.chart) S.chart.destroy();
    S.chart = new Chart($("fc-chart").getContext("2d"), {
      type: "line", data: { labels: labels, datasets: datasets },
      options: { responsive: true, maintainAspectRatio: false, animation: false, layout: { padding: { right: 16, top: 22 } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10, color: "#64748b", font: { size: 11 } } }, y: { beginAtZero: true, grid: { color: "#eef2f6" }, ticks: { color: "#64748b", font: { size: 11 }, callback: function (v) { return num(v); } } } },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 10, usePointStyle: true, font: { size: 11 } } } } },
      plugins: [labelPlugin]
    });
    $("fc-chart-note").textContent = (d.intervals_note || "") + " The model predicts the target week only. Intermediate weekly forecasts are not generated.";
  }

  // ---- risk table ----
  function badge(l) { return '<span class="badge ' + (RISK_CLASS[l] || "badge-nodata") + '">' + l + "</span>"; }
  function sorted() {
    var rows = S.rows.slice(), term = S.search.trim().toLowerCase();
    if (term) rows = rows.filter(function (r) { return (r.district_name || "").toLowerCase().indexOf(term) !== -1; });
    var k = S.sort.key, dir = S.sort.dir === "asc" ? 1 : -1;
    rows.sort(function (a, b) { var av = k === "risk_rank" ? RISK_RANK[a.risk_level] : a[k], bv = k === "risk_rank" ? RISK_RANK[b.risk_level] : b[k]; if (av == null) return 1; if (bv == null) return -1; return typeof av === "string" ? av.localeCompare(bv) * dir : (av - bv) * dir; });
    return rows;
  }
  function paintTable() {
    var rows = sorted(), body = $("fc-risk-body");
    if (!rows.length) { body.innerHTML = ""; setState("fc-risk-state", S.search ? "No district matches this search." : "No data available.", "empty"); $("fc-pager").hidden = true; return; }
    var pages = Math.max(1, Math.ceil(rows.length / S.pageSize)); if (S.page > pages) S.page = pages; if (S.page < 1) S.page = 1;
    var start = (S.page - 1) * S.pageSize, pageRows = rows.slice(start, start + S.pageSize);
    body.innerHTML = pageRows.map(function (r) {
      return '<tr><td title="' + (r.risk_basis || "").replace(/"/g, "&quot;") + '">' + r.district_name + "</td><td>" + badge(r.risk_level) + "</td><td>" + num(r.predicted_cases) + "</td><td>" + num(r.threshold_used) + "</td><td class='" + (Number(r.predicted_surge_score) > 0 ? "score-over" : "score-under") + "'>" + signed(r.predicted_surge_score) + "</td></tr>";
    }).join("");
    setState("fc-risk-state", "", null);
    $("fc-pager").hidden = false;
    $("fc-page-status").textContent = "Showing " + (start + 1) + "–" + (start + pageRows.length) + " of " + rows.length + " · page " + S.page + " of " + pages;
    $("fc-prev").disabled = S.page <= 1; $("fc-next").disabled = S.page >= pages;
  }
  function renderRisk(d) {
    S.rows = d.rows || []; S.page = 1;
    var m = d.selected_models || {};
    $("fc-risk-subhead").textContent = m.same_model ? "Counts and risk both from " + (m.count && m.count.model_name) + "." :
      "Predicted cases from " + (m.count && m.count.model_name) + "; risk from " + (m.surge && m.surge.model_name) + ". Different models, not combined.";
    $("fc-risk-note").textContent = [d.score_definition, d.risk_assessment_note].filter(Boolean).join(" ");
    renderDistrictPanel();
    paintTable();
  }

  // Selected-district interpretation, populated exclusively from the risk row
  // (backend values; nothing recomputed here).
  function renderDistrictPanel() {
    var panel = $("fc-district-panel");
    if (!S.selected) { panel.hidden = true; return; }
    var row = null;
    for (var i = 0; i < S.rows.length; i++) { if (S.rows[i].district_id === S.selected) { row = S.rows[i]; break; } }
    if (!row) { panel.hidden = true; return; }
    panel.hidden = false;
    $("fc-di-title").textContent = row.district_name + " — district interpretation";
    $("fc-di-models").textContent = "Predicted cases: " + (row.count_model_name || "—") + " · risk: " + (row.surge_model_name || "—");
    $("fc-di-count-label").textContent = "Count-model predicted cases (" + (row.count_model_name || "—") + ")";
    $("fc-di-surge-label").textContent = "Surge-model predicted cases (" + (row.surge_model_name || "—") + ")";
    $("fc-di-risk").innerHTML = badge(row.risk_level);
    $("fc-di-count").textContent = num(row.predicted_cases);
    $("fc-di-surge").textContent = num(row.surge_predicted_cases);
    $("fc-di-threshold").textContent = num(row.threshold_used);
    $("fc-di-score").textContent = signed(row.predicted_surge_score);
    $("fc-di-completeness").textContent = row.reporting_completeness === null || row.reporting_completeness === undefined
      ? "Not reported" : num(row.reporting_completeness, 1) + "%";
    $("fc-di-weeks").textContent = (row.issue_label || "—") + " → " + (row.target_label || "—");
    $("fc-di-horizon").textContent = row.forecast_horizon_weeks + " weeks";
    $("fc-di-basis").textContent = row.risk_basis || "";
  }

  // ---- map ----
  function style(f) { var sel = S.selected, is = sel && f.properties.district_id === sel; return { fillColor: RISK_COLOURS[f.properties.risk_level] || RISK_COLOURS["No data"], fillOpacity: sel && !is ? 0.35 : 0.85, color: is ? "#0f172a" : "#fff", weight: is ? 2.5 : 0.5 }; }
  function renderMap(d) {
    if (typeof L === "undefined") { setState("fc-map-state", "Map library unavailable.", "error"); return; }
    if (!d.features || !d.features.length) { setState("fc-map-state", "District boundary unavailable.", "empty"); return; }
    setState("fc-map-state", "", null);
    if (!S.map) S.map = L.map("fc-map", { zoomControl: true, attributionControl: false, scrollWheelZoom: false });
    if (S.layer) { S.map.removeLayer(S.layer); S.layer = null; } S.feats = {};
    S.layer = L.geoJSON(d, { style: style, onEachFeature: function (f, layer) {
      S.feats[f.properties.district_id] = layer;
      layer.bindTooltip('<strong>' + f.properties.district_name + "</strong><br>Risk: " + f.properties.risk_level + "<br>Predicted: " + num(f.properties.predicted_cases), { sticky: true, className: "map-tooltip" });
      layer.on("click", function () { var s = $("fc-district"); if (s) { s.value = f.properties.district_id; s.dispatchEvent(new Event("change")); } });
    } }).addTo(S.map);
    S.nat = S.layer.getBounds(); S.map.fitBounds(S.nat, { padding: [8, 8] });
    applyMapSel();
  }
  function applyMapSel() {
    if (!S.layer) return; S.layer.setStyle(style);
    if (S.selected && S.feats[S.selected]) { S.map.fitBounds(S.feats[S.selected].getBounds(), { padding: [24, 24], maxZoom: 9 }); S.feats[S.selected].bringToFront(); }
    else if (S.nat) S.map.fitBounds(S.nat, { padding: [8, 8] });
  }

  function loadAll() {
    var p = params(); S.selected = p.district_id && p.district_id !== "all" ? p.district_id : null;
    S.gen += 1; var gen = S.gen; if (S.abort) S.abort.abort();
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null; S.abort = ctrl; var sig = ctrl ? ctrl.signal : undefined;
    // Hide the district panel until fresh data arrives so a failed/slow load
    // cannot leave a previous district's interpretation visible.
    $("fc-district-panel").hidden = true;
    ["fc-chart-state", "fc-risk-state"].forEach(function (id) { setState(id, "Loading…", "loading"); });
    var s = guard(gen, function (b) { renderSummary(b.data); }, summaryError); get("/api/dashboard/summary", p, sig).then(s.ok, s.fail);
    var w = guard(gen, function (b) { renderChart(b.data); }, function (e) { setState("fc-chart-state", errText(e), "error"); }); get("/api/dashboard/weekly-series", p, sig).then(w.ok, w.fail);
    var r = guard(gen, function (b) { renderRisk(b.data); }, function (e) { setState("fc-risk-state", errText(e), "error"); }); get("/api/dashboard/risk-table", p, sig).then(r.ok, r.fail);
    var mk = [p.horizon, p.issue_year, p.issue_week].join("|");
    if (mk !== S.mapKey || !S.layer) { setState("fc-map-state", "Loading…", "loading"); S.mapKey = mk; var m = guard(gen, function (b) { renderMap(b.data); }, function (e) { setState("fc-map-state", errText(e), "error"); }); get("/api/dashboard/map", p, sig).then(m.ok, m.fail); }
    else applyMapSel();
  }

  function init() {
    if (!$("fc-risk-table")) return;
    ["fc-horizon", "fc-district", "fc-issue"].forEach(function (id) { var el = $(id); if (el) el.addEventListener("change", loadAll); });
    $("fc-search").addEventListener("input", function (e) { S.search = e.target.value; S.page = 1; paintTable(); });
    document.querySelectorAll("#fc-risk-table th.sortable .th-sort").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var th = btn.closest("th"), k = th.dataset.key;
        if (S.sort.key === k) S.sort.dir = S.sort.dir === "asc" ? "desc" : "asc"; else { S.sort.key = k; S.sort.dir = th.dataset.type === "text" ? "asc" : "desc"; }
        document.querySelectorAll("#fc-risk-table th.sortable").forEach(function (o) { o.setAttribute("aria-sort", "none"); o.classList.remove("sorted-asc", "sorted-desc"); });
        th.setAttribute("aria-sort", S.sort.dir === "asc" ? "ascending" : "descending"); th.classList.add(S.sort.dir === "asc" ? "sorted-asc" : "sorted-desc");
        S.page = 1; paintTable();
      });
    });
    $("fc-prev").addEventListener("click", function () { S.page -= 1; paintTable(); });
    $("fc-next").addEventListener("click", function () { S.page += 1; paintTable(); });
    loadAll();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
