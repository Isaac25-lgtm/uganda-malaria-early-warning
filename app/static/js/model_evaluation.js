/* Model Evaluation — reads dashboard model-performance + Phase 29B explainability
 * APIs. No winner/metric/threshold is computed here; values are displayed as
 * returned. A generation guard makes the latest horizon selection win.
 */
(function () {
  "use strict";
  function $(id) { return document.getElementById(id); }
  function num(v, dp) {
    if (v === null || v === undefined) return "—";
    var n = Number(v); if (!isFinite(n)) return "—";
    return n.toLocaleString("en-GB", { minimumFractionDigits: dp || 0, maximumFractionDigits: dp === undefined ? 0 : dp });
  }
  var state = { gen: 0, abort: null, impGen: 0, impAbort: null };

  function setState(msg, kind) {
    var el = $("me-state"); el.hidden = !msg; el.textContent = msg || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
  }

  function getJSON(url, signal) {
    return fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" }, signal: signal })
      .then(function (r) {
        var t = r.headers.get("content-type") || "";
        if (t.indexOf("application/json") === -1) throw { status: r.status === 401 ? 401 : 0 };
        return r.json().then(function (b) { if (!r.ok) throw { status: r.status, body: b }; return b; });
      });
  }

  function tag(labels) {
    return (labels || []).map(function (l) { return '<span class="tag tag-' + l.toLowerCase() + '">' + l + "</span>"; }).join("") || '<span class="cell-muted">—</span>';
  }

  function renderPerf(rows, horizon) {
    var scoped = rows.filter(function (r) { return String(r.horizon) === String(horizon); });
    $("me-count-body").innerHTML = scoped.map(function (r) {
      return "<tr class='" + (r.selected_for_count ? "is-selected" : "") + "'><td>" + r.model_name + "</td><td class='tag-cell'>" +
        tag(r.selected_labels) + "</td><td>" + num(r.mae, 1) + "</td><td>" + num(r.rmse, 1) + "</td><td>" +
        num(r.mape, 1) + "</td><td>" + num(r.wape, 1) + "</td><td>" + num(r.r_squared, 3) + "</td></tr>";
    }).join("");
    $("me-surge-body").innerHTML = scoped.map(function (r) {
      return "<tr class='" + (r.selected_for_surge ? "is-selected" : "") + "'><td>" + r.model_name + "</td><td class='tag-cell'>" +
        tag(r.selected_labels) + "</td><td>" + num(r.sensitivity, 3) + "</td><td>" + num(r.specificity, 3) + "</td><td>" +
        num(r.precision, 3) + "</td><td>" + num(r.f1, 3) + "</td><td>" + num(r.auc, 3) + "</td></tr>";
    }).join("");
  }

  function renderSelection(d) {
    var c = d.count_selection, s = d.surge_selection;
    $("me-selection").innerHTML =
      '<div class="lifecycle-note lifecycle-note-info"><span class="lifecycle-note-title">Count model: ' + d.selected_count_model + "</span>" +
      '<span class="lifecycle-note-body">Selected by ' + c.metric_name + " = " + num(c.metric_value, 3) + ". " + (c.metric_rule || "") + "</span></div>" +
      '<div class="lifecycle-note lifecycle-note-info"><span class="lifecycle-note-title">Surge model: ' + d.selected_surge_model + "</span>" +
      '<span class="lifecycle-note-body">Selected by ' + s.metric_name + " = " + num(s.metric_value, 3) +
      " (tie-breaker " + s.tie_breaker_name + " = " + num(s.tie_breaker_value, 3) + "). " + (s.metric_rule || "") + "</span></div>" +
      '<div class="lifecycle-note lifecycle-note-method"><span class="lifecycle-note-body">' + d.accuracy_policy + "</span></div>";
  }

  function renderValidation(d) {
    $("me-validation").innerHTML = (d.statements || []).map(function (s) { return "<li>" + s + "</li>"; }).join("") +
      "<li><strong>" + d.numbered_folds + " verified folds</strong> per horizon.</li>";
  }

  function importState(msg, kind) {
    var el = $("me-importance-state"); el.hidden = !msg; el.textContent = msg || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
  }

  function renderImportance(body) {
    var d = body.data;
    var isPerm = d.method === "permutation";
    var grouped = !!d.families;
    $("me-importance-note").textContent = (d.model_name || "") + " · " + (isPerm ? "signed permutation MAE change" : d.importance_definition) +
      " · " + d.total_features + " features · " + (grouped ? "predictor families" : "individual features");
    importState("", null);
    if (grouped) {
      $("me-importance-head").innerHTML = "<th>Rank</th><th>Family</th><th>Features</th><th>Value</th>";
      $("me-importance-body").innerHTML = (d.families || []).map(function (g) {
        var val = isPerm ? (g.signed_mean_mae_change >= 0 ? "+" : "") + num(g.signed_mean_mae_change, 3) : num(g.normalized_percentage, 2) + "%";
        return "<tr><td>" + g.family_rank + "</td><td>" + g.family + "</td><td>" + g.feature_count + "</td><td class='" +
          (isPerm && g.signed_mean_mae_change < 0 ? "score-under" : "") + "'>" + val + "</td></tr>";
      }).join("");
    } else {
      $("me-importance-head").innerHTML = "<th>Rank</th><th>Feature</th><th>Family</th><th>Value</th>";
      $("me-importance-body").innerHTML = (d.features || []).map(function (f) {
        var val = isPerm ? (f.signed_mean_mae_change >= 0 ? "+" : "") + num(f.signed_mean_mae_change, 3) : num(f.normalized_percentage, 2) + "%";
        return "<tr><td>" + f.rank + "</td><td>" + f.label + "</td><td>" + f.family + "</td><td class='" +
          (isPerm && f.signed_mean_mae_change < 0 ? "score-under" : "") + "'>" + val + "</td></tr>";
      }).join("");
    }
    $("me-importance-limits").textContent = (body.limitations || []).join(" ");
  }

  function renderThreshold(body) {
    var d = body.data;
    $("me-threshold-body").innerHTML = (d.risk_bands || []).map(function (b) {
      return "<tr><td><span class='badge badge-" + b.risk_level.replace(/\s+/g, "").toLowerCase().replace("veryhigh", "vhigh").replace("moderate", "mod") + "'>" +
        b.risk_level + "</span></td><td>" + b.rule + "</td><td>" + b.meaning + "</td></tr>";
    }).join("");
    $("me-threshold-note").textContent = [d.risk_interpretation, d.surge_score_definition].filter(Boolean).join(" ");
  }

  function activeTab() {
    var t = document.querySelector(".tab.is-active");
    return t ? t.dataset.tab : "count";
  }

  function loadAll() {
    var horizon = $("me-horizon").value;
    state.gen += 1; var gen = state.gen;
    if (state.abort) state.abort.abort();
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    state.abort = controller;
    var signal = controller ? controller.signal : undefined;
    setState("Loading…", "loading");

    Promise.all([
      getJSON("/api/dashboard/model-performance?horizon=" + horizon, signal),
      getJSON("/api/explainability/model-selection?horizon=" + horizon, signal),
      getJSON("/api/explainability/validation-method", signal),
      getJSON("/api/explainability/threshold-method?horizon=" + horizon, signal)
    ]).then(function (res) {
      if (gen !== state.gen) return;
      renderPerf(res[0].data.rows || [], horizon);
      renderSelection(res[1].data);
      renderValidation(res[2].data);
      renderThreshold(res[3]);
      setState("", null);
      loadImportance();
    }).catch(function (err) {
      if (err && err.name === "AbortError") return;
      if (gen !== state.gen) return;
      setState(err && err.status === 401 ? "Your session has ended. Sign in again." :
        (err && err.status === 503 ? "Some model evidence is not available yet." : "Model evidence could not be loaded."), "error");
    });
  }

  // Feature-importance has its OWN generation guard + AbortController so a slow
  // earlier request (e.g. Random Forest/native) can never overwrite a newer one
  // (e.g. XGBoost/permutation). Latest importance selection always wins.
  function loadImportance() {
    var horizon = $("me-horizon").value, model = $("me-model").value, method = $("me-method").value;
    var grouped = $("me-view").value === "families";
    state.impGen += 1; var gen = state.impGen;
    if (state.impAbort) state.impAbort.abort();
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    state.impAbort = ctrl;
    importState("Loading feature importance…", "loading");
    getJSON("/api/explainability/feature-importance?model=" + model + "&horizon=" + horizon +
            "&method=" + method + "&limit=20&grouped=" + (grouped ? "true" : "false"), ctrl ? ctrl.signal : undefined)
      .then(function (body) { if (gen !== state.impGen) return; renderImportance(body); })
      .catch(function (err) {
        if (err && err.name === "AbortError") return;
        if (gen !== state.impGen) return;
        $("me-importance-body").innerHTML = "";
        importState(err && err.status === 401 ? "Your session has ended. Sign in again." : "Importance evidence unavailable.", "error");
      });
  }

  function switchTab(name) {
    document.querySelectorAll(".tab").forEach(function (t) {
      var on = t.dataset.tab === name;
      t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".tabpanel").forEach(function (p) { p.hidden = p.dataset.panel !== name; });
  }

  function init() {
    if (!$("me-horizon")) return;
    document.querySelectorAll(".tab").forEach(function (t) {
      t.addEventListener("click", function () { switchTab(t.dataset.tab); });
      t.addEventListener("keydown", function (e) {
        var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
        var i = tabs.indexOf(t);
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          var next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
          next.focus(); switchTab(next.dataset.tab);
        }
      });
    });
    $("me-horizon").addEventListener("change", loadAll);
    $("me-model").addEventListener("change", loadImportance);
    $("me-method").addEventListener("change", loadImportance);
    $("me-view").addEventListener("change", loadImportance);
    loadAll();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
