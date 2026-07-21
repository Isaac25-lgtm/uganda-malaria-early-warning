/* Settings & Governance (Phase 29E corrective pass).
 *
 * Renders the eleven read-only governance sections from the canonical endpoint
 * GET /api/system/governance, and manages the two display preferences that have
 * genuinely observable behaviour. Preference storage/validation lives in the
 * shared preferences module (static/js/preferences.js) so the Settings page and
 * every other page use one implementation.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  if (!$("gov-state")) return;

  var PREFS = (window.MEWS && window.MEWS.preferences) || null;

  // ---- preferences ------------------------------------------------------
  function applyPrefsToForm(prefs) {
    if ($("pref-default_horizon")) $("pref-default_horizon").value = String(prefs.default_horizon);
    if ($("pref-reduce_motion")) $("pref-reduce_motion").checked = !!prefs.reduce_motion;
  }

  function readForm() {
    return {
      default_horizon: parseInt($("pref-default_horizon").value, 10),
      reduce_motion: $("pref-reduce_motion").checked
    };
  }

  function prefState(msg, kind) {
    var el = $("pref-state");
    el.hidden = !msg;
    el.textContent = msg || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
  }

  function initPrefs() {
    if (!PREFS) return;
    applyPrefsToForm(PREFS.load());
    $("pref-save").addEventListener("click", function () {
      var ok = PREFS.save(readForm());
      var stored = PREFS.load();
      applyPrefsToForm(stored);
      PREFS.apply();                       // take effect immediately on this page
      prefState(ok ? "Display preferences saved in this browser. They apply on every page."
                   : "Preferences could not be saved in this browser.", ok ? "empty" : "error");
    });
    $("pref-reset").addEventListener("click", function () {
      PREFS.clear();
      applyPrefsToForm(PREFS.defaults());
      PREFS.apply();
      prefState("Display preferences reset to defaults.", "empty");
    });
  }

  // ---- rendering helpers -------------------------------------------------
  function esc(v) {
    if (v === null || v === undefined) return "—";
    return String(v).replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }
  function yesno(v) { return v === true ? "Yes" : v === false ? "No" : esc(v); }

  function dl(el, pairs) {
    el.innerHTML = pairs.filter(function (p) { return p[1] !== undefined; })
      .map(function (p) {
        return '<div class="gov-row"><dt>' + esc(p[0]) + "</dt><dd>" +
          (p[2] === "bool" ? yesno(p[1]) : esc(p[1])) + "</dd></div>";
      }).join("");
  }
  function rows(tableId, data) {
    $(tableId).querySelector("tbody").innerHTML = data.map(function (r) {
      return "<tr>" + r.map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
    }).join("");
  }
  function title(k) {
    return String(k).replace(/_/g, " ").replace(/\b\w/g, function (m) { return m.toUpperCase(); });
  }

  // ---- sections ----------------------------------------------------------
  function render(d, warnings) {
    var o = d.system_overview;
    dl($("sec-overview"), [
      ["System name", o.system_name], ["Application version", o.application_version],
      ["Active districts", o.active_districts], ["Production issue week", o.production_issue_week],
      ["Forecast horizons", (o.forecast_horizons || []).join(", ") + " weeks"],
      ["Approved models", o.approved_model_count], ["Production forecast rows", o.production_forecast_rows],
      ["Overall project state", o.overall_project_state], ["Dashboard", o.dashboard_status],
      ["Explainability services", o.explainability_status], ["Lifecycle pages", o.lifecycle_status],
      ["Reports module", o.reports_status]
    ]);
    $("sec-overview-ts").textContent = "Generated (server): " + o.generated_at_utc;

    var a = d.authentication;
    dl($("sec-auth"), [
      ["Method", a.method], ["Session transport", a.session_transport],
      ["Signed with secret key", a.session_signed_with_secret_key, "bool"],
      ["Cookie HttpOnly", a.cookie_httponly, "bool"], ["Cookie SameSite", a.cookie_samesite],
      ["Cookie Secure", a.cookie_secure, "bool"],
      ["CSRF configured", a.csrf_configured, "bool"],
      ["CSRF scope", a.csrf_scope],
      ["Reauthentication (minutes)", a.reauthentication_minutes],
      ["Identity fields stored", (a.identity_fields_stored || []).join(", ")],
      ["Identity version", a.identity_version],
      ["Currently authenticated", a.currently_authenticated, "bool"],
      ["This session age (seconds)", a.current_session_age_seconds],
      ["Reauthentication in (seconds)", a.current_session_expires_in_seconds],
      ["Roles supported", (a.roles_supported && a.roles_supported.length) ? a.roles_supported.join(", ") : "None"]
    ]);
    $("sec-auth-note").textContent = a.login_vs_data_note + " " + a.credential_storage_note;

    var pv = d.data_provenance;
    var lw = pv.latest_weeks || {};
    rows("sec-provenance", [
      ["Surveillance", esc(pv.surveillance.source) + " · " + esc(pv.surveillance.freshness_note)],
      ["Latest malaria week", esc(lw.latest_malaria_week)],
      ["Latest completeness week", esc(lw.latest_completeness_week)],
      ["Latest processed climate week", esc(lw.latest_processed_climate_week)],
      ["Reporting completeness", esc(pv.reporting_completeness.source) + " · " + esc(pv.reporting_completeness.note)],
      ["Rainfall", esc(pv.climate.rainfall)], ["Temperature", esc(pv.climate.temperature)],
      ["Humidity", esc(pv.climate.humidity)],
      ["Forecast evidence generated", esc(pv.forecast_evidence_generated_at_utc)],
      ["Dataset signature", esc(pv.dataset_signature)]
    ]);

    var pl = d.pipeline;
    $("sec-pipeline-state").textContent = "Overall state: " + esc(pl.overall_state) +
      " · issue " + esc(pl.issue_label) + " · verified " + esc(pl.verified_at_utc);
    rows("sec-pipeline", (pl.stages || []).map(function (s) {
      return [esc(s.stage), esc(s.status),
              s.output_count === null || s.output_count === undefined ? "—" : esc(s.output_count),
              esc(s.unavailable_reason || "")];
    }));

    var m = d.models;
    dl($("sec-models"), [
      ["Approved models", (m.approved_models || []).join(", ")],
      ["Approved predictors", m.approved_predictor_count],
      ["Validation method", m.validation_method], ["Numbered folds", m.numbered_folds],
      ["Production artifact status", m.production_artifact_status],
      ["Production manifest version", m.production_manifest_version],
      ["Dataset signature", m.dataset_signature]
    ]);
    rows("sec-models-table", Object.keys(m.selected_models_by_horizon || {}).map(function (h) {
      var s = m.selected_models_by_horizon[h];
      return [esc(h) + " weeks", esc(s.count_model), esc(s.surge_model)];
    }));
    rows("sec-models-artifacts", Object.keys(m.model_card_status || {}).map(function (name) {
      return [esc(name), esc(m.model_card_status[name]),
              esc((m.validation_artifact_status || {})[name] || "—")];
    }));
    // Excluded methods/predictors are DERIVED: "Absent" means proven absent.
    rows("sec-models-excluded",
      (m.excluded_methods || []).map(function (x) {
        return [esc(title(x.method)), "Method",
                x.present_in_approved_models ? "PRESENT (unexpected)" : "Absent (verified)"];
      }).concat((m.excluded_predictors || []).map(function (x) {
        return [esc(title(x.variable)), "Predictor",
                x.present_in_approved_predictors ? "PRESENT (unexpected)" : "Absent (verified)"];
      })));
    $("sec-models-rules").textContent = "Count selection: " + esc(m.selection_rules.count) +
      " Surge selection: " + esc(m.selection_rules.surge);

    var f = d.forecasts;
    dl($("sec-forecasts"), [
      ["Production issue week", f.production_issue_week],
      ["Publication status", f.publication_status], ["Published", f.published, "bool"],
      ["Generation timestamp", f.generation_timestamp_utc],
      ["Model count", f.model_count],
      ["Expected forecast rows", f.expected_forecast_rows],
      ["Actual forecast rows", f.actual_forecast_rows],
      ["Row counts agree", f.row_counts_agree, "bool"],
      ["Null predicted cases", f.null_predicted_cases],
      ["Duplicate forecast keys", f.duplicate_forecast_keys],
      ["Single production issue", f.single_production_issue, "bool"],
      ["Target weeks agree with database", f.target_weeks_agree_with_database, "bool"],
      ["Schema version", f.schema_version],
      ["Threshold method", f.threshold_method], ["Threshold unit", f.threshold_unit],
      ["Threshold history cutoff", f.threshold_history_cutoff],
      ["Surge score", f.surge_score_definition],
      ["Threshold is a probability", f.threshold_is_probability, "bool"],
      ["Confirms an outbreak", f.threshold_confirms_outbreak, "bool"],
      ["Risk interpretation", f.risk_interpretation]
    ]);
    rows("sec-forecast-targets", Object.keys(f.published_target_weeks || {}).map(function (h) {
      var t = f.published_target_weeks[h];
      var dbt = (f.database_target_weeks || {})[h];
      return [esc(h) + " weeks", esc(t.target_label), esc(t.target_week_start_date),
              dbt ? esc(dbt.target_epi_year + "W" + String(dbt.target_epi_week).padStart(2, "0")) : "—"];
    }));

    var dq = d.data_quality_rules;
    var dqHtml;
    if (dq.available) {
      var controls = Object.keys(dq.leakage_controls || {}).map(function (k) {
        return "<tr><td>" + esc(title(k)) + "</td><td>" + yesno(dq.leakage_controls[k]) +
          "</td><td>" + esc((dq.leakage_control_sources || {})[k]) + "</td></tr>";
      }).join("");
      var defs = (dq.definitions || []).map(function (p) {
        return "<dt>" + esc(p[0]) + "</dt><dd>" + esc(p[1]) + "</dd>";
      }).join("");
      dqHtml = '<dl class="gov-grid">' +
        '<div class="gov-row"><dt>Expected district-weeks</dt><dd>' + esc(dq.coverage.expected_district_weeks) + "</dd></div>" +
        '<div class="gov-row"><dt>Received district-weeks</dt><dd>' + esc(dq.coverage.received_district_weeks) + "</dd></div>" +
        '<div class="gov-row"><dt>Missing district-weeks (absent)</dt><dd>' + esc(dq.coverage.missing_district_weeks) + "</dd></div>" +
        '<div class="gov-row"><dt>Rows with all nine climate lags</dt><dd>' + esc(dq.rows_with_complete_climate_lags) + "</dd></div></dl>" +
        '<div class="table-scroll"><table class="data-table"><thead><tr><th>Leakage control</th><th>Verified</th><th>Evidence source</th></tr></thead><tbody>' +
        controls + '</tbody></table></div><dl class="print-dl gov-defs">' + defs + "</dl>";
    } else {
      dqHtml = '<p class="page-sub">Data-quality evidence is currently unavailable: ' + esc(dq.reason) + "</p>";
    }
    $("sec-dq").innerHTML = dqHtml;

    var rp = d.reports;
    rows("sec-reports", (rp.reports || []).map(function (r) {
      var links = '<a href="' + esc(r.url) + '">Open</a>';
      if (r.print_url) links += ' · <a href="' + esc(r.print_url) + '" target="_blank" rel="noopener">Print view</a>';
      return [esc(r.title),
              (r.formats || []).map(function (x) { return esc(x.toUpperCase()); }).join(", "),
              links];
    }));
    $("sec-reports-deferred").textContent = "Deferred: " + (rp.deferred || []).join(", ") + ".";

    var it = d.integrity;
    dl($("sec-integrity"), [
      ["Production forecast rows", it.production_forecast_rows],
      ["Active districts", it.active_districts],
      ["Approved models", it.approved_model_count],
      ["Null predicted cases", it.null_predicted_cases],
      ["Duplicate forecast keys", it.duplicate_forecast_keys],
      ["Expected composite keys", it.expected_composite_keys],
      ["Actual composite keys", it.actual_composite_keys],
      ["Missing composite keys", it.missing_composite_keys],
      ["Unexpected composite keys", it.unexpected_composite_keys],
      ["Composite key set complete", it.composite_key_set_complete, "bool"],
      ["Single production issue", it.single_production_issue, "bool"],
      ["Deterministic fingerprint (SHA-256)", it.forecast_fingerprint_sha256]
    ]);
    $("sec-integrity-ts").textContent = "Verified (server): " + it.verified_at_utc;

    $("sec-prefs-note").textContent = d.display_preferences.note;

    $("sec-research-status").textContent = d.research.status;
    $("sec-research-use").textContent = d.research.responsible_use;
    $("sec-research-limits").innerHTML = (d.research.limitations || [])
      .map(function (l) { return "<li>" + esc(l) + "</li>"; }).join("");

    var wb = $("gov-warnings");
    if (warnings && warnings.length) {
      wb.hidden = false;
      wb.innerHTML = "<strong>Warnings:</strong> " + warnings.map(esc).join(" ");
    } else {
      wb.hidden = true;
    }
  }

  function setState(msg, kind) {
    var el = $("gov-state");
    el.hidden = !msg;
    el.textContent = msg || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
  }

  function load() {
    setState("Loading governance evidence…", "loading");
    $("gov-content").hidden = true;
    fetch("/api/system/governance", {
      credentials: "same-origin", headers: { Accept: "application/json" }
    }).then(function (r) {
      var type = r.headers.get("content-type") || "";
      if (type.indexOf("application/json") === -1) throw { status: r.status === 401 ? 401 : 0 };
      return r.json().then(function (b) { if (!r.ok) throw { status: r.status, reason: b.reason }; return b; });
    }).then(function (body) {
      render(body.data, body.warnings);
      setState("", null);
      $("gov-content").hidden = false;
    }).catch(function (err) {
      $("gov-content").hidden = true;   // clear stale values before showing an error
      var msg = "Governance evidence could not be loaded.";
      if (err && err.status === 401) msg = "Your session has expired. Please log in again.";
      else if (err && err.reason) msg = err.reason;
      setState(msg, "error");
    });
  }

  initPrefs();   // preferences work even if governance evidence fails
  load();
})();
