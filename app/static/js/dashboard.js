/* Malaria Early Warning Dashboard - panel data layer.
 *
 * Every panel is loaded from its own API endpoint and renders only what the API
 * returns. Nothing here invents a value: no interval is drawn unless the
 * selected count model published one, no probability is shown because none
 * exists, and every "—" means the API returned null.
 */
(function () {
  "use strict";

  var RISK_COLOURS = {
    "Very High": "#dc2626",
    High: "#ea580c",
    Moderate: "#f59e0b",
    Low: "#16a34a",
    "No data": "#94a3b8"
  };
  var RISK_RANK = { "Very High": 4, High: 3, Moderate: 2, Low: 1, "No data": -1 };
  var RISK_CLASS = {
    "Very High": "badge-vhigh",
    High: "badge-high",
    Moderate: "badge-mod",
    Low: "badge-low",
    "No data": "badge-nodata"
  };
  var TREND_LABEL = {
    increasing: "Increasing",
    decreasing: "Decreasing",
    stable: "Stable",
    unknown: "Unknown"
  };

  var MSG_LOADING = "Loading dashboard data…";
  var MSG_NO_DATA = "No data available.";
  var MSG_API = "API unavailable. The dashboard could not reach the server.";
  var MSG_FORECAST = "Forecast output unavailable.";
  var MSG_BOUNDARY = "District boundary unavailable.";

  var state = {
    riskRows: [],
    sort: { key: "risk_rank", dir: "desc" },
    search: "",
    page: 1,
    pageSize: 25,
    range: 52,
    weekly: null,
    chart: null,
    map: null,
    mapLayer: null,
    mapKey: null,
    mapFeatures: {},
    // Latest-request-wins: every loadAll() bumps this generation. A response is
    // applied only if its generation is still current when it resolves.
    generation: 0,
    abort: null
  };

  // ---- utilities --------------------------------------------------------
  function $(id) {
    return document.getElementById(id);
  }

  function num(value, dp) {
    if (value === null || value === undefined || value === "") return "—";
    var n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("en-GB", {
      minimumFractionDigits: dp || 0,
      maximumFractionDigits: dp === undefined ? 0 : dp
    });
  }

  function pct(value, dp) {
    if (value === null || value === undefined) return "—";
    return num(value, dp === undefined ? 1 : dp) + "%";
  }

  function signed(value, dp) {
    if (value === null || value === undefined) return "—";
    var n = Number(value);
    return (n > 0 ? "+" : "") + num(n, dp === undefined ? 0 : dp);
  }

  function setState(id, message, kind) {
    var el = $(id);
    if (!el) return;
    el.textContent = message || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
    el.hidden = !message;
  }

  function show(el, visible) {
    if (el) el.hidden = !visible;
  }

  function describeError(err, context) {
    if (!err) return MSG_API;
    if (err.kind === "auth") {
      return err.message || "Your session has ended. Sign in again to load dashboard data.";
    }
    if (err.kind === "invalid") return err.message || "Invalid dashboard filter.";
    if (err.kind === "unavailable") {
      var reason = (err.message || "").toLowerCase();
      if (context === "map" && (reason.indexOf("boundary") !== -1 || reason.indexOf("web-map") !== -1)) {
        return MSG_BOUNDARY;
      }
      return err.message || MSG_FORECAST;
    }
    return MSG_API;
  }

  // ---- parameters -------------------------------------------------------
  function currentParams() {
    var params = {};
    var horizon = $("filter-horizon");
    var district = $("filter-district");
    var issue = $("filter-issue");
    if (horizon && horizon.value) params.horizon = horizon.value;
    if (district && district.value) params.district_id = district.value;
    if (issue && issue.value) {
      var parts = issue.value.split("-");
      if (parts.length === 2) {
        params.issue_year = parts[0];
        params.issue_week = parts[1];
      }
    }
    return params;
  }

  function apiUrl(path, params) {
    var query = Object.keys(params)
      .map(function (key) {
        return encodeURIComponent(key) + "=" + encodeURIComponent(params[key]);
      })
      .join("&");
    return path + (query ? "?" + query : "");
  }

  function fetchSection(path, params, signal) {
    return fetch(apiUrl(path, params), {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal: signal
    }).then(function (response) {
      var type = response.headers.get("content-type") || "";
      if (type.indexOf("application/json") === -1) {
        // A session timeout can return an HTML login page to fetch.
        throw { kind: response.status === 401 || response.redirected ? "auth" : "error" };
      }
      return response.json().then(function (body) {
        if (response.status === 401) throw { kind: "auth", message: body.reason };
        if (response.status === 400) throw { kind: "invalid", message: body.reason };
        if (response.status === 503) throw { kind: "unavailable", message: body.reason };
        if (!response.ok) throw { kind: "error", message: body.reason };
        return body;
      });
    }).catch(function (err) {
      // A superseded request aborts intentionally; surface it as a distinct kind
      // so callers can drop it silently instead of showing an API error.
      if (err && (err.name === "AbortError" || err.kind === "aborted")) {
        throw { kind: "aborted" };
      }
      throw err;
    });
  }

  // Wrap a section load so a stale response (older generation) or an abort is
  // dropped without touching any panel's loading/success/error state.
  function guarded(generation, onData, onError) {
    return {
      apply: function (body) {
        if (generation !== state.generation) return; // superseded
        onData(body);
      },
      fail: function (err) {
        if (err && err.kind === "aborted") return; // intentional cancellation
        if (generation !== state.generation) return; // superseded
        onError(err);
      }
    };
  }

  function initDistrictCombobox() {
    var source = $("filter-district");
    var input = $("filter-district-search");
    var list = $("district-options");
    if (!source || !input || !list) return;

    var options = Array.prototype.slice.call(list.querySelectorAll(".district-option"));
    var empty = list.querySelector(".district-option-empty");
    var active = null;

    function selectedOption() {
      return options.find(function (option) { return option.dataset.value === source.value; });
    }

    function syncSelection() {
      var selected = selectedOption() || options[0];
      input.value = selected ? selected.dataset.label : "";
      options.forEach(function (option) {
        option.setAttribute("aria-selected", option === selected ? "true" : "false");
      });
    }

    function visibleOptions() {
      return options.filter(function (option) { return !option.hidden; });
    }

    function setActive(option) {
      options.forEach(function (item) { item.classList.remove("is-active"); });
      active = option || null;
      if (active) {
        active.classList.add("is-active");
        input.setAttribute("aria-activedescendant", active.id);
        active.scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }

    function openList() {
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }

    function closeList() {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      setActive(null);
    }

    function filterOptions(query) {
      var term = String(query || "").trim().toLowerCase();
      options.forEach(function (option) {
        option.hidden = Boolean(term) && option.dataset.label.toLowerCase().indexOf(term) === -1;
      });
      var visible = visibleOptions();
      if (empty) empty.hidden = visible.length !== 0;
      openList();
      setActive(visible[0] || null);
    }

    function choose(option) {
      if (!option) return;
      source.value = option.dataset.value;
      syncSelection();
      closeList();
      source.dispatchEvent(new Event("change", { bubbles: true }));
    }

    input.addEventListener("focus", function () {
      input.select();
      filterOptions("");
    });
    input.addEventListener("input", function () { filterOptions(input.value); });
    input.addEventListener("keydown", function (event) {
      var visible = visibleOptions();
      var index = visible.indexOf(active);
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (list.hidden) filterOptions(input.value);
        visible = visibleOptions();
        index = visible.indexOf(active);
        var step = event.key === "ArrowDown" ? 1 : -1;
        var next = index < 0 ? 0 : Math.max(0, Math.min(visible.length - 1, index + step));
        setActive(visible[next] || null);
      } else if (event.key === "Enter" && !list.hidden) {
        event.preventDefault();
        choose(active);
      } else if (event.key === "Escape") {
        event.preventDefault();
        syncSelection();
        closeList();
      }
    });
    input.addEventListener("blur", function () {
      window.setTimeout(function () {
        var exact = options.find(function (option) {
          return option.dataset.label.toLowerCase() === input.value.trim().toLowerCase();
        });
        if (exact && exact.dataset.value !== source.value) choose(exact);
        else if (!exact) syncSelection();
        closeList();
      }, 120);
    });
    options.forEach(function (option) {
      option.addEventListener("mousedown", function (event) { event.preventDefault(); });
      option.addEventListener("click", function () { choose(option); });
    });
    source.addEventListener("change", syncSelection);
    document.addEventListener("click", function (event) {
      if (!list.hidden && !event.target.closest(".district-combobox")) {
        syncSelection();
        closeList();
      }
    });
    syncSelection();
  }

  // ---- summary ----------------------------------------------------------
  var NATIONAL_LEAD =
    "This dashboard combines recent malaria reports, climate signals and validated models to support " +
    "early review of possible district-level malaria surges. It shows the latest observed burden, " +
    "target-week forecasts and districts that may require attention.";
  var DISTRICT_LEAD =
    "This view summarises recent malaria reporting, target-week forecasts and surge risk for the " +
    "selected district.";

  function renderSummary(data) {
    var scopeLabel = data.scope === "district" ? data.district_name : "All districts";
    $("page-title").textContent = data.scope === "district"
      ? data.district_name + " Malaria Early Warning Overview"
      : "National Malaria Early Warning Overview";
    var lead = $("page-lead");
    if (lead) lead.textContent = data.scope === "district" ? DISTRICT_LEAD : NATIONAL_LEAD;
    var issue = data.issue_label || "—";

    $("kpi-cases-value").textContent = num(data.confirmed_malaria_cases);
    $("kpi-cases-hint").textContent = issue;
    var change = data.previous_week_change_percent;
    $("kpi-cases-note").textContent =
      change === null || change === undefined
        ? "No comparable previous week."
        : signed(change, 1) + "% vs previous week · " + scopeLabel;

    $("kpi-completeness-value").textContent = pct(data.reporting_completeness);
    $("kpi-completeness-hint").textContent = issue;
    $("kpi-completeness-note").textContent = data.reporting_completeness === null
      ? "Not reported for this week."
      : "HMIS 033b reporting — not a measure of malaria burden.";

    $("kpi-risk-value").textContent =
      data.districts_at_surge_risk === null ? "—" : num(data.districts_at_surge_risk);
    $("kpi-risk-hint").textContent = "of " + num(data.districts_in_scope) + " in view";
    $("kpi-risk-note").textContent = "High or Very High · " + (data.surge_model_name || "—") + " · warrants review, not confirmed outbreak";

    $("kpi-horizon-value").textContent = data.forecast_horizon_weeks + " weeks";
    $("kpi-horizon-hint").textContent = data.target_label ? "Target " + data.target_label : "Selected";
    $("kpi-horizon-note").textContent = data.same_model
      ? "Count and surge: " + data.count_model_name
      : "Count: " + data.count_model_name + " · Surge: " + data.surge_model_name;

    var context = $("page-scope-note");
    if (context) {
      context.innerHTML = [
        ["Issue week", issue],
        ["Horizon", data.forecast_horizon_weeks + " weeks"],
        ["Target week", data.target_label || "—"],
        ["Count model", data.count_model_name || "—"],
        ["Surge model", data.surge_model_name || "—"]
      ]
        .map(function (pair) {
          return '<span class="ctx"><i>' + pair[0] + "</i>" + pair[1] + "</span>";
        })
        .join("");
    }

    var updated = $("footer-updated");
    if (updated) {
      updated.textContent =
        "Issue " + issue + (data.issue_week_start_date ? " (week starting " + data.issue_week_start_date + ")" : "");
    }
  }

  function summaryError(err) {
    var message = describeError(err, "summary");
    ["cases", "completeness", "risk", "horizon"].forEach(function (key) {
      $("kpi-" + key + "-value").textContent = "—";
      $("kpi-" + key + "-note").textContent = message;
    });
    var updated = $("footer-updated");
    if (updated) updated.textContent = "Last updated: unavailable";
  }

  // ---- weekly chart -----------------------------------------------------
  // Separates the observed period from the forecast period. The shaded band is
  // the primary signal, so the distinction survives even when a single forecast
  // target leaves no room for a label.
  var forecastDivider = {
    id: "forecastDivider",
    beforeDatasetsDraw: function (chart, args, options) {
      var index = options.index;
      if (index === null || index === undefined || index < 0) return;
      var x = chart.scales.x.getPixelForValue(index);
      var area = chart.chartArea;
      var ctx = chart.ctx;
      ctx.save();
      ctx.fillStyle = "rgba(37, 99, 235, 0.06)";
      ctx.fillRect(x, area.top, area.right - x, area.bottom - area.top);
      ctx.restore();
    },
    afterDatasetsDraw: function (chart, args, options) {
      var index = options.index;
      if (index === null || index === undefined || index < 0) return;
      var x = chart.scales.x.getPixelForValue(index);
      var area = chart.chartArea;
      var ctx = chart.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1;
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#64748b";
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.textAlign = "right";
      if (x - area.left > ctx.measureText("observed").width + 12) {
        ctx.fillText("observed", x - 6, area.top + 12);
      }
      // Only label the forecast side when the band is wide enough to hold it.
      ctx.textAlign = "left";
      if (area.right - x > ctx.measureText("forecast").width + 12) {
        ctx.fillText("forecast", x + 6, area.top + 12);
      }
      ctx.restore();
    }
  };

  // Direct labels: draw the forecast value, threshold value and target week next
  // to their single markers. Pure Chart.js custom plugin — no datalabels package.
  var directLabels = {
    id: "directLabels",
    afterDatasetsDraw: function (chart, args, options) {
      var forecastMeta = chart.getDatasetMeta(1);
      var thresholdMeta = chart.getDatasetMeta(2);
      var ctx = chart.ctx;
      var area = chart.chartArea;

      function lastPoint(meta, values) {
        for (var i = values.length - 1; i >= 0; i--) {
          if (values[i] !== null && values[i] !== undefined && meta.data[i]) {
            return { el: meta.data[i], value: values[i] };
          }
        }
        return null;
      }

      function chip(point, text, colour, place) {
        if (!point) return null; // hide a label when its source value is missing
        ctx.font = "600 11px Inter, system-ui, sans-serif";
        var padX = 6, padY = 3, h = 18;
        var w = ctx.measureText(text).width + padX * 2;
        var x = point.el.x, y = point.el.y;
        // Keep the box inside the plot; default to the left of the marker.
        var bx = x - w - 10;
        if (bx < area.left + 2) bx = x + 10;
        if (bx + w > area.right - 2) bx = area.right - 2 - w;
        var by = place === "below" ? y + 8 : y - h - 8;
        if (by < area.top + 2) by = y + 8;
        if (by + h > area.bottom - 2) by = area.bottom - 2 - h;
        ctx.save();
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1;
        roundRect(ctx, bx, by, w, h, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = colour;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, bx + padX, by + h / 2 + 0.5);
        ctx.restore();
        return { x: bx, y: by, w: w, h: h };
      }

      var forecastPoint = lastPoint(forecastMeta, chart.data.datasets[1].data);
      var thresholdPoint = lastPoint(thresholdMeta, chart.data.datasets[2].data);

      var fBox = null;
      if (forecastPoint) {
        fBox = chip(forecastPoint, "Forecast " + num(forecastPoint.value), "#2563eb",
          forecastPoint.value <= (thresholdPoint ? thresholdPoint.value : Infinity) ? "below" : "above");
      }
      if (thresholdPoint) {
        // Place threshold opposite the forecast to avoid overlap.
        var place = forecastPoint && forecastPoint.value <= thresholdPoint.value ? "above" : "below";
        chip(thresholdPoint, "Threshold " + num(thresholdPoint.value), "#dc2626", place);
      }

      // Target-week label pinned under the forecast marker on the axis.
      if (forecastPoint && options.targetLabel) {
        ctx.save();
        ctx.font = "600 11px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#1e3a8a";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        var tx = Math.min(Math.max(forecastPoint.el.x, area.left + 20), area.right - 20);
        ctx.fillText(options.targetLabel, tx, area.bottom + 4);
        ctx.restore();
      }
    }
  };

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function visibleRows(data) {
    var observed = data.rows.filter(function (row) {
      return row.period === "observed";
    });
    var forecast = data.rows.filter(function (row) {
      return row.period === "forecast";
    });
    if (state.range !== "all") {
      observed = observed.slice(-Number(state.range));
    }
    return observed.concat(forecast);
  }

  function renderWeekly(data) {
    state.weekly = data;
    var badge = $("weekly-model-badge");
    if (badge) {
      badge.textContent = data.count_model_name ? "Count model: " + data.count_model_name : "";
      badge.hidden = !data.count_model_name;
    }

    var rows = visibleRows(data);
    if (!rows.length) {
      show($("weekly-chart").parentNode, false);
      setState("weekly-state", MSG_NO_DATA, "empty");
      return;
    }
    if (typeof Chart === "undefined") {
      show($("weekly-chart").parentNode, false);
      setState("weekly-state", "Chart library unavailable.", "error");
      return;
    }

    var labels = rows.map(function (row) {
      return row.week_label;
    });
    var lastObserved = -1;
    rows.forEach(function (row, index) {
      if (row.period === "observed") lastObserved = index;
    });

    var datasets = [
      {
        label: "Observed confirmed cases",
        data: rows.map(function (row) {
          return row.period === "observed" ? row.actual_cases : null;
        }),
        borderColor: "#0d9488",
        backgroundColor: "#0d9488",
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.2,
        spanGaps: false
      },
      {
        label: "Forecast cases · " + (data.count_model_name || "selected count model"),
        data: rows.map(function (row) {
          return row.period === "forecast" ? row.predicted_cases : null;
        }),
        borderColor: "#2563eb",
        backgroundColor: "#2563eb",
        borderWidth: 2,
        borderDash: [6, 4],
        pointRadius: 5,
        pointHoverRadius: 7,
        pointStyle: "rectRot",
        spanGaps: false,
        // The final forecast target sits on the right edge of the plot, so the
        // marker must be allowed to paint outside the clipping box.
        clip: false
      },
      {
        label: "Surge threshold at forecast target",
        data: rows.map(function (row) {
          return row.surge_threshold;
        }),
        borderColor: "#dc2626",
        backgroundColor: "#dc2626",
        borderWidth: 2,
        pointRadius: 7,
        pointHoverRadius: 9,
        pointStyle: "crossRot",
        showLine: false,
        spanGaps: false,
        clip: false
      }
    ];

    if (data.intervals_available) {
      datasets.push({
        label: "Prediction interval (lower)",
        data: rows.map(function (row) {
          return row.prediction_interval_lower;
        }),
        borderColor: "#bfdbfe",
        backgroundColor: "rgba(191, 219, 254, 0.45)",
        borderWidth: 1,
        pointRadius: 0,
        fill: "+1",
        spanGaps: false
      });
      datasets.push({
        label: "Prediction interval (upper)",
        data: rows.map(function (row) {
          return row.prediction_interval_upper;
        }),
        borderColor: "#bfdbfe",
        backgroundColor: "rgba(191, 219, 254, 0.45)",
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        spanGaps: false
      });
    }

    if (state.chart) state.chart.destroy();
    state.chart = new Chart($("weekly-chart").getContext("2d"), {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        // Right padding keeps the forecast-target markers off the plot edge;
        // bottom padding reserves room for the direct target-week label.
        layout: { padding: { right: 16, top: 20, bottom: 18 } },
        scales: {
          x: {
            offset: true,
            grid: { display: false },
            ticks: { maxTicksLimit: 12, autoSkip: true, color: "#64748b", font: { size: 11 } }
          },
          y: {
            beginAtZero: true,
            grid: { color: "#eef2f6" },
            ticks: {
              color: "#64748b",
              font: { size: 11 },
              callback: function (value) {
                return num(value);
              }
            },
            title: { display: true, text: "Confirmed cases", color: "#64748b", font: { size: 11 } }
          }
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 10, usePointStyle: true, color: "#475569", font: { size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                if (context.parsed.y === null) return null;
                return context.dataset.label + ": " + num(context.parsed.y, 0);
              }
            }
          },
          forecastDivider: { index: lastObserved >= 0 && lastObserved < rows.length - 1 ? lastObserved : null },
          directLabels: { targetLabel: data.target_label || (rows.length ? rows[rows.length - 1].week_label : null) }
        }
      },
      plugins: [forecastDivider, directLabels]
    });

    show($("weekly-chart").parentNode, true);
    setState("weekly-state", "", null);

    var note = [];
    note.push(data.threshold_note);
    note.push(data.intervals_note);
    if (data.scope === "national" && data.national_threshold_note) note.push(data.national_threshold_note);
    $("weekly-note").textContent = note.filter(Boolean).join(" ");
  }

  function weeklyError(err) {
    show($("weekly-chart").parentNode, false);
    setState("weekly-state", describeError(err, "weekly"), "error");
    $("weekly-note").textContent = "";
  }

  // ---- risk table -------------------------------------------------------
  function riskBadge(level) {
    return '<span class="badge ' + (RISK_CLASS[level] || "badge-nodata") + '">' + level + "</span>";
  }

  // Update a sortable header's visible label without destroying its sort button.
  function setHeaderLabel(thId, label, sub) {
    var th = $(thId);
    if (!th) return;
    var target = th.querySelector(".th-label") || th;
    target.innerHTML = label + (sub ? '<i class="th-sub">' + sub + "</i>" : "");
  }

  function sortedRows() {
    var rows = state.riskRows.slice();
    var term = state.search.trim().toLowerCase();
    if (term) {
      rows = rows.filter(function (row) {
        return (row.district_name || "").toLowerCase().indexOf(term) !== -1;
      });
    }
    var key = state.sort.key;
    var dir = state.sort.dir === "asc" ? 1 : -1;
    rows.sort(function (a, b) {
      var av = key === "risk_rank" ? RISK_RANK[a.risk_level] : a[key];
      var bv = key === "risk_rank" ? RISK_RANK[b.risk_level] : b[key];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (typeof av === "string") return av.localeCompare(bv) * dir;
      return (av - bv) * dir;
    });
    return rows;
  }

  function riskRowHtml(row) {
    var completeness =
      row.reporting_completeness === null
        ? '<span class="cell-missing" title="Not reported">Not reported</span>'
        : pct(row.reporting_completeness);
    var observedFlag = row.observed_cases_missing
      ? ' <span class="cell-missing" title="Observed cases missing for the issue week">no obs.</span>'
      : "";
    return (
      "<tr>" +
      "<td>" + row.district_name + observedFlag + "</td>" +
      '<td title="' + (row.risk_basis || "").replace(/"/g, "&quot;") + '">' + riskBadge(row.risk_level) + "</td>" +
      "<td>" + num(row.predicted_cases) + "</td>" +
      "<td>" + num(row.surge_predicted_cases) + "</td>" +
      "<td>" + num(row.threshold_used) + "</td>" +
      '<td class="' + (Number(row.predicted_surge_score) > 0 ? "score-over" : "score-under") + '">' +
      signed(row.predicted_surge_score) + "</td>" +
      "<td>" + (TREND_LABEL[row.trend] || "Unknown") + "</td>" +
      "<td>" + completeness + "</td>" +
      "<td>" + row.forecast_horizon_weeks + "w</td>" +
      "</tr>"
    );
  }

  function paintRiskTable() {
    // Sort and search the COMPLETE set first, then slice the current page.
    var rows = sortedRows();
    var body = $("risk-tbody");
    if (!rows.length) {
      body.innerHTML = "";
      show($("risk-table").parentNode, false);
      updatePager(0, 0);
      setState("risk-state", state.search ? "No district matches this search." : MSG_NO_DATA, "empty");
      return;
    }

    var pageCount = Math.max(1, Math.ceil(rows.length / state.pageSize));
    if (state.page > pageCount) state.page = pageCount;
    if (state.page < 1) state.page = 1;
    var start = (state.page - 1) * state.pageSize;
    var pageRows = rows.slice(start, start + state.pageSize);

    body.innerHTML = pageRows.map(riskRowHtml).join("");
    show($("risk-table").parentNode, true);
    setState("risk-state", "", null);
    updatePager(rows.length, pageCount, start, pageRows.length);
  }

  function updatePager(total, pageCount, start, shown) {
    var pager = $("risk-pager");
    if (!pager) return;
    if (!total) {
      pager.hidden = true;
      return;
    }
    pager.hidden = false;
    var first = start + 1;
    var last = start + shown;
    $("risk-page-status").textContent =
      "Showing " + first + "–" + last + " of " + total + " districts · page " +
      state.page + " of " + pageCount;
    var prev = $("risk-prev");
    var next = $("risk-next");
    prev.disabled = state.page <= 1;
    next.disabled = state.page >= pageCount;
  }

  function renderRisk(data) {
    state.riskRows = data.rows || [];

    var countModel = (data.selected_models && data.selected_models.count.model_name) || "count model";
    var surgeModel = (data.selected_models && data.selected_models.surge.model_name) || "surge model";
    // The model name sits under each metric so the reader can never mistake which
    // model produced a column. The label lives inside the sort button, so update
    // only the label span and leave the button intact.
    setHeaderLabel("th-predicted", "Predicted cases", countModel);
    setHeaderLabel("th-surge-cases", "Surge-model cases", surgeModel);
    setHeaderLabel("th-risk", "Risk level", surgeModel);

    // New data set: start at the first page.
    state.page = 1;

    var counts = data.risk_counts || {};
    var summary = ["Very High", "High", "Moderate", "Low", "No data"]
      .filter(function (level) {
        return counts[level];
      })
      .map(function (level) {
        return counts[level] + " " + level;
      })
      .join(" · ");

    $("risk-subhead").textContent =
      (data.selected_models && data.selected_models.same_model
        ? "Counts and risk both from " + countModel + "."
        : "Predicted cases from " + countModel + "; risk assessment from " + surgeModel + ". These are different models and are not combined.") +
      (summary ? " " + summary + "." : "");

    var note = [data.score_definition, data.threshold_method_note, data.risk_assessment_note];
    if (!data.surge_probability_available) {
      note.unshift("No trained surge probability exists in these outputs, so risk is a count-threshold assessment.");
    }
    $("risk-note").textContent = note.filter(Boolean).join(" ");

    paintRiskTable();
  }

  function riskError(err) {
    state.riskRows = [];
    $("risk-tbody").innerHTML = "";
    show($("risk-table").parentNode, false);
    setState("risk-state", describeError(err, "risk"), "error");
    $("risk-subhead").textContent = "";
    $("risk-note").textContent = "";
  }

  // ---- map --------------------------------------------------------------
  function featureStyle(feature) {
    var selected = state.selectedDistrict;
    var isSelected = selected && feature.properties.district_id === selected;
    return {
      fillColor: RISK_COLOURS[feature.properties.risk_level] || RISK_COLOURS["No data"],
      fillOpacity: selected && !isSelected ? 0.35 : 0.85,
      color: isSelected ? "#0f172a" : "#ffffff",
      weight: isSelected ? 2.5 : 0.5
    };
  }

  function tooltipHtml(props) {
    var rows = [
      ["Risk level", props.risk_level],
      ["Predicted cases · " + (props.count_model_name || "count"), num(props.predicted_cases)],
      ["Surge-model cases · " + (props.surge_model_name || "surge"), num(props.surge_predicted_cases)],
      ["Surge threshold", num(props.threshold_used)],
      ["Surge score", signed(props.predicted_surge_score)],
      ["Horizon", props.forecast_horizon_weeks + " weeks" + (props.target_label ? " → " + props.target_label : "")]
    ];
    return (
      '<div class="map-tip"><strong>' +
      props.district_name +
      "</strong>" +
      rows
        .map(function (row) {
          return "<span><i>" + row[0] + "</i>" + row[1] + "</span>";
        })
        .join("") +
      '<em>Risk warrants review; it does not confirm an outbreak.</em></div>'
    );
  }

  function renderMap(data) {
    if (typeof L === "undefined") {
      setState("map-state", "Map library unavailable.", "error");
      show($("risk-map"), false);
      return;
    }
    if (!data.features || !data.features.length) {
      setState("map-state", MSG_BOUNDARY, "empty");
      show($("risk-map"), false);
      return;
    }

    show($("risk-map"), true);
    setState("map-state", "", null);

    if (!state.map) {
      // No tile layer is added: the choropleth renders entirely offline.
      // SVG (Leaflet's default) is used rather than canvas so each district is a
      // real element that can be styled, brought to front and hit-tested.
      state.map = L.map("risk-map", {
        zoomControl: true,
        attributionControl: false,
        scrollWheelZoom: false
      });
    }
    if (state.mapLayer) {
      state.map.removeLayer(state.mapLayer);
      state.mapLayer = null;
    }
    state.mapFeatures = {};

    state.mapLayer = L.geoJSON(data, {
      style: featureStyle,
      onEachFeature: function (feature, layer) {
        state.mapFeatures[feature.properties.district_id] = layer;
        layer.bindTooltip(tooltipHtml(feature.properties), { sticky: true, className: "map-tooltip" });
        layer.on("click", function () {
          var select = $("filter-district");
          if (select) {
            select.value = feature.properties.district_id;
            select.dispatchEvent(new Event("change"));
          }
        });
      }
    }).addTo(state.map);

    state.map.fitBounds(state.mapLayer.getBounds(), { padding: [8, 8] });
    state.nationalBounds = state.mapLayer.getBounds();
    applyMapSelection();

    var missing = data.missing_geometry_districts || [];
    $("map-note").textContent =
      "Source: District boundaries — UBOS, simplified for display. Joined on district_id." +
      (missing.length ? " Districts without geometry: " + missing.length + "." : "") +
      " " + (data.risk_assessment_note || "");
  }

  function clearMapLabels() {
    if (state.labelLayer) {
      state.map.removeLayer(state.labelLayer);
      state.labelLayer = null;
    }
  }

  // In district view, permanently label the selected district and any neighbour
  // whose label does not collide (pixel-space greedy placement). National view
  // carries no permanent labels — hover tooltips remain in both views.
  function drawMapLabels() {
    clearMapLabels();
    var selected = state.selectedDistrict;
    if (!selected || !state.mapFeatures[selected] || !state.map) return;

    state.labelLayer = L.layerGroup().addTo(state.map);
    var placed = [];

    function labelBox(latlng, name) {
      var p = state.map.latLngToContainerPoint(latlng);
      var w = name.length * 6.2 + 12;
      return { x: p.x - w / 2, y: p.y - 9, w: w, h: 18, cx: p.x, cy: p.y };
    }
    function overlaps(a, b) {
      return !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
    }
    function addLabel(latlng, name, cls) {
      L.tooltip({ permanent: true, direction: "center", className: cls, opacity: 1 })
        .setLatLng(latlng)
        .setContent(name)
        .addTo(state.labelLayer);
    }

    // Selected district always gets a label.
    var selCenter = state.mapFeatures[selected].getBounds().getCenter();
    var selName = (state.mapFeatures[selected].feature.properties.district_name) || selected;
    placed.push(labelBox(selCenter, selName));
    addLabel(selCenter, selName, "map-label map-label-selected");

    // Neighbours: only those visible in the current viewport, nearest first,
    // added while collision-free so labels stay readable.
    var viewport = state.map.getBounds();
    var candidates = [];
    Object.keys(state.mapFeatures).forEach(function (id) {
      if (id === selected) return;
      var layer = state.mapFeatures[id];
      var center = layer.getBounds().getCenter();
      if (!viewport.contains(center)) return;
      candidates.push({
        center: center,
        name: (layer.feature.properties.district_name) || id,
        dist: selCenter.distanceTo(center)
      });
    });
    candidates.sort(function (a, b) { return a.dist - b.dist; });

    for (var i = 0; i < candidates.length; i++) {
      var box = labelBox(candidates[i].center, candidates[i].name);
      var collides = placed.some(function (b) { return overlaps(box, b); });
      if (!collides) {
        placed.push(box);
        addLabel(candidates[i].center, candidates[i].name, "map-label map-label-neighbour");
      }
    }
  }

  function applyMapSelection() {
    if (!state.mapLayer) return;
    state.mapLayer.setStyle(featureStyle);
    var selected = state.selectedDistrict;
    if (selected && state.mapFeatures[selected]) {
      state.map.fitBounds(state.mapFeatures[selected].getBounds(), { padding: [24, 24], maxZoom: 9 });
      state.mapFeatures[selected].bringToFront();
      // Labels are placed after the pan/zoom settles so pixel positions are final.
      state.map.once("moveend zoomend", drawMapLabels);
      // Also draw immediately in case no movement is triggered.
      drawMapLabels();
    } else {
      clearMapLabels();
      if (state.nationalBounds) state.map.fitBounds(state.nationalBounds, { padding: [8, 8] });
    }
  }

  function mapError(err) {
    show($("risk-map"), false);
    setState("map-state", describeError(err, "map"), "error");
  }

  // ---- climate ----------------------------------------------------------
  var CLIMATE_CLASS = { rainfall: "climate-rain", temperature: "climate-temp", humidity: "climate-humid" };
  // Inline SVGs mirror the stroke style of templates/_icons.html, which Jinja
  // macros cannot supply to client-rendered cards.
  var SVG_ATTRS =
    'class="ic" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" ' +
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"';
  var CLIMATE_ICON = {
    rainfall:
      "<svg " + SVG_ATTRS + '><path d="M6 14a4 4 0 010-8 5 5 0 019.6-1.5A3.5 3.5 0 1118 14z"/>' +
      '<line x1="8" y1="18" x2="7" y2="21"/><line x1="12" y1="18" x2="11" y2="21"/>' +
      '<line x1="16" y1="18" x2="15" y2="21"/></svg>',
    temperature:
      "<svg " + SVG_ATTRS + '><path d="M14 14.8V4a2 2 0 10-4 0v10.8a4 4 0 104 0z"/></svg>',
    humidity:
      "<svg " + SVG_ATTRS + '><path d="M12 3s5 5.5 5 9a5 5 0 11-10 0c0-3.5 5-9 5-9z"/></svg>'
  };

  function renderClimate(data) {
    var rows = data.rows || [];
    if (!rows.length || data.status !== "available") {
      $("climate-cards").innerHTML = "";
      setState("climate-state", MSG_NO_DATA, "empty");
      $("climate-subhead").textContent = "";
      return;
    }
    setState("climate-state", "", null);

    $("climate-subhead").textContent =
      "Data week " + (data.data_week_label || "—") + " · " + data.aggregation_label;

    // Every lag is shown with equal weight. No lag is highlighted by the forecast
    // horizon, because the pipeline computes no lag-malaria correlation.
    $("climate-cards").innerHTML = rows
      .map(function (row) {
        var measures = row.measures || {};
        var current = measures.current || {};
        var lagHtml = ["lag_4", "lag_8", "lag_12"]
          .map(function (key) {
            var weeks = key.split("_")[1];
            var measure = measures[key] || {};
            return (
              '<div class="lag-row"><span class="lag-name">' + weeks + "-week lag</span>" +
              '<span class="lag-val">' +
              (measure.value === null || measure.value === undefined ? "—" : num(measure.value, 1) + " " + row.unit) +
              "</span></div>"
            );
          })
          .join("");
        return (
          '<div class="climate-col">' +
          '<div class="climate-col-head"><span class="climate-icon ' + CLIMATE_CLASS[row.variable] +
          '" aria-hidden="true">' + (CLIMATE_ICON[row.variable] || "") + "</span>" +
          '<span class="climate-name">' + row.label + '</span>' +
          '<span class="climate-src">' + row.source + "</span></div>" +
          '<div class="climate-current"><span class="climate-value">' +
          (current.value === null || current.value === undefined ? "—" : num(current.value, 1) + " " + row.unit) +
          '</span><span class="climate-unit">current · ' + row.unit_label + "</span></div>" +
          '<div class="lag-list">' + lagHtml + "</div>" +
          '<div class="climate-cov">' + current.districts_with_value + " of " + row.coverage.districts_expected + " districts reporting</div>" +
          "</div>"
        );
      })
      .join("");

    $("climate-note").textContent = [data.aggregation_label, data.correlation_note, data.causation_note]
      .filter(Boolean)
      .join(" ");
  }

  function climateError(err) {
    $("climate-cards").innerHTML = "";
    $("climate-subhead").textContent = "";
    $("climate-note").textContent = "";
    setState("climate-state", describeError(err, "climate"), "error");
  }

  // ---- model performance ------------------------------------------------
  function selectedTag(row) {
    var tags = (row.selected_labels || [])
      .map(function (label) {
        return '<span class="tag tag-' + label.toLowerCase() + '">' + label + "</span>";
      })
      .join("");
    return tags || '<span class="cell-muted">—</span>';
  }

  function renderPerformance(data, horizon) {
    var rows = (data.rows || []).filter(function (row) {
      return String(row.horizon) === String(horizon);
    });
    if (!rows.length) {
      ["perf-count-tbody", "perf-surge-tbody"].forEach(function (id) { $(id).innerHTML = ""; });
      $("perf-selection").innerHTML = "";
      setState("performance-state", data.message || MSG_NO_DATA, "empty");
      return;
    }
    setState("performance-state", "", null);

    // Count Forecasting tab — regression metrics only (no accuracy field exists).
    $("perf-count-tbody").innerHTML = rows
      .map(function (row) {
        return (
          '<tr class="' + (row.selected_for_count ? "is-selected" : "") + '">' +
          "<td>" + row.model_name + '<i class="cell-sub">' + (row.model_scope || "") + "</i></td>" +
          '<td class="tag-cell">' + (row.selected_for_count ? '<span class="tag tag-count">Count</span>' : '<span class="cell-muted">—</span>') + "</td>" +
          "<td>" + num(row.mae, 1) + "</td><td>" + num(row.rmse, 1) + "</td><td>" + num(row.mape, 1) +
          "</td><td>" + num(row.wape, 1) + "</td><td>" + num(row.r_squared, 3) + "</td></tr>"
        );
      })
      .join("");

    // Surge Detection tab — classification metrics only.
    $("perf-surge-tbody").innerHTML = rows
      .map(function (row) {
        return (
          '<tr class="' + (row.selected_for_surge ? "is-selected" : "") + '">' +
          "<td>" + row.model_name + '<i class="cell-sub">' + (row.model_scope || "") + "</i></td>" +
          '<td class="tag-cell">' + (row.selected_for_surge ? '<span class="tag tag-surge">Surge</span>' : '<span class="cell-muted">—</span>') + "</td>" +
          "<td>" + num(row.sensitivity, 3) + "</td><td>" + num(row.specificity, 3) + "</td><td>" +
          num(row.precision, 3) + "</td><td>" + num(row.f1, 3) + "</td><td>" + num(row.auc, 3) + "</td></tr>"
        );
      })
      .join("");

    // Model Selection tab — the two selected models for this horizon + real rules.
    var selectedByHorizon = (data.selected_models || {})[String(horizon)] || {};
    var count = selectedByHorizon.count || {};
    var surge = selectedByHorizon.surge || {};
    var rules = data.selection_rules || {};
    $("perf-selection").innerHTML =
      '<div class="selection-card"><span class="selection-role">Count model</span>' +
      '<span class="selection-model">' + (count.model_name || "—") + "</span>" +
      '<p class="selection-rule">The count model is selected using the lowest WAPE when available. ' +
      "RMSE is used when WAPE is unavailable.</p></div>" +
      '<div class="selection-card"><span class="selection-role">Surge model</span>' +
      '<span class="selection-model">' + (surge.model_name || "—") + "</span>" +
      '<p class="selection-rule">The surge method must first meet sensitivity and specificity ' +
      "guardrails, then is selected using F2 with sensitivity, PR-AUC, precision and specificity " +
      "as ordered tie-breakers. Accuracy alone is not used.</p></div>";

    $("performance-subhead").textContent = "Horizon " + horizon + " weeks · national validation across all districts.";
    $("performance-note").textContent = data.scope_note || "";
  }

  function performanceError(err) {
    ["perf-count-tbody", "perf-surge-tbody"].forEach(function (id) { $(id).innerHTML = ""; });
    $("perf-selection").innerHTML = "";
    setState("performance-state", describeError(err, "performance"), "error");
    $("performance-subhead").textContent = "";
    $("performance-note").textContent = "";
  }

  // ---- orchestration ----------------------------------------------------
  function loadAll() {
    var params = currentParams();
    state.selectedDistrict = params.district_id && params.district_id !== "all" ? params.district_id : null;

    // Supersede any in-flight generation: bump the counter and abort old fetches.
    state.generation += 1;
    var generation = state.generation;
    if (state.abort) state.abort.abort();
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    state.abort = controller;
    var signal = controller ? controller.signal : undefined;

    ["kpi-cases", "kpi-completeness", "kpi-risk", "kpi-horizon"].forEach(function (id) {
      $(id + "-note").textContent = MSG_LOADING;
    });
    setState("weekly-state", MSG_LOADING, "loading");
    setState("risk-state", MSG_LOADING, "loading");
    setState("climate-state", MSG_LOADING, "loading");
    setState("performance-state", MSG_LOADING, "loading");

    var summary = guarded(generation, function (body) { renderSummary(body.data); }, summaryError);
    fetchSection("/api/dashboard/summary", params, signal).then(summary.apply, summary.fail);

    var weekly = guarded(generation, function (body) { renderWeekly(body.data); }, weeklyError);
    fetchSection("/api/dashboard/weekly-series", params, signal).then(weekly.apply, weekly.fail);

    var risk = guarded(generation, function (body) { renderRisk(body.data); }, riskError);
    fetchSection("/api/dashboard/risk-table", params, signal).then(risk.apply, risk.fail);

    var climate = guarded(generation, function (body) { renderClimate(body.data); }, climateError);
    fetchSection("/api/dashboard/climate-summary", params, signal).then(climate.apply, climate.fail);

    var perf = guarded(generation, function (body) { renderPerformance(body.data, params.horizon || 4); }, performanceError);
    fetchSection("/api/dashboard/model-performance", params, signal).then(perf.apply, perf.fail);

    // The map always shows every district, so it only reloads when the issue or
    // horizon changes. A district-only change just restyles and zooms — that path
    // is synchronous and always reflects the current selection.
    var mapKey = [params.horizon, params.issue_year, params.issue_week].join("|");
    if (mapKey !== state.mapKey || !state.mapLayer) {
      setState("map-state", MSG_LOADING, "loading");
      state.mapKey = mapKey;
      var map = guarded(generation, function (body) { renderMap(body.data); }, mapError);
      fetchSection("/api/dashboard/map", params, signal).then(map.apply, map.fail);
    } else {
      applyMapSelection();
    }
  }

  function bind() {
    initDistrictCombobox();
    ["filter-horizon", "filter-district", "filter-issue"].forEach(function (id) {
      var el = $(id);
      if (el) el.addEventListener("change", loadAll);
    });

    var search = $("risk-search");
    if (search) {
      search.addEventListener("input", function (event) {
        state.search = event.target.value;
        state.page = 1; // a new search resets pagination
        paintRiskTable();
      });
    }

    // Sorting is driven by a real <button> inside each sortable header, so it
    // works with Enter/Space via native button behaviour. aria-sort stays on th.
    document.querySelectorAll("#risk-table th.sortable .th-sort").forEach(function (button) {
      button.addEventListener("click", function () {
        var th = button.closest("th");
        var key = th.dataset.key;
        if (state.sort.key === key) {
          state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
        } else {
          state.sort.key = key;
          state.sort.dir = th.dataset.type === "text" ? "asc" : "desc";
        }
        document.querySelectorAll("#risk-table th.sortable").forEach(function (other) {
          other.setAttribute("aria-sort", "none");
          other.classList.remove("sorted-asc", "sorted-desc");
        });
        th.setAttribute("aria-sort", state.sort.dir === "asc" ? "ascending" : "descending");
        th.classList.add(state.sort.dir === "asc" ? "sorted-asc" : "sorted-desc");
        state.page = 1; // a new sort resets pagination
        paintRiskTable();
      });
    });

    var prev = $("risk-prev");
    var next = $("risk-next");
    if (prev) prev.addEventListener("click", function () { state.page -= 1; paintRiskTable(); });
    if (next) next.addEventListener("click", function () { state.page += 1; paintRiskTable(); });

    document.querySelectorAll(".chart-toolbar .chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        document.querySelectorAll(".chart-toolbar .chip").forEach(function (other) {
          other.classList.remove("is-active");
        });
        chip.classList.add("is-active");
        state.range = chip.dataset.range;
        if (state.weekly) renderWeekly(state.weekly);
      });
    });

    bindTabs();
    bindHowToggles();
  }

  // ---- model-performance tabs (WAI-ARIA tablist) ------------------------
  function bindTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll(".tabs .tab"));
    if (!tabs.length) return;

    function activate(tab) {
      tabs.forEach(function (other) {
        var selected = other === tab;
        other.setAttribute("aria-selected", selected ? "true" : "false");
        other.classList.toggle("is-active", selected);
        other.tabIndex = selected ? 0 : -1;
        var panel = $(other.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      });
    }

    tabs.forEach(function (tab, index) {
      tab.addEventListener("click", function () { activate(tab); });
      tab.addEventListener("keydown", function (event) {
        var next = null;
        if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
        else if (event.key === "ArrowLeft") next = tabs[(index - 1 + tabs.length) % tabs.length];
        else if (event.key === "Home") next = tabs[0];
        else if (event.key === "End") next = tabs[tabs.length - 1];
        if (next) {
          event.preventDefault();
          activate(next);
          next.focus();
        }
      });
    });
  }

  // ---- level-2 "How to read this" expandable notes ----------------------
  function bindHowToggles() {
    document.querySelectorAll(".how-toggle").forEach(function (toggle) {
      toggle.addEventListener("click", function () {
        var note = $(toggle.getAttribute("aria-controls"));
        if (!note) return;
        var open = note.hidden;
        note.hidden = !open;
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    });
  }

  function init() {
    if (!$("risk-map")) return; // not the dashboard page
    bind();
    loadAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
