/* Reports & Exports (Phase 29D).
 *
 * Formats and filter relevance come from the server registry rendered into the
 * page, never from decorative frontend config. Generation uses latest-request-
 * wins with an AbortController, disables the button while active so repeated
 * clicks cannot create a generation storm, and clears any stale download or
 * error state on every new attempt.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  if (!$("rp-generate")) return;

  var S = { gen: 0, abort: null, busy: false };

  function setState(msg, kind) {
    var el = $("rp-state");
    el.hidden = !msg;
    el.textContent = msg || "";
    el.className = "state-block" + (kind ? " state-" + kind : "");
  }

  function selected() {
    var radio = document.querySelector(".report-radio:checked");
    return {
      type: radio ? radio.value : null,
      formats: radio ? (radio.dataset.formats || "").split(",").filter(Boolean) : [],
      filters: radio ? (radio.dataset.filters || "").split(",").filter(Boolean) : []
    };
  }

  function applyRequestedReport() {
    try {
      var requested = new URLSearchParams(window.location.search).get("report_type");
      if (!requested) return;
      Array.prototype.some.call(document.querySelectorAll(".report-radio"), function (radio) {
        if (radio.value !== requested) return false;
        radio.checked = true;
        return true;
      });
    } catch (e) {
      // A malformed query cannot prevent the default report from loading.
    }
  }

  // Show only the filters this report actually supports, and only the formats
  // the server registry lists for it.
  function syncControls() {
    var s = selected();
    document.querySelectorAll(".report-card").forEach(function (card) {
      card.classList.toggle("is-active", card.querySelector(".report-radio").checked);
    });

    var wantsDistrict = s.filters.indexOf("district_required") !== -1 ||
                        s.filters.indexOf("district_optional") !== -1;
    var required = s.filters.indexOf("district_required") !== -1;
    show("wrap-horizon", s.filters.indexOf("horizon") !== -1);
    show("wrap-district", wantsDistrict);
    show("wrap-risk", s.filters.indexOf("risk_level") !== -1);
    $("rp-district").required = required;
    var districtLabel = document.querySelector('label[for="rp-district-search"]');
    if (districtLabel) districtLabel.textContent = required ? "District (required)" : "District";
    if (required && $("rp-district").value === "all") {
      // Keep "All" selectable but make the requirement explicit in the note.
      $("rp-note").textContent = "This report requires a specific district.";
    } else {
      $("rp-note").textContent = "";
    }

    // "html" is delivered by the print view, not by /generate, so it is never
    // offered as a downloadable format — that would produce a confusing 400.
    var downloadable = s.formats.filter(function (f) { return f !== "html"; });
    var fmt = $("rp-format");
    var previous = fmt.value;
    fmt.innerHTML = downloadable.map(function (f) {
      return '<option value="' + f + '">' + f.toUpperCase() + "</option>";
    }).join("");
    if (downloadable.indexOf(previous) !== -1) fmt.value = previous;
    $("rp-generate").disabled = downloadable.length === 0;
    updatePrintLink();
  }

  function show(id, visible) {
    var el = $(id);
    if (el) el.hidden = !visible;
  }

  function params() {
    var s = selected();
    var p = { report_type: s.type };
    if (s.filters.indexOf("horizon") !== -1) p.horizon = $("rp-horizon").value;
    if (s.filters.indexOf("district_required") !== -1 || s.filters.indexOf("district_optional") !== -1) {
      p.district_id = $("rp-district").value;
    }
    if (s.filters.indexOf("risk_level") !== -1) p.risk_level = $("rp-risk").value;
    var issue = $("rp-issue");
    if (issue && issue.value) {
      var bits = issue.value.split("-");
      if (bits.length === 2) { p.issue_year = bits[0]; p.issue_week = bits[1]; }
    }
    return p;
  }

  function query(extra) {
    var p = params();
    if (extra) Object.keys(extra).forEach(function (k) { p[k] = extra[k]; });
    return Object.keys(p).filter(function (k) { return p[k] !== null && p[k] !== undefined && p[k] !== ""; })
      .map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(p[k]); }).join("&");
  }

  function updatePrintLink() {
    var s = selected();
    var link = $("rp-print");
    var supported = s.formats.indexOf("html") !== -1;
    link.hidden = !supported;
    if (supported) link.href = "/reports/print?" + query();
  }

  function filenameFrom(disposition, fallback) {
    if (!disposition) return fallback;
    var match = /filename="?([^";]+)"?/i.exec(disposition);
    return match ? match[1] : fallback;
  }

  // Invalidate any in-flight generation: bump the generation counter, abort the
  // request and reset the busy/button state. Called on every filter change so a
  // stale response can neither download a file nor update state after the
  // selection has moved on (latest-request-wins).
  function invalidate() {
    S.gen += 1;
    if (S.abort) { S.abort.abort(); S.abort = null; }
    S.busy = false;
    $("rp-generate").disabled = false;
    $("rp-generate").textContent = "Generate report";
  }

  function generate() {
    if (S.busy) return;                 // no parallel generation storm
    S.gen += 1;
    var gen = S.gen;
    if (S.abort) S.abort.abort();
    var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    S.abort = ctrl;

    S.busy = true;
    $("rp-generate").disabled = true;
    $("rp-generate").textContent = "Generating…";
    setState("Generating report…", "loading");

    // `format` is only meaningful for /generate (the print view has its own
    // route), so it is added here rather than in the shared param builder.
    fetch("/reports/generate?" + query({ format: $("rp-format").value }), {
      credentials: "same-origin",
      signal: ctrl ? ctrl.signal : undefined
    }).then(function (response) {
      var type = response.headers.get("content-type") || "";
      if (!response.ok) {
        if (type.indexOf("application/json") !== -1) {
          return response.json().then(function (body) { throw { status: response.status, reason: body.reason }; });
        }
        throw { status: response.status };
      }
      var name = filenameFrom(response.headers.get("content-disposition"), "report");
      return response.blob().then(function (blob) { return { blob: blob, name: name }; });
    }).then(function (result) {
      if (gen !== S.gen) return;        // superseded
      var url = URL.createObjectURL(result.blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = result.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      setState("Report ready: " + result.name, "empty");
    }).catch(function (err) {
      if (err && err.name === "AbortError") return;
      if (gen !== S.gen) return;
      var message = "The report could not be generated.";
      if (err && err.status === 401) message = "Your session has expired. Please log in again.";
      else if (err && err.reason) message = err.reason;
      setState(message, "error");       // stale success state is replaced
    }).then(function () {
      if (gen !== S.gen) return;
      S.busy = false;
      $("rp-generate").disabled = false;
      $("rp-generate").textContent = "Generate report";
    });
  }

  // Governance links select a report through the same server-rendered allowlist.
  applyRequestedReport();

  // Any change to the report or its filters supersedes an active generation.
  document.querySelectorAll(".report-radio").forEach(function (radio) {
    radio.addEventListener("change", function () { invalidate(); setState("", null); syncControls(); });
  });
  ["rp-horizon", "rp-district", "rp-risk", "rp-issue", "rp-format"].forEach(function (id) {
    var el = $(id);
    if (el) el.addEventListener("change", function () { invalidate(); setState("", null); updatePrintLink(); });
  });
  $("rp-generate").addEventListener("click", generate);

  // Searchable, keyboard-accessible district combobox (shared pattern).
  if (window.MEWS && window.MEWS.initCombobox) {
    window.MEWS.initCombobox({ source: "rp-district", input: "rp-district-search",
                              list: "rp-district-options", wrapClass: "district-combobox" });
  }

  syncControls();
})();
