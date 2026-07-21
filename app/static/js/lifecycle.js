/* Lifecycle pages — shared read-only presentation engine.
 *
 * Declarative: a page container carries data-endpoint; child elements with
 * data-field="dotted.path" are populated from the API `data`. No epidemiological
 * result, metric, threshold or risk is computed here — values are displayed as
 * returned. A request-generation guard ensures the latest load wins and aborts
 * superseded requests; an API error never leaves stale values on screen.
 */
(function () {
  "use strict";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function fmt(value, kind) {
    if (value === null || value === undefined || value === "") return "—";
    if (kind === "pct") return Number(value).toLocaleString("en-GB", { maximumFractionDigits: 1 }) + "%";
    if (kind === "num") {
      var n = Number(value);
      if (!isFinite(n)) return "—";
      return n.toLocaleString("en-GB", { maximumFractionDigits: 0 });
    }
    if (kind === "num1") return Number(value).toLocaleString("en-GB", { maximumFractionDigits: 1 });
    return String(value);
  }

  function dig(obj, path) {
    return path.split(".").reduce(function (acc, key) {
      return acc === null || acc === undefined ? undefined : acc[key];
    }, obj);
  }

  function setState(page, kind, message) {
    var el = $("[data-lifecycle-state]", page);
    if (!el) return;
    if (!kind) { el.hidden = true; el.textContent = ""; el.className = "state-block"; return; }
    el.hidden = false;
    el.className = "state-block state-" + kind;
    el.textContent = message;
  }

  function describe(status) {
    if (status === 401) return { kind: "error", msg: "Your session has ended. Sign in again to load this page." };
    if (status === 400) return { kind: "error", msg: "Invalid request." };
    if (status === 503) return { kind: "empty", msg: "This evidence is not available yet." };
    return { kind: "error", msg: "This page could not reach the server." };
  }

  function bind(page, data) {
    $all("[data-field]", page).forEach(function (el) {
      var value = dig(data, el.getAttribute("data-field"));
      el.textContent = fmt(value, el.getAttribute("data-format"));
    });
    // Optional list rendering: <ul data-list="path" data-item-key="title">…
    $all("[data-list]", page).forEach(function (host) {
      var items = dig(data, host.getAttribute("data-list"));
      host.innerHTML = "";
      if (!Array.isArray(items)) return;
      var key = host.getAttribute("data-item-key");
      var subKey = host.getAttribute("data-item-sub");
      var statusKey = host.getAttribute("data-item-status");
      items.forEach(function (item) {
        var li = document.createElement("li");
        li.className = "stage-item";
        var status = statusKey ? dig(item, statusKey) : null;
        li.innerHTML =
          (status ? '<span class="stage-status stage-' + String(status).replace(/\s+/g, "-").toLowerCase() + '">' + status + "</span>" : "") +
          '<span class="stage-title">' + (key ? dig(item, key) : item) + "</span>" +
          (subKey && dig(item, subKey) ? '<span class="stage-detail">' + dig(item, subKey) + "</span>" : "");
        host.appendChild(li);
      });
    });
  }

  function load(page, onData) {
    var endpoint = page.getAttribute("data-endpoint");
    if (!endpoint) return;
    page.__gen = (page.__gen || 0) + 1;
    var gen = page.__gen;
    if (page.__abort) page.__abort.abort();
    var controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    page.__abort = controller;

    setState(page, "loading", "Loading…");
    fetch(endpoint, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal: controller ? controller.signal : undefined
    })
      .then(function (resp) {
        var type = resp.headers.get("content-type") || "";
        if (type.indexOf("application/json") === -1) {
          throw { status: resp.status === 401 || resp.redirected ? 401 : 0 };
        }
        return resp.json().then(function (body) {
          if (!resp.ok) throw { status: resp.status, body: body };
          return body;
        });
      })
      .then(function (body) {
        if (gen !== page.__gen) return; // superseded
        bind(page, body.data || {});
        setState(page, null);
        page.setAttribute("data-loaded", "true");
        if (onData) onData(body);
      })
      .catch(function (err) {
        if (err && (err.name === "AbortError")) return; // intentional
        if (gen !== page.__gen) return;
        var d = describe(err && err.status);
        setState(page, d.kind, d.msg);
      });
  }

  function init() {
    $all("[data-lifecycle-page]").forEach(function (page) {
      load(page);
      // Optional explicit retry button.
      var retry = $("[data-lifecycle-retry]", page);
      if (retry) retry.addEventListener("click", function () { load(page); });
    });
    // "How to read this" disclosures use native <details>; nothing to wire.
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.Lifecycle = { load: load, bind: bind, fmt: fmt };
})();
