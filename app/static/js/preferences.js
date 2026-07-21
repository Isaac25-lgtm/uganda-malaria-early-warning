/* Shared display preferences (Phase 29E corrective pass).
 *
 * Loaded on every page BEFORE the page's own deferred script, so a stored
 * preference is already applied when that script reads its controls. This is
 * what makes the preferences genuinely observable across pages.
 *
 * Only two preferences exist, because only these two have real behaviour:
 *   default_horizon - sets the initial value of a horizon <select> where a page
 *                     exposes one. It changes nothing on the server and no
 *                     analytical definition; it only chooses an allowed
 *                     starting filter.
 *   reduce_motion   - adds a class that disables CSS transitions/animations.
 *
 * The store is versioned and strictly allowlisted. Malformed, unknown, tampered
 * or old-version data is ignored and the defaults are used; a storage failure
 * never blocks a page.
 */
(function () {
  "use strict";

  var PREF_KEY = "mews.preferences.v1";
  var PREF_VERSION = 1;
  var ALLOW = {
    default_horizon: { type: "enum", values: [4, 8, 12], def: 4 },
    reduce_motion: { type: "bool", def: false }
  };
  // Horizon selects exposed by the various pages.
  var HORIZON_SELECTS = ["filter-horizon", "fc-horizon", "me-horizon", "rp-horizon", "climate-horizon"];

  function defaults() {
    var out = {};
    Object.keys(ALLOW).forEach(function (k) { out[k] = ALLOW[k].def; });
    return out;
  }

  function validate(raw) {
    var out = defaults();
    if (!raw || typeof raw !== "object" || raw.version !== PREF_VERSION ||
        !raw.preferences || typeof raw.preferences !== "object") {
      return out;
    }
    Object.keys(ALLOW).forEach(function (k) {
      var spec = ALLOW[k];
      var v = raw.preferences[k];
      if (spec.type === "bool" && typeof v === "boolean") out[k] = v;
      else if (spec.type === "enum" && spec.values.indexOf(v) !== -1) out[k] = v;
    });
    return out;
  }

  function load() {
    try {
      return validate(JSON.parse(window.localStorage.getItem(PREF_KEY)));
    } catch (e) {
      return defaults();
    }
  }

  function save(prefs) {
    var clean = validate({ version: PREF_VERSION, preferences: prefs });
    try {
      window.localStorage.setItem(PREF_KEY,
        JSON.stringify({ version: PREF_VERSION, preferences: clean }));
      return true;
    } catch (e) {
      return false;
    }
  }

  function clear() {
    try { window.localStorage.removeItem(PREF_KEY); return true; } catch (e) { return false; }
  }

  function applyMotion(prefs) {
    var root = document.documentElement;
    if (prefs.reduce_motion) root.classList.add("reduce-motion");
    else root.classList.remove("reduce-motion");
  }

  function applyHorizon(prefs) {
    // Set the initial value only; the page's own script reads it afterwards.
    var applied = [];
    HORIZON_SELECTS.forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      var wanted = String(prefs.default_horizon);
      var has = Array.prototype.some.call(el.options, function (o) { return o.value === wanted; });
      if (has && el.value !== wanted) {
        el.value = wanted;
        applied.push(id);
      }
    });
    return applied;
  }

  function apply() {
    var prefs = load();
    applyMotion(prefs);
    var applied = applyHorizon(prefs);
    return { prefs: prefs, appliedTo: applied };
  }

  window.MEWS = window.MEWS || {};
  window.MEWS.preferences = {
    KEY: PREF_KEY, VERSION: PREF_VERSION, ALLOW: ALLOW,
    defaults: defaults, validate: validate, load: load, save: save, clear: clear, apply: apply
  };

  // Motion can be applied immediately; horizon needs the DOM.
  applyMotion(load());
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
