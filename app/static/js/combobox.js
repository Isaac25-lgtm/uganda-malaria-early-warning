/* Shared accessible searchable combobox (Phase 29D corrective pass).
 *
 * This reusable implementation follows the established main-dashboard pattern:
 * a hidden <select> as the source of truth, a search <input> with role="combobox",
 * and a listbox of options. It supports mouse
 * selection, type-to-filter, ArrowUp/ArrowDown/Enter/Escape keyboard control,
 * and clearing. Selecting an option dispatches a native "change" event on the
 * hidden <select>, so callers listen to that select exactly as before.
 */
(function () {
  "use strict";

  function initCombobox(opts) {
    var source = document.getElementById(opts.source);
    var input = document.getElementById(opts.input);
    var list = document.getElementById(opts.list);
    if (!source || !input || !list) return;

    var options = Array.prototype.slice.call(list.querySelectorAll(".district-option"));
    var empty = list.querySelector(".district-option-empty");
    var active = null;

    function selectedOption() {
      return options.find(function (o) { return o.dataset.value === source.value; });
    }
    function syncSelection() {
      var selected = selectedOption() || options[0];
      input.value = selected ? selected.dataset.label : "";
      options.forEach(function (o) {
        o.setAttribute("aria-selected", o === selected ? "true" : "false");
      });
    }
    function visibleOptions() {
      return options.filter(function (o) { return !o.hidden; });
    }
    function setActive(option) {
      options.forEach(function (o) { o.classList.remove("is-active"); });
      active = option || null;
      if (active) {
        active.classList.add("is-active");
        input.setAttribute("aria-activedescendant", active.id);
        active.scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }
    function openList() { list.hidden = false; input.setAttribute("aria-expanded", "true"); }
    function closeList() { list.hidden = true; input.setAttribute("aria-expanded", "false"); setActive(null); }
    function filterOptions(query) {
      var term = String(query || "").trim().toLowerCase();
      options.forEach(function (o) {
        o.hidden = Boolean(term) && o.dataset.label.toLowerCase().indexOf(term) === -1;
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

    input.addEventListener("focus", function () { input.select(); filterOptions(""); });
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
        var exact = options.find(function (o) {
          return o.dataset.label.toLowerCase() === input.value.trim().toLowerCase();
        });
        if (exact && exact.dataset.value !== source.value) choose(exact);
        else if (!exact) syncSelection();
        closeList();
      }, 120);
    });
    options.forEach(function (option) {
      option.addEventListener("mousedown", function (e) { e.preventDefault(); });
      option.addEventListener("click", function () { choose(option); });
    });
    source.addEventListener("change", syncSelection);
    document.addEventListener("click", function (event) {
      if (!list.hidden && !event.target.closest("." + (opts.wrapClass || "district-combobox"))) {
        syncSelection();
        closeList();
      }
    });
    syncSelection();
  }

  window.MEWS = window.MEWS || {};
  window.MEWS.initCombobox = initCombobox;
})();
