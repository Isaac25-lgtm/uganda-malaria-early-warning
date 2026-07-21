/* Malaria Early Warning Dashboard - foundation JS.
 * Minimal by design: only the mobile navigation toggle is wired here.
 * Charts, the district risk map and live filters arrive in later prompts.
 */
(function () {
  "use strict";

  // Close the off-canvas sidebar after a navigation choice on small screens.
  document.addEventListener("click", function (event) {
    var link = event.target.closest(".nav-item");
    if (link && document.body.classList.contains("nav-open")) {
      document.body.classList.remove("nav-open");
    }
  });
})();
