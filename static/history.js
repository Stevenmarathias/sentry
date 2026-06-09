// Sentry History (Pass 13): "Quiz this lecture" per-row action.
//
// Each .quiz-this-btn lives inside its row's outer <a class="session-row">.
// Without intercepting the click, the browser would follow the outer
// anchor first (which opens the saved/regenerated quiz) and the new
// per-lecture quiz route would never be hit. So the handler explicitly:
//   - stopPropagation so the row's anchor doesn't see the click
//   - preventDefault on the anchor activation
//   - swap the button into a "Generating…" loading state
//   - navigate to the data-quiz-url
//
// The server's quiz call can take ~10s on a first generation; the cache
// is module-level so subsequent visits are near-instant.

document.querySelectorAll(".quiz-this-btn").forEach((btn) => {
  const label = btn.querySelector(".quiz-this-label");
  const spinner = btn.querySelector(".quiz-this-spinner");

  btn.addEventListener("click", (e) => {
    // The row-link is the click target's ancestor; cancel its default and
    // stop the bubble before navigating to the per-lecture quiz URL.
    e.preventDefault();
    e.stopPropagation();

    if (btn.disabled) return;
    btn.disabled = true;
    if (label) label.hidden = true;
    if (spinner) spinner.hidden = false;

    // Brief defer so the loading-state paint lands before the request.
    setTimeout(() => {
      const url = btn.dataset.quizUrl;
      if (url) window.location = url;
    }, 0);
  });

  // Keyboard activation on the button itself follows the click path; the
  // row anchor's Enter handler would otherwise navigate first.
  btn.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    e.stopPropagation();
    btn.click();
  });
});
