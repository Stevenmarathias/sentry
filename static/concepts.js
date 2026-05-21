// Sentry concept browser: client-side sort toggle over the rendered table.

const tbody = document.getElementById("concept-rows");
const sortButtons = document.querySelectorAll(".sort-btn");

function sortRows(mode) {
  if (!tbody) return;
  const rows = Array.from(tbody.querySelectorAll("tr"));
  rows.sort((a, b) => {
    if (mode === "name") {
      return a.dataset.name.localeCompare(b.dataset.name);
    }
    if (mode === "recency") {
      // data-recency is the last-mention session file + timestamp, which
      // sorts chronologically as a string; ties fall back to importance.
      const cmp = b.dataset.recency.localeCompare(a.dataset.recency);
      if (cmp !== 0) return cmp;
      return Number(b.dataset.importance) - Number(a.dataset.importance);
    }
    return Number(b.dataset.importance) - Number(a.dataset.importance);
  });
  rows.forEach((row) => tbody.appendChild(row));
}

sortButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    sortButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sortRows(btn.dataset.sort);
  });
});
