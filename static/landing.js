// Sentry landing page (Pass 8): class cards navigate to /class/<name>;
// the kebab is now slim (rename + delete only); audio picker + Start
// Session form live on the per-class overview page.

function classHomeUrl(name) {
  return window.SENTRY_CLASS_HOME_URL.replace("__C__", encodeURIComponent(name));
}

async function postJSON(url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

// ---- Audio input picker (new-class form only) -----------------------------
//
// The existing-class cards no longer have inline forms; only the
// "Create new class" form on this page still has an audio-device <select>.
// The per-class overview page has its own copy of this populate routine.
async function populateAudioDevices() {
  let devices = [];
  try {
    const res = await fetch("/audio_devices");
    if (res.ok) devices = await res.json();
  } catch (err) {
    return;  // leave every picker hidden
  }
  if (!Array.isArray(devices) || devices.length === 0) return;

  document.querySelectorAll(".audio-device-select").forEach((sel) => {
    sel.innerHTML = "";
    let defaultIdx = -1;
    devices.forEach((d, i) => {
      const opt = document.createElement("option");
      opt.value = String(d.index);
      opt.textContent = d.is_default ? `${d.name} (default)` : d.name;
      sel.appendChild(opt);
      if (d.is_default) defaultIdx = i;
    });
    if (defaultIdx >= 0) sel.selectedIndex = defaultIdx;
    const wrap = sel.closest(".audio-picker");
    if (wrap) wrap.hidden = false;
  });
}
populateAudioDevices();


// ---- New-class form -------------------------------------------------------

const newToggle = document.getElementById("new-class-toggle");
const newForm = document.getElementById("new-class-form");
if (newToggle && newForm) {
  newToggle.addEventListener("click", () => {
    newToggle.hidden = true;
    newForm.hidden = false;
    document.getElementById("new-class-input").focus();
  });
}

// ---- Kebab menus ----------------------------------------------------------

function closeAllMenus() {
  document.querySelectorAll(".kebab-menu").forEach((m) => { m.hidden = true; });
}
// A click anywhere else, or the Escape key, dismisses an open menu.
document.addEventListener("click", closeAllMenus);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeAllMenus();
});

// ---- Class cards: whole-card click navigates to /class/<name> -------------
//
// The kebab keeps working independently because every click whose target
// is inside `.kebab-wrap` early-returns from the card handler. The kebab
// button itself also calls `stopPropagation` so it doesn't bubble up to
// the card OR to the document-level "close menus" handler.

function openClassHome(card) {
  const name = card.dataset.class;
  if (name) window.location = classHomeUrl(name);
}

document.querySelectorAll(".class-card[data-class]").forEach((card) => {
  const kebab = card.querySelector(".kebab-btn");
  const menu = card.querySelector(".kebab-menu");

  // Click anywhere outside the kebab area = open the class overview.
  card.addEventListener("click", (e) => {
    if (e.target.closest(".kebab-wrap")) return;
    openClassHome(card);
  });
  // Keyboard equivalent for tabindex+role="button": Enter and Space.
  card.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    if (e.target.closest(".kebab-wrap")) return;
    e.preventDefault();
    openClassHome(card);
  });

  if (!kebab || !menu) return;

  kebab.addEventListener("click", (e) => {
    e.stopPropagation();
    const wasOpen = !menu.hidden;
    closeAllMenus();
    menu.hidden = wasOpen;
  });
  // Menu container swallows clicks so picking an item doesn't both close
  // the menu via the document handler AND navigate via the card handler.
  menu.addEventListener("click", (e) => e.stopPropagation());

  menu.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      menu.hidden = true;
      handleAction(btn.dataset.action, card.dataset.class, card);
    });
  });
});

function handleAction(action, name, card) {
  // Pass 8: the kebab is rename + delete only. Concepts / History / Exam
  // moved to the per-class overview page; their actions are no longer
  // surfaced here.
  if (action === "rename") {
    renameClass(name);
  } else if (action === "delete") {
    deleteClass(name, card);
  }
}

// ---- Rename / delete ------------------------------------------------------

async function renameClass(name) {
  const next = prompt(`Rename "${name}" to:`, name);
  if (next === null) return;
  const trimmed = next.trim();
  if (!trimmed || trimmed === name) return;
  const res = await postJSON("/rename_class", { old: name, new: trimmed });
  if (res.ok) {
    window.location.reload();
  } else {
    alert(res.error || "Rename failed.");
  }
}

async function deleteClass(name, card) {
  const count = card.dataset.sessions || "0";
  const typed = prompt(
    `This will delete ${count} session(s) and the concept memory for ` +
    `"${name}". The folder is moved to your Trash, so it can be recovered.\n\n` +
    `Type the class name to confirm:`
  );
  if (typed === null) return;
  if (typed.trim() !== name) {
    alert("The name did not match — nothing was deleted.");
    return;
  }
  const res = await postJSON("/delete_class",
                             { name: name, confirm: typed.trim() });
  if (res.ok) {
    window.location.reload();
  } else {
    alert(res.error || "Delete failed.");
  }
}
