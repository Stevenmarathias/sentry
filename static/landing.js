// Sentry landing page: class cards, kebab menus, new-class form,
// rename / delete actions.

function conceptsUrl(name) {
  return window.SENTRY_CONCEPTS_URL.replace("__C__", encodeURIComponent(name));
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

document.querySelectorAll(".class-card[data-class]").forEach((card) => {
  const name = card.dataset.class;
  const kebab = card.querySelector(".kebab-btn");
  const menu = card.querySelector(".kebab-menu");
  if (!kebab || !menu) return;

  kebab.addEventListener("click", (e) => {
    e.stopPropagation();
    const wasOpen = !menu.hidden;
    closeAllMenus();
    menu.hidden = wasOpen;
  });
  menu.addEventListener("click", (e) => e.stopPropagation());

  menu.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      menu.hidden = true;
      handleAction(btn.dataset.action, name, card);
    });
  });
});

function handleAction(action, name, card) {
  if (action === "concepts") {
    window.location = conceptsUrl(name);
  } else if (action === "history") {
    window.location = window.SENTRY_HISTORY_URL;
  } else if (action === "rename") {
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
