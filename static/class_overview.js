// Sentry per-class overview (Pass 8 + Pass 11):
//   - populates the audio-device picker on the Start Session form;
//   - wires the per-class color picker so a change live-updates the
//     page's --class-accent* custom properties AND persists to
//     meta.json via POST /class/<name>/color.

// ---- Audio input picker (Pass 8) ------------------------------------------

async function populateAudioDevices() {
  let devices = [];
  try {
    const res = await fetch("/audio_devices");
    if (res.ok) devices = await res.json();
  } catch (err) {
    return;
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


// ---- Per-class color picker (Pass 11) -------------------------------------

const colorInput = document.getElementById("class-color-input");
const colorHex = document.getElementById("class-color-hex");
const colorStatus = document.getElementById("class-color-status");
const className = document.body.dataset.className || "";

// Push a hex + its precomputed variants onto the live page so the themed
// elements (.is-themed, .start-card.themed) update without a reload.
function applyAccent(hex, variants) {
  const body = document.body;
  body.style.setProperty("--class-accent", hex);
  if (variants) {
    body.style.setProperty("--class-accent-soft", variants.soft);
    body.style.setProperty("--class-accent-softer", variants.softer);
    body.style.setProperty("--class-accent-border", variants.border);
    body.style.setProperty("--class-accent-glow", variants.glow);
  }
  if (colorHex) colorHex.textContent = hex;
}

function flashStatus(text, isError) {
  if (!colorStatus) return;
  colorStatus.textContent = text;
  colorStatus.classList.toggle("error", Boolean(isError));
  colorStatus.hidden = false;
  // Hide the status note after a beat so it doesn't linger.
  clearTimeout(flashStatus._t);
  flashStatus._t = setTimeout(() => { colorStatus.hidden = true; }, 1800);
}

if (colorInput && className) {
  // Remember the last server-confirmed value so a failed save can revert.
  let lastSaved = colorInput.value;

  colorInput.addEventListener("change", async () => {
    const next = colorInput.value;
    // Update the visible hex + page accent optimistically so the picker
    // feels instant; the server response then confirms or reverts.
    if (colorHex) colorHex.textContent = next;
    document.body.style.setProperty("--class-accent", next);

    try {
      const res = await fetch(
        `/class/${encodeURIComponent(className)}/color`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ color: next }),
        },
      );
      const data = await res.json();
      if (data.ok) {
        applyAccent(data.accent_color, data.variants);
        lastSaved = data.accent_color;
        flashStatus("Saved", false);
      } else {
        colorInput.value = lastSaved;
        applyAccent(lastSaved);
        flashStatus(data.error || "Could not save color.", true);
      }
    } catch (err) {
      colorInput.value = lastSaved;
      applyAccent(lastSaved);
      flashStatus("Could not reach server.", true);
    }
  });
}


// ---- YouTube import (Pass 14) ---------------------------------------------
//
// POST the URL → get a job_id → poll /import_status/<id> every 1.5s for
// stage updates. On "done", navigate to the resulting session's quiz view.
// On "error", show the friendly server-supplied message and re-enable the
// form. The background work happens server-side; this is just UI plumbing.

const importForm = document.getElementById("import-form");
const importUrlInput = document.getElementById("import-url");
const importBtn = document.querySelector(".import-btn");
const importStatus = document.getElementById("import-status");
const importStatusStage = document.getElementById("import-status-stage");

const IMPORT_POLL_MS = 1500;

function showImportStatus(text, isError) {
  if (!importStatus) return;
  importStatus.hidden = false;
  importStatus.classList.toggle("error", Boolean(isError));
  if (importStatusStage) importStatusStage.textContent = text;
}

function lockImportForm(locked) {
  if (importBtn) {
    importBtn.disabled = locked;
    importBtn.textContent = locked ? "Importing…" : "Import";
  }
  if (importUrlInput) importUrlInput.disabled = locked;
}

async function pollImportJob(jobId) {
  // Poll forever (server timeouts aside) — caption jobs finish in seconds,
  // audio jobs take minutes. The user can navigate away or refresh; the
  // background job continues regardless.
  while (true) {
    let data;
    try {
      const res = await fetch(`/import_status/${encodeURIComponent(jobId)}`);
      data = await res.json();
    } catch (err) {
      showImportStatus("Lost connection — retrying…", true);
      await new Promise((r) => setTimeout(r, IMPORT_POLL_MS));
      continue;
    }
    if (!data.ok) {
      showImportStatus(data.error || "Unknown job.", true);
      lockImportForm(false);
      return;
    }
    if (data.status === "error") {
      showImportStatus(data.error || "Import failed.", true);
      lockImportForm(false);
      return;
    }
    if (data.status === "done") {
      showImportStatus("Done — opening quiz…", false);
      if (data.result && data.result.quiz_url) {
        // Brief delay so the "Done" line is visible before navigation.
        setTimeout(() => { window.location = data.result.quiz_url; }, 300);
      } else {
        lockImportForm(false);
      }
      return;
    }
    // Running — surface the latest stage.
    showImportStatus(data.stage || "Working…", false);
    await new Promise((r) => setTimeout(r, IMPORT_POLL_MS));
  }
}

if (importForm) {
  importForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = (importUrlInput.value || "").trim();
    if (!url) return;
    lockImportForm(true);
    showImportStatus("Starting…", false);
    try {
      const res = await fetch(
        `/class/${encodeURIComponent(className)}/import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url }),
        },
      );
      const data = await res.json();
      if (!data.ok) {
        showImportStatus(data.error || "Could not start import.", true);
        lockImportForm(false);
        return;
      }
      pollImportJob(data.job_id);
    } catch (err) {
      showImportStatus("Could not reach server.", true);
      lockImportForm(false);
    }
  });
}


// ---- Pass D2.3: paste-transcript import -----------------------------------
//
// Hosted equivalent of the YouTube import — POSTs {title, transcript} to
// /class/<n>/import_text and reuses the same pollImportJob + redirect-to-
// quiz flow. The form has its own status surface (#paste-status) so a
// concurrent YouTube import (local mode only) keeps its own. Either form
// being in flight locks the other to avoid two simultaneous jobs.

const pasteForm = document.getElementById("paste-form");
const pasteTitle = document.getElementById("paste-title");
const pasteTranscript = document.getElementById("paste-transcript");
const pasteBtn = document.querySelector(".paste-btn");
const pasteStatus = document.getElementById("paste-status");
const pasteStatusStage = document.getElementById("paste-status-stage");

function showPasteStatus(text, isError) {
  if (!pasteStatus) return;
  pasteStatus.hidden = false;
  pasteStatus.classList.toggle("error", Boolean(isError));
  if (pasteStatusStage) pasteStatusStage.textContent = text;
}

function lockPasteForm(locked) {
  if (pasteBtn) {
    pasteBtn.disabled = locked;
    pasteBtn.textContent = locked ? "Working…" : "Create from transcript";
  }
  if (pasteTitle) pasteTitle.disabled = locked;
  if (pasteTranscript) pasteTranscript.disabled = locked;
  // Also lock the YouTube form if it's on the page — only one job at a time.
  if (importBtn) importBtn.disabled = locked;
  if (importUrlInput) importUrlInput.disabled = locked;
}

// Re-purpose pollImportJob's status surface for the paste flow by swapping
// the function it writes into. Simpler than parameterising pollImportJob.
async function pollPasteJob(jobId) {
  while (true) {
    let data;
    try {
      const res = await fetch(`/import_status/${encodeURIComponent(jobId)}`);
      data = await res.json();
    } catch (err) {
      showPasteStatus("Lost connection — retrying…", true);
      await new Promise((r) => setTimeout(r, IMPORT_POLL_MS));
      continue;
    }
    if (!data.ok) {
      showPasteStatus(data.error || "Unknown job.", true);
      lockPasteForm(false);
      return;
    }
    if (data.status === "error") {
      showPasteStatus(data.error || "Import failed.", true);
      lockPasteForm(false);
      return;
    }
    if (data.status === "done") {
      showPasteStatus("Done — opening quiz…", false);
      if (data.result && data.result.quiz_url) {
        setTimeout(() => { window.location = data.result.quiz_url; }, 300);
      } else {
        lockPasteForm(false);
      }
      return;
    }
    showPasteStatus(data.stage || "Working…", false);
    await new Promise((r) => setTimeout(r, IMPORT_POLL_MS));
  }
}

if (pasteForm) {
  pasteForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = (pasteTitle.value || "").trim();
    const transcript = (pasteTranscript.value || "").trim();
    if (!transcript) {
      showPasteStatus("Paste a transcript first.", true);
      return;
    }
    lockPasteForm(true);
    showPasteStatus("Starting…", false);
    try {
      const res = await fetch(
        `/class/${encodeURIComponent(className)}/import_text`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, transcript }),
        },
      );
      const data = await res.json();
      if (!data.ok) {
        showPasteStatus(data.error || "Could not start import.", true);
        lockPasteForm(false);
        return;
      }
      pollPasteJob(data.job_id);
    } catch (err) {
      showPasteStatus("Could not reach server.", true);
      lockPasteForm(false);
    }
  });
}
