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
