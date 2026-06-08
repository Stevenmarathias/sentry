// Sentry per-class overview (Pass 8): populates the audio-device picker
// on the Start Session form. The list is fetched from /audio_devices once
// at page load; if it comes back empty (or the request fails) the picker
// stays hidden and /start falls through to the system default — same
// behaviour as on the landing page.

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
