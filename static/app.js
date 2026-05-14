// Sentry web frontend: subscribes to /events (SSE) and drives the UI.

const statusEl = document.getElementById("status");
const meterText = document.getElementById("meter-text");
const meterBar = document.getElementById("meter-bar");
const meterThreshold = document.getElementById("meter-threshold");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const endBtn = document.getElementById("end-btn");
const transcriptEl = document.getElementById("transcript");
const quizSection = document.getElementById("quiz-section");
const quizContent = document.getElementById("quiz-content");

const panels = {
  board_content: document.getElementById("board-content"),
  explanation: document.getElementById("explanation"),
  watch_out_for: document.getElementById("watch-out-for"),
};

// Meter bar is scaled so the threshold marker sits at 50% of the track.
const METER_FULL_SCALE_MULTIPLIER = 2;

let transcriptStarted = false;
let sessionEnded = false;

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", Boolean(isError));
}

function setPanel(el, text) {
  if (text && text.trim()) {
    el.textContent = text;
    el.classList.remove("empty");
  } else {
    el.textContent = "—";
    el.classList.add("empty");
  }
}

function updateMeter(diff, threshold, armed) {
  meterText.textContent = `${diff.toFixed(2)} / ${threshold.toFixed(2)} ${armed ? "[armed]" : "[idle]"}`;
  const fullScale = threshold * METER_FULL_SCALE_MULTIPLIER || 1;
  const pct = Math.min(100, (diff / fullScale) * 100);
  meterBar.style.width = `${pct}%`;
  meterBar.classList.toggle("armed", Boolean(armed));
  meterThreshold.style.left = `${100 / METER_FULL_SCALE_MULTIPLIER}%`;
}

function addTranscript(time, text) {
  if (!transcriptStarted) {
    transcriptEl.innerHTML = "";
    transcriptStarted = true;
  }
  const line = document.createElement("p");
  line.className = "transcript-line";
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = time;
  line.appendChild(ts);
  line.appendChild(document.createTextNode(" " + text));
  transcriptEl.appendChild(line);
  // Newest at the bottom — keep it scrolled into view.
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function handleEvent(event) {
  switch (event.type) {
    case "meter":
      updateMeter(event.diff, event.threshold, event.armed);
      break;
    case "status":
      setStatus(event.text, false);
      break;
    case "error":
      setStatus(`Error: ${event.text}`, true);
      break;
    case "feedback":
      setPanel(panels.board_content, event.data.board_content);
      setPanel(panels.explanation, event.data.explanation);
      setPanel(panels.watch_out_for, event.data.watch_out_for);
      break;
    case "transcript":
      addTranscript(event.time, event.text);
      break;
  }
}

function connect() {
  const source = new EventSource("/events");
  source.onmessage = (e) => {
    try {
      handleEvent(JSON.parse(e.data));
    } catch (err) {
      console.error("Bad SSE payload", err);
    }
  };
  source.onerror = () => {
    if (sessionEnded) {
      source.close();
      return;
    }
    setStatus("Reconnecting…", true);
    // EventSource auto-reconnects; no manual retry needed.
  };
}

analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  try {
    const res = await fetch("/analyze", { method: "POST" });
    const data = await res.json();
    if (!data.accepted) setStatus(data.message, false);
  } catch (err) {
    setStatus("Could not reach server", true);
  } finally {
    // Brief debounce so rapid clicks don't spam the backend.
    setTimeout(() => { analyzeBtn.disabled = false; }, 1000);
  }
});

clearBtn.addEventListener("click", () => {
  setPanel(panels.board_content, "");
  setPanel(panels.explanation, "");
  setPanel(panels.watch_out_for, "");
  setStatus("Cleared.", false);
});

endBtn.addEventListener("click", async () => {
  if (!confirm("End the session? This stops the camera and microphone and generates a practice quiz.")) {
    return;
  }
  analyzeBtn.disabled = true;
  endBtn.disabled = true;
  endBtn.textContent = "Generating quiz…";
  setStatus("Generating quiz…", false);
  try {
    const res = await fetch("/end_session", { method: "POST" });
    const data = await res.json();
    if (data.ok) {
      sessionEnded = true;
      quizContent.textContent = data.quiz;
      quizSection.hidden = false;
      quizSection.scrollIntoView({ behavior: "smooth" });
      endBtn.textContent = "Session Ended";
      setStatus("Session ended.", false);
    } else {
      setStatus(`Error: ${data.error}`, true);
      endBtn.disabled = false;
      endBtn.textContent = "End Session";
      analyzeBtn.disabled = false;
    }
  } catch (err) {
    setStatus("Could not reach server", true);
    endBtn.disabled = false;
    endBtn.textContent = "End Session";
    analyzeBtn.disabled = false;
  }
});

connect();
