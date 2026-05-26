// Sentry web frontend: live-lecture SSE view + interactive quiz view.

// ---- Live view elements ----
const liveView = document.getElementById("live-view");
const quizView = document.getElementById("quiz-view");
const statusEl = document.getElementById("status");
const meterText = document.getElementById("meter-text");
const meterBar = document.getElementById("meter-bar");
const meterThreshold = document.getElementById("meter-threshold");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const pauseBtn = document.getElementById("pause-btn");
const endBtn = document.getElementById("end-btn");
const transcriptEl = document.getElementById("transcript");
const videoEl = document.getElementById("video");
const boardTs = document.getElementById("board-ts");
const badgeLegend = document.getElementById("badge-legend");
const modeToggle = document.getElementById("mode-toggle");
const meterLabel = document.getElementById("meter-label");

// ---- Quiz view elements ----
const quizCards = document.getElementById("quiz-cards");
const summaryBanner = document.getElementById("summary-banner");
const backBtn = document.getElementById("back-btn");
const pdfBtn = document.getElementById("pdf-btn");

const panels = {
  board_content: document.getElementById("board-content"),
  explanation: document.getElementById("explanation"),
  watch_out_for: document.getElementById("watch-out-for"),
};

// Meter bar is scaled so the threshold marker sits at 50% of the track.
const METER_FULL_SCALE_MULTIPLIER = 2;

let transcriptStarted = false;
let eventSource = null;
let isPaused = false;  // mirrors the server's pause state for the live session
let quizResults = [];  // per-question: "correct" | "partial" | "incorrect" | null

// =====================================================================
// Live lecture view
// =====================================================================

// Render a duration (seconds) as MM:SS, or H:MM:SS once it passes an hour.
// Returns null when there is nothing meaningful to show.
function formatElapsed(seconds) {
  if (seconds == null || isNaN(seconds)) return null;
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

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
  meterBar.classList.remove("paused", "new-slide");
  meterText.textContent = `${diff.toFixed(2)} / ${threshold.toFixed(2)} ${armed ? "[armed]" : "[idle]"}`;
  const fullScale = threshold * METER_FULL_SCALE_MULTIPLIER || 1;
  const pct = Math.min(100, (diff / fullScale) * 100);
  meterBar.style.width = `${pct}%`;
  meterBar.classList.toggle("armed", Boolean(armed));
  meterThreshold.style.left = `${100 / METER_FULL_SCALE_MULTIPLIER}%`;
}

// While paused the camera worker emits a "paused" meter — show a flat gray bar.
function showPausedMeter() {
  meterText.textContent = "paused";
  meterBar.style.width = "100%";
  meterBar.classList.remove("armed", "new-slide");
  meterBar.classList.add("paused");
}

// Slide mode meter: hash distance from the last captured slide, with a
// "new slide" highlight once we cross the trigger threshold.
function updateSlideMeter(event) {
  meterBar.classList.remove("paused", "armed");
  meterBar.classList.toggle("new-slide", Boolean(event.new_slide));
  const fullScale = (event.threshold * METER_FULL_SCALE_MULTIPLIER) || 1;
  const pct = Math.min(100, (event.distance / fullScale) * 100);
  meterBar.style.width = `${pct}%`;
  meterText.textContent =
    `${event.distance} / ${event.threshold} ${event.new_slide ? "[new slide]" : "[same slide]"}`;
  meterThreshold.style.left = `${100 / METER_FULL_SCALE_MULTIPLIER}%`;
}

function addTranscript(label, text) {
  if (!transcriptStarted) {
    transcriptEl.innerHTML = "";
    transcriptStarted = true;
  }
  const line = document.createElement("p");
  line.className = "transcript-line";
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = label;
  line.appendChild(ts);
  line.appendChild(document.createTextNode(" " + text));
  transcriptEl.appendChild(line);
  // Newest at the bottom — keep it scrolled into view.
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// An italic gray annotation marking a pause gap in the transcript.
function addPauseMarker(text) {
  if (!transcriptStarted) {
    transcriptEl.innerHTML = "";
    transcriptStarted = true;
  }
  const line = document.createElement("p");
  line.className = "transcript-pause";
  line.textContent = text;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function handleEvent(event) {
  switch (event.type) {
    case "meter":
      if (event.paused) {
        showPausedMeter();
      } else if (event.mode === "slide") {
        updateSlideMeter(event);
      } else {
        updateMeter(event.diff, event.threshold, event.armed);
      }
      break;
    case "status":
      setStatus(event.text, false);
      break;
    case "pause_marker":
      addPauseMarker(event.text);
      break;
    case "error":
      setStatus(`Error: ${event.text}`, true);
      break;
    case "feedback":
      setPanel(panels.board_content, event.data.board_content);
      setPanel(panels.explanation, event.data.explanation);
      setPanel(panels.watch_out_for, event.data.watch_out_for);
      if (boardTs) {
        const label = formatElapsed(event.elapsed);
        boardTs.textContent = label ? ` · ${label} elapsed` : "";
      }
      break;
    case "transcript":
      // Timestamps are shown as elapsed time from session start; fall back to
      // the wall clock if the server did not supply an elapsed value.
      addTranscript(formatElapsed(event.elapsed) || event.time, event.text);
      break;
  }
}

function connect() {
  eventSource = new EventSource("/events");
  eventSource.onmessage = (e) => {
    try {
      handleEvent(JSON.parse(e.data));
    } catch (err) {
      console.error("Bad SSE payload", err);
    }
  };
  eventSource.onerror = () => {
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
    // Brief debounce so rapid clicks don't spam the backend. Stay disabled
    // if the session was paused while the request was in flight.
    setTimeout(() => { analyzeBtn.disabled = isPaused; }, 1000);
  }
});

clearBtn.addEventListener("click", () => {
  setPanel(panels.board_content, "");
  setPanel(panels.explanation, "");
  setPanel(panels.watch_out_for, "");
  setStatus("Cleared.", false);
});

// Pause/resume: the button label and analyze availability follow server state.
function setPauseUI(paused) {
  isPaused = paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  analyzeBtn.disabled = paused;
}

pauseBtn.addEventListener("click", async () => {
  pauseBtn.disabled = true;
  try {
    const res = await fetch("/toggle_pause", { method: "POST" });
    const data = await res.json();
    if (data.ok) {
      setPauseUI(data.paused);
      setStatus(data.paused ? "Paused." : "Resumed.", false);
    } else {
      setStatus(data.error || "Could not toggle pause.", true);
    }
  } catch (err) {
    setStatus("Could not reach server", true);
  } finally {
    pauseBtn.disabled = false;
  }
});

// Mid-session mode toggle. The chip's text + class flip in place; the meter
// label follows, and the next meter SSE event swaps the visualisation since
// the server tags meters with `mode`.
function applyMode(mode) {
  if (!modeToggle) return;
  modeToggle.textContent = mode === "slide" ? "📊 Slide mode" : "✏️ Board mode";
  modeToggle.classList.toggle("mode-slide", mode === "slide");
  modeToggle.classList.toggle("mode-board", mode !== "slide");
  if (meterLabel) {
    meterLabel.textContent = mode === "slide" ? "slide diff" : "motion";
  }
}

if (modeToggle) {
  modeToggle.addEventListener("click", async () => {
    modeToggle.disabled = true;
    try {
      const res = await fetch("/toggle_mode", { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        applyMode(data.mode);
      } else {
        setStatus(data.error || "Could not toggle mode.", true);
      }
    } catch (err) {
      setStatus("Could not reach server", true);
    } finally {
      modeToggle.disabled = false;
    }
  });
}

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
      enterQuizMode(data.quiz);
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

// =====================================================================
// Quiz view
// =====================================================================

backBtn.addEventListener("click", () => {
  // "/" for a live session; /history when a past session's quiz was re-opened.
  window.location = document.body.dataset.backUrl || "/";
});

if (pdfBtn) {
  // The route responds with Content-Disposition: attachment, so navigating to
  // it downloads the PDF without leaving the quiz view.
  pdfBtn.addEventListener("click", () => {
    if (pdfBtn.dataset.url) window.location = pdfBtn.dataset.url;
  });
}

function enterQuizMode(quiz) {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  videoEl.src = "";  // stop the MJPEG stream
  liveView.hidden = true;
  quizView.hidden = false;
  renderQuiz(quiz);
  window.scrollTo(0, 0);
}

function renderQuiz(quiz) {
  const questions = (quiz && quiz.questions) || [];
  quizResults = questions.map(() => null);
  quizCards.innerHTML = "";
  // The badge legend explains the orange "FROM PRIOR LECTURE" badge — only
  // worth showing when this quiz actually carries a recurring question.
  if (badgeLegend) {
    badgeLegend.hidden = !questions.some((q) => q.recurring);
  }
  if (questions.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No quiz questions were generated for this session.";
    quizCards.appendChild(empty);
  }
  questions.forEach((q, i) => quizCards.appendChild(buildCard(q, i)));
  updateSummary();
}

function recordResult(index, outcome) {
  quizResults[index] = outcome;
  updateSummary();
}

function updateSummary() {
  const total = quizResults.length;
  const answered = quizResults.filter((r) => r !== null).length;
  if (total === 0 || answered < total) {
    summaryBanner.hidden = true;
    return;
  }
  // MCQ/fill-blank count as correct/incorrect; a short-answer "partial" is
  // reported separately (worth half a point).
  const correct = quizResults.filter((r) => r === "correct").length;
  const partial = quizResults.filter((r) => r === "partial").length;
  let text = `You got ${correct}/${total} correct`;
  if (partial > 0) text += `, ${partial} partial`;
  text += ".";
  const wasHidden = summaryBanner.hidden;
  summaryBanner.textContent = text;
  summaryBanner.hidden = false;
  if (wasHidden) {
    summaryBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ---- card scaffolding ----

function makeCard(q, index, typeLabel) {
  const card = document.createElement("article");
  card.className = "panel quiz-card";

  const num = document.createElement("div");
  num.className = "q-number";
  num.textContent = `Question ${index + 1} · ${typeLabel}`;
  card.appendChild(num);

  // Questions on concepts carried over from earlier lectures get flagged.
  if (q.recurring) {
    const badge = document.createElement("div");
    badge.className = "recurring-badge";
    badge.textContent = "FROM PRIOR LECTURE — likely on exam.";
    card.appendChild(badge);
  }

  const text = document.createElement("p");
  text.className = "q-text";
  text.textContent = q.question || "";
  card.appendChild(text);
  return card;
}

function addExplanationSlot(card) {
  const exp = document.createElement("div");
  exp.className = "q-explanation";
  card.appendChild(exp);
  return exp;
}

function addSource(card, q) {
  const src = document.createElement("div");
  src.className = "q-source";
  // source_display carries the elapsed-time label; source_timestamp (wall
  // clock) is the fallback if a quiz predates the display-layer change.
  // Exam questions self-label ("From: 2026-05-13 session — …") so we skip the
  // "Source: " prefix there to avoid "Source: From: …".
  const label = q.source_display || q.source_timestamp || "—";
  src.textContent = label.startsWith("From:") ? label : `Source: ${label}`;
  card.appendChild(src);
}

function revealExplanation(exp, text, tone) {
  if (tone) exp.classList.add(`tone-${tone}`);
  if (text) exp.textContent = text;
  exp.classList.add("show");
}

function normalize(s) {
  return (s || "").trim().toLowerCase();
}

function buildCard(q, index) {
  switch (q.type) {
    case "mcq": return buildMcq(q, index);
    case "fill_blank": return buildFillBlank(q, index);
    case "short_answer": return buildShortAnswer(q, index);
    default: return buildShortAnswer(q, index);
  }
}

// ---- MCQ ----

// Fisher-Yates shuffle of an MCQ's choices that also tracks where the correct
// answer landed. Render-time only — the quiz JSON (server cache, PDF) is left
// untouched, so re-studying a quiz tests the concept, not memorised positions.
function shuffleChoices(choices, correctIndex) {
  const items = choices.map((text, i) => ({ text, correct: i === correctIndex }));
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return {
    choices: items.map((it) => it.text),
    correctIndex: items.findIndex((it) => it.correct),
  };
}

function buildMcq(q, index) {
  const card = makeCard(q, index, "Multiple choice");

  // Shuffle before building buttons so the click handlers below bind against
  // the post-shuffle correctIndex and grade correctly.
  const shuffled = shuffleChoices(q.choices || [], q.correct_index);

  const choices = document.createElement("div");
  choices.className = "choices";
  const buttons = shuffled.choices.map((choice) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "choice-btn";
    btn.textContent = choice;
    choices.appendChild(btn);
    return btn;
  });
  card.appendChild(choices);

  const exp = addExplanationSlot(card);
  addSource(card, q);

  buttons.forEach((btn, i) => {
    btn.addEventListener("click", () => {
      if (card.dataset.answered) return;
      card.dataset.answered = "1";
      buttons.forEach((b, j) => {
        b.disabled = true;
        if (j === shuffled.correctIndex) b.classList.add("correct");
      });
      const isCorrect = i === shuffled.correctIndex;
      if (!isCorrect) btn.classList.add("wrong");
      revealExplanation(exp, q.explanation);
      recordResult(index, isCorrect ? "correct" : "incorrect");
    });
  });
  return card;
}

// ---- Fill in the blank ----

function buildFillBlank(q, index) {
  const card = makeCard(q, index, "Fill in the blank");

  const row = document.createElement("div");
  row.className = "answer-row";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "answer-input";
  input.placeholder = "Your answer";
  const checkBtn = document.createElement("button");
  checkBtn.type = "button";
  checkBtn.className = "check-btn";
  checkBtn.textContent = "Check";
  row.appendChild(input);
  row.appendChild(checkBtn);
  card.appendChild(row);

  const exp = addExplanationSlot(card);
  addSource(card, q);

  function submit() {
    if (card.dataset.answered) return;
    const accepted = [q.correct_answer]
      .concat(q.acceptable_variants || [])
      .map(normalize);
    const isCorrect = accepted.includes(normalize(input.value));
    card.dataset.answered = "1";
    input.disabled = true;
    checkBtn.disabled = true;
    input.classList.add(isCorrect ? "correct" : "wrong");
    let text = q.explanation || "";
    if (!isCorrect) {
      text = `Correct answer: ${q.correct_answer}` + (text ? `\n\n${text}` : "");
    }
    revealExplanation(exp, text, isCorrect ? "correct" : "incorrect");
    recordResult(index, isCorrect ? "correct" : "incorrect");
  }

  checkBtn.addEventListener("click", submit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submit();
  });
  return card;
}

// ---- Short answer ----

function buildShortAnswer(q, index) {
  const card = makeCard(q, index, "Short answer");

  const textarea = document.createElement("textarea");
  textarea.className = "answer-textarea";
  textarea.rows = 4;
  textarea.placeholder = "Your answer";
  card.appendChild(textarea);

  const gradeBtn = document.createElement("button");
  gradeBtn.type = "button";
  gradeBtn.className = "grade-btn";
  gradeBtn.textContent = "Grade my answer";
  card.appendChild(gradeBtn);

  const exp = addExplanationSlot(card);
  addSource(card, q);

  gradeBtn.addEventListener("click", async () => {
    if (card.dataset.answered) return;
    const userAnswer = textarea.value.trim();
    if (!userAnswer) {
      textarea.focus();
      return;
    }
    gradeBtn.disabled = true;
    gradeBtn.textContent = "Grading…";
    try {
      const res = await fetch("/grade_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q.question,
          reference_answer: q.reference_answer || "",
          user_answer: userAnswer,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      card.dataset.answered = "1";
      textarea.disabled = true;
      gradeBtn.textContent = "Graded";
      const verdict = data.verdict || "incorrect";

      const badge = document.createElement("span");
      badge.className = `verdict verdict-${verdict}`;
      badge.textContent = verdict;
      gradeBtn.after(badge);

      let text = data.feedback || "";
      if (q.reference_answer) {
        text += (text ? "\n\n" : "") + `Reference answer: ${q.reference_answer}`;
      }
      revealExplanation(exp, text, verdict);
      recordResult(index, verdict);
    } catch (err) {
      gradeBtn.disabled = false;
      gradeBtn.textContent = "Grade my answer";
      revealExplanation(exp, `Could not grade your answer: ${err.message}`);
    }
  });
  return card;
}

// =====================================================================
// Boot
// =====================================================================

if (window.SENTRY_QUIZ) {
  // Session already ended (e.g. a page refresh) — go straight to the quiz.
  enterQuizMode(window.SENTRY_QUIZ);
} else {
  // The pause button's label is server-rendered; sync JS state to it so a
  // refresh mid-pause keeps the UI consistent.
  setPauseUI(pauseBtn.textContent.trim() === "Resume");
  connect();
}
