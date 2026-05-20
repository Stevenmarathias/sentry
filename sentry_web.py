"""Sentry (web) — real-time lecture feedback in the browser.

A Flask rebuild of Sentry v1.0. The backend owns the camera and microphone,
runs board-change detection and continuous audio transcription, and calls
Claude (Opus 4.7) for the 3-panel structured feedback and end-of-session
quizzes. The browser is a pure view: it pulls the live preview as an MJPEG
stream and receives meter / status / feedback / transcript updates over
Server-Sent Events.

Phase 1: class-agnostic live feedback.
Phase 2: class selection at startup, continuous Whisper transcription, a
per-class session markdown file, and an interactive end-of-session quiz.
Phase 3: per-class cross-lecture concept memory with importance scoring;
end-of-session quizzes mix today's content with recurring concepts.

The camera / detection / analysis / transcription logic is reused from
sentry.py (v1.0) — only the Tkinter UI is replaced. Run this instead of
sentry.py; sentry.py is left untouched.

    ANTHROPIC_API_KEY=... .venv/bin/python sentry_web.py
    # then open http://127.0.0.1:5000
"""
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

# Reuse v1.0's camera / detection / analysis / transcription logic. sentry.py
# only launches the Tk UI under `if __name__ == "__main__"`, so importing it
# here is side-effect free (no window is created).
from sentry import (
    AUDIO_SAMPLE_RATE,
    CAMERA_INDEX,
    COOLDOWN_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_ID,
    Analyzer,
    ChangeDetector,
    Transcriber,
)


# ---- Configuration -----------------------------------------------------------

PREVIEW_WIDTH = 640          # downscale width for the MJPEG preview stream
PREVIEW_FPS = 15
METER_INTERVAL = 0.25        # seconds between motion-meter broadcasts
SSE_KEEPALIVE = 15.0         # seconds idle before an SSE keepalive comment
AUDIO_CHUNK_SECONDS = 20.0   # length of each audio chunk handed to Whisper

RECENT_SESSIONS = 3          # window for the recency term of importance_score
MAX_RECURRING_CONCEPTS = 5   # high-importance past concepts offered to the quiz

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

QUIZ_SYSTEM_PROMPT = (
    "You are Sentry, a study assistant. You are given TODAY'S lecture session "
    "log — board analyses and audio transcripts, each tagged with an HH:MM:SS "
    "timestamp — and sometimes a short list of recurring high-importance "
    "concepts from earlier lectures in this class. Generate a practice quiz, "
    "following these rules strictly.\n\n"
    "COVERAGE: For today's lecture, you MUST include at least one question "
    "about every named concept, term, person, event, formula, framework, "
    "claim, or technique that explicitly appears in today's transcript. Do not "
    "omit any. If five named items are mentioned, all five must appear.\n\n"
    "GROUNDING: Base every question about today's lecture on content that "
    "actually appears in today's transcript. Do not extrapolate to related "
    "topics from background knowledge unless the transcript explicitly "
    "introduces them. If the transcript is thin on a topic, write a thinner "
    "question — do not pad with outside material.\n\n"
    "TRANSPARENCY: Every question must include a source_timestamp. For a "
    "question about today's lecture, use the HH:MM:SS timestamp from today's "
    "transcript where the concept was discussed; if it recurs, cite the first "
    "occurrence.\n\n"
    "SIZING: The number of today's-lecture questions should reflect transcript "
    "coverage — a 30-minute lecture with 6 named concepts should yield roughly "
    "6-10 questions. Do not pad to hit a target number.\n\n"
    "RECURRING CONCEPTS: If a list of recurring high-importance concepts from "
    "earlier lectures is provided below the transcript, keep the quiz mostly "
    "today's transcript (about 80% of the questions), then add 1-2 questions "
    "that test those recurring concepts. For each recurring-concept question, "
    "set \"recurring\": true and set its source_timestamp to the exact "
    "\"PAST: ...\" string supplied for that concept. If no recurring concepts "
    "are provided, generate the quiz entirely from today's transcript and do "
    "not mark any question as recurring.\n\n"
    "FORMAT: Return a JSON object with a \"questions\" array. Use a mix of the "
    "three question types:\n"
    "- \"mcq\": provide \"question\", \"choices\" (exactly four answer strings, "
    "with no \"A.\"/\"B.\" prefixes), \"correct_index\" (0-3), \"explanation\", "
    "and \"source_timestamp\".\n"
    "- \"fill_blank\": provide \"question\" (mark the blank with \"___\"), "
    "\"correct_answer\", \"acceptable_variants\" (other accepted spellings or "
    "phrasings — may be an empty list), \"explanation\", and "
    "\"source_timestamp\".\n"
    "- \"short_answer\": provide \"question\", \"reference_answer\", and "
    "\"source_timestamp\".\n"
    "Set \"recurring\": true only on questions that test a recurring concept "
    "from an earlier lecture; today's-lecture questions omit it. Mix "
    "multiple-choice, fill-in-the-blank, and short-answer types across the "
    "quiz."
)

# Structured-output schema for the quiz. Type-specific fields are optional so
# one item shape can carry all three question types; the prompt enforces which
# fields each type must supply. `recurring` flags a past-lecture question.
QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["mcq", "fill_blank", "short_answer"],
                    },
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                    "correct_index": {"type": "integer"},
                    "correct_answer": {"type": "string"},
                    "acceptable_variants": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reference_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "source_timestamp": {"type": "string"},
                    "recurring": {"type": "boolean"},
                },
                "required": ["type", "question", "source_timestamp"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

GRADE_SYSTEM_PROMPT = (
    "You are grading a student's short-answer response to a lecture-review "
    "question. Compare their answer to the reference answer and decide a "
    "verdict: 'correct' if it captures the key point, 'partial' if it gets the "
    "main idea but misses a detail, 'incorrect' if it misses or contradicts "
    "the key point. Be generous with 'partial' — a student who got the main "
    "idea but missed a detail should get partial credit. Then write feedback "
    "of 2-3 sentences, addressed to the student, explaining what they got "
    "right and what they got wrong."
)

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["correct", "partial", "incorrect"],
        },
        "feedback": {"type": "string"},
    },
    "required": ["verdict", "feedback"],
    "additionalProperties": False,
}

CONCEPT_EXTRACTION_SYSTEM_PROMPT = (
    "You are Sentry's concept extractor. You are given the log of a single "
    "lecture session — board analyses and audio transcripts, each tagged with "
    "an HH:MM:SS timestamp. Identify every named concept that was actually "
    "discussed in this lecture and return them as structured JSON.\n\n"
    "A named concept is a specific term, person, event, formula, framework, "
    "claim, or technique that the lecture introduces or discusses by name. "
    "Include only concepts grounded in the transcript — do not add related "
    "ideas from background knowledge. Skip filler, chit-chat, and anything "
    "that is not lecture material.\n\n"
    "For each concept provide:\n"
    "- name: a short canonical name (e.g. 'heteroskedasticity'), in the form a "
    "student would look up; lowercase unless it is a proper noun.\n"
    "- category: one of term, person, event, formula, framework, claim, "
    "technique, other.\n"
    "- first_mention_timestamp: the HH:MM:SS timestamp where the concept is "
    "first mentioned in the transcript.\n"
    "- brief_definition: one sentence describing how the concept was used in "
    "THIS lecture.\n\n"
    "If the transcript contains no real lecture concepts (for example it is "
    "empty or junk), return an empty concepts array."
)

CONCEPT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "term", "person", "event", "formula",
                            "framework", "claim", "technique", "other",
                        ],
                    },
                    "first_mention_timestamp": {"type": "string"},
                    "brief_definition": {"type": "string"},
                },
                "required": [
                    "name", "category",
                    "first_mention_timestamp", "brief_definition",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


# ---- Helpers -----------------------------------------------------------------

def sanitize_class_name(raw: str) -> str:
    """Reduce a user-entered class name to a safe single-path-segment string."""
    cleaned = re.sub(r"[^\w \-]", "", raw or "").strip()
    return re.sub(r"\s+", " ", cleaned)


def list_classes() -> list[str]:
    """Existing class names = subdirectories of the sessions directory."""
    if not SESSIONS_DIR.exists():
        return []
    return sorted(d.name for d in SESSIONS_DIR.iterdir() if d.is_dir())


# ---- Session -----------------------------------------------------------------

class Session:
    """A single lecture session, tagged to a class, backed by one markdown file.

    Both workers append to this file as events happen; every append is locked
    and the file is closed each time, so a crash leaves a valid partial file.
    The generated quiz is cached on the instance so a refresh or repeat request
    does not regenerate it.
    """

    def __init__(self, class_name: str):
        self.class_name = class_name
        self.started_at = datetime.now()
        self.ended = False
        self.quiz: Optional[dict] = None

        class_dir = SESSIONS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y-%m-%d_%H%M")
        self.file_path = class_dir / f"{stamp}.md"

        self._lock = threading.Lock()
        self._append(
            f"# Sentry Session — {class_name}\n\n"
            f"**Started:** {self.started_at:%Y-%m-%d %H:%M}\n\n"
            f"---\n\n"
        )

    def _append(self, text: str) -> None:
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(text)

    def append_board_analysis(self, result: dict) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._append(
            f"### 🟦 Board analysis — {ts}\n\n"
            f"**On the board:** {result.get('board_content', '')}\n\n"
            f"**Simple explanation:** {result.get('explanation', '')}\n\n"
            f"**Watch out for:** {result.get('watch_out_for', '')}\n\n"
        )

    def append_transcript(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._append(f"**🎙️ {ts}** — {text}\n\n")

    def append_quiz(self, quiz_markdown: str) -> None:
        self._append(f"\n---\n\n## Practice Quiz\n\n{quiz_markdown}\n")

    def read(self) -> str:
        with self._lock:
            return self.file_path.read_text(encoding="utf-8")


class AppState:
    """Process-global handle to the active session, shared by both workers."""
    session: Optional[Session] = None


state = AppState()


# ---- Event bus ---------------------------------------------------------------

class EventBus:
    """Fan-out pub/sub. Each browser tab's /events stream gets its own queue;
    both workers broadcast onto the same bus."""

    def __init__(self):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Slow client — drop the event rather than blocking a worker.
                pass


bus = EventBus()


# ---- Camera worker -----------------------------------------------------------

class CameraWorker:
    """Owns the camera and board analysis. A single instance is reused across
    sessions: start()/stop() are cycle-safe so a new session starts clean.

    The capture thread reads frames, runs change detection, and auto-triggers
    analysis. Analysis runs on its own short-lived thread so the capture loop
    never stalls; results are broadcast on the bus and appended to the session
    file.
    """

    def __init__(self):
        self.analyzer = Analyzer()
        self.detector = ChangeDetector()

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._last_analysis_time = 0.0
        self._analyzing = False
        self._last_meter_emit = 0.0

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._stop_evt.clear()
        self.detector = ChangeDetector()
        self._latest_frame = None
        self._analyzing = False
        self._last_analysis_time = 0.0

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self._started = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._started = False

    # ---- capture loop -------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_evt.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            settled = self.detector.update(frame)
            now = time.time()

            if now - self._last_meter_emit > METER_INTERVAL:
                self._last_meter_emit = now
                bus.broadcast({
                    "type": "meter",
                    "diff": round(self.detector.last_diff, 2),
                    "threshold": round(self.detector.threshold, 2),
                    "armed": bool(self.detector._was_changing),
                })

            if (settled
                    and not self._analyzing
                    and now - self._last_analysis_time > COOLDOWN_SECONDS):
                self._launch_analysis(frame.copy())

            time.sleep(0.03)

    # ---- preview ------------------------------------------------------------

    def get_preview_jpeg(self) -> Optional[bytes]:
        """Latest frame, downscaled and JPEG-encoded for the MJPEG stream."""
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = PREVIEW_WIDTH / float(w)
        small = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # ---- analysis -----------------------------------------------------------

    def trigger_analysis(self) -> tuple[bool, str]:
        """Manual "Analyze Now". Returns (accepted, message)."""
        if not self._started:
            return False, "Session not running."
        if self._analyzing:
            return False, "Already analyzing…"
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return False, "No frame yet…"
        self._launch_analysis(frame)
        return True, "Analyzing…"

    def _launch_analysis(self, frame: np.ndarray) -> None:
        if self._analyzing:
            return
        self._analyzing = True
        self._last_analysis_time = time.time()
        bus.broadcast({"type": "status", "text": "Analyzing…"})
        threading.Thread(
            target=self._run_analysis, args=(frame,), daemon=True
        ).start()

    def _run_analysis(self, frame: np.ndarray) -> None:
        try:
            result = self.analyzer.analyze(frame)
            bus.broadcast({"type": "feedback", "data": result})
            bus.broadcast({"type": "status", "text": "Watching the board…"})
            if state.session is not None:
                state.session.append_board_analysis(result)
        except Exception as exc:
            bus.broadcast({"type": "error", "text": str(exc)})
            bus.broadcast({"type": "status", "text": f"Error: {exc}"})
        finally:
            self._analyzing = False


# ---- Audio worker ------------------------------------------------------------

class AudioWorker:
    """Owns the microphone and continuous transcription. A single instance is
    reused across sessions; start()/stop() are cycle-safe.

    A sounddevice callback accumulates samples; the chunk loop drains every
    AUDIO_CHUNK_SECONDS, transcribes the chunk with Whisper (base model, lazy
    loaded on first use), broadcasts a `transcript` event, and appends to the
    session file. Microphone or transcription failures broadcast an error and
    leave the rest of the app running.
    """

    def __init__(self):
        self.transcriber = Transcriber()  # lazy: model loads on first chunk

        self._stream: Optional[sd.InputStream] = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._model_loaded = False

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._stop_evt.clear()
        with self._lock:
            self._chunks = []
        try:
            self._stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            bus.broadcast({"type": "error", "text": f"Audio: {exc}"})
            return
        self._started = True
        self._thread = threading.Thread(target=self._chunk_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Join generously: the loop does one final drain + transcription so the
        # last spoken chunk lands in the session file before the quiz is built.
        if self._thread is not None:
            self._thread.join(timeout=25)
            self._thread = None
        self._started = False

    # ---- capture + transcription -------------------------------------------

    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._chunks.append(indata.copy())

    def _drain(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._chunks:
                return None
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []
        return audio

    def _chunk_loop(self) -> None:
        while True:
            stopped = self._stop_evt.wait(AUDIO_CHUNK_SECONDS)
            audio = self._drain()
            if audio is not None and audio.size > 0:
                self._process(audio)
            if stopped:
                break

    def _process(self, audio: np.ndarray) -> None:
        try:
            if not self._model_loaded:
                bus.broadcast(
                    {"type": "status", "text": "Loading transcription model…"}
                )
            text = self.transcriber.transcribe(audio, AUDIO_SAMPLE_RATE)
            if not self._model_loaded:
                self._model_loaded = True
                bus.broadcast({"type": "status", "text": "Watching the board…"})
            text = text.strip()
            if not text:
                return
            ts = datetime.now().strftime("%H:%M:%S")
            bus.broadcast({"type": "transcript", "time": ts, "text": text})
            if state.session is not None:
                state.session.append_transcript(text)
        except Exception as exc:
            bus.broadcast({"type": "error", "text": f"Transcription: {exc}"})


# ---- Cross-lecture concept memory --------------------------------------------
#
# Each class accumulates a concept store at sessions/<class>/concepts.json.
# At End Session a concept-extraction call lists today's named concepts; they
# are merged into the store, importance scores are recomputed, and the highest-
# importance concepts that did NOT come up today are offered to the quiz.

def normalize_concept(name: str) -> str:
    """Light normalization for matching concept names across sessions."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def extract_concepts(session_markdown: str) -> list[dict]:
    """Extract named concepts from a session log via a separate Claude call.

    Independent of quiz generation. Returns [] on any failure or empty result
    so a junk transcript never blocks the quiz.
    """
    try:
        client = camera_worker.analyzer.client
        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=6000,
            thinking={"type": "adaptive"},
            system=CONCEPT_EXTRACTION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Lecture session log:\n\n{session_markdown}",
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": CONCEPT_EXTRACTION_SCHEMA,
                }
            },
        )
        text = next(b.text for b in message.content if b.type == "text")
        concepts = json.loads(text).get("concepts", [])
        return concepts if isinstance(concepts, list) else []
    except Exception as exc:
        print(f"Warning: concept extraction failed ({exc}); "
              f"continuing with today's transcript only.")
        return []


def load_concepts(class_dir: Path) -> Optional[dict]:
    """Load a class's concept store. Returns None if it is missing or corrupt
    (the caller then treats this as a first session and overwrites the file)."""
    path = class_dir / "concepts.json"
    if not path.exists():
        return None
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(store, dict) or not isinstance(
                store.get("concepts"), list):
            raise ValueError("unexpected structure")
        return store
    except Exception as exc:
        print(f"Warning: {path} is empty or corrupt ({exc}); "
              f"treating as a first session and overwriting it.")
        return None


def save_concepts(class_dir: Path, store: dict) -> None:
    """Write the per-class concept store back to concepts.json."""
    path = class_dir / "concepts.json"
    path.write_text(
        json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def recompute_importance(concepts: list[dict]) -> None:
    """Recompute importance_score for every concept, in place.

    importance_score = occurrences + 0.3 * (occurrences in the last 3 sessions).
    Session files are named <YYYY-MM-DD_HHMM>.md, so sorting them lexically
    sorts them chronologically.
    """
    all_sessions = sorted({
        occ.get("session_file", "")
        for c in concepts
        for occ in c.get("occurrences", [])
    })
    recent = set(all_sessions[-RECENT_SESSIONS:])
    for c in concepts:
        occurrences = c.get("occurrences", [])
        recent_count = sum(
            1 for o in occurrences if o.get("session_file", "") in recent
        )
        c["importance_score"] = round(
            len(occurrences) + 0.3 * recent_count, 2
        )


def merge_concepts(store: Optional[dict], extracted: list[dict],
                   session_file: str, class_name: str) -> dict:
    """Merge a session's extracted concepts into the per-class store.

    A concept already present (matched on a normalized name) gains a new
    occurrence; a new concept is appended. Importance scores are then
    recomputed across the whole store so they stay current.
    """
    if store is None or not isinstance(store.get("concepts"), list):
        store = {"class_name": class_name, "concepts": []}

    by_norm = {
        normalize_concept(c.get("name", "")): c for c in store["concepts"]
    }
    for ex in extracted:
        norm = normalize_concept(ex.get("name", ""))
        if not norm:
            continue
        occurrence = {
            "session_file": session_file,
            "timestamp": ex.get("first_mention_timestamp", ""),
            "definition": ex.get("brief_definition", ""),
        }
        existing = by_norm.get(norm)
        if existing is not None:
            existing["occurrences"].append(occurrence)
            existing["category"] = ex.get(
                "category", existing.get("category", "other")
            )
        else:
            concept = {
                "name": ex.get("name", ""),
                "category": ex.get("category", "other"),
                "occurrences": [occurrence],
                "importance_score": 0.0,
            }
            store["concepts"].append(concept)
            by_norm[norm] = concept

    recompute_importance(store["concepts"])
    store["class_name"] = class_name
    store["last_updated"] = datetime.now().isoformat(timespec="seconds")
    return store


def pick_recurring_concepts(store: Optional[dict],
                            extracted: list[dict]) -> list[dict]:
    """Highest-importance stored concepts that did NOT appear in today's
    session — the recurring concepts the quiz should also test."""
    if not store or not store.get("concepts"):
        return []
    today = {normalize_concept(c.get("name", "")) for c in extracted}
    candidates = [
        c for c in store["concepts"]
        if normalize_concept(c.get("name", "")) not in today
    ]
    candidates.sort(
        key=lambda c: c.get("importance_score", 0.0), reverse=True
    )
    return candidates[:MAX_RECURRING_CONCEPTS]


def _earliest_occurrence(concept: dict) -> dict:
    """The concept's first recorded occurrence (sessions sort chronologically)."""
    occurrences = concept.get("occurrences", [])
    if not occurrences:
        return {}
    return min(occurrences, key=lambda o: o.get("session_file", ""))


def format_recurring_block(recurring: list[dict]) -> str:
    """Render recurring concepts for the quiz prompt, each with a PAST tag."""
    lines = []
    for c in recurring:
        occ = _earliest_occurrence(c)
        session_file = occ.get("session_file", "")
        date = session_file[:10] if len(session_file) >= 10 else session_file
        past = f"PAST: {date} — {occ.get('timestamp', '')}"
        lines.append(
            f"- \"{c.get('name', '')}\" ({c.get('category', 'other')}): "
            f"{occ.get('definition', '')} "
            f"[for a question on this concept, set source_timestamp exactly "
            f"to \"{past}\"]"
        )
    return "\n".join(lines)


# ---- Quiz generation + grading -----------------------------------------------

def generate_quiz(session_markdown: str,
                  recurring_concepts: list[dict]) -> dict:
    """Send today's session log (plus optional recurring concepts) to Claude.

    Returns a structured quiz dict. When recurring_concepts is non-empty the
    quiz mixes today's content with 1-2 questions on those past-lecture
    concepts; otherwise it is generated entirely from today's transcript.
    """
    user_text = f"TODAY'S LECTURE SESSION LOG:\n\n{session_markdown}"
    if recurring_concepts:
        user_text += (
            "\n\n---\n\nRECURRING HIGH-IMPORTANCE CONCEPTS FROM EARLIER "
            "LECTURES IN THIS CLASS (these did NOT come up today — add 1-2 "
            "quiz questions covering them, each marked \"recurring\": true):"
            "\n\n" + format_recurring_block(recurring_concepts)
        )
    else:
        user_text += (
            "\n\n---\n\n(No recurring concepts from earlier lectures — "
            "generate the quiz entirely from today's transcript.)"
        )

    client = camera_worker.analyzer.client
    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=12000,
        thinking={"type": "adaptive"},
        system=QUIZ_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
        output_config={"format": {"type": "json_schema", "schema": QUIZ_SCHEMA}},
    )
    text = next(b.text for b in message.content if b.type == "text")
    return json.loads(text)


def grade_short_answer(question: str, reference_answer: str,
                       user_answer: str) -> dict:
    """Grade a student's short answer against the reference via Claude."""
    client = camera_worker.analyzer.client
    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system=GRADE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Reference answer:\n{reference_answer}\n\n"
                f"Student's answer:\n{user_answer}"
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": GRADE_SCHEMA}},
    )
    text = next(b.text for b in message.content if b.type == "text")
    return json.loads(text)


def render_quiz_markdown(quiz: dict) -> str:
    """Render the structured quiz as human-readable Markdown for the .md file."""
    lines: list[str] = []
    answer_key: list[str] = []
    for i, q in enumerate(quiz.get("questions", []), start=1):
        ts = q.get("source_timestamp", "—")
        qtype = q.get("type")
        question = q.get("question", "")
        if qtype == "mcq":
            choices = q.get("choices", [])
            lines.append(f"**{i}. (Multiple choice)** {question}\n")
            for j, choice in enumerate(choices):
                lines.append(f"- {chr(65 + j)}. {choice}")
            lines.append(f"\n*Source: {ts}*\n")
            ci = q.get("correct_index", 0)
            letter = chr(65 + ci) if 0 <= ci < len(choices) else "?"
            answer_key.append(f"**{i}.** {letter} — {q.get('explanation', '')}")
        elif qtype == "fill_blank":
            lines.append(f"**{i}. (Fill in the blank)** {question}\n")
            lines.append(f"*Source: {ts}*\n")
            answer_key.append(
                f"**{i}.** {q.get('correct_answer', '')} — "
                f"{q.get('explanation', '')}"
            )
        elif qtype == "short_answer":
            lines.append(f"**{i}. (Short answer)** {question}\n")
            lines.append(f"*Source: {ts}*\n")
            answer_key.append(
                f"**{i}.** Reference answer: {q.get('reference_answer', '')}"
            )
        else:
            lines.append(f"**{i}.** {question}\n")
            lines.append(f"*Source: {ts}*\n")
    body = "\n".join(lines)
    key = "\n\n".join(answer_key)
    return f"{body}\n---\n\n### Answer Key\n\n{key}"


# ---- Flask app ---------------------------------------------------------------

app = Flask(__name__)
camera_worker = CameraWorker()
audio_worker = AudioWorker()


@app.route("/")
def landing():
    """Class-selection landing screen."""
    return render_template("landing.html", classes=list_classes())


@app.route("/start", methods=["POST"])
def start():
    """Begin a session tagged to a class, then hand off to the main page."""
    choice = request.form.get("class_select", "")
    raw = request.form.get("new_class", "") if choice == "__new__" else choice
    name = sanitize_class_name(raw)
    if not name:
        return redirect(url_for("landing"))

    # Cleanly tear down any prior session's workers before starting fresh.
    camera_worker.stop()
    audio_worker.stop()

    state.session = Session(name)
    try:
        camera_worker.start()
    except Exception as exc:
        bus.broadcast({"type": "error", "text": f"Camera: {exc}"})
    audio_worker.start()  # broadcasts its own error if the mic is unavailable
    return redirect(url_for("session_page"))


@app.route("/session")
def session_page():
    """The main Sentry page. Requires an active session.

    If the session has already ended, the cached quiz is handed to the template
    so a page refresh lands straight on the interactive quiz.
    """
    if state.session is None:
        return redirect(url_for("landing"))
    return render_template(
        "index.html",
        class_name=state.session.class_name,
        quiz=state.session.quiz,
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the live camera preview."""
    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        frame_interval = 1.0 / PREVIEW_FPS
        while True:
            jpeg = camera_worker.get_preview_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + jpeg + b"\r\n"
            time.sleep(frame_interval)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events")
def events():
    """Server-Sent Events: meter, status, feedback, transcript, error."""
    def stream():
        q = bus.subscribe()
        # Prime the new client with the current resting status.
        yield _sse({"type": "status", "text": "Watching the board…"})
        try:
            while True:
                try:
                    event = q.get(timeout=SSE_KEEPALIVE)
                    yield _sse(event)
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    accepted, message = camera_worker.trigger_analysis()
    return jsonify({"accepted": accepted, "message": message})


@app.route("/end_session", methods=["POST"])
def end_session():
    """Stop the workers, extract concepts, build a quiz, update class memory.

    The quiz is cached on the session so a refresh or repeat request returns
    the same quiz without regenerating it (and without merging concepts twice).
    """
    if state.session is None:
        return jsonify({"ok": False, "error": "No active session."}), 400

    sess = state.session
    if sess.quiz is not None:
        return jsonify({"ok": True, "quiz": sess.quiz})

    camera_worker.stop()
    audio_worker.stop()
    bus.broadcast({"type": "status", "text": "Generating quiz…"})

    try:
        markdown = sess.read()
        class_dir = sess.file_path.parent

        # Concept extraction — a separate call, independent of the quiz.
        extracted = extract_concepts(markdown)

        # Recurring concepts are chosen from the store as it stands BEFORE
        # today's merge, and only when today's concepts were extractable.
        if extracted:
            store = load_concepts(class_dir)
            recurring = pick_recurring_concepts(store, extracted)
        else:
            store = None
            recurring = []

        quiz = generate_quiz(markdown, recurring)

        # Merge today's concepts into the per-class store and rescore.
        if extracted:
            store = merge_concepts(
                store, extracted, sess.file_path.name, sess.class_name
            )
            save_concepts(class_dir, store)

        sess.quiz = quiz
        sess.ended = True
        sess.append_quiz(render_quiz_markdown(quiz))
        bus.broadcast({"type": "status", "text": "Session ended."})
        return jsonify({"ok": True, "quiz": quiz})
    except Exception as exc:
        bus.broadcast({"type": "error", "text": f"Quiz: {exc}"})
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/grade_answer", methods=["POST"])
def grade_answer():
    """Grade one short-answer response against its reference answer."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    reference_answer = (data.get("reference_answer") or "").strip()
    user_answer = (data.get("user_answer") or "").strip()
    if not user_answer:
        return jsonify({
            "verdict": "incorrect",
            "feedback": "No answer was submitted.",
        })
    try:
        return jsonify(
            grade_short_answer(question, reference_answer, user_answer)
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---- Entry point -------------------------------------------------------------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY is not set; API calls will fail.")
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("Sentry web UI: http://127.0.0.1:5000")
    # use_reloader=False: the reloader spawns a second process that would fight
    # over the camera. threaded=True: /video_feed and /events are long-lived.
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
