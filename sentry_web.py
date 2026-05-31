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
import shutil
import threading
import time
from datetime import datetime, timedelta
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
    send_file,
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

# Slide-capture mode (Pass 3): perceptual-hash trigger for slide-based lectures.
SLIDE_HASH_THRESHOLD = 10    # Hamming distance (out of 64) that counts as a new slide
SLIDE_COOLDOWN_SECONDS = 3   # minimum gap between slide captures (skips brief occlusions)

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

EXAM_SYSTEM_PROMPT = (
    "You are Sentry, a study assistant. You are given the highest-importance "
    "concepts from a student's class, drawn from a semester of recorded "
    "lectures. Each concept comes with a short definition and a transcript "
    "snippet showing it in context. Generate a 20-question consolidated "
    "practice exam.\n\n"
    "COVERAGE: Every concept in the list must appear in at least one "
    "question. Distribute coverage across all concepts — do not stack many "
    "questions on the same one while ignoring others.\n\n"
    "SIZING: Generate exactly 20 questions, with roughly a 60/20/20 split — "
    "about 12 multiple-choice, 4 fill-in-the-blank, and 4 short-answer.\n\n"
    "GROUNDING: Base each question on the definition and snippet provided "
    "for the concept it covers. Do not extrapolate to unrelated material; if "
    "a concept's snippet is thin, write a thinner question rather than "
    "padding with outside knowledge.\n\n"
    "SOURCING: For each question include `source_session` (the markdown "
    "filename the concept came from, e.g. \"2026-05-13_1430.md\") and "
    "`source_timestamp` (HH:MM:SS). Copy them exactly from the concept's "
    "source line above its snippet.\n\n"
    "FORMAT: Same JSON schema as the end-of-session quiz, with an added "
    "source_session field. mcq: question, choices (exactly 4, no letter "
    "prefixes), correct_index (0-3), explanation, source_timestamp, "
    "source_session. fill_blank: question (blank as \"___\"), "
    "correct_answer, acceptable_variants (may be empty), explanation, "
    "source_timestamp, source_session. short_answer: question, "
    "reference_answer, source_timestamp, source_session. Mix all three "
    "types across the exam."
)

EXAM_SCHEMA = {
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
                    "source_session": {"type": "string"},
                },
                "required": [
                    "type", "question",
                    "source_timestamp", "source_session",
                ],
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


def list_audio_devices() -> list[dict]:
    """Available input devices for the landing-page picker.

    Filters to devices with at least one input channel and flags whichever
    sounddevice considers the current input default. Returns [] on any
    enumeration failure so the picker just stays hidden.
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    # sd.default.device is an _InputOutputPair (not a list/tuple), but it's
    # subscript-accessible: index 0 is the input default. When unset it can be
    # -1; in that case fall back to query_devices(kind="input").
    default_in = -1
    try:
        default_in = int(sd.default.device[0])
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    default_in_name: Optional[str] = None
    if default_in < 0:
        try:
            di = sd.query_devices(kind="input")
            if isinstance(di, dict):
                default_in_name = di.get("name")
        except Exception:
            pass
    out: list[dict] = []
    matched_default = False
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        is_def = (i == default_in)
        # Fall-back name match — only the first occurrence wins, in case two
        # devices share a display name (USB hub on macOS often duplicates).
        if (not is_def and not matched_default
                and default_in_name and d.get("name") == default_in_name):
            is_def = True
        if is_def:
            matched_default = True
        out.append({
            "index": i,
            "name": d.get("name", f"Device {i}"),
            "is_default": is_def,
        })
    return out


# ---- Display helpers (Pass 2A) -----------------------------------------------
#
# Timestamps are stored as wall clock in the session markdown (reproducible),
# but shown to students as time elapsed from session start. These helpers do
# that conversion at render time; the markdown is never rewritten.

SESSION_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}\.md$")
PAST_RE = re.compile(r"PAST:\s*(\d{4}-\d{2}-\d{2})\s*—\s*(.+?)\s*$")
STARTED_RE = re.compile(
    r"\*\*Started:\*\*\s*(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})"
)
BOARD_HEADER_RE = re.compile(
    r"^#{2,3}.*Board analysis.*?(\d{1,2}:\d{2}:\d{2})", re.M
)
TRANSCRIPT_RE = re.compile(r"^\*\*.*?(\d{1,2}:\d{2}:\d{2})\*\*", re.M)
# Pause/resume annotation, e.g. "*— paused 19:30:00, resumed 19:35:00 —*".
PAUSE_MARKER_RE = re.compile(
    r"paused\s+(\d{1,2}:\d{2}:\d{2}),\s*resumed\s+(\d{1,2}:\d{2}:\d{2})"
)
# Capture-mode header line written by Session.__init__ — parsed by /history.
MODE_RE = re.compile(r"\*\*Mode:\*\*\s*(Board|Slide)", re.IGNORECASE)
# Inline marker appended by Session.append_mode_switch_marker on mid-session
# toggle — its presence tells /history the session wasn't single-mode.
MODE_SWITCH_RE = re.compile(r"switched to (Board|Slide) mode", re.IGNORECASE)
# Audio input device written into the session markdown header (Pass 5).
# Missing from pre-existing sessions; falls back to "System default".
AUDIO_INPUT_RE = re.compile(r"\*\*Audio input:\*\*\s*([^\n]+)")


def format_elapsed(total_seconds: float) -> str:
    """Render a duration as MM:SS, or H:MM:SS once it passes an hour."""
    total = max(0, int(total_seconds))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _clock_to_seconds(clock: str) -> Optional[int]:
    """Parse an HH:MM:SS (or HH:MM) wall-clock string to seconds since midnight."""
    parts = (clock or "").strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = nums[0], nums[1], 0
    else:
        return None
    return h * 3600 + m * 60 + s


def elapsed_since(wall_clock: str, start: datetime) -> Optional[str]:
    """Elapsed time from `start` to a wall-clock string, as MM:SS / H:MM:SS.

    Both are treated as the same day; a negative result (a session running
    past midnight) wraps forward by 24h. Returns None if `wall_clock` will
    not parse.
    """
    secs = _clock_to_seconds(wall_clock)
    if secs is None:
        return None
    start_secs = start.hour * 3600 + start.minute * 60 + start.second
    delta = secs - start_secs
    if delta < 0:
        delta += 24 * 3600
    return format_elapsed(delta)


def _start_from_filename(stem: str) -> Optional[datetime]:
    """Session start from a `YYYY-MM-DD_HHMM` file stem (minute precision)."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})$", stem)
    if not m:
        return None
    try:
        return datetime.strptime(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}", "%Y-%m-%d %H:%M"
        )
    except ValueError:
        return None


def _start_from_markdown(text: str) -> Optional[datetime]:
    """Session start parsed from the `**Started:**` line of a session file."""
    m = STARTED_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _session_start_for_date(class_dir: Path, date: str,
                            clock: str) -> Optional[datetime]:
    """Best-effort start of the past session a 'PAST:' mention belongs to.

    Session files are `<date>_<HHMM>.md`; when a date has several, pick the
    one whose start is the latest at or before the mention's wall-clock time.
    """
    if not class_dir.is_dir():
        return None
    candidates = []
    for p in class_dir.glob(f"{date}_*.md"):
        m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{2})(\d{2})$", p.stem)
        if m:
            candidates.append(
                (int(m.group(1)) * 3600 + int(m.group(2)) * 60, p)
            )
    if not candidates:
        return None
    candidates.sort()
    chosen_secs = candidates[0][0]
    mention = _clock_to_seconds(clock)
    if mention is not None:
        at_or_before = [c for c in candidates if c[0] <= mention]
        if at_or_before:
            chosen_secs = at_or_before[-1][0]
    try:
        return datetime.strptime(date, "%Y-%m-%d") + timedelta(
            seconds=chosen_secs
        )
    except ValueError:
        return None


def display_timestamp(source_timestamp: str, session_start: datetime,
                      class_dir: Path) -> str:
    """Convert a quiz question's source_timestamp into an elapsed-time label.

    Today's-lecture timestamps (HH:MM:SS) are measured from `session_start`.
    A `PAST: <date> — <HH:MM:SS>` tag keeps its date prefix and converts only
    the time, measured from that past session's start. Anything that will not
    parse is returned unchanged.
    """
    raw = (source_timestamp or "").strip()
    if not raw:
        return "—"
    past = PAST_RE.match(raw)
    if past:
        date, clock = past.group(1), past.group(2)
        start = _session_start_for_date(class_dir, date, clock)
        elapsed = elapsed_since(clock, start) if start else None
        return f"PAST: {date} — {elapsed} elapsed" if elapsed else raw
    elapsed = elapsed_since(raw, session_start)
    return f"{elapsed} elapsed" if elapsed else raw


def annotate_quiz(quiz: dict, session_start: datetime,
                  class_dir: Path) -> dict:
    """Add an elapsed-time `source_display` to each quiz question, in place.

    Display-layer only — `source_timestamp` keeps its wall-clock value so the
    session markdown stays reproducible.
    """
    for q in quiz.get("questions", []):
        q["source_display"] = display_timestamp(
            q.get("source_timestamp", ""), session_start, class_dir
        )
    return quiz


def parse_quiz_markdown(md: str) -> Optional[dict]:
    """Rebuild a structured quiz dict from a session file's Practice Quiz section.

    The inverse of render_quiz_markdown — lets /history re-open a saved quiz
    with no extra API call. Returns None when there is no quiz section or it
    cannot be parsed, so the caller can fall back to regenerating one.
    """
    try:
        marker = md.find("## Practice Quiz")
        if marker == -1:
            return None
        body, _, key_part = md[marker:].partition("### Answer Key")

        type_map = {
            "multiple choice": "mcq",
            "fill in the blank": "fill_blank",
            "short answer": "short_answer",
        }
        q_re = re.compile(
            r"\*\*(\d+)\.\s*\(([^)]+)\)\*\*\s*(.+?)(?=\n\*\*\d+\.\s*\(|\Z)",
            re.S,
        )
        questions: dict = {}
        order: list = []
        for m in q_re.finditer(body):
            num = int(m.group(1))
            qtype = type_map.get(m.group(2).strip().lower(), "short_answer")
            block = m.group(3)
            src_m = re.search(r"\*Source:\s*(.+?)\*", block)
            source = src_m.group(1).strip() if src_m else ""
            q = {
                "type": qtype,
                "question": block.partition("\n")[0].strip(),
                "source_timestamp": source,
                "recurring": source.startswith("PAST:"),
            }
            if qtype == "mcq":
                q["choices"] = [
                    cm.group(1).strip()
                    for cm in re.finditer(
                        r"^-\s*[A-Z]\.\s*(.+?)\s*$", block, re.M
                    )
                ]
            questions[num] = q
            order.append(num)

        if not order:
            return None

        key_re = re.compile(
            r"\*\*(\d+)\.\*\*\s*(.+?)(?=\n\*\*\d+\.\*\*|\Z)", re.S
        )
        for m in key_re.finditer(key_part):
            q = questions.get(int(m.group(1)))
            if not q:
                continue
            payload = m.group(2).strip()
            if q["type"] == "mcq":
                letter, _, expl = payload.partition(" — ")
                letter = letter.strip()
                idx = ord(letter[0]) - 65 if letter else 0
                choices = q.get("choices", [])
                q["correct_index"] = idx if 0 <= idx < len(choices) else 0
                q["explanation"] = expl.strip()
            elif q["type"] == "fill_blank":
                answer, _, expl = payload.partition(" — ")
                q["correct_answer"] = answer.strip()
                q["acceptable_variants"] = []
                q["explanation"] = expl.strip()
            else:
                ref = payload
                if ref.startswith("Reference answer:"):
                    ref = ref[len("Reference answer:"):].strip()
                q["reference_answer"] = ref

        return {"questions": [questions[n] for n in order]}
    except Exception as exc:
        print(f"Warning: could not parse saved quiz ({exc}); regenerating.")
        return None


def session_metrics(md_path: Path, concept_store: Optional[dict]) -> dict:
    """Parse one session markdown file into the metrics shown on /history."""
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        text = ""
    pre_quiz = text.split("## Practice Quiz")[0]

    board_times = BOARD_HEADER_RE.findall(pre_quiz)
    transcript_times = TRANSCRIPT_RE.findall(pre_quiz)
    start = _start_from_markdown(text) or _start_from_filename(md_path.stem)

    duration = "—"
    all_secs = [s for s in (_clock_to_seconds(t)
                            for t in board_times + transcript_times)
                if s is not None]
    if start is not None and all_secs:
        start_secs = start.hour * 3600 + start.minute * 60 + start.second
        delta = max(all_secs) - start_secs
        if delta < 0:
            delta += 24 * 3600
        # Exclude time the session was paused, so the duration matches the
        # elapsed clock the student saw live.
        for pa, ra in PAUSE_MARKER_RE.findall(pre_quiz):
            gap = (_clock_to_seconds(ra) or 0) - (_clock_to_seconds(pa) or 0)
            if gap < 0:
                gap += 24 * 3600
            delta -= gap
        duration = format_elapsed(max(0, delta))

    concept_count = 0
    if concept_store:
        concept_count = sum(
            1 for c in concept_store.get("concepts", [])
            if any(o.get("session_file") == md_path.name
                   for o in c.get("occurrences", []))
        )

    mode_match = MODE_RE.search(text)
    mode = mode_match.group(1).lower() if mode_match else "board"
    toggled = bool(MODE_SWITCH_RE.search(pre_quiz))

    audio_match = AUDIO_INPUT_RE.search(text)
    audio_input = audio_match.group(1).strip() if audio_match else "System default"

    return {
        "filename": md_path.name,
        "date": start.strftime("%Y-%m-%d %H:%M") if start else md_path.stem,
        "duration": duration,
        "board_count": len(board_times),
        "transcript_count": len(transcript_times),
        "concept_count": concept_count,
        "has_quiz": "## Practice Quiz" in text,
        "mode": mode,
        "toggled": toggled,
        "audio_input": audio_input,
    }


def _mention_label(occ: dict) -> str:
    """A `YYYY-MM-DD — MM:SS elapsed` label for one concept occurrence."""
    if not occ:
        return "—"
    session_file = occ.get("session_file", "")
    clock = occ.get("timestamp", "")
    date = session_file[:10]
    stem = session_file[:-3] if session_file.endswith(".md") else session_file
    start = _start_from_filename(stem)
    elapsed = elapsed_since(clock, start) if start else None
    if elapsed:
        return f"{date} — {elapsed} elapsed"
    return f"{date} — {clock}" if clock else (date or "—")


# ---- Session -----------------------------------------------------------------

class Session:
    """A single lecture session, tagged to a class, backed by one markdown file.

    Both workers append to this file as events happen; every append is locked
    and the file is closed each time, so a crash leaves a valid partial file.
    The generated quiz is cached on the instance so a refresh or repeat request
    does not regenerate it.
    """

    def __init__(self, class_name: str, mode: str = "board",
                 audio_device_index: Optional[int] = None,
                 audio_device_name: str = "System default"):
        self.class_name = class_name
        self.mode = mode if mode in ("board", "slide") else "board"
        # Audio input picker (Pass 5). None means "use whichever device
        # sounddevice picks", which is the existing pre-Pass-5 behavior.
        self.audio_device_index = audio_device_index
        self.audio_device_name = audio_device_name or "System default"
        self.started_at = datetime.now()
        self.ended = False
        self.quiz: Optional[dict] = None

        # Pause/resume: paused_seconds accumulates completed pauses so the
        # elapsed clock can exclude any time the session spent paused.
        self.paused = False
        self.paused_seconds = 0.0
        self._pause_started_at: Optional[datetime] = None

        class_dir = SESSIONS_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y-%m-%d_%H%M")
        self.file_path = class_dir / f"{stamp}.md"

        self._lock = threading.Lock()
        mode_label = "Slide" if self.mode == "slide" else "Board"
        self._append(
            f"# Sentry Session — {class_name}\n\n"
            f"**Started:** {self.started_at:%Y-%m-%d %H:%M}\n\n"
            f"**Mode:** {mode_label}\n\n"
            f"**Audio input:** {self.audio_device_name}\n\n"
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

    def append_pause_marker(self, paused_at: str, resumed_at: str) -> None:
        """Append an italic transcript annotation marking a pause gap."""
        self._append(f"*— paused {paused_at}, resumed {resumed_at} —*\n\n")

    def append_mode_switch_marker(self, new_mode: str) -> str:
        """Append an italic transcript marker for a mid-session mode toggle.

        Uses elapsed-time formatting (same as quiz source timestamps) and
        returns the label so the live UI can show the same string. The
        markdown header keeps the *starting* mode; these inline markers
        carry the timeline — same pattern as pause/resume.
        """
        label = format_elapsed(self.elapsed())
        mode_label = "Slide" if new_mode == "slide" else "Board"
        self._append(f"*— switched to {mode_label} mode at {label} —*\n\n")
        return label

    def mark_paused(self) -> None:
        """Record the start of a pause (idempotent)."""
        if self.paused:
            return
        self.paused = True
        self._pause_started_at = datetime.now()

    def mark_resumed(self) -> tuple[str, str]:
        """End the current pause; return its (paused_at, resumed_at) wall clocks."""
        resumed = datetime.now()
        paused = self._pause_started_at or resumed
        self.paused_seconds += (resumed - paused).total_seconds()
        self.paused = False
        self._pause_started_at = None
        return paused.strftime("%H:%M:%S"), resumed.strftime("%H:%M:%S")

    def elapsed(self) -> float:
        """Seconds since start, excluding any time spent paused."""
        total = (datetime.now() - self.started_at).total_seconds()
        total -= self.paused_seconds
        if self.paused and self._pause_started_at is not None:
            total -= (datetime.now() - self._pause_started_at).total_seconds()
        return max(0.0, total)

    def read(self) -> str:
        with self._lock:
            return self.file_path.read_text(encoding="utf-8")


class AppState:
    """Process-global handle to the active session, shared by both workers."""
    session: Optional[Session] = None


state = AppState()


def _session_elapsed() -> Optional[float]:
    """Seconds since the active session started, or None if there is none.

    Excludes time spent paused, so the elapsed clock does not advance while
    the session is paused. Broadcast with live events so the browser can show
    elapsed-time labels without parsing wall-clock strings against a start time.
    """
    if state.session is None:
        return None
    return state.session.elapsed()


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


# ---- Slide-detection helpers (Pass 3) ----------------------------------------
#
# Slide mode triggers analysis on perceptual-hash changes instead of motion
# settling. dHash resizes the frame to 9x8 grayscale and compares adjacent
# pixels — small lighting changes are absorbed; a new slide flips most bits.

def _compute_dhash(frame: np.ndarray) -> int:
    """64-bit difference hash of a frame (resized to 9×8 grayscale)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def _hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit hashes."""
    return bin(a ^ b).count("1")


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
        self._paused = False

        # Capture mode (Pass 3): "board" = motion-diff trigger (existing);
        # "slide" = perceptual-hash trigger over `_last_captured_hash`.
        self._mode = "board"
        self._last_captured_hash: Optional[int] = None

    # ---- lifecycle ----------------------------------------------------------

    def start(self, mode: str = "board") -> None:
        if self._started:
            return
        self._stop_evt.clear()
        self.detector = ChangeDetector()
        self._latest_frame = None
        self._analyzing = False
        self._paused = False
        self._last_analysis_time = 0.0
        self._mode = mode if mode in ("board", "slide") else "board"
        self._last_captured_hash = None

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

    def pause(self) -> None:
        """Suspend board-change detection and auto-analysis (camera idles)."""
        self._paused = True

    def resume(self) -> None:
        """Resume detection. In slide mode, clear the last-captured hash so
        the first frame after resume is captured — the user has likely
        advanced slides during the pause. Harmless no-op in board mode."""
        self._last_captured_hash = None
        self._paused = False

    def set_mode(self, mode: str) -> None:
        """Switch capture mode mid-session and reset the now-stale baseline.

        Mirrors how resume() clears baselines so the very next frame is
        treated as new content. _capture_loop reads self._mode every frame
        (Pass 3 dispatch), so the new trigger takes effect immediately.
        """
        if mode not in ("board", "slide") or mode == self._mode:
            return
        self._mode = mode
        self._last_captured_hash = None
        # Re-instantiate the motion detector — same reset start() performs.
        self.detector = ChangeDetector()

    # ---- capture loop -------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_evt.is_set():
            if self._paused:
                # Idle: no frame reads, no detection, no auto-trigger. Keep
                # broadcasting a "paused" meter so the UI shows the state.
                now = time.time()
                if now - self._last_meter_emit > METER_INTERVAL:
                    self._last_meter_emit = now
                    bus.broadcast({"type": "meter", "paused": True})
                time.sleep(0.1)
                continue
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            now = time.time()
            if self._mode == "slide":
                self._slide_step(frame, now)
            else:
                self._board_step(frame, now)

            time.sleep(0.03)

    def _board_step(self, frame: np.ndarray, now: float) -> None:
        """Motion-diff trigger (board mode): analyse when motion settles."""
        settled = self.detector.update(frame)
        if now - self._last_meter_emit > METER_INTERVAL:
            self._last_meter_emit = now
            bus.broadcast({
                "type": "meter",
                "mode": "board",
                "diff": round(self.detector.last_diff, 2),
                "threshold": round(self.detector.threshold, 2),
                "armed": bool(self.detector._was_changing),
            })
        if (settled
                and not self._analyzing
                and now - self._last_analysis_time > COOLDOWN_SECONDS):
            self._launch_analysis(frame.copy())

    def _slide_step(self, frame: np.ndarray, now: float) -> None:
        """Perceptual-hash trigger (slide mode): analyse on a new slide."""
        h = _compute_dhash(frame)
        if self._last_captured_hash is None:
            # No baseline yet — treat as brand new so the first frame (or the
            # first frame after a resume) gets captured.
            dist = 64
        else:
            dist = _hamming(h, self._last_captured_hash)

        if now - self._last_meter_emit > METER_INTERVAL:
            self._last_meter_emit = now
            bus.broadcast({
                "type": "meter",
                "mode": "slide",
                "distance": int(min(dist, 64)),
                "threshold": SLIDE_HASH_THRESHOLD,
                "new_slide": dist > SLIDE_HASH_THRESHOLD,
            })

        if (dist > SLIDE_HASH_THRESHOLD
                and not self._analyzing
                and now - self._last_analysis_time > SLIDE_COOLDOWN_SECONDS):
            self._last_captured_hash = h
            self._launch_analysis(frame.copy())

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
        if self._paused:
            return False, "Paused — resume to analyze."
        if self._analyzing:
            return False, "Already analyzing…"
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return False, "No frame yet…"
        # In slide mode, set the hash so the cooldown doesn't lapse onto an
        # identical frame and immediately auto-trigger again.
        if self._mode == "slide":
            self._last_captured_hash = _compute_dhash(frame)
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
            bus.broadcast({
                "type": "feedback",
                "data": result,
                "elapsed": _session_elapsed(),
            })
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
        self._paused = False
        self._model_loaded = False

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._stop_evt.clear()
        self._paused = False
        with self._lock:
            self._chunks = []
        # Read the chosen input device from the session at stream-open time
        # (Pass 5) — never cache it on the worker. None falls through to the
        # system default, matching pre-Pass-5 behavior.
        device = (state.session.audio_device_index
                  if state.session is not None else None)
        stream_kwargs = {
            "samplerate": AUDIO_SAMPLE_RATE,
            "channels": 1,
            "dtype": "float32",
            "callback": self._callback,
        }
        if device is not None:
            stream_kwargs["device"] = device
        try:
            self._stream = sd.InputStream(**stream_kwargs)
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

    def pause(self) -> None:
        """Stop recording; drop the partial chunk so the gap is clean."""
        if not self._started or self._paused:
            return
        self._paused = True
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
        with self._lock:
            self._chunks = []

    def resume(self) -> None:
        """Restart capture from an empty buffer (no overlap, no duplicates)."""
        if not self._started or not self._paused:
            return
        with self._lock:
            self._chunks = []
        if self._stream is not None:
            try:
                self._stream.start()
            except Exception as exc:
                bus.broadcast({"type": "error", "text": f"Audio resume: {exc}"})
        self._paused = False

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
            bus.broadcast({
                "type": "transcript",
                "time": ts,
                "elapsed": _session_elapsed(),
                "text": text,
            })
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


# ---- Semester practice exam (Pass 4) -----------------------------------------
#
# /class/<name>/exam composes one consolidated exam drawing from the whole
# semester. It reuses concepts.json (Pass 1), the existing quiz UI (Pass 2),
# and the PDF route (Pass 2B). Results are cached in-process so a refresh of
# the URL returns the same exam, while a fresh visit (no ?ts param) generates
# a new one and redirects to its timestamped URL.

EXAM_SNIPPET_HALF_WINDOW = 600   # characters of session markdown on each side
EXAM_TOP_CONCEPTS = 15           # candidates sent to Claude (importance-ranked)
EXAM_MIN_CONCEPTS = 5            # below this, render the empty state instead

exam_cache: dict = {}            # (class_name, ts) -> {"exam","session_count","concept_count","error"}


def session_snippet(class_dir: Path, session_file: str, timestamp: str,
                    half_window: int = EXAM_SNIPPET_HALF_WINDOW) -> Optional[str]:
    """Return a chunk of session markdown surrounding `timestamp`.

    Returns None when the file is missing; falls back to the opening of the
    session if the literal timestamp string can't be located in the body.
    """
    path = class_dir / session_file
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    pre = text.split("## Practice Quiz")[0]
    idx = pre.find(timestamp)
    if idx < 0:
        return pre[:half_window * 2]
    start = max(0, idx - half_window)
    end = min(len(pre), idx + half_window)
    return pre[start:end]


def generate_practice_exam(class_name: str) -> dict:
    """Build a 20-question semester practice exam from a class's concept memory.

    Returns {"exam", "session_count", "concept_count", "error", "detail"}. A
    `not_enough_concepts` error covers missing concepts.json, too few stored
    concepts, or all referenced sessions deleted; `api` covers a Claude failure.
    """
    class_dir = SESSIONS_DIR / class_name
    if not class_dir.is_dir():
        return {"error": "not_enough_concepts"}

    store = load_concepts(class_dir)
    all_concepts = store.get("concepts", []) if store else []
    if len(all_concepts) < EXAM_MIN_CONCEPTS:
        return {"error": "not_enough_concepts"}

    top = sorted(
        all_concepts,
        key=lambda c: c.get("importance_score", 0.0),
        reverse=True,
    )[:EXAM_TOP_CONCEPTS]

    # Pull each concept's most recent occurrence + a transcript snippet; drop
    # concepts whose session file has been deleted (Pass 4 edge case).
    selected: list[dict] = []
    for c in top:
        occs = c.get("occurrences", [])
        if not occs:
            continue
        latest = max(occs, key=lambda o: o.get("session_file", ""))
        sf = latest.get("session_file", "")
        if not sf or not (class_dir / sf).is_file():
            continue
        ts = latest.get("timestamp", "")
        snippet = session_snippet(class_dir, sf, ts)
        if not snippet:
            continue
        selected.append({
            "name": c.get("name", ""),
            "category": c.get("category", "other"),
            "definition": latest.get("definition", ""),
            "source_session": sf,
            "source_timestamp": ts,
            "snippet": snippet,
        })

    if len(selected) < 3:
        return {"error": "not_enough_concepts"}

    parts = []
    for c in selected:
        parts.append(
            f"### {c['name']} ({c['category']})\n"
            f"- Definition: {c['definition']}\n"
            f"- Source: {c['source_session']} at {c['source_timestamp']}\n"
            f"- Snippet:\n{c['snippet']}\n"
        )
    user_text = (
        f"PRACTICE-EXAM CONCEPTS for class \"{class_name}\":\n\n"
        + "\n---\n\n".join(parts)
        + "\n\nGenerate the 20-question practice exam now."
    )

    try:
        client = camera_worker.analyzer.client
        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=EXAM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            output_config={
                "format": {"type": "json_schema", "schema": EXAM_SCHEMA},
            },
        )
        text = next(b.text for b in message.content if b.type == "text")
        exam = json.loads(text)
    except Exception as exc:
        return {"error": "api", "detail": str(exc)}

    return {
        "exam": exam,
        "session_count": len(list(class_dir.glob("*.md"))),
        "concept_count": len(all_concepts),
        "error": None,
    }


def annotate_exam(exam: dict, class_dir: Path) -> dict:
    """Add `source_display` ('From: DATE session — MM:SS elapsed') per question.

    Also force-clears `recurring`: every exam question is from a prior lecture
    by definition, so the FROM PRIOR LECTURE badge would be meaningless.
    """
    for q in exam.get("questions", []):
        sf = q.get("source_session", "")
        ts = q.get("source_timestamp", "")
        date = sf[:10] if sf else ""
        stem = sf[:-3] if sf.endswith(".md") else sf
        start = _start_from_filename(stem) if stem else None
        elapsed = elapsed_since(ts, start) if start else None
        if elapsed and date:
            q["source_display"] = f"From: {date} session — {elapsed} elapsed"
        elif date:
            q["source_display"] = f"From: {date} session — {ts or '—'}"
        else:
            q["source_display"] = f"From: {sf or '—'}"
        q["recurring"] = False
    return exam


# ---- Quiz PDF export (Pass 2B) -----------------------------------------------

def build_quiz_pdf(quiz: dict, class_name: str, session_date: str, *,
                   title_suffix: str = "Practice Quiz",
                   subtitle_prefix: str = "Session: "):
    """Render a quiz dict to a printable PDF, returned as an in-memory buffer.

    title_suffix / subtitle_prefix let the exam path render
    "Practice Exam / Generated from N sessions · M concepts" instead of the
    end-of-session "Practice Quiz / Session: DATE".

    reportlab is imported lazily so a missing dependency only breaks the PDF
    route — pause/resume and the rest of the app keep working.
    """
    from io import BytesIO
    from xml.sax.saxutils import escape
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
    )

    grey = HexColor("#666666")
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11,
                          leading=15)
    title = ParagraphStyle("title", fontName="Times-Bold", fontSize=18,
                           leading=22)
    subtitle = ParagraphStyle("subtitle", fontName="Times-Roman", fontSize=11,
                              leading=14, textColor=grey)
    qheader = ParagraphStyle("qheader", fontName="Times-Bold", fontSize=14,
                             leading=17, spaceAfter=6)
    choice = ParagraphStyle("choice", parent=body, leftIndent=20, spaceAfter=2)
    source = ParagraphStyle("source", fontName="Times-Italic", fontSize=9,
                            leading=12, textColor=grey, spaceBefore=4)
    keyhdr = ParagraphStyle("keyhdr", fontName="Times-Bold", fontSize=14,
                            leading=17, spaceAfter=8)
    keyitem = ParagraphStyle("keyitem", parent=body, spaceAfter=6)

    type_labels = {
        "mcq": "Multiple choice",
        "fill_blank": "Fill in the blank",
        "short_answer": "Short answer",
    }
    questions = quiz.get("questions", [])

    flow = [
        Paragraph(escape(f"{class_name} — {title_suffix}"), title),
        Paragraph(escape(f"{subtitle_prefix}{session_date}"), subtitle),
        Spacer(1, 18),
    ]

    for i, q in enumerate(questions, start=1):
        qtype = q.get("type", "")
        label = type_labels.get(qtype, "Question")
        block = [Paragraph(escape(f"Question {i}  ({label})"), qheader)]

        qtext = q.get("question", "")
        if q.get("recurring"):
            qtext = "[FROM PRIOR LECTURE] " + qtext
        block.append(Paragraph(escape(qtext), body))

        if qtype == "mcq":
            block.append(Spacer(1, 4))
            for j, ch in enumerate(q.get("choices", [])):
                block.append(
                    Paragraph(f"{chr(65 + j)}. {escape(ch)}", choice)
                )
        elif qtype == "fill_blank":
            block.append(Spacer(1, 4))
            block.append(Paragraph("Answer: " + "_" * 40, body))
        elif qtype == "short_answer":
            block.append(Spacer(1, 4))
            block.append(Paragraph("Answer:", body))
            block.append(Spacer(1, 54))  # blank space to write a response

        src = q.get("source_display") or q.get("source_timestamp") or "—"
        # Exam questions self-label with "From: <date> session — ..."; the
        # "Source: " prefix is only added when the label isn't already framed.
        prefix = "" if src.startswith("From:") else "Source: "
        block.append(Paragraph(escape(f"{prefix}{src}"), source))
        # Keep each question (header through source) on a single page.
        flow.append(KeepTogether(block))
        flow.append(Spacer(1, 14))

    flow.append(Spacer(1, 6))
    flow.append(HRFlowable(width="100%", thickness=1, color=grey))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("Answer Key", keyhdr))

    for i, q in enumerate(questions, start=1):
        qtype = q.get("type", "")
        expl = q.get("explanation", "")
        if qtype == "mcq":
            choices = q.get("choices", [])
            ci = q.get("correct_index", 0)
            if 0 <= ci < len(choices):
                answer = f"{chr(65 + ci)}. {choices[ci]}"
            else:
                answer = "—"
            text = f"<b>{i}.</b> {escape(answer)}"
            if expl:
                text += f" — {escape(expl)}"
        elif qtype == "fill_blank":
            text = f"<b>{i}.</b> {escape(q.get('correct_answer', ''))}"
            if expl:
                text += f" — {escape(expl)}"
        elif qtype == "short_answer":
            text = (f"<b>{i}.</b> Reference answer: "
                    f"{escape(q.get('reference_answer', ''))}")
        else:
            text = f"<b>{i}.</b>"
        flow.append(Paragraph(text, keyitem))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title=f"{class_name} Practice Quiz",
    )
    doc.build(flow)
    buf.seek(0)
    return buf


# ---- Flask app ---------------------------------------------------------------

app = Flask(__name__)
camera_worker = CameraWorker()
audio_worker = AudioWorker()


def class_overview(name: str) -> dict:
    """Per-class summary card data for the landing page."""
    class_dir = SESSIONS_DIR / name
    md_files = sorted(class_dir.glob("*.md")) if class_dir.is_dir() else []
    store = load_concepts(class_dir)
    latest = "—"
    if md_files:
        start = _start_from_filename(md_files[-1].stem)
        latest = start.strftime("%Y-%m-%d") if start else md_files[-1].stem[:10]
    return {
        "name": name,
        "session_count": len(md_files),
        "concept_count": len(store.get("concepts", [])) if store else 0,
        "latest": latest,
    }


def _stop_active_session_if_class(class_name: str) -> None:
    """Tear down the live session if it belongs to `class_name`.

    The single-process design has no child process to pkill — the in-flight
    session is `state.session` plus the two workers. Stopping them here ensures
    a class folder is never renamed or trashed mid-write.
    """
    if state.session is not None and state.session.class_name == class_name:
        camera_worker.stop()
        audio_worker.stop()
        state.session = None


def _pause_session() -> None:
    """Pause the live session: camera, audio, and the elapsed clock."""
    sess = state.session
    if sess is None or sess.paused:
        return
    camera_worker.pause()
    audio_worker.pause()
    sess.mark_paused()
    bus.broadcast({"type": "status", "text": "Paused."})


def _resume_session() -> None:
    """Resume the live session and record the pause gap in the transcript."""
    sess = state.session
    if sess is None or not sess.paused:
        return
    paused_at, resumed_at = sess.mark_resumed()
    camera_worker.resume()
    audio_worker.resume()
    sess.append_pause_marker(paused_at, resumed_at)
    bus.broadcast({
        "type": "pause_marker",
        "text": f"— paused {paused_at}, resumed {resumed_at} —",
    })
    bus.broadcast({"type": "status", "text": "Watching the board…"})


@app.route("/")
def landing():
    """Class-selection landing screen — one card per class."""
    classes = [class_overview(n) for n in list_classes()]
    return render_template("landing.html", classes=classes)


@app.route("/audio_devices")
def audio_devices():
    """JSON list of input devices for the landing-page picker (Pass 5)."""
    return jsonify(list_audio_devices())


@app.route("/start", methods=["POST"])
def start():
    """Begin a session tagged to a class, then hand off to the main page."""
    choice = request.form.get("class_select", "")
    raw = request.form.get("new_class", "") if choice == "__new__" else choice
    name = sanitize_class_name(raw)
    if not name:
        return redirect(url_for("landing"))

    mode = request.form.get("capture_mode", "board")
    if mode not in ("board", "slide"):
        mode = "board"

    # Resolve the optional audio_device selection against the live device list
    # — a forged or stale index silently falls back to the system default.
    audio_device_index: Optional[int] = None
    audio_device_name = "System default"
    raw_device = request.form.get("audio_device", "").strip()
    if raw_device:
        try:
            idx = int(raw_device)
            for dev in list_audio_devices():
                if dev["index"] == idx:
                    audio_device_index = idx
                    audio_device_name = dev["name"]
                    break
        except ValueError:
            pass

    # Cleanly tear down any prior session's workers before starting fresh.
    camera_worker.stop()
    audio_worker.stop()

    state.session = Session(
        name, mode=mode,
        audio_device_index=audio_device_index,
        audio_device_name=audio_device_name,
    )
    try:
        camera_worker.start(mode=mode)
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
        session_id=state.session.file_path.stem,
        paused=state.session.paused,
        mode=state.session.mode,
        audio_input=state.session.audio_device_name,
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
        resting = ("Paused." if (state.session and state.session.paused)
                   else "Watching the board…")
        yield _sse({"type": "status", "text": resting})
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


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    """Pause or resume the live session (camera, audio, and elapsed clock)."""
    sess = state.session
    if sess is None or sess.ended:
        return jsonify({"ok": False, "error": "No active session."}), 400
    if sess.paused:
        _resume_session()
    else:
        _pause_session()
    return jsonify({"ok": True, "paused": sess.paused})


@app.route("/toggle_mode", methods=["POST"])
def toggle_mode():
    """Flip capture mode mid-session. Header stays as starting mode; an inline
    italic marker records the switch in the transcript (and is surfaced live)."""
    sess = state.session
    if sess is None or sess.ended:
        return jsonify({"ok": False, "error": "No active session."}), 400
    new_mode = "slide" if sess.mode == "board" else "board"
    sess.mode = new_mode
    camera_worker.set_mode(new_mode)
    label = sess.append_mode_switch_marker(new_mode)
    mode_label = "Slide" if new_mode == "slide" else "Board"
    # Reuse the pause_marker SSE channel — same italic-gray transcript line.
    bus.broadcast({
        "type": "pause_marker",
        "text": f"— switched to {mode_label} mode at {label} —",
    })
    bus.broadcast({"type": "status", "text": f"Switched to {mode_label} mode."})
    return jsonify({"ok": True, "mode": new_mode})


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

    # Ending while paused: resume first so the workers and the pause clock
    # close out cleanly (the pause gap is still recorded in the transcript).
    if sess.paused:
        _resume_session()

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

        sess.ended = True
        # Markdown is appended from the raw quiz (wall-clock source_timestamp);
        # annotate_quiz then adds elapsed-time source_display for the browser.
        sess.append_quiz(render_quiz_markdown(quiz))
        sess.quiz = annotate_quiz(quiz, sess.started_at, class_dir)
        bus.broadcast({"type": "status", "text": "Session ended."})
        return jsonify({"ok": True, "quiz": sess.quiz})
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


# ---- Quiz PDF download (Pass 2B) ---------------------------------------------

QUIZ_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}$")


@app.route("/quiz/<class_name>/<session_id>/pdf")
def quiz_pdf(class_name: str, session_id: str):
    """Download a session's quiz — or a semester practice exam — as a PDF.

    A session_id of the form "exam-<cache_ts>" pulls from the practice-exam
    cache (Pass 4); otherwise the existing session path runs, using the cached
    quiz when it matches the active session and falling back to the markdown.
    """
    name = sanitize_class_name(class_name)
    if not name:
        return redirect(url_for("landing"))

    # Practice-exam PDFs (Pass 4): the quiz view rendered by /class/<n>/exam
    # uses session_id="exam-<ts>"; the PDF button reaches us here.
    if session_id.startswith("exam-"):
        cached = exam_cache.get((name, session_id[5:]))
        if not cached or not cached.get("exam"):
            return jsonify({"error": "Exam not in cache."}), 404
        try:
            subtitle = (f"{cached['session_count']} sessions · "
                        f"{cached['concept_count']} concepts")
            pdf = build_quiz_pdf(
                cached["exam"], name, subtitle,
                title_suffix="Practice Exam",
                subtitle_prefix="Generated from ",
            )
        except Exception as exc:
            return jsonify({"error": f"PDF generation failed: {exc}"}), 500
        download = f"{name.replace(' ', '_')}_practice_exam.pdf"
        return send_file(pdf, mimetype="application/pdf",
                         as_attachment=True, download_name=download)

    if not QUIZ_ID_RE.match(session_id):
        return redirect(url_for("landing"))
    class_dir = SESSIONS_DIR / name

    quiz = None
    sess = state.session
    if (sess is not None and sess.class_name == name
            and sess.file_path.stem == session_id and sess.quiz is not None):
        quiz = sess.quiz

    if quiz is None:
        md_path = class_dir / f"{session_id}.md"
        if not md_path.is_file():
            return redirect(url_for("landing"))
        md = md_path.read_text(encoding="utf-8")
        quiz = parse_quiz_markdown(md)
        if quiz is None:
            return jsonify({"error": "No saved quiz for this session."}), 404
        start = (_start_from_markdown(md)
                 or _start_from_filename(session_id)
                 or datetime.now())
        annotate_quiz(quiz, start, class_dir)

    try:
        pdf = build_quiz_pdf(quiz, name, session_id[:10])
    except Exception as exc:
        return jsonify({"error": f"PDF generation failed: {exc}"}), 500

    download = f"{name.replace(' ', '_')}_{session_id[:10]}_quiz.pdf"
    return send_file(pdf, mimetype="application/pdf",
                     as_attachment=True, download_name=download)


# ---- History, concepts, class management (Pass 2A) ---------------------------

@app.route("/history")
def history():
    """Every session across every class, newest-first, with parsed metrics."""
    classes = []
    for name in list_classes():
        class_dir = SESSIONS_DIR / name
        store = load_concepts(class_dir)
        sessions = [
            session_metrics(md, store)
            for md in sorted(class_dir.glob("*.md"), reverse=True)
        ]
        classes.append({"name": name, "sessions": sessions})
    return render_template("history.html", classes=classes)


@app.route("/history/session/<class_name>/<filename>")
def history_session(class_name: str, filename: str):
    """Re-open a past session's quiz: read it from the markdown, else regenerate."""
    name = sanitize_class_name(class_name)
    if not name or not SESSION_FILE_RE.match(filename):
        return redirect(url_for("history"))
    md_path = SESSIONS_DIR / name / filename
    if not md_path.is_file():
        return redirect(url_for("history"))

    md = md_path.read_text(encoding="utf-8")
    quiz = parse_quiz_markdown(md)
    if quiz is None:
        # No saved quiz (older session) — regenerate from the transcript only.
        transcript = md.split("## Practice Quiz")[0]
        try:
            quiz = generate_quiz(transcript, [])
        except Exception as exc:
            bus.broadcast({"type": "error", "text": f"Quiz: {exc}"})
            quiz = {"questions": []}

    start = (_start_from_markdown(md)
             or _start_from_filename(md_path.stem)
             or datetime.now())
    annotate_quiz(quiz, start, md_path.parent)
    audio_match = AUDIO_INPUT_RE.search(md)
    audio_input = audio_match.group(1).strip() if audio_match else "System default"
    return render_template(
        "index.html",
        class_name=name,
        quiz=quiz,
        session_id=md_path.stem,
        history_mode=True,
        back_url=url_for("history"),
        mode="board",
        audio_input=audio_input,
    )


@app.route("/class/<class_name>/exam")
def class_exam(class_name: str):
    """Semester practice exam — top 15 concepts → 20 questions via Claude.

    A bare visit always regenerates and redirects to ?ts=<new>; refreshing the
    timestamped URL is a cache hit. Process restart clears the cache, so the
    next bare visit naturally regenerates.
    """
    name = sanitize_class_name(class_name)
    if not name or not (SESSIONS_DIR / name).is_dir():
        return redirect(url_for("landing"))

    ts = request.args.get("ts", "")
    if ts:
        cached = exam_cache.get((name, ts))
        if cached and cached.get("exam"):
            return render_template(
                "index.html",
                class_name=name,
                quiz=cached["exam"],
                session_id=f"exam-{ts}",
                history_mode=True,
                back_url=url_for("landing"),
                mode="board",
                is_exam=True,
                exam_session_count=cached["session_count"],
                exam_concept_count=cached["concept_count"],
            )
        # Stale link (process restart, etc.) — regenerate below.

    result = generate_practice_exam(name)
    if result.get("error") == "not_enough_concepts":
        return render_template(
            "exam_empty.html",
            class_name=name,
            message=("Not enough concepts yet — record a few more sessions "
                     "and try again."),
        )
    if result.get("error"):
        return render_template(
            "exam_empty.html",
            class_name=name,
            message=f"Could not generate exam: "
                    f"{result.get('detail') or result['error']}",
        )

    annotate_exam(result["exam"], SESSIONS_DIR / name)
    new_ts = datetime.now().strftime("%Y%m%d%H%M%S")
    exam_cache[(name, new_ts)] = result
    return redirect(url_for("class_exam", class_name=name, ts=new_ts))


@app.route("/class/<class_name>/concepts")
def class_concepts(class_name: str):
    """Browse a class's accumulated concept memory (concepts.json) as a table."""
    name = sanitize_class_name(class_name)
    store = load_concepts(SESSIONS_DIR / name)
    concepts = []
    for c in (store.get("concepts", []) if store else []):
        occ = c.get("occurrences", [])
        first = min(occ, key=lambda o: o.get("session_file", "")) if occ else {}
        last = max(occ, key=lambda o: o.get("session_file", "")) if occ else {}
        concepts.append({
            "name": c.get("name", ""),
            "category": c.get("category", "other"),
            "importance": c.get("importance_score", 0.0),
            "count": len(occ),
            "first": _mention_label(first),
            "last": _mention_label(last),
            "last_key": (last.get("session_file", "")
                         + last.get("timestamp", "")),
        })
    concepts.sort(key=lambda c: c["importance"], reverse=True)
    return render_template("concepts.html", class_name=name, concepts=concepts)


@app.route("/rename_class", methods=["POST"])
def rename_class():
    """Rename a class folder atomically and keep concepts.json's name in sync."""
    data = request.get_json(silent=True) or {}
    old = sanitize_class_name(data.get("old", ""))
    new = sanitize_class_name(data.get("new", ""))
    if not old or not new:
        return jsonify({"ok": False, "error": "Invalid class name."}), 400
    src, dst = SESSIONS_DIR / old, SESSIONS_DIR / new
    if not src.is_dir():
        return jsonify({"ok": False, "error": "Class not found."}), 404
    if new == old:
        return jsonify({"ok": True})
    if dst.exists():
        return jsonify(
            {"ok": False, "error": "A class with that name already exists."}
        ), 409

    _stop_active_session_if_class(old)
    try:
        os.rename(src, dst)
    except OSError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    store = load_concepts(dst)
    if store is not None:
        store["class_name"] = new
        save_concepts(dst, store)
    return jsonify({"ok": True})


@app.route("/delete_class", methods=["POST"])
def delete_class():
    """Move a class folder to ~/.Trash after a typed-name confirmation."""
    data = request.get_json(silent=True) or {}
    name = sanitize_class_name(data.get("name", ""))
    confirm = sanitize_class_name(data.get("confirm", ""))
    src = SESSIONS_DIR / name
    if not name or not src.is_dir():
        return jsonify({"ok": False, "error": "Class not found."}), 404
    if confirm != name:
        return jsonify(
            {"ok": False, "error": "Confirmation did not match."}
        ), 400

    _stop_active_session_if_class(name)
    trash = Path.home() / ".Trash"
    try:
        trash.mkdir(parents=True, exist_ok=True)
        dest = trash / name
        if dest.exists():
            dest = trash / f"{name} {datetime.now():%Y-%m-%d %H%M%S}"
        shutil.move(str(src), str(dest))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


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
