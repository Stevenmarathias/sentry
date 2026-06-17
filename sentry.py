"""Sentry — real-time lecture feedback from a laptop webcam.

Captures frames from an outward-facing USB webcam, auto-detects when the
whiteboard changes, sends the settled frame to Claude (Opus 4.7) for analysis,
and renders structured feedback in a floating Tkinter UI. Audio capture +
Whisper transcription is optional and gets attached to the next analysis.
"""
# Pass D1.1: lazy annotations so type hints referencing optional libraries
# (tk.Text, sd.InputStream, cv2.VideoCapture, …) stay as strings and never
# evaluate at class/def time. Lets sentry_web import this module on a
# hardware-less host where those libs failed to load.
from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from typing import List, Optional

import anthropic
import numpy as np

# Pass D1.1: hardware-dependent and UI-only imports are guarded so this
# module can still be *imported* on a server with no PortAudio, no OpenCV
# system libs, no Tk, no Whisper install. sentry_web.py only needs a handful
# of constants and a few classes (Analyzer, ChangeDetector, Transcriber)
# from here, and CameraWorker / AudioWorker in sentry_web.py are kept route-
# gated so the missing libs are never *used* on a hardware-less host.
# Running sentry.py itself (the Tk live-feedback UI) still requires all of
# these; main() will error clearly if they're missing.
try:
    import cv2
    _HAS_CV2 = True
except (OSError, ImportError):
    cv2 = None
    _HAS_CV2 = False

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except (OSError, ImportError):
    sd = None
    _HAS_SOUNDDEVICE = False

try:
    import whisper
    _HAS_WHISPER = True
except (OSError, ImportError):
    whisper = None
    _HAS_WHISPER = False

try:
    import tkinter as tk
    from tkinter import ttk
    _HAS_TK = True
except (OSError, ImportError):
    tk = None
    ttk = None
    _HAS_TK = False

try:
    from PIL import Image, ImageTk
    _HAS_IMAGETK = True
except (OSError, ImportError):
    Image = None
    ImageTk = None
    _HAS_IMAGETK = False


# ---- Configuration -----------------------------------------------------------

CAMERA_INDEX = int(os.environ.get("SENTRY_CAMERA_INDEX", "0"))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CHANGE_THRESHOLD = float(os.environ.get("SENTRY_CHANGE_THRESHOLD", "2.0"))
STABILITY_FRAMES = 30        # consecutive low-diff frames to confirm settled
COOLDOWN_SECONDS = 8.0       # min seconds between auto-triggered analyses
AUDIO_SAMPLE_RATE = 16000
MODEL_ID = "claude-sonnet-4-6"
WHISPER_MODEL = os.environ.get("SENTRY_WHISPER_MODEL", "small")
MAX_IMAGE_DIM = 1568         # downscale longest edge before sending


# JSON schema for Claude's structured response.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "board_content": {
            "type": "string",
            "description": "A short summary orienting the student - the lecture topic and a couple of key items, not a full transcription of every equation. One sentence, under 25 words.",
        },
        "explanation": {
            "type": "string",
            "description": "A simple explanation of the concept, in 1-2 sentences a student can grasp at a glance.",
        },
        "watch_out_for": {
            "type": "string",
            "description": "One common mistake or subtle point to watch out for, in 1-2 sentences.",
        },
    },
    "required": ["board_content", "explanation", "watch_out_for"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = (
    "You are Sentry, a real-time lecture assistant. The image is from a laptop "
    "webcam pointing outward at a professor and their whiteboard. Your job is "
    "to help a student follow along. Be concise, accurate, and focused on what "
    "is actually on the board right now. If audio context from the professor "
    "is provided, use it to disambiguate, but do not invent content that is "
    "not visible or stated. Keep every field tight and scannable: "
    "board_content, explanation, and watch_out_for are each at most 1-2 "
    "sentences. Prefer omitting secondary nuance over exceeding the limit."
)


# ---- Helpers -----------------------------------------------------------------

def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Linear-interpolation resample of mono float audio to 16 kHz."""
    if sample_rate == 16000:
        return audio.astype(np.float32, copy=False)
    target_len = int(round(audio.shape[0] * 16000 / float(sample_rate)))
    if target_len <= 0:
        return audio.astype(np.float32, copy=False)
    src_idx = np.linspace(0.0, audio.shape[0] - 1, target_len)
    resampled = np.interp(src_idx, np.arange(audio.shape[0]), audio)
    return resampled.astype(np.float32)


def encode_frame_jpeg_b64(frame: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Downscale (if needed) and JPEG-encode a BGR frame as base64 ASCII."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


# ---- Board change detection --------------------------------------------------

class ChangeDetector:
    """Fires once when the board transitions from changing -> settled."""

    def __init__(self,
                 threshold: float = CHANGE_THRESHOLD,
                 stability_frames: int = STABILITY_FRAMES):
        self.threshold = threshold
        self.stability_frames = stability_frames
        self.last_diff = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._stable_count = 0
        self._was_changing = False

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (15, 15), 0)

    def update(self, frame: np.ndarray) -> bool:
        gray = self._preprocess(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        mean_diff = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
        self.last_diff = mean_diff
        self._prev_gray = gray

        if mean_diff > self.threshold:
            self._was_changing = True
            self._stable_count = 0
            return False

        if self._was_changing:
            self._stable_count += 1
            if self._stable_count >= self.stability_frames:
                self._was_changing = False
                self._stable_count = 0
                return True
        return False


# ---- Claude analyzer ---------------------------------------------------------

class Analyzer:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze(self, frame: np.ndarray, transcript: Optional[str] = None) -> dict:
        b64 = encode_frame_jpeg_b64(frame)
        prompt = (
            "Analyze the current state of the board in this lecture frame. "
            "Produce structured feedback."
        )
        if transcript:
            prompt += f"\n\nProfessor audio (transcribed):\n{transcript}"

        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            },
            {"type": "text", "text": prompt},
        ]

        message = self.client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
            },
        )
        text = next(b.text for b in message.content if b.type == "text")
        return json.loads(text)


# ---- Audio recording + transcription -----------------------------------------

class AudioRecorder:
    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._chunks: List[np.ndarray] = []
        # Pass D1.1: annotation dropped (was Optional[sd.InputStream]) so this
        # class body still executes when sounddevice failed to import.
        self._stream = None
        self._lock = threading.Lock()
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self) -> None:
        if self.recording:
            return
        with self._lock:
            self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.recording = True

    def stop(self) -> Optional[np.ndarray]:
        if not self.recording:
            return None
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self.recording = False
        with self._lock:
            if not self._chunks:
                return None
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []
        return audio


class Transcriber:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = whisper.load_model(self._model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self._ensure_loaded()
        audio = resample_to_16k(audio, sample_rate)
        result = self._model.transcribe(audio, fp16=False)
        return result["text"].strip()


# ---- Tkinter UI --------------------------------------------------------------

class SentryApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sentry")
        self.root.attributes("-topmost", True)
        self.root.geometry("440x640")

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.detector = ChangeDetector()
        self.analyzer = Analyzer()
        self.audio = AudioRecorder()
        self.transcriber = Transcriber()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._last_analysis_time = 0.0
        self._analyzing = False
        self._stop_evt = threading.Event()
        self._settle_evt = threading.Event()
        self._last_preview = 0.0
        self._last_meter = 0.0
        self._result_queue: "queue.Queue[dict]" = queue.Queue()

        self._build_ui()

        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

        self.root.after(33, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        self.preview = tk.Label(self.root, bg="black")
        self.preview.pack(fill="x", **pad)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        self.analyze_btn = ttk.Button(
            btn_frame, text="Analyze Now", command=self._analyze_now)
        self.analyze_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.audio_btn = ttk.Button(
            btn_frame, text="+ Audio", command=self._toggle_audio)
        self.audio_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.clear_btn = ttk.Button(
            btn_frame, text="Clear", command=self._clear)
        self.clear_btn.pack(side="left", expand=True, fill="x", padx=2)

        self.status = tk.Label(
            self.root, text="Watching the board…", anchor="w", fg="#888")
        self.status.pack(fill="x", **pad)

        self.meter = tk.Label(
            self.root, text="", anchor="w", fg="#aaa",
            font=("TkDefaultFont", 9))
        self.meter.pack(fill="x", padx=8)

        self.board_text = self._make_section("On the board")
        self.expl_text = self._make_section("Simple explanation")
        self.warn_text = self._make_section("Watch out for")

    def _make_section(self, title: str) -> tk.Text:
        frame = tk.LabelFrame(self.root, text=title, padx=6, pady=4)
        frame.pack(fill="both", expand=True, padx=8, pady=4)
        widget = tk.Text(frame, height=4, wrap="word", bd=0, relief="flat")
        widget.pack(fill="both", expand=True)
        widget.configure(state="disabled")
        return widget

    # ---- Capture / change detection loop -----------------------------------

    def _capture_loop(self) -> None:
        """Camera read + change detection only. Never touches Tk."""
        while not self._stop_evt.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            if self.detector.update(frame):
                self._settle_evt.set()

            time.sleep(0.03)

    def _tick(self) -> None:
        """Main-thread UI pump: preview, meter, auto-trigger, results.

        All Tk access happens here. The capture thread only writes shared
        state — cross-thread Tk calls hang or fail to render on macOS.
        """
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        now = time.time()
        if frame is not None:
            if now - self._last_preview > 0.1:
                self._last_preview = now
                self._update_preview(frame)
            if now - self._last_meter > 0.25:
                self._last_meter = now
                self._update_meter(self.detector.last_diff)

        if self._settle_evt.is_set():
            self._settle_evt.clear()
            if (frame is not None
                    and not self._analyzing
                    and now - self._last_analysis_time > COOLDOWN_SECONDS):
                self._launch_analysis(frame)

        try:
            while True:
                self._handle_result(self._result_queue.get_nowait())
        except queue.Empty:
            pass

        self.root.after(33, self._tick)

    def _update_meter(self, diff: float) -> None:
        armed = "armed" if self.detector._was_changing else "idle"
        self.meter.configure(
            text=f"motion: {diff:.2f}   threshold: {self.detector.threshold:.2f}"
                 f"   [{armed}]"
        )

    def _update_preview(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        target_w = 420
        scale = target_w / w
        small = cv2.resize(frame, (target_w, int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=img)
        self.preview.image = img  # prevent GC

    # ---- Analysis lifecycle ------------------------------------------------

    def _analyze_now(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            self._set_status("No frame yet…")
            return
        self._launch_analysis(frame)

    def _launch_analysis(self, frame: np.ndarray) -> None:
        if self._analyzing:
            return
        self._analyzing = True
        self._last_analysis_time = time.time()
        self._set_status("Analyzing…")

        # Stop the stream on the main thread (quick, touches the button),
        # but hand transcription to the worker so the UI stays responsive.
        audio_data: Optional[np.ndarray] = None
        if self.audio.recording:
            audio_data = self.audio.stop()
            self.audio_btn.configure(text="+ Audio")

        threading.Thread(
            target=self._run_analysis,
            args=(frame, audio_data),
            daemon=True,
        ).start()

    def _run_analysis(self, frame: np.ndarray,
                      audio_data: Optional[np.ndarray]) -> None:
        try:
            transcript: Optional[str] = None
            if audio_data is not None and audio_data.size > 0:
                transcript = self.transcriber.transcribe(
                    audio_data, AUDIO_SAMPLE_RATE)
            result = self.analyzer.analyze(frame, transcript)
            self._result_queue.put({"ok": True, "data": result})
        except Exception as exc:
            self._result_queue.put({"ok": False, "error": str(exc)})

    def _handle_result(self, msg: dict) -> None:
        self._analyzing = False
        if msg["ok"]:
            data = msg["data"]
            self._set_text(self.board_text, data.get("board_content", ""))
            self._set_text(self.expl_text, data.get("explanation", ""))
            self._set_text(self.warn_text, data.get("watch_out_for", ""))
            self._set_status("Watching the board…")
        else:
            self._set_status(f"Error: {msg['error']}")

    # ---- UI mutators -------------------------------------------------------

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _toggle_audio(self) -> None:
        if self.audio.recording:
            self.audio.stop()
            self.audio_btn.configure(text="+ Audio")
            self._set_status("Audio off.")
        else:
            try:
                self.audio.start()
                self.audio_btn.configure(text="Stop Audio")
                self._set_status("Recording audio…")
            except Exception as exc:
                self._set_status(f"Audio error: {exc}")

    def _clear(self) -> None:
        for w in (self.board_text, self.expl_text, self.warn_text):
            self._set_text(w, "")
        self._set_status("Cleared.")

    def _on_close(self) -> None:
        self._stop_evt.set()
        try:
            if self.audio.recording:
                self.audio.stop()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---- Entry point -------------------------------------------------------------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY is not set; API calls will fail.")
    SentryApp().run()


if __name__ == "__main__":
    main()
