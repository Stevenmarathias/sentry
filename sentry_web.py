"""Sentry (web) — real-time lecture feedback in the browser.

A Flask rebuild of Sentry v1.0. The backend owns the camera, runs board-change
detection, and calls Claude (Opus 4.7) for the 3-panel structured feedback. The
browser is a pure view: it pulls the live preview as an MJPEG stream and
receives meter / status / feedback updates over Server-Sent Events.

The camera, detector, and analyzer logic are reused from sentry.py (v1.0) —
only the Tkinter UI is replaced. Run this instead of sentry.py; sentry.py is
left untouched.

    ANTHROPIC_API_KEY=... .venv/bin/python sentry_web.py
    # then open http://127.0.0.1:5000
"""
import json
import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# Reuse v1.0's camera / detection / analysis logic. sentry.py only launches the
# Tk UI under `if __name__ == "__main__"`, so importing it here is side-effect
# free (no window is created).
from sentry import (
    CAMERA_INDEX,
    COOLDOWN_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    Analyzer,
    ChangeDetector,
)


# ---- Configuration -----------------------------------------------------------

PREVIEW_WIDTH = 640          # downscale width for the MJPEG preview stream
PREVIEW_FPS = 15
METER_INTERVAL = 0.25        # seconds between motion-meter broadcasts
SSE_KEEPALIVE = 15.0         # seconds idle before an SSE keepalive comment


# ---- Camera worker -----------------------------------------------------------

class CameraWorker:
    """Owns the camera and all background work; broadcasts events to browsers.

    A single instance runs for the life of the process. The capture thread
    reads frames, runs change detection, and auto-triggers analysis. Analysis
    runs on its own short-lived thread so the capture loop never stalls.
    Browser tabs subscribe via `subscribe()` and receive dict events.
    """

    def __init__(self):
        self.detector = ChangeDetector()
        self.analyzer = Analyzer()

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._last_analysis_time = 0.0
        self._analyzing = False
        self._last_meter_emit = 0.0

        self._stop_evt = threading.Event()
        self._started = False
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._started = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    # ---- pub/sub ------------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event: dict) -> None:
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Slow client — drop the event rather than blocking capture.
                pass

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
                self._broadcast({
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
        self._broadcast({"type": "status", "text": "Analyzing…"})
        threading.Thread(
            target=self._run_analysis, args=(frame,), daemon=True
        ).start()

    def _run_analysis(self, frame: np.ndarray) -> None:
        try:
            result = self.analyzer.analyze(frame)
            self._broadcast({"type": "feedback", "data": result})
            self._broadcast({"type": "status", "text": "Watching the board…"})
        except Exception as exc:
            self._broadcast({"type": "error", "text": str(exc)})
            self._broadcast({"type": "status", "text": f"Error: {exc}"})
        finally:
            self._analyzing = False


# ---- Flask app ---------------------------------------------------------------

app = Flask(__name__)
worker = CameraWorker()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the live camera preview."""
    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        frame_interval = 1.0 / PREVIEW_FPS
        while True:
            jpeg = worker.get_preview_jpeg()
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
    """Server-Sent Events: meter, status, feedback, and error updates."""
    def stream():
        q = worker.subscribe()
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
            worker.unsubscribe(q)

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
    accepted, message = worker.trigger_analysis()
    return jsonify({"accepted": accepted, "message": message})


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---- Entry point -------------------------------------------------------------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY is not set; API calls will fail.")
    worker.start()
    print("Sentry web UI: http://127.0.0.1:5000")
    # use_reloader=False: the reloader spawns a second process that would fight
    # over the camera. threaded=True: /video_feed and /events are long-lived.
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
