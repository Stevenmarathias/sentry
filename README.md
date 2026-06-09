# Sentry

A local Flask web app that turns lectures into study material. Capture a
live lecture through your webcam and microphone, **or** import a YouTube
video by link. Sentry transcribes the lecture (Whisper for live audio,
YouTube captions when available, audio + Whisper when they aren't), then
uses Claude to extract concepts, build a per-class memory across the
whole semester, generate quizzes, write per-concept explanations, and
compose a practice exam over everything you've covered.

It runs on `http://127.0.0.1:5000`. Everything is stored on disk under
`sessions/<class>/` — no cloud account, no database.

## What it does

- **Live capture, two modes.** Board mode auto-triggers on motion
  (whiteboards, chalkboards). Slide mode triggers on perceptual-hash
  changes (slide decks, projectors) so each distinct slide is captured
  once and brief occlusions like the professor walking past are
  ignored. You can flip modes mid-session.
- **YouTube import.** Paste a link on a class's overview page. Sentry
  tries captions first; if there are none, it downloads the audio with
  yt-dlp and transcribes it locally with Whisper. The import runs as a
  background job and surfaces live stage updates so the page doesn't
  freeze on long videos.
- **Per-class concept memory.** A `concepts.json` per class
  accumulates named concepts across every session (live + imported),
  weighted by frequency and recency. New quizzes mix today's material
  with 1–2 recurring concepts from past lectures, flagged with a
  "FROM PRIOR LECTURE — likely on exam" badge.
- **Quizzes** in three places: at the end of a live session, on demand
  for any single past lecture from History ("Quiz this lecture"), and
  automatically after every YouTube import. Three question types per
  quiz (MCQ, fill-in-blank, short answer); MCQ choices reshuffle on
  every reload. Every question links back to a source timestamp.
- **Semester practice exam.** `/class/<name>/exam` composes a
  20-question exam over the most important concepts across every
  session in the class.
- **Per-concept "in depth" explanations.** Click any concept on
  `/class/<name>/concepts` for a Claude-written explanation grounded
  in the lecture's brief definitions of it.
- **Customizable per-class color.** Each class has an accent color
  (auto-derived from the class name; pick your own from the overview
  page). It themes the landing-card stripe and the class's overview
  page.
- **Quality of life.** Pause / resume mid-session with a clean
  transcript gap. Download any quiz as PDF. Sessions, concepts,
  history, quizzes, exams all persist on disk.

## Requirements

- **Python 3.13** (this is what the project is developed and tested on).
- **System dependency: ffmpeg** — required for YouTube audio extraction.
  Install on macOS with `brew install ffmpeg`.
- **An Anthropic API key.** Quiz / exam / concept-explanation /
  short-answer grading are all Claude API calls — each one bills your
  Anthropic account. The server warns at startup if `ANTHROPIC_API_KEY`
  is unset and proceeds (so non-LLM routes still work), but any feature
  that talks to Claude will fail until you set it.
- **The Whisper "small" model.** Downloads automatically on first use
  (cached locally). You can override the model name with the
  `SENTRY_WHISPER_MODEL` environment variable if you want a different
  size.

Python packages (in `requirements.txt`):

- `anthropic` — Claude API client
- `flask` — web framework
- `opencv-python`, `pillow` — camera + image handling
- `openai-whisper` — local speech-to-text
- `sounddevice`, `numpy` — microphone capture
- `reportlab` — PDF export
- `yt-dlp` *(added in Pass 14)* — YouTube downloader. Intentionally
  unpinned because YouTube changes its frontend often; `pip install -U
  yt-dlp` whenever an import suddenly stops working.
- `youtube-transcript-api` *(added in Pass 14)* — primary caption path.

## Setup

```bash
brew install ffmpeg                              # macOS system dep
git clone https://github.com/Stevenmarathias/sentry.git
cd sentry
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

> Use `.venv/bin/python -m pip …` for installs — the `.venv/bin/pip`
> shebang is stale and will install into the wrong interpreter.

## Running it

```bash
cd ~/Desktop/sentry
.venv/bin/python sentry_web.py
```

Then open <http://127.0.0.1:5000>.

The server keeps running until you Ctrl+C it.

## Development notes / gotchas

A few things worth knowing if you're going to edit and rerun.

- **No auto-reload.** The server runs with `use_reloader=False` on
  purpose — the Flask reloader spawns a second process and the two
  would fight over the camera. After editing Python, templates,
  CSS, or JS, stop the server (Ctrl+C) and relaunch it. For CSS /
  JS changes, also hard-refresh the browser (Cmd+Shift+R on macOS)
  so the cached old asset doesn't stick.

- **macOS AirPlay Receiver squats on port 5000.** If the server
  fails to bind with `Address already in use`, disable AirPlay
  Receiver: System Settings → General → AirDrop & Handoff → turn
  off "AirPlay Receiver". (This is also exactly what the startup
  error message tells you to do.)

- **`yt-dlp` may need updates.** YouTube changes its frontend
  regularly and yt-dlp ships fixes within days/weeks. If a previously
  working import starts failing with extractor errors, update it:

  ```bash
  .venv/bin/python -m pip install -U yt-dlp
  ```

- **System audio during live capture.** If you want to capture a live
  *online* lecture (Zoom, a stream, a video call) with the live
  capture flow, you need a virtual audio device like
  [BlackHole](https://github.com/ExistentialAudio/BlackHole) routed
  through a macOS Multi-Output Device so the mic input sees both your
  microphone and the system audio. **This is largely unnecessary now**
  — for YouTube content, paste the link into the Import card on a
  class's overview page instead.

- **Sessions live on disk.** Everything Sentry knows about a class is
  under `sessions/<class>/`: one markdown file per session, plus
  `concepts.json` and `meta.json`. Delete the folder to delete the
  class. Renaming / deleting from the UI moves the folder cleanly.
  The `sessions/` directory is in `.gitignore` — none of your
  recorded material is committed.

## Built with

- Python 3.13 + Flask
- OpenCV (board capture, slide perceptual hashing)
- OpenAI Whisper (local transcription)
- Anthropic Claude Opus 4.7 (quiz / exam / concept extraction / short
  answer grading / per-concept explanations)
- yt-dlp + youtube-transcript-api (YouTube import)
- reportlab (quiz PDF export)
