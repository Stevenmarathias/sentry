# Sentry

A study companion that watches your lectures and helps you study for the exam.

Sentry runs in your browser during class. It watches the board or slides through your webcam, transcribes the lecture audio with Whisper, and at the end of each session generates an interactive quiz using Claude. Over the semester it builds a per-class concept map — recurring concepts get tagged as "likely on the exam" and woven into future quizzes.

## What it does

- **Live capture, two modes.** Board mode auto-triggers on motion (chalkboards, whiteboards). Slide mode triggers on perceptual-hash changes (slide decks, projectors) — each distinct slide gets captured once, ignoring brief occlusions like the professor walking past. Pick the mode when starting the session.
- **Continuous audio transcription** via Whisper. Per-class session logs saved to `sessions/<class>/<date>_HHMM.md`.
- **End-of-session quiz.** Generates a structured quiz with three question types: multiple choice (click to check), fill-in-blank (accepts variants), and short answer (graded by Claude with correct/partial/incorrect verdicts). Every question links back to a source timestamp in the transcript.
- **Cross-lecture memory.** A per-class `concepts.json` accumulates named concepts across sessions, weighted by frequency and recency. New quizzes mix today's material with 1–2 recurring concepts from past lectures, flagged with a "FROM PRIOR LECTURE — likely on exam" badge.
- **History and concept browser.** `/history` shows every past session for a class with parsed duration, capture mode, and a link to the quiz. `/class/<name>/concepts` shows the accumulated knowledge graph, sortable by importance, recency, or alphabetical.
- **Quality-of-life.** Pause/resume mid-session (clean transcript gap, paused-time excluded from elapsed clock). Download any quiz as PDF. MCQ choices reshuffle on every reload so you can't memorize positions.

## Setup

```bash
brew install ffmpeg
git clone https://github.com/Stevenmarathias/sentry.git
cd sentry
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
.venv/bin/python sentry_web.py
```

Then open http://127.0.0.1:5000.

## Built with

- Python 3.13 + Flask
- OpenCV (board and slide capture)
- OpenAI Whisper (transcription)
- Anthropic Claude Opus 4.7 (quiz generation, concept extraction, short-answer grading)
- reportlab (PDF export)

## Status

Feature complete for real classroom use. Tested end-to-end with both motion-triggered board capture and perceptual-hash slide capture; cross-lecture memory correctly surfaces recurring concepts in subsequent quizzes.
