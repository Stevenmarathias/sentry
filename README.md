# Sentry

A laptop-mounted webcam app that watches lectures in real time and gives AI-powered feedback using Claude.

## What it does
- Captures frames from a USB webcam pointed at a whiteboard
- Auto-detects when the board changes
- Sends frames to Claude (Opus 4.7) for analysis
- Displays concise feedback in three panels: what's on the board, simple explanation, and what to watch out for
- Optionally captures audio and transcribes with OpenAI Whisper

## Setup

```bash
brew install python-tk@3.13 ffmpeg
git clone https://github.com/Stevenmarathias/sentry.git
cd sentry
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python sentry.py
```

## Built with
- Python 3.13 + Tkinter
- OpenCV for camera capture
- Anthropic Claude API for vision analysis
- OpenAI Whisper for audio transcription

## Status
v1.0 — working prototype
