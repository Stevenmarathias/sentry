"""Pass D2.2 diagnostic — run on the Render shell to see which YouTube
transcript-fetch path (if any) works from the deployed server's IP.

YouTube blocks datacenter IPs aggressively. yt-dlp already failed on
Render in deploy logs; this script confirms whether youtube-transcript-api
also fails, whether yt-dlp can at least reach the metadata endpoint, and
which (if any) of the three call shapes used by the app works in
production. Read-only and free — no Anthropic / paid API calls.

Usage on the Render shell:
    python yt_diagnostic.py

Throwaway. Safe to delete after we have the results.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path


# Three well-known captioned videos from different uploaders so a single
# video being de-listed / age-gated doesn't poison the signal. Each has
# robust human-uploaded captions in production.
TEST_VIDEOS = [
    ("RcGyVTAoXEU", "Kelly McGonigal — How to make stress your friend (TED)"),
    ("lBx86QE4k3s", "Ann Patchett — The Love of My Life (TED)"),
    ("UF8uR6Z6KLc", "Steve Jobs — 2005 Stanford Commencement Address"),
]

# Limit snippet length so a multi-paragraph success doesn't drown the table.
SNIPPET_CHARS = 120


def _trim(s: str, n: int = SNIPPET_CHARS) -> str:
    """One-line preview for the result table — collapse whitespace, clip."""
    flat = re.sub(r"\s+", " ", (s or "").strip())
    return flat[:n] + ("…" if len(flat) > n else "")


def _short_exc() -> str:
    """Last exception's `Type: message` — one line, no traceback."""
    et, ev, _ = sys.exc_info()
    name = et.__name__ if et else "?"
    msg = str(ev) if ev else ""
    return _trim(f"{name}: {msg}", 200)


# ---- Method A: youtube-transcript-api -------------------------------------
#
# Matches the primary path in sentry_web.fetch_captions — newer instance API:
#     api = YouTubeTranscriptApi(); fetched = api.fetch(video_id)
# A success means the app's no-cost caption path works on this host.

def method_a_transcript_api(video_id: str) -> tuple[bool, str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return False, "youtube-transcript-api not installed: " + _short_exc()
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        text = " ".join(
            s.text for s in fetched if getattr(s, "text", "")
        ).strip()
        if not text:
            return False, "empty transcript returned"
        return True, _trim(text)
    except Exception:
        return False, _short_exc()


# ---- Method B: yt-dlp subtitle download -----------------------------------
#
# Mirrors the VTT fallback in sentry_web.fetch_captions — writesubtitles +
# writeautomaticsub + skip_download + extract_info(download=True). Writes
# any .vtt files into a TemporaryDirectory and reports whether anything
# landed on disk.

def method_b_ytdlp_subs(video_id: str) -> tuple[bool, str]:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False, "yt-dlp not installed: " + _short_exc()
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with tempfile.TemporaryDirectory(prefix="yt-diag-subs-") as tmp:
            import yt_dlp
            opts = {
                "quiet": True, "no_warnings": True, "noprogress": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            vtts = sorted(Path(tmp).glob("*.vtt"))
            if not vtts:
                return False, "no .vtt files written"
            # Light-touch text extraction — enough to show captions actually
            # arrived; we're not reusing the app's parser here.
            text_lines: list[str] = []
            for raw in vtts[0].read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if (not line or line.startswith("WEBVTT") or "-->" in line
                        or line.isdigit()):
                    continue
                text_lines.append(re.sub(r"<[^>]+>", "", line))
                if sum(len(s) for s in text_lines) > 400:
                    break
            text = " ".join(text_lines).strip()
            if not text:
                return False, f"wrote {len(vtts)} .vtt but parsed to empty"
            return True, _trim(text) + f"  [{len(vtts)} vtt file(s)]"
    except Exception:
        return False, _short_exc()


# ---- Method C: yt-dlp metadata only ---------------------------------------
#
# Matches sentry_web.fetch_video_metadata — extract_info(url, download=False).
# Tells us whether yt-dlp can reach YouTube at all from this host. If even
# this fails, captioned-or-not is moot until network access is restored.

def method_c_ytdlp_metadata(video_id: str) -> tuple[bool, str]:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False, "yt-dlp not installed: " + _short_exc()
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = (info or {}).get("title") or ""
        duration = (info or {}).get("duration") or 0
        if not title and not duration:
            return False, "extract_info returned no title/duration"
        return True, _trim(f"title={title!r} duration={duration}s")
    except Exception:
        return False, _short_exc()


METHODS = [
    ("A: youtube-transcript-api",      method_a_transcript_api),
    ("B: yt-dlp subs (VTT)",           method_b_ytdlp_subs),
    ("C: yt-dlp metadata only",        method_c_ytdlp_metadata),
]


def main() -> None:
    # Show the resolved lib versions so a "doesn't work" reading is anchored
    # to the actual environment — not the docs.
    print("=== Pass D2.2 YouTube transcript diagnostic ===")
    print(f"python: {sys.version.split()[0]}  cwd: {os.getcwd()}")
    for modname in ("youtube_transcript_api", "yt_dlp"):
        try:
            mod = __import__(modname)
            ver = getattr(mod, "__version__", "(no __version__)")
            print(f"{modname}: {ver}")
        except Exception:
            print(f"{modname}: NOT IMPORTABLE — {_short_exc()}")
    print()

    # Per-attempt log + tally per method.
    results: dict[str, list[bool]] = {label: [] for label, _ in METHODS}
    width = max(len(label) for label, _ in METHODS)

    for vid, label_v in TEST_VIDEOS:
        print(f"--- {vid}  ({label_v}) ---")
        for label_m, fn in METHODS:
            try:
                ok, snippet = fn(vid)
            except Exception:
                # Defensive — should be unreachable since each method
                # already catches Exception internally.
                ok, snippet = False, "outer guard: " + _short_exc()
                traceback.print_exc()
            results[label_m].append(ok)
            tag = "OK  " if ok else "FAIL"
            print(f"  {label_m.ljust(width)}  {tag}  {snippet}")
        print()

    # Summary table.
    print("=== Summary ===")
    total = len(TEST_VIDEOS)
    for label_m, _ in METHODS:
        hits = sum(results[label_m])
        print(f"  {label_m.ljust(width)}  {hits}/{total} succeeded")

    # Bottom-line guidance for the user reading the shell output.
    print()
    a_ok = sum(results["A: youtube-transcript-api"])
    b_ok = sum(results["B: yt-dlp subs (VTT)"])
    c_ok = sum(results["C: yt-dlp metadata only"])
    if a_ok:
        print("✓ youtube-transcript-api works from this host — the app's "
              "primary captions path should work.")
    elif b_ok:
        print("✗ youtube-transcript-api is blocked, but yt-dlp can fetch "
              "subs — fall back path should work.")
    elif c_ok:
        print("✗ Both caption paths are blocked, but yt-dlp can reach "
              "YouTube for metadata — the block is targeted at captions.")
    else:
        print("✗ Everything failed — YouTube is fully blocking this host's "
              "IP; transcript fetching requires a proxy or alternate source.")


if __name__ == "__main__":
    main()
