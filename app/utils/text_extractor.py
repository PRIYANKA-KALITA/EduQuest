"""
Text Extractor Module
=====================
Responsibility: Extract clean text from any source type.
This module is completely decoupled from Gemini / AI logic.

Supported sources:
  - YouTube links (transcript API → web scrape fallback)
  - PDF files     (pdfplumber → pypdf fallback)
  - Video files   (audio via ffmpeg → Whisper → frame fallback)
  - Plain text    (pass-through)
"""

import os
import re
import tempfile

# Detect if running on Render cloud (Render sets this env var automatically)
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────

def extract_text(source: str, source_type: str) -> dict:
    """
    Entry point. Returns a dict:
      {
        "text":   str,          # extracted content (may be empty)
        "method": str,          # how it was extracted
        "error":  str | None    # first non-fatal issue encountered
      }
    Raises only on unrecoverable failures.
    """
    source_type = source_type.lower().strip()

    if source_type == "link":
        return _extract_from_link(source)
    elif source_type == "pdf":
        return _extract_from_pdf(source)
    elif source_type == "video":
        return _extract_from_video(source)
    else:
        # Plain text — pass through (capped)
        text = str(source)[:25000]
        return {"text": text, "method": "plain_text", "error": None}


# ─────────────────────────────────────────────────────────────────
# YouTube / Link
# ─────────────────────────────────────────────────────────────────

def _get_youtube_id(url: str):
    patterns = [
        r"shorts/([A-Za-z0-9_-]{11})",
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _extract_from_link(url: str) -> dict:
    video_id = _get_youtube_id(url)

    if video_id:
        # ── Try YouTube transcript API (instant, most accurate) ──
        # Add timeout protection — on Render, cloud IPs can be throttled by YouTube
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            import socket

            # Set a global socket timeout so we never hang more than 10s on YouTube
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                try:
                    transcript = transcript_list.find_transcript(['en', 'hi', 'en-US', 'en-GB', 'en-IN'])
                except Exception:
                    # Fallback: pick first available language
                    transcript = next(iter(transcript_list))

                segments = transcript.fetch()
                text = " ".join(
                    s["text"] if isinstance(s, dict) else s.text
                    for s in segments
                )[:14000]
                print(f"[TextExtractor] YouTube transcript ({transcript.language_code}): {len(text)} chars")
                return {"text": text, "method": "youtube_transcript", "error": None}
            finally:
                socket.setdefaulttimeout(old_timeout)

        except Exception as e:
            print(f"[TextExtractor] Transcript unavailable: {e}")

        # ── Fallback: scrape page title + description ──
        # Use a proper browser UA to avoid bot detection on Render
        try:
            import requests
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            r = requests.get(
                f"https://www.youtube.com/watch?v={video_id}",
                timeout=8,
                headers=headers,
            )
            title = re.search(r"<title>(.*?)</title>", r.text)
            # Try multiple regex patterns for description (YouTube changes their HTML)
            desc = (
                re.search(r'"shortDescription":"(.*?)"', r.text) or
                re.search(r'"description":\{"simpleText":"(.*?)"', r.text)
            )
            title_text = title.group(1).replace("- YouTube", "").strip() if title else ""
            desc_text  = desc.group(1)[:3000] if desc else ""
            # Unescape common escape sequences from YouTube's JSON
            desc_text = desc_text.replace("\\n", " ").replace("\\u0026", "&")

            text = f"Video Title: {title_text}\nDescription: {desc_text}"
            print(f"[TextExtractor] YouTube metadata scraped: {len(text)} chars")
            return {
                "text": text,
                "method": "youtube_metadata_scrape",
                "error": "No transcript available; used metadata only.",
            }
        except Exception as e:
            print(f"[TextExtractor] YouTube scrape failed: {e}")
            # Still return the video_id so the AI can synthesize from the URL context
            return {
                "text": f"Video ID: {video_id} (from YouTube)",
                "method": "youtube_metadata_scrape",
                "error": f"Could not fetch transcript or metadata: {e}",
            }

    # ── Generic URL: scrape body text ──
    try:
        import requests
        from html.parser import HTMLParser

        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self._buf, self._skip = [], False
            def handle_starttag(self, tag, _):
                if tag in ("script", "style", "nav", "footer", "head"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer", "head"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip and data.strip():
                    self._buf.append(data.strip())
            def text(self):
                return " ".join(self._buf)

        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        parser = _Strip()
        parser.feed(r.text)
        text = parser.text()[:25000]
        print(f"[TextExtractor] Generic URL scraped: {len(text)} chars")
        return {"text": text, "method": "generic_url_scrape", "error": None}
    except Exception as e:
        return {"text": "", "method": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────

def _extract_from_pdf(path: str) -> dict:
    # ── pdfplumber (best formatting awareness) ──
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:30]:  # cap at 30 pages
                t = page.extract_text()
                if t:
                    text += t + "\n"
        text = text[:30000]
        if len(text) > 100:
            print(f"[TextExtractor] PDF via pdfplumber: {len(text)} chars")
            return {"text": text, "method": "pdfplumber", "error": None}
    except Exception as e:
        print(f"[TextExtractor] pdfplumber failed: {e}")

    # ── pypdf fallback ──
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = ""
        for page in reader.pages[:30]:
            t = page.extract_text()
            if t:
                text += t + "\n"
        text = text[:30000]
        if len(text) > 100:
            print(f"[TextExtractor] PDF via pypdf: {len(text)} chars")
            return {"text": text, "method": "pypdf", "error": None}
    except Exception as e:
        print(f"[TextExtractor] pypdf failed: {e}")

    return {"text": "", "method": "failed",
            "error": "Could not extract text from PDF."}


# ─────────────────────────────────────────────────────────────────
# Video  (audio → transcript preferred; frames as fallback)
# ─────────────────────────────────────────────────────────────────

def _extract_from_video(path: str) -> dict:
    """
    Strategy:
      1. Extract audio with ffmpeg (if available — skipped on Render)
      2. Transcribe with faster-whisper tiny model (skipped on Render)
      3. Fall back to key-frame images for Llama 4 Vision
    """
    # On Render, ffmpeg is NOT installed and whisper model download would timeout.
    # Skip straight to frame extraction which uses Llama 4 Vision via Groq (fast & cloud-ready).
    if IS_RENDER:
        print("[TextExtractor] Render detected → skipping ffmpeg/whisper, going straight to frame extraction")
    else:
        # ── Step 1 & 2: Audio → Transcript (local only) ──
        transcript = _transcribe_video_audio(path)
        if transcript and len(transcript) > 100:
            print(f"[TextExtractor] Video transcript: {len(transcript)} chars")
            return {"text": transcript, "method": "video_audio_transcription", "error": None}

    # ── Step 3: Frame extraction (vision fallback) ──
    print("[TextExtractor] Falling back to frame extraction for Llama 4 Vision")
    frames = _extract_key_frames(path)
    return {
        "text": "",
        "method": "video_frames",
        "frames": frames,
        "error": "Using visual frame analysis." if frames else "Frame extraction also failed.",
    }


def _transcribe_video_audio(video_path: str):
    """Extract audio with ffmpeg; transcribe with faster-whisper. Local-only."""
    try:
        import subprocess, shutil

        if not shutil.which("ffmpeg"):
            print("[TextExtractor] ffmpeg not found on PATH.")
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"[TextExtractor] ffmpeg error: {result.stderr[:200]}")
            os.unlink(audio_path)
            return None

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_path, beam_size=1)
            transcript = " ".join(seg.text for seg in segments)
            return transcript[:30000]
        except ImportError:
            print("[TextExtractor] faster-whisper not installed.")
            return None
        finally:
            try: os.unlink(audio_path)
            except: pass

    except Exception as e:
        print(f"[TextExtractor] Audio transcription error: {e}")
        return None


def _extract_key_frames(video_path: str, num_frames: int = 6) -> list:
    """Extract small JPEG frames as base64 inline_data dicts for Llama 4 Vision."""
    import base64
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[TextExtractor] cv2 could not open video: {video_path}")
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1000

        frames = []
        for i in range(num_frames):
            pos = int(total * (i + 1) / (num_frames + 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret: break
                if not ret: continue

            h, w = frame.shape[:2]
            new_h = 480
            frame = cv2.resize(frame, (int(w * new_h / h), new_h))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frames.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.tobytes()).decode(),
                }
            })
        cap.release()
        print(f"[TextExtractor] Extracted {len(frames)} video frames")
        return frames
    except Exception as e:
        print(f"[TextExtractor] Frame extraction failed: {e}")
        return []
