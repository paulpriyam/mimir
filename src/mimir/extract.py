"""Turn a file into items that can be embedded and searched.

An `Item` is one thing that gets a vector and/or keyword text:
  text items   (doc chunks, PDF pages, OCR text, transcripts) -> the text tower
  image items  (a picture, a video frame)                     -> the vision tower
  audio items  (a 30 s window of sound)                       -> the audio tower

Every item carries `loc`/`loc_end`: a PDF page number, or seconds into audio/video, so a hit
can open at the right place. Media items point at a real file on disk (a frame written to a
temp dir, an audio window written as wav), which is what the worker's image/audio ops want.

Native macOS only, on purpose: Apple Vision for OCR, PDFKit for page text, PyAV (bundled
ffmpeg) for decoding. No Homebrew, no system ffmpeg.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif", ".bmp", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aiff", ".aif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".ts"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv"}

AUDIO_WINDOW = 30.0   # the model listens to 30 s at a time
AUDIO_HOP = 25.0      # 5 s overlap, so a word on a boundary is still whole in one window
VIDEO_EVERY = 4.0     # a frame every 4 s, plus the soundtrack as audio windows
SAMPLE_RATE = 16000
CHUNK = 800
CHUNK_OVERLAP = 100


@dataclass
class Item:
    kind: str                  # image | ocr | chunk | page | audio | frame
    modality: str              # text | image | audio
    loc: float | None = None
    loc_end: float | None = None
    text: str | None = None
    media: str | None = None
    media_kind: str | None = None
    excerpt: str | None = None


def kind_of(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix == ".pdf":
        return "pdf"
    if suffix in TEXT_SUFFIXES:
        return "doc"
    return None


# --- OCR (Apple Vision) --------------------------------------------------------------

def ocr(path: Path) -> str:
    """The text Apple Vision finds in an image. Empty string when there is none."""
    import Vision
    from Foundation import NSURL

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)          # accurate
    request.setUsesLanguageCorrection_(True)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"vision OCR failed: {error}")
    lines = []
    for observation in (request.results() or []):
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append(candidates[0].string())
    return "\n".join(lines)


# --- Audio and video decoding (PyAV) -------------------------------------------------

def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """The file's audio as mono int16 at 16 kHz."""
    import av

    container = av.open(str(path))
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise RuntimeError("no audio stream")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pieces: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            pieces.append(resampled.to_ndarray().reshape(-1))
    for resampled in resampler.resample(None):
        pieces.append(resampled.to_ndarray().reshape(-1))
    samples = np.concatenate(pieces) if pieces else np.zeros(0, np.int16)
    return samples.astype(np.int16, copy=False), SAMPLE_RATE


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype(np.int16, copy=False).tobytes())


def audio_items(path: Path, workdir: Path) -> list[Item]:
    """Overlapping 30 s windows, each written as its own wav so it can be embedded alone."""
    samples, rate = decode_audio(path)
    if samples.size == 0:
        return []
    window, hop = int(AUDIO_WINDOW * rate), int(AUDIO_HOP * rate)
    items: list[Item] = []
    for index, start in enumerate(range(0, samples.size, hop)):
        piece = samples[start:start + window]
        if piece.size < rate:            # under a second: not worth a vector
            break
        out = workdir / f"{path.stem}-{index:04d}.wav"
        write_wav(out, piece, rate)
        items.append(Item("audio", "audio", loc=start / rate, loc_end=(start + piece.size) / rate,
                          media=str(out), media_kind="audio"))
        if start + window >= samples.size:
            break
    return items


def video_items(path: Path, workdir: Path) -> list[Item]:
    """A frame every few seconds, plus the soundtrack as audio windows."""
    import av
    from PIL import Image

    items: list[Item] = []
    try:
        items += audio_items(path, workdir)
    except Exception:
        pass                          # a silent clip is normal

    container = av.open(str(path))
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        return items
    next_target = 0.0
    for frame in container.decode(stream):
        seconds = float(frame.time or 0.0)
        if seconds + 1e-6 < next_target:
            continue
        image: Image.Image = frame.to_image()
        out = workdir / f"{path.stem}-{int(seconds):05d}.jpg"
        image.convert("RGB").save(out, "JPEG", quality=90)
        items.append(Item("frame", "image", loc=seconds, loc_end=seconds + VIDEO_EVERY,
                          media=str(out), media_kind="image"))
        next_target = seconds + VIDEO_EVERY
    return items


# --- Documents ----------------------------------------------------------------------

def pdf_items(path: Path) -> list[Item]:
    """One item per page, carrying the page number as its place."""
    from Foundation import NSURL
    from Quartz import PDFDocument

    document = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(path)))
    items: list[Item] = []
    for index in range(document.pageCount()):
        text = (document.pageAtIndex_(index).string() or "").strip()
        if text:
            items.append(Item("page", "text", loc=float(index + 1), text=text, excerpt=text[:200]))
    return items


def text_items(path: Path) -> list[Item]:
    text = path.read_text(errors="ignore").strip()
    items: list[Item] = []
    for start in range(0, len(text), max(CHUNK - CHUNK_OVERLAP, 1)):
        piece = text[start:start + CHUNK].strip()
        if piece:
            items.append(Item("chunk", "text", loc=float(start), text=piece, excerpt=piece[:200]))
    return items


# --- Dispatch -----------------------------------------------------------------------

def transcripts(path: Path) -> list[Item]:
    """Transcript segments for audio/video, if the stt extra is installed and enabled."""
    from .transcribe import transcribe_items

    try:
        return transcribe_items(path)
    except Exception as error:
        # A silent clip is normal, not a failure: only real errors are worth reporting.
        if "no audio stream" not in str(error):
            print(f"  ! transcript failed for {path.name}: {type(error).__name__}: {error}")
        return []


def to_jpeg(path: Path, workdir: Path) -> Path:
    """The vision tower reads jpeg/png; anything else is converted with sips first."""
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return path
    out = workdir / f"{path.stem}.jpg"
    subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def extract(path: Path, kind: str, workdir: Path) -> list[Item]:
    if kind == "image":
        items = [Item("image", "image", media=str(to_jpeg(path, workdir)), media_kind="image")]
        found = ocr(path).strip()
        if found:
            items.append(Item("ocr", "text", text=found, excerpt=found[:200]))
        return items
    if kind == "audio":
        return audio_items(path, workdir) + transcripts(path)
    if kind == "video":
        return video_items(path, workdir) + transcripts(path)
    if kind == "pdf":
        return pdf_items(path)
    if kind == "doc":
        return text_items(path)
    return []
