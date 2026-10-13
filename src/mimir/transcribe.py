"""Speech-to-text: timestamped transcript segments, for exact words inside audio and video.

Why this exists at all, given that audio embeddings already work: an embedding cannot hold a
code. The eval has a clip that says "ZQ-7734" and writes it nowhere; the audio tower returns
nothing for that query, and only a transcript can. That is the whole case for Whisper.

Whisper's own `load_audio` shells out to the `ffmpeg` binary. We never use it: the audio is
decoded here with PyAV and handed over as a float32 array, so there is still no system ffmpeg.

Known limit, measured: Whisper heard the spoken code "ZQ-7734" as "EQ-7734". Spoken letter
codes are not reliable; the numeric half survives, which is why a keyword match on part of a
query still qualifies. Do not promise exact recall for spoken alphanumerics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TRANSCRIBE, WHISPER_MODEL


def available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe_items(path: Path) -> list["Item"]:  # noqa: F821 - Item imported lazily below
    """Timestamped transcript segments as text items, or [] if transcription is off/unavailable."""
    if not TRANSCRIBE:
        return []
    try:
        import mlx_whisper
    except ImportError:
        return []
    from .extract import Item, decode_audio

    samples, rate = decode_audio(path)
    if samples.size == 0:
        return []
    audio = samples.astype(np.float32) / 32768.0
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL)
    items: list[Item] = []
    for segment in result.get("segments") or []:
        text = (segment.get("text") or "").strip()
        if text:
            items.append(Item("transcript", "text", loc=float(segment["start"]),
                              loc_end=float(segment["end"]), text=text, excerpt=text[:200]))
    return items
