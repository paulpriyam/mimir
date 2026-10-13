"""Speech-to-text: timestamped transcript segments, for exact words inside audio and video.

Why this exists at all, given that audio embeddings already work: an embedding cannot hold a
code. The eval has a clip that says "ZQ-7734" and writes it nowhere; the audio tower returns
nothing for that query, and only a transcript can. That is the whole case for Whisper.

Whisper's own `load_audio` shells out to the `ffmpeg` binary. We never use it: the audio is
decoded with PyAV and handed over as a float32 array, so there is still no system ffmpeg.

It runs in a subprocess (whisper_worker.py) rather than in-process, because mlx_whisper caches
its model in a class attribute and would otherwise stay resident across the later indexing
phases - which is what phased indexing is supposed to prevent.

Known limit, measured: Whisper heard the spoken code "ZQ-7734" as "EQ-7734". Spoken letter
codes are not reliable; the numeric half survives, which is why a keyword match on part of a
query still qualifies. Do not promise exact recall for spoken alphanumerics.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import TRANSCRIBE, WHISPER_MODEL


def available() -> bool:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


class Transcriber:
    """Subprocess client for the speech worker. One process, one model, released when closed."""

    def __init__(self, model: str = WHISPER_MODEL, worker: Path | None = None):
        worker = worker or Path(__file__).with_name("whisper_worker.py")
        self.proc = subprocess.Popen(
            [sys.executable, str(worker), "--model", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
        )
        self._next_id = 0
        hello = self._read()
        if not hello.get("ready"):
            raise RuntimeError(f"speech worker failed to start: {hello.get('error')}")
        self.info = hello["info"]

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("speech worker exited")
        return json.loads(line)

    def _call(self, op: str, **payload) -> dict:
        assert self.proc.stdin is not None
        self._next_id += 1
        self.proc.stdin.write(json.dumps({"id": self._next_id, "op": op, **payload}) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"speech worker {op} failed: {reply.get('error')}")
        return reply

    def transcribe(self, path: str | Path) -> list:
        """Transcript segments for one file as text items. A silent clip yields [], not an error."""
        if not TRANSCRIBE:
            return []
        from .extract import Item

        reply = self._call("transcribe", inputs=[str(path)])
        segments = (reply.get("results") or [[]])[0] or []
        return [
            Item("transcript", "text", loc=float(s["start"]), loc_end=float(s["end"]),
                 text=s["text"], excerpt=s["text"][:200])
            for s in segments
        ]

    def close(self) -> None:
        try:
            self._call("quit")
        except Exception:
            pass
        finally:
            self.proc.terminate()
