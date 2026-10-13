"""Client for the caption worker: one subprocess, one model, released when the phase ends.

Like the embedder, this is a process rather than a function call so the model's memory is
actually returned to the system between indexing phases, not just dereferenced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .config import CAPTION_MAX_TOKENS, CAPTION_MODEL, CAPTION_PROMPT


def available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        return False
    return True


class Captioner:
    def __init__(self, model: str = CAPTION_MODEL, prompt: str = CAPTION_PROMPT,
                 max_tokens: int = CAPTION_MAX_TOKENS, worker: Path | None = None):
        worker = worker or Path(__file__).with_name("caption_worker.py")
        self.prompt = prompt
        self.proc = subprocess.Popen(
            [sys.executable, str(worker), "--model", model, "--prompt", prompt,
             "--max-tokens", str(max_tokens)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
        )
        self._next_id = 0
        hello = self._read()
        if not hello.get("ready"):
            raise RuntimeError(f"caption worker failed to start: {hello.get('error')}")
        self.info = hello["info"]

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("caption worker exited")
        return json.loads(line)

    def _call(self, op: str, **payload) -> dict:
        assert self.proc.stdin is not None
        self._next_id += 1
        self.proc.stdin.write(json.dumps({"id": self._next_id, "op": op, **payload}) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"caption worker {op} failed: {reply.get('error')}")
        return reply

    def caption(self, path: str | Path) -> str:
        """One short sentence for one image. Empty string if the image cannot be read.

        Retried once: on a cold start right after the weights are fetched the model has been
        seen to emit an empty completion, and a blank caption silently makes that image
        unfindable, which is worse than paying for one more forward pass.
        """
        for attempt in range(2):
            reply = self._call("caption", inputs=[str(path)])
            captions = reply.get("captions") or [""]
            text = (captions[0] or "").strip()
            if text or attempt:
                return text
        return ""

    def close(self) -> None:
        try:
            self._call("quit")
        except Exception:
            pass
        finally:
            self.proc.terminate()
