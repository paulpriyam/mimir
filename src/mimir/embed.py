"""Embedders.

`FakeEmbedder` is deterministic and needs no model: it hashes words into buckets, so vectors
grow with lexical overlap. That is enough to exercise indexing, search and the eval harness
end to end (and in CI) without a 1 GB download. It is NOT a quality signal.

`WorkerEmbedder` drives the reused MLX worker (see embedding_worker.py, derived from DigUp,
MIT). The worker prints {"ready": true, "info": {...}} first, then answers one JSON object per
line. All MLX work stays on the worker's single thread.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DIM, MODEL_ID, QUERY_PROMPT, DOC_PROMPT


def prompt(template: str, text: str) -> str:
    return template.format(text=text)


class Embedder(Protocol):
    info: dict
    dim: int

    def embed_text(self, texts: list[str], template: str = QUERY_PROMPT) -> np.ndarray: ...
    def embed_media(self, kind: str, paths: list[str]) -> tuple[np.ndarray, list[str | None]]: ...
    def close(self) -> None: ...


class FakeEmbedder:
    """Hash-based stand-in. Deterministic, dependency-free, lexical only."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.info = {"model": "fake", "dim": dim, "dtype": "f16"}

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, np.float32)
        for word in text.lower().split():
            word = word.strip(".,;:!?()[]\"'")
            if not word:
                continue
            bucket = int(hashlib.sha1(word.encode()).hexdigest()[:8], 16) % self.dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_text(self, texts: list[str], template: str = QUERY_PROMPT) -> np.ndarray:
        # The template is applied for fidelity to the real worker, but the fake ignores it.
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_media(self, kind: str, paths: list[str]) -> tuple[np.ndarray, list[str | None]]:
        # Media has no text to hash; keep it out of the way so fake runs stay text-only.
        vectors = np.stack([self._one(Path(p).stem) for p in paths]) if paths else np.zeros((0, self.dim), np.float32)
        return vectors, [None] * len(paths)

    def close(self) -> None:
        pass


class WorkerEmbedder:
    """Subprocess client for the MLX worker. One process, one thread, one model."""

    def __init__(self, model: str = MODEL_ID, text_only: bool = False, worker: Path | None = None):
        worker = worker or Path(__file__).with_name("embedding_worker.py")
        cmd = [sys.executable, str(worker), "--model", model]
        if text_only:
            cmd.append("--text-only")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        self._next_id = 0
        hello = self._read()
        if not hello.get("ready"):
            raise RuntimeError(f"worker failed to start: {hello.get('error')}")
        self.info = hello["info"]
        self.dim = int(self.info["dim"])

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited")
        return json.loads(line)

    def _call(self, op: str, **payload) -> dict:
        assert self.proc.stdin is not None
        self._next_id += 1
        request = {"id": self._next_id, "op": op, **payload}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"worker {op} failed: {reply.get('error')}")
        return reply

    @staticmethod
    def _decode(reply: dict, count: int) -> np.ndarray:
        raw = base64.b64decode(reply["vectors"])
        return np.frombuffer(raw, dtype="<f4").reshape(count, reply["dim"]).copy()

    def embed_text(self, texts: list[str], template: str = QUERY_PROMPT) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        reply = self._call("text", inputs=[prompt(template, t) for t in texts])
        return self._decode(reply, len(texts))

    def embed_media(self, kind: str, paths: list[str]) -> tuple[np.ndarray, list[str | None]]:
        if not paths:
            return np.zeros((0, self.dim), np.float32), []
        reply = self._call(kind, inputs=paths)
        return self._decode(reply, len(paths)), reply.get("errors", [None] * len(paths))

    def close(self) -> None:
        try:
            self._call("quit")
        except Exception:
            pass
        finally:
            self.proc.terminate()
