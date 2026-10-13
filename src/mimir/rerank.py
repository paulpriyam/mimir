"""The reranker: a cross-encoder that reads the query and a candidate together.

Why a separate process. The reranker is a GGUF served by llama.cpp's `llama-server --reranking`,
which is a different runtime from the MLX models. Starting it lazily and keeping it for the
session means instant search never pays for it, and deep search pays the load once.

Why it is worth having at all, measured: instant search cannot separate a real paraphrase from
an absurd query, because both land in the same 0.66-0.68 cosine band ("percussion kit you sit
behind" scores 0.674 against a drum; "the source code for the Linux kernel scheduler" scores
0.662 against a flower). Read together, the cross-encoder puts 0.0104 on the drum and 0.0001 on
everything for the absurd query - a 100x gap.

The known trap this guards against: some GGUF reranker conversions return a flat score for
every document, which looks fine until you notice nothing is ever ranked. `self_check()`
asserts the scores actually vary before the reranker is trusted.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import LLAMA_SERVER, RERANK_MODEL, RERANK_PORT


class RerankerUnavailable(RuntimeError):
    """Raised when the runtime is missing, so callers can fall back to instant search."""


class Reranker:
    """Owns a llama-server process serving a rank-pooling GGUF."""

    def __init__(self, binary: Path = LLAMA_SERVER, model: Path = RERANK_MODEL,
                 port: int = RERANK_PORT, context: int = 2048, threads: int | None = None):
        self.binary, self.model, self.port = Path(binary), Path(model), port
        self.base = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        if not self.binary.exists():
            raise RerankerUnavailable(f"llama-server not found at {self.binary}; "
                                      f"run scripts/fetch_reranker.py")
        if not self.model.exists():
            raise RerankerUnavailable(f"reranker model not found at {self.model}; "
                                      f"run scripts/fetch_reranker.py")
        command = [
            str(self.binary), "--reranking", "-m", str(self.model),
            "--host", "127.0.0.1", "--port", str(port), "-c", str(context), "--no-warmup",
        ]
        if threads:
            command += ["-t", str(threads)]
        self.proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RerankerUnavailable(f"llama-server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as response:
                    if b"ok" in response.read():
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise RerankerUnavailable(f"llama-server not ready after {timeout:.0f}s")

    def score(self, query: str, documents: list[str], timeout: float = 300.0) -> list[float]:
        """A relevance score per document, in the order given."""
        if not documents:
            return []
        import json

        payload = json.dumps({"query": query, "documents": documents}).encode()
        request = urllib.request.Request(f"{self.base}/v1/rerank", data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        scores = [0.0] * len(documents)
        for item in body.get("results", body):
            scores[item["index"]] = float(item.get("relevance_score", 0.0))
        return scores

    def self_check(self) -> tuple[bool, str]:
        """Assert the scores vary. A flat-scoring conversion must never be trusted silently."""
        documents = [
            "A red, white and blue drum with two drumsticks on top.",
            "An emperor penguin in front of a blue background.",
            "Sunflower with yellow petals and green center.",
        ]
        good = self.score("percussion kit you sit behind", documents)
        absurd = self.score("the source code for the Linux kernel scheduler", documents)
        spread = max(good) - min(good)
        if spread < 1e-6:
            return False, f"scores are flat ({good}); this conversion cannot rank"
        if good.index(max(good)) != 0:
            return False, f"the drum did not win for a drum query: {good}"
        if max(absurd) >= max(good):
            return False, f"an absurd query scored as high as a real one: {max(absurd)} vs {max(good)}"
        return True, (f"drum {max(good):.4f} vs best-other {sorted(good)[-2]:.4f}; "
                      f"absurd query tops out at {max(absurd):.4f}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def available() -> bool:
    return Path(LLAMA_SERVER).exists() and Path(RERANK_MODEL).exists()


def threads_default() -> int:
    return max(4, (os.cpu_count() or 8) - 2)
