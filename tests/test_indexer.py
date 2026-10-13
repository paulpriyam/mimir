"""The phasing contract: one model at a time.

This is the whole point of phased indexing - peak memory is the largest single model, not the
sum - so it is pinned by a test rather than left to inspection. An empty folder exercises all
three phases with no file dependencies at all.
"""

from pathlib import Path

import numpy as np

from mimir.indexer import index_folder
from mimir.store import Store


def test_models_are_loaded_one_phase_at_a_time(tmp_path):
    live: list[str] = []
    events: list[tuple[str, str]] = []

    class Model:
        info = {"model": "fake"}
        dim = 4

        def __init__(self, role: str):
            self.role = role
            live.append(role)
            events.append(("open", role))

        def close(self) -> None:
            assert live and live[-1] == self.role, f"{self.role} closed out of order, live={live}"
            live.pop()
            events.append(("close", self.role))

        def transcribe(self, path):
            raise AssertionError("no audio or video in an empty folder")

        def caption(self, path):
            raise AssertionError("no images in an empty folder")

        def embed_text(self, texts, template=None):
            return np.zeros((len(texts), 4), np.float32)

        def embed_media(self, kind, paths):
            return np.zeros((len(paths), 4), np.float32), [None] * len(paths)

    root = tmp_path / "empty"
    root.mkdir()
    store = Store(tmp_path / "index.sqlite")
    files, skipped = index_folder(store, root, lambda: Model("embed"),
                                  lambda: Model("speech"), lambda: Model("caption"))
    store.close()

    assert files == 0
    assert events == [
        ("open", "speech"), ("close", "speech"),
        ("open", "caption"), ("close", "caption"),
        ("open", "embed"), ("close", "embed"),
    ]
    assert live == []


def test_a_missing_model_just_skips_its_phase(tmp_path):
    live: list[str] = []

    class Embedder:
        info = {"model": "fake"}
        dim = 4

        def close(self):
            live.remove("embed")

        def embed_text(self, texts, template=None):
            return np.zeros((len(texts), 4), np.float32)

        def embed_media(self, kind, paths):
            return np.zeros((len(paths), 4), np.float32), [None] * len(paths)

    root = tmp_path / "empty"
    root.mkdir()
    store = Store(tmp_path / "index.sqlite")
    # Factories return None when a model is switched off or unavailable.
    files, _ = index_folder(store, root, lambda: (live.append("embed"), Embedder())[1], lambda: None, lambda: None)
    store.close()
    assert files == 0
    assert live == []
