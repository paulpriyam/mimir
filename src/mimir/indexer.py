"""Walk a folder, extract every file, embed the items, write the index.

Models are loaded once per run and fed in batches grouped by modality, so a folder of images
never thrashes the audio tower. Media the worker produces (frames, audio windows) goes to a
temp dir that is deleted as soon as it has been embedded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .config import DOC_PROMPT
from .extract import Item, extract, kind_of
from .store import Store, content_hash, vec_blob
from .vectors import normalize

TEXT_BATCH = 16
MEDIA_BATCH = {"image": 4, "audio": 2}


def file_info(path: Path, kind: str, items: list[Item]) -> dict:
    info: dict[str, str] = {}
    if kind == "pdf":
        info["pages"] = str(len([i for i in items if i.kind == "page"]))
    if kind in {"audio", "video"}:
        ends = [i.loc_end for i in items if i.modality == "audio" and i.loc_end]
        if ends:
            info["duration"] = f"{max(ends):.0f}"
    if kind == "image":
        try:
            from PIL import Image

            with Image.open(path) as image:
                info["width"], info["height"] = str(image.width), str(image.height)
        except Exception:
            pass
    return info


def embed_items(embedder, items: list[Item]) -> dict[int, np.ndarray]:
    """Vector for every text/image/audio item, batched by what the tower needs."""
    vectors: dict[int, np.ndarray] = {}
    text_items = [i for i in items if i.modality == "text" and i.text]
    for start in range(0, len(text_items), TEXT_BATCH):
        batch = text_items[start:start + TEXT_BATCH]
        rows = normalize(embedder.embed_text([i.text for i in batch], DOC_PROMPT))
        for item, row in zip(batch, rows):
            vectors[id(item)] = row
    for media_kind in ("image", "audio"):
        media_items = [i for i in items if i.media_kind == media_kind]
        size = MEDIA_BATCH[media_kind]
        for start in range(0, len(media_items), size):
            batch = media_items[start:start + size]
            rows, errors = embedder.embed_media(media_kind, [i.media for i in batch])
            rows = normalize(rows)
            for item, row, error in zip(batch, rows, errors):
                if error:
                    print(f"  ! {media_kind} failed: {error}")
                    continue
                vectors[id(item)] = row
    return vectors


def index_file(store: Store, embedder, path: Path, kind: str) -> int:
    """Index one file. Returns the number of segments written."""
    with tempfile.TemporaryDirectory(prefix="mimir-") as temp:
        items = extract(path, kind, Path(temp))
        if not items:
            return 0
        vectors = embed_items(embedder, items)
        stat = path.stat()
        file_id = store.add_file(str(path), kind, stat.st_size, stat.st_mtime,
                                 content_hash(path.read_bytes()), file_info(path, kind, items))
        store.add_segment(file_id, "name", "text", text=path.name)
        written = 1
        for item in items:
            vector = vectors.get(id(item))
            if vector is None and item.text is None:
                continue
            store.add_segment(
                file_id, item.kind, item.modality, loc=item.loc, loc_end=item.loc_end,
                vec=vec_blob(vector) if vector is not None else None,
                excerpt=item.excerpt, text=item.text,
                model_id=embedder.info.get("model"), dim=embedder.dim if vector is not None else None,
            )
            written += 1
        store.commit()
        return written


def index_folder(store: Store, embedder, root: Path, skip_hidden: bool = True) -> tuple[int, int]:
    files = skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if skip_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
            skipped += 1
            continue
        kind = kind_of(path)
        if kind is None:
            skipped += 1
            continue
        try:
            index_file(store, embedder, path, kind)
            files += 1
        except Exception as error:
            skipped += 1
            print(f"  ! {path.name}: {type(error).__name__}: {error}")
    store.set_meta("model", str(embedder.info.get("model")))
    store.set_meta("dim", str(embedder.dim))
    return files, skipped
