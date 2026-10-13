"""Walk a folder and index it in phases, one model at a time.

Indexing used to be per file: extract (which called Whisper) then embed (which called the
embedder), so the speech model and the embedding model were resident together. Now it runs in
three passes and each pass owns exactly one model:

    1. speech    Whisper transcribes every audio and video file, then goes away
    2. captions  SmolVLM describes every image, then goes away
    3. embed     EmbeddingGemma vectors everything, using the transcript and caption text

Peak memory becomes the largest single model instead of their sum, which is what makes an
8 GB Mac viable. Each model is a separate process (or, for Whisper, uncached), so "goes away"
means the memory is actually returned, not just dereferenced.
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
SPEECH_KINDS = {"audio", "video"}


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


def index_file(store: Store, embedder, path: Path, kind: str,
               transcript: list[Item] | None = None, caption: str | None = None) -> int:
    """Index one file. Returns the number of segments written."""
    with tempfile.TemporaryDirectory(prefix="mimir-") as temp:
        items = extract(path, kind, Path(temp), transcript=transcript, caption=caption)
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


def discover(root: Path, skip_hidden: bool = True) -> list[tuple[Path, str]]:
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if skip_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        kind = kind_of(path)
        if kind:
            found.append((path, kind))
    return found


def index_folder(store: Store, root: Path, embedder_factory, transcriber_factory=None,
                 captioner_factory=None, skip_hidden: bool = True) -> tuple[int, int]:
    """Phase 1 speech, phase 2 captions, phase 3 embeddings. Each factory returns a model with
    `.close()`, or None when that model is unavailable or switched off."""
    targets = discover(root, skip_hidden)
    files = skipped = 0

    # Phase 1: speech, with only Whisper loaded.
    transcripts: dict[Path, list[Item]] = {}
    transcriber = transcriber_factory() if transcriber_factory else None
    if transcriber:
        speech = [(path, kind) for path, kind in targets if kind in SPEECH_KINDS]
        print(f"phase 1/3: transcribing {len(speech)} audio/video files", flush=True)
        try:
            for path, _ in speech:
                try:
                    transcripts[path] = transcriber.transcribe(path)
                except Exception as error:
                    print(f"  ! transcript failed for {path.name}: {type(error).__name__}: {error}")
        finally:
            transcriber.close()

    # Phase 2: captions, with only the vision-language model loaded.
    captions: dict[Path, str] = {}
    captioner = captioner_factory() if captioner_factory else None
    if captioner:
        pictures = [path for path, kind in targets if kind == "image"]
        print(f"phase 2/3: captioning {len(pictures)} images", flush=True)
        try:
            for path in pictures:
                try:
                    captions[path] = captioner.caption(path)
                except Exception as error:
                    print(f"  ! caption failed for {path.name}: {type(error).__name__}: {error}")
        finally:
            captioner.close()

    # Phase 3: embeddings, with only the embedder loaded.
    print(f"phase 3/3: embedding {len(targets)} files", flush=True)
    embedder = embedder_factory()
    try:
        for path, kind in targets:
            try:
                index_file(store, embedder, path, kind,
                           transcript=transcripts.get(path), caption=captions.get(path))
                files += 1
            except Exception as error:
                skipped += 1
                print(f"  ! {path.name}: {type(error).__name__}: {error}")
        store.set_meta("model", str(embedder.info.get("model")))
        store.set_meta("dim", str(embedder.dim))
    finally:
        embedder.close()

    store.set_meta("root", str(root))
    return files, skipped
