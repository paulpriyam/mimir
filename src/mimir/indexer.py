"""Walk a folder and index it in phases, one model at a time.

Indexing used to be per file: extract (which called Whisper) then embed (which called the
embedder), so the speech model and the embedding model were resident together. Now it runs in
three passes and each pass owns exactly one model:

    1. speech    Whisper transcribes every audio and video file, then goes away
    2. captions  SmolVLM describes every image, and every frame of every video, then goes away
    3. embed     EmbeddingGemma vectors everything, using the transcript and caption text

Peak memory becomes the largest single model instead of their sum, which is what makes an
8 GB Mac viable. Each model is a separate process (or, for Whisper, uncached), so "goes away"
means the memory is actually returned, not just dereferenced.

Video frames are cut and captioned in phase 2 rather than during embedding, so that a frame is
not just a picture: it carries a sentence. Without that a video's moments all reduce to the
file name, which makes them impossible to keyword-match and impossible to rerank - measured as
video-moment R@1 falling from 0.86 to 0.29 under deep search.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DOC_PROMPT
from .extract import Item, extract, kind_of, video_audio, video_frames
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


def index_file(store: Store, embedder, path: Path, kind: str, workdir: Path,
               transcript: list[Item] | None = None, caption: str | None = None,
               prepared: list[Item] | None = None) -> int:
    """Index one file. `prepared` is the item list already built by an earlier phase."""
    items = prepared if prepared is not None else extract(path, kind, workdir,
                                                          transcript=transcript, caption=caption)
    stat = path.stat()
    file_id = store.add_file(str(path), kind, stat.st_size, stat.st_mtime,
                             content_hash(path.read_bytes()), file_info(path, kind, items))
    # The name segment is written even when nothing was extractable: a scanned PDF with no text
    # layer should still be findable by its file name, and recording the file stops the next
    # pass from re-attempting it forever.
    store.add_segment(file_id, "name", "text", text=path.name)
    written = 1
    if items:
        vectors = embed_items(embedder, items)
        for item in items:
            vector = vectors.get(id(item))
            if vector is None and item.text is None:
                continue
            store.add_segment(
                file_id, item.kind, item.modality, loc=item.loc, loc_end=item.loc_end,
                vec=vec_blob(vector) if vector is not None else None,
                excerpt=item.excerpt, text=item.text,
                model_id=embedder.info.get("model"),
                dim=embedder.dim if vector is not None else None,
            )
            written += 1
    store.commit()
    return written


def discover(root: Path, skip_hidden: bool = True,
             include_video: bool = False) -> list[tuple[Path, str]]:
    """Every indexable file. Video is left out unless asked for: it is by far the most
    expensive thing to index (a folder of 6 hours of video cost ~2.5 hours, almost all of it
    writing captions for frames), and one long episode can turn a one-minute index into hours."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if skip_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        kind = kind_of(path)
        if kind is None:
            continue
        if kind == "video" and not include_video:
            continue
        found.append((path, kind))
    return found


@dataclass
class IndexResult:
    """What one pass did. `changed` is what a watch loop waits for."""
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed

    def describe(self) -> str:
        return (f"{self.added} new, {self.updated} changed, {self.unchanged} unchanged, "
                f"{self.removed} removed, {self.skipped} skipped")


def _under(path: str, root: Path) -> bool:
    return path == str(root) or path.startswith(str(root) + "/")


def index_folder(store: Store, root: Path, embedder_factory, transcriber_factory=None,
                 captioner_factory=None, skip_hidden: bool = True,
                 include_video: bool = False) -> IndexResult:
    """One incremental pass: phase 1 speech, phase 2 captions, phase 3 embeddings.

    Only files whose size or modification time changed are looked at, so a re-scan of an
    unchanged folder costs a few stat calls and loads no model at all. Files that vanished are
    dropped. Each phase owns exactly one model and closes it before the next opens, which keeps
    peak memory at the largest single model rather than their sum.
    """
    root = root.resolve()
    targets = discover(root, skip_hidden, include_video)
    if not include_video:
        videos = sum(1 for p in root.rglob("*") if p.is_file() and kind_of(p) == "video")
        if videos:
            print(f"skipping {videos} video files (pass --video to index them; "
                  f"frames and transcripts make them far slower than everything else)", flush=True)

    result = IndexResult()

    # What disappeared: indexed files under this root that are no longer on disk.
    for path in list(store.file_fingerprints()):
        if _under(path, root) and not Path(path).exists():
            if store.delete_file(path):
                result.removed += 1

    # What needs work: new files, or ones whose size/mtime moved. A touched file whose contents
    # are identical is left alone, so a sync that rewrites mtimes costs nothing.
    todo: list[tuple[Path, str]] = []
    known = store.file_fingerprints()
    for path, kind in targets:
        try:
            stat = path.stat()
        except OSError:
            result.skipped += 1
            continue
        previous = known.get(str(path))
        if previous and previous[0] == stat.st_size and abs(previous[1] - stat.st_mtime) < 1e-6:
            result.unchanged += 1
            continue
        if previous and previous[2] and previous[0] == stat.st_size:
            if content_hash(path.read_bytes()) == previous[2]:
                result.unchanged += 1
                continue
        todo.append((path, kind))

    if not todo:
        return result

    # One workdir for the whole run: frames cut in phase 2 must still be there in phase 3.
    with tempfile.TemporaryDirectory(prefix="mimir-") as temp:
        workdir = Path(temp)

        # Phase 1: speech, with only Whisper loaded. A model is only started if its phase has
        # work: a folder of documents should never pay to load Whisper or the caption model.
        transcripts: dict[Path, list[Item]] = {}
        speech = [(path, kind) for path, kind in todo if kind in SPEECH_KINDS]
        transcriber = transcriber_factory() if (speech and transcriber_factory) else None
        if transcriber:
            print(f"phase 1/3: transcribing {len(speech)} audio/video files", flush=True)
            try:
                for path, _ in speech:
                    try:
                        transcripts[path] = transcriber.transcribe(path)
                    except Exception as error:
                        print(f"  ! transcript failed for {path.name}: {type(error).__name__}: {error}")
            finally:
                transcriber.close()

        # Phase 2: captions, with only the vision-language model loaded. Videos are cut into
        # frames here so each frame can be captioned, and so the cut is done once.
        captions: dict[Path, str] = {}
        prepared: dict[Path, list[Item]] = {}
        pictures = [path for path, kind in todo if kind == "image"]
        videos = [path for path, kind in todo if kind == "video"]
        captioner = captioner_factory() if ((pictures or videos) and captioner_factory) else None
        if captioner:
            print(f"phase 2/3: captioning {len(pictures)} images and {len(videos)} videos' frames",
                  flush=True)
            try:
                for path in pictures:
                    try:
                        captions[path] = captioner.caption(path)
                    except Exception as error:
                        print(f"  ! caption failed for {path.name}: {type(error).__name__}: {error}")
                for path in videos:
                    try:
                        items: list[Item] = []
                        for frame in video_frames(path, workdir):
                            items.append(frame)
                            text = captioner.caption(frame.media)
                            if text:
                                # The caption is searchable text placed at the frame's moment,
                                # so a keyword hit lands on that second of the video.
                                items.append(Item("caption", "text", loc=frame.loc,
                                                  loc_end=frame.loc_end, text=text,
                                                  excerpt=text[:200]))
                        items += video_audio(path, workdir)
                        items += transcripts.get(path, [])
                        prepared[path] = items
                    except Exception as error:
                        print(f"  ! frames failed for {path.name}: {type(error).__name__}: {error}")
            finally:
                captioner.close()

        # Phase 3: embeddings, with only the embedder loaded.
        print(f"phase 3/3: embedding {len(todo)} changed files", flush=True)
        embedder = embedder_factory()
        try:
            for path, kind in todo:
                existed = str(path) in known
                try:
                    index_file(store, embedder, path, kind, workdir,
                               transcript=transcripts.get(path), caption=captions.get(path),
                               prepared=prepared.get(path))
                    if existed:
                        result.updated += 1
                    else:
                        result.added += 1
                except Exception as error:
                    result.skipped += 1
                    print(f"  ! {path.name}: {type(error).__name__}: {error}")
            store.set_meta("model", str(embedder.info.get("model")))
            store.set_meta("dim", str(embedder.dim))
        finally:
            embedder.close()

    store.set_meta("root", str(root))
    return result
