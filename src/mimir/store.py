"""SQLite storage: files, segments (each with a vector), and an FTS5 keyword index.

Design notes (deliberate, so later changes are cheap):
- Every segment row records the `model_id`, `dim` and `dtype` that produced its vector. That
  lets us mix model versions, shrink dimensions later (Matryoshka), and re-embed incrementally
  without a migration.
- `content_hash` on the file and `text_hash` on the segment make dedupe and "re-embed only what
  changed" trivial.
- The FTS5 rowid IS the segment id, so a keyword hit and a vector hit point at the same thing.
- Vectors are L2-normalized float16 blobs. Search is brute force: a matrix-vector product is
  milliseconds on Apple Silicon up to ~1M rows, so no ANN index until the numbers say otherwise.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DTYPE, DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS files(
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    mtime        REAL NOT NULL,
    content_hash TEXT,
    info         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS files_kind ON files(kind);
CREATE INDEX IF NOT EXISTS files_hash ON files(content_hash);

CREATE TABLE IF NOT EXISTS segments(
    id        INTEGER PRIMARY KEY,
    file      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind      TEXT NOT NULL,
    loc       REAL,
    loc_end   REAL,
    modality  TEXT NOT NULL,
    model_id  TEXT,
    dim       INTEGER,
    dtype     TEXT,
    vec       BLOB,
    excerpt   TEXT,
    text_hash TEXT
);
CREATE INDEX IF NOT EXISTS segments_file ON segments(file);
CREATE INDEX IF NOT EXISTS segments_model ON segments(model_id, dim);

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    body,
    original UNINDEXED,
    file UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Segment:
    id: int
    file: int
    kind: str
    loc: float | None
    loc_end: float | None
    modality: str
    vec: bytes | None
    excerpt: str | None


class Store:
    """One SQLite connection. Not thread-safe: callers that use it from several threads (the
    MCP server runs tools on a worker thread) must serialize access themselves. The connection
    is opened with check_same_thread=False so that is possible rather than an error."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- meta ------------------------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.db.commit()

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # --- files -----------------------------------------------------------------------
    def add_file(self, path: str, kind: str, size: int, mtime: float,
                 chash: str | None = None, info: dict | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO files(path, kind, size, mtime, content_hash, info)
               VALUES(?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   kind = excluded.kind, size = excluded.size, mtime = excluded.mtime,
                   content_hash = excluded.content_hash, info = excluded.info
               RETURNING id""",
            (path, kind, size, mtime, chash, json.dumps(info or {})),
        )
        file_id = cur.fetchone()["id"]
        # A re-indexed file replaces its segments; the FTS rows go with them.
        old = [r["id"] for r in self.db.execute("SELECT id FROM segments WHERE file = ?", (file_id,))]
        for segment_id in old:
            self.db.execute("DELETE FROM fts WHERE rowid = ?", (segment_id,))
        self.db.execute("DELETE FROM segments WHERE file = ?", (file_id,))
        self.db.commit()
        return file_id

    def file_fingerprints(self) -> dict[str, tuple[int, float, str | None]]:
        """Every indexed file as {path: (size, mtime, content_hash)}, for deciding what changed."""
        return {
            row["path"]: (row["size"], row["mtime"], row["content_hash"])
            for row in self.db.execute("SELECT path, size, mtime, content_hash FROM files")
        }

    def delete_file(self, path: str) -> bool:
        """Drop a file and everything indexed from it."""
        row = self.db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return False
        for segment in self.db.execute("SELECT id FROM segments WHERE file = ?", (row["id"],)):
            self.db.execute("DELETE FROM fts WHERE rowid = ?", (segment["id"],))
        self.db.execute("DELETE FROM files WHERE id = ?", (row["id"],))   # cascades to segments
        self.db.commit()
        return True

    def file_kinds(self) -> dict[int, str]:
        return {r["id"]: r["kind"] for r in self.db.execute("SELECT id, kind FROM files")}

    def file_by_path(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()

    # --- segments --------------------------------------------------------------------
    def add_segment(self, file_id: int, kind: str, modality: str, *, loc: float | None = None,
                    loc_end: float | None = None, vec: bytes | None = None, excerpt: str | None = None,
                    text: str | None = None, model_id: str | None = None, dim: int | None = None,
                    dtype: str = DTYPE, text_hash: str | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO segments(file, kind, loc, loc_end, modality, model_id, dim, dtype, vec, excerpt, text_hash)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
            (file_id, kind, loc, loc_end, modality, model_id, dim, dtype, vec, excerpt, text_hash),
        )
        segment_id = cur.fetchone()["id"]
        if text:
            self.db.execute("INSERT INTO fts(rowid, body, original, file) VALUES(?, ?, ?, ?)",
                            (segment_id, text, text, file_id))
        return segment_id

    def commit(self) -> None:
        self.db.commit()

    # --- reading vectors -------------------------------------------------------------
    def vector_matrix(self):
        """(matrix float16, segment ids, file ids, modalities) for every segment with a vector."""
        rows = self.db.execute(
            "SELECT id, file, modality, vec FROM segments WHERE vec IS NOT NULL ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, DIM), np.float16), np.array([], np.int64), np.array([], np.int64), np.array([], object)
        matrix = np.zeros((len(rows), DIM), np.float16)
        segment_ids = np.zeros(len(rows), np.int64)
        file_ids = np.zeros(len(rows), np.int64)
        modalities = np.empty(len(rows), object)
        for i, row in enumerate(rows):
            blob = row["vec"]
            matrix[i] = np.frombuffer(blob, dtype="<f2", count=DIM)
            segment_ids[i] = row["id"]
            file_ids[i] = row["file"]
            modalities[i] = row["modality"]
        return matrix, segment_ids, file_ids, modalities

    def segment(self, segment_id: int) -> Segment | None:
        row = self.db.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
        if not row:
            return None
        return Segment(row["id"], row["file"], row["kind"], row["loc"], row["loc_end"],
                       row["modality"], row["vec"], row["excerpt"])

    def segment_text(self, segment_id: int) -> str | None:
        row = self.db.execute("SELECT coalesce(original, body) AS t FROM fts WHERE rowid = ?",
                              (segment_id,)).fetchone()
        return row["t"] if row else None

    def count(self) -> tuple[int, int]:
        files = self.db.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
        segments = self.db.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
        return files, segments


def vec_blob(vector: np.ndarray) -> bytes:
    """A vector as a little-endian float16 blob (L2-normalize before calling)."""
    return np.asarray(vector, dtype="<f2").tobytes()
