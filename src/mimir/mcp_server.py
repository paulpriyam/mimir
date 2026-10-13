"""MCP server: let Claude, Cursor and other MCP clients search the local index.

Read-only and local. There is no network call, no file is ever modified, and results can only
come from folders the user already chose to index, so the server cannot read anything the
index does not already contain. Snippets are returned rather than whole files.

The model loads lazily on the first search: starting the server costs nothing.

    mimir mcp            # stdio, ready to be pointed at by an MCP client

Client config (Claude Desktop / Cursor) is a command plus args:

    {"command": "uv", "args": ["run", "--directory", "/path/to/mimir", "mimir", "mcp"]}
"""

from __future__ import annotations

import threading
from pathlib import Path

from .config import MAIN_DB, QUERY_PROMPT
from .deep import deep_search
from .search import Hit, Searcher
from .store import Store


def hit_to_dict(hit: Hit, root: Path) -> dict:
    """One result as a plain dict: enough to open the thing at the right place."""
    try:
        path = str(Path(hit.path).relative_to(root))
    except ValueError:
        path = hit.path
    out = {
        "path": path,
        "absolute_path": hit.path,
        "kind": hit.kind,
        "score": round(hit.score, 3),
        "excerpt": (hit.excerpt or "").strip()[:400],
    }
    if hit.loc is not None:
        # PDF page, or seconds into audio/video, so the client can jump straight there.
        out["loc"] = round(hit.loc, 2)
        if hit.loc_end is not None:
            out["loc_end"] = round(hit.loc_end, 2)
    if hit.z is not None:
        out["z"] = round(hit.z, 2)
    if hit.cosine is not None:
        out["cosine"] = round(hit.cosine, 3)
    return out


class MimirSearch:
    """The search the tools expose. Holds the store and one embedder; loads the model on demand."""

    def __init__(self, index_dir: str | Path, embedder=None):
        self.index_dir = Path(index_dir).expanduser()
        self.store = Store(self.index_dir / MAIN_DB)
        self.searcher = Searcher(self.store)
        # Results are reported relative to the folder the user indexed, which is what a client
        # should show. Falls back to the index directory when no root was recorded.
        self.root = Path(self.store.meta("root") or self.index_dir)
        self._embedder = embedder
        self._reranker = None
        self._reranker_checked = False
        # MCP and the web UI run request handlers on worker threads, and SQLite connections are
        # not thread-safe: one lock serializes every call that touches the store.
        self._lock = threading.Lock()

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import WorkerEmbedder

            self._embedder = WorkerEmbedder(text_only=True)
        return self._embedder

    @property
    def reranker(self):
        """The cross-encoder, started once on first use, or None when it has not been fetched."""
        if not self._reranker_checked:
            self._reranker_checked = True
            from .rerank import Reranker, RerankerUnavailable, available

            if available():
                try:
                    self._reranker = Reranker()
                except RerankerUnavailable:
                    self._reranker = None
        return self._reranker

    def search(self, query: str, limit: int = 10, kind: str | None = None,
               deep: bool = False) -> list[dict]:
        limit = max(1, min(limit, 50))
        kinds = {kind} if kind else None
        with self._lock:
            vector = self.embedder.embed_text([query], QUERY_PROMPT)[0]
            reranker = self.reranker if deep else None
            if reranker is not None:
                hits = deep_search(self.store, self.searcher, reranker, query,
                                   query_vector=vector, limit=limit, kinds=kinds)
            else:
                hits = self.searcher.search(query, query_vector=vector, kinds=kinds, limit=limit)
            return [hit_to_dict(hit, self.root) for hit in hits]

    def info(self) -> dict:
        files, segments = self.store.count()
        return {"index": str(self.index_dir), "root": str(self.root), "files": files,
                "segments": segments, "model": self.store.meta("model"),
                "deep_available": self.reranker is not None}

    def open_file(self, path: str, reveal: bool = False) -> bool:
        """Open a file in its default app, or show it in Finder. Never reads or changes it."""
        import subprocess

        if not Path(path).exists():
            return False
        command = ["open", "-R", path] if reveal else ["open", path]
        return subprocess.run(command, capture_output=True).returncode == 0

    def read(self, path: str) -> str:
        """The text the index holds for a file: its chunks, pages or transcript."""
        with self._lock:
            return self._read(path)

    def _read(self, path: str) -> str:
        row = self.store.file_by_path(path)
        if row is None:
            return ""
        pieces = []
        for segment in self.store.db.execute(
            "SELECT id, kind, loc FROM segments WHERE file = ? AND vec IS NOT NULL ORDER BY loc", (row["id"],)
        ):
            text = self.store.segment_text(segment["id"])
            if text:
                where = f"[{segment['kind']} @{segment['loc']:.0f}] " if segment["loc"] is not None else ""
                pieces.append(where + text)
        return "\n\n".join(pieces)

    def close(self) -> None:
        if self._reranker is not None:
            self._reranker.close()
        if self._embedder is not None:
            self._embedder.close()
        self.store.close()


def build_server(index_dir: str | Path):
    # mcp 2.x renamed FastMCP to MCPServer; accept either so the extra is not version-brittle.
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer

    search = MimirSearch(index_dir)
    server = MCPServer("mimir")

    @server.tool()
    def mimir_search(query: str, limit: int = 10, kind: str | None = None,
                     deep: bool = False) -> list[dict]:
        """Search the user's own files by meaning and by exact words.

        Covers pictures, screenshots (their text), documents, PDF pages, recordings and video.
        A result's `loc` is a PDF page or a number of seconds, so it can be opened at the exact
        place. `kind` filters to one of: image, pdf, doc, audio, video.

        `deep` reranks candidates with a cross-encoder: slower, finds more, and needs the
        reranker runtime from scripts/fetch_reranker.py.

        Returns [] when nothing is a good enough match, rather than a weak guess.
        """
        return search.search(query, limit=limit, kind=kind, deep=deep)

    @server.tool()
    def mimir_read(path: str) -> str:
        """The indexed text of one file (its pages, chunks or transcript), by absolute path."""
        return search.read(path)

    return server, search


def serve(index_dir: str | Path) -> int:
    server, search = build_server(index_dir)
    try:
        server.run()          # stdio
    finally:
        search.close()
    return 0
