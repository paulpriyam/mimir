"""Command line: index, search, eval, info.

`--fake` swaps in the hash-based stand-in embedder so the whole pipeline runs end to end
without downloading a model. Use it for wiring and CI, never for quality.

    mimir index  --fake ~/notes
    mimir search --fake "retry with backoff"
    mimir eval   --fake evals/smoke.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DOC_PROMPT, INDEX_DIR, MAIN_DB, QUERY_PROMPT, RERANK_TOP
from .embed import FakeEmbedder, WorkerEmbedder
from .evals import evaluate, format_report
from .search import Searcher
from .store import Store, content_hash, vec_blob
from .vectors import normalize

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
CHUNK = 800
OVERLAP = 100


def chunks(text: str, size: int = CHUNK, overlap: int = OVERLAP):
    text = text.strip()
    if not text:
        return
    step = max(size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            yield start, piece


def open_store(args) -> Store:
    directory = Path(args.index_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return Store(directory / MAIN_DB)


def make_embedder(args):
    return FakeEmbedder() if args.fake else WorkerEmbedder(text_only=True)


def cmd_index(args) -> int:
    root = Path(args.path).expanduser().resolve()
    store, embedder = open_store(args), make_embedder(args)
    added = skipped = 0
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(errors="ignore")
            stat = path.stat()
            file_id = store.add_file(str(path), "doc", stat.st_size, stat.st_mtime,
                                     content_hash(path.read_bytes()))
            store.add_segment(file_id, "name", "text", text=path.name)
            pieces = list(chunks(text))
            if pieces:
                vectors = normalize(embedder.embed_text([p for _, p in pieces], DOC_PROMPT))
                for (offset, piece), vector in zip(pieces, vectors):
                    store.add_segment(file_id, "chunk", "text", loc=float(offset), vec=vec_blob(vector),
                                      excerpt=piece[:200], text=piece, model_id=embedder.info.get("model"),
                                      dim=embedder.dim)
            store.commit()
            added += 1
        store.set_meta("model", str(embedder.info.get("model")))
        store.set_meta("dim", str(embedder.dim))
    finally:
        embedder.close()
    files, segments = store.count()
    print(f"indexed {added} files ({skipped} skipped); index now holds {files} files, {segments} segments")
    store.close()
    return 0


def cmd_search(args) -> int:
    store, embedder = open_store(args), make_embedder(args)
    try:
        vector = embedder.embed_text([args.query], QUERY_PROMPT)[0]
    finally:
        embedder.close()
    hits = Searcher(store).search(args.query, query_vector=vector, limit=args.limit)
    if not hits:
        print("nothing found")
    for hit in hits:
        where = ""
        if hit.loc is not None:
            where = f" @{hit.loc:.0f}" + (f"-{hit.loc_end:.0f}" if hit.loc_end else "")
        print(f"{hit.score:5.2f}  {hit.kind:<10} {Path(hit.path).name}{where}")
        if hit.excerpt:
            print(f"        {hit.excerpt.strip()[:160]}")
    store.close()
    return 0


def cmd_eval(args) -> int:
    store, embedder = open_store(args), make_embedder(args)
    try:
        report = evaluate(store, embedder, args.eval_file, depth=args.depth)
    finally:
        embedder.close()
    print(format_report(report, depth=args.depth))
    store.close()
    return 0


def cmd_info(args) -> int:
    store = open_store(args)
    files, segments = store.count()
    print(f"index:  {store.path}")
    print(f"files:  {files}")
    print(f"segments: {segments}")
    print(f"model:  {store.meta('model', '(none)')}  dim {store.meta('dim', '?')}")
    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimir", description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default=str(INDEX_DIR), help="where the SQLite index lives")
    parser.add_argument("--fake", action="store_true", help="use the hash-based stand-in embedder")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index a folder (text files for now)")
    p.add_argument("path")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="run an eval file and print the score table")
    p.add_argument("eval_file")
    p.add_argument("-n", "--depth", type=int, default=10)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("info", help="what the index holds")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
