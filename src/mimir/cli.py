"""Command line: index, search, eval, info.

`--fake` swaps in the hash-based stand-in embedder so the whole pipeline runs end to end
without downloading a model. Use it for wiring and CI, never for quality.

    mimir index  --fake ~/notes
    mimir index  ~/mimir-testbed
    mimir search "a penguin"
    mimir search "where they talk about the flight delay"
    mimir eval   evals/media.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import INDEX_DIR, MAIN_DB, QUERY_PROMPT
from .embed import FakeEmbedder, WorkerEmbedder
from .evals import evaluate, format_report
from .indexer import index_folder
from .search import Searcher
from .store import Store


def open_store(args) -> Store:
    directory = Path(args.index_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return Store(directory / MAIN_DB)


def make_embedder(args, text_only: bool = True):
    if args.fake:
        return FakeEmbedder()
    return WorkerEmbedder(text_only=text_only)


def cmd_index(args) -> int:
    root = Path(args.path).expanduser().resolve()
    store = open_store(args)
    embedder = make_embedder(args, text_only=False)  # images and audio need the full model
    try:
        files, skipped = index_folder(store, embedder, root)
    finally:
        embedder.close()
    total_files, segments = store.count()
    print(f"indexed {files} files ({skipped} skipped); index now holds {total_files} files, {segments} segments")
    store.close()
    return 0


def cmd_search(args) -> int:
    store = open_store(args)
    embedder = make_embedder(args, text_only=True)  # a query is text
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
        print(f"{hit.score:5.2f}  {hit.kind:<9} {Path(hit.path).name}{where}")
        if hit.excerpt:
            print(f"        {hit.excerpt.strip()[:150]}")
    store.close()
    return 0


def cmd_eval(args) -> int:
    store = open_store(args)
    embedder = make_embedder(args, text_only=True)
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
    print(f"index:    {store.path}")
    print(f"files:    {files}")
    print(f"segments: {segments}")
    print(f"model:    {store.meta('model', '(none)')}  dim {store.meta('dim', '?')}")
    rows = store.db.execute(
        "SELECT modality, kind, COUNT(*) AS n FROM segments GROUP BY modality, kind ORDER BY n DESC"
    ).fetchall()
    for row in rows:
        print(f"  {row['modality']:<6} {row['kind']:<8} {row['n']}")
    store.close()
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import serve

    return serve(args.index_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimir", description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default=str(INDEX_DIR), help="where the SQLite index lives")
    parser.add_argument("--fake", action="store_true", help="use the hash-based stand-in embedder")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index a folder")
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

    p = sub.add_parser("mcp", help="serve the index to MCP clients (Claude, Cursor) over stdio")
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
