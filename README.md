# mimir

Local search over your own files by **meaning and exact words**, with evals that decide what
stays. Pictures, screenshots, documents, PDFs, recordings and video go into one index; a query
ranks them all against each other.

Everything runs on the machine. Nothing is uploaded, and the index never leaves `~/.mimir`.

Named after the Norse figure who guarded the well of wisdom. The tool is meant to answer
"where did I put the thing", so the name fits.

## Status at a glance

| | |
|---|---|
| Tests | 36 passing |
| Search quality (synthetic testbed, 77 queries) | instant **R@1 0.86 / R@5 0.90** · deep **R@1 0.86 / R@5 0.94** · negatives **0/8** false positives |
| Query latency | instant **11 ms**, deep **62 ms** (p50) |
| Indexing | 50 real files in ~38 s · re-scan with no changes in **0.08 s** · peak **3.2 GB** |
| Interfaces | CLI, local web UI, MCP server (Claude/Cursor) |

## What works

**Indexing** — PDFs (page by page, via PDFKit), documents (`docx`, `doc`, `rtf`, `odt`, `html`
via macOS `textutil`; `txt`, `md`, `rst`, `csv` directly), images (with Apple Vision OCR and a
SmolVLM caption), audio (overlapping 30 s windows), and video (a frame every 4 s plus its
soundtrack) — off by default, see below.

**Speech** — Whisper turbo produces timestamped transcripts, so exact words inside a recording
are findable and a hit lands on the moment.

**Search** — hybrid: vector similarity on the query's own scale, plus an FTS5 keyword index for
codes, IDs and names an embedding cannot hold. Exact lookups ("INV-20931") flip the weights.

**Two modes** — *instant* compares vectors only; *deep* loosens the gate and reranks with a
cross-encoder. Both are in the CLI, the UI and the MCP tools.

**Interfaces** — `mimir ui` serves a local page with search-as-you-type and page/moment links.
`mimir mcp` exposes the same search to Claude and Cursor over stdio.

**Incremental indexing** — a pass only touches files whose size or mtime changed, drops files
that vanished, and loads no model at all when nothing changed. `--watch` keeps it running.

## Getting started

```bash
uv sync --extra embed --extra media --extra stt --extra dev   # core + models + extraction
uv run pytest                                                  # 36 tests

uv run mimir index ~/Downloads          # index a folder (video skipped; --video to include)
uv run mimir search "asset acknowledgment form"
uv run mimir ui                         # http://127.0.0.1:8765
uv run mimir index ~/Downloads --watch  # keep it current as files appear
```

Optional extras:

```bash
python scripts/fetch_reranker.py   # enables deep search (llama.cpp + Qwen3-Reranker, ~680 MB)
python scripts/make_testbed.py     # builds the synthetic testbed the evals use
```

## How it works

```
files ──► extract ──► phase 1  Whisper      transcripts, timestamped
                 ──► phase 2  SmolVLM      one caption per image and per video frame
                 ──► phase 3  EmbeddingGemma 2   one vector per chunk, page, frame, 30 s of sound
                          │
                          ▼
                   SQLite: vectors + FTS5 keyword index + metadata
                          │
query ──► EmbeddingGemma 2 (text) ──► meaning score ──┐
      ──► FTS5 ─────────────────────► exact words ────┴──► fuse ──► (deep: rerank) ──► results
```

Indexing runs in **phases, each owning exactly one model**, so peak memory is the largest single
model rather than their sum. That is what makes the whole thing fit on a laptop, and it is
pinned by a test.

## Design decisions, with the measurements behind them

These were not guesses. Each one is recorded where the constant lives, and re-derivable with
the scripts in `scripts/`.

- **Scores are centred per query, not thresholded raw.** Raw cosines sit in a narrow band — two
  unrelated images measure 0.680 — so absolute thresholds cannot work. DigUp measured this
  approach beating reciprocal-rank fusion (R@1 0.88 vs 0.73), and the same behaviour shows here.
- **A meaning match must clear two gates**: stand out on the query's scale *and* be close in
  absolute terms. Centring alone destroys the "nothing here resembles this" signal.
- **`COS_FLOOR = 0.68`** is the corpus's own typical self-similarity, not a tuned constant.
  `scripts/sweep_gate.py` shows the trade: 0.66 buys three paraphrases but lets an absurd query
  ("the source code for the Linux kernel scheduler") return a flower photo.
- **Video is opt-in.** 48 files / 6.3 hours in one real Downloads folder cost ~2.5 hours to
  index, almost all of it writing frame captions, while every other file there took a minute.
- **Deep mode is opt-in and unproven as a default.** The cross-encoder is good at *ranking*
  (R@5 0.94 vs 0.90, and 3 empty misses instead of 6) but poor at *rejecting* — six of eight
  negatives score as plausibly as a real match. It trades the honest-nothing promise for recall.
- **Whisper runs in its own process** because `mlx_whisper` caches its model in a class
  attribute; in-process it stayed resident across the later phases and peak memory measured
  3846 MB instead of 3037 MB.
- **Captions exist because evals asked for them**, not because they were planned. They lifted
  paraphrase R@1 0.60 → 0.67 and fixed a query that previously returned nothing at all.

## Known limitations

- **Every quality number is measured on synthetic media** (`scripts/make_testbed.py`), not on a
  real corpus. The per-query scaling is corpus-sensitive, so expect different absolutes.
- **Deep mode does not beat instant on R@1**, only on R@5 — see above.
- **`--watch` polls** (default every 30 s) rather than using FSEvents, so new files appear within
  an interval rather than instantly.
- **Scanned PDFs with no text layer** are recorded and findable by file name, but their contents
  are not searchable — no OCR pass on PDF pages yet.
- **Video frames are captioned one at a time**, which is the dominant indexing cost.
- Not supported: `.pages`, `.numbers`, `.key`, code files, email.

## Roadmap

**Next, in the order I'd do them**

1. **Fold the watcher and the UI into one command** — today they are separate processes; daily
   use wants one thing to start.
2. **A menu-bar shell** around the existing web page, for a global hotkey (⇧⌘Space) and no
   browser tab. Wrapping the page, not reimplementing the engine.
3. **Run the evals on a real corpus** — the plan's own Step 0, still unfulfilled.
4. **Decide whether deep becomes the default**, now that frame captions removed the confound.
5. **Derive `COS_FLOOR` per index** instead of hardcoding 0.68.
6. **OCR PDF pages** that have no text layer, instead of only recording them.

**Later**

- Date and place filters from EXIF and mtime ("photos from Goa last December").
- Ask-your-files with citations (a small local LLM over retrieved passages).
- More sources: Mail, Notes, Messages, browser history.
- Apple Vision image classification labels and face detection.
- ColQwen/ColPali page embeddings for tables, slides and charts.
- Near-duplicate finder.
- Spotlight integration.
- An ANN index — unnecessary below roughly a million vectors, where brute force is milliseconds.

## Layout

```
src/mimir/
  store.py       SQLite: files, segments, FTS5, fingerprints
  vectors.py     cosine, the per-query meaning scale, rank fusion
  search.py      hybrid retrieval and fusion
  deep.py        loose gate + reranking
  extract.py     files -> items (PDFKit, textutil, Vision, PyAV)
  indexer.py     phased, incremental indexing
  embed.py       EmbeddingGemma 2 client + a fake embedder for tests
  transcribe.py  Whisper client          whisper_worker.py
  caption.py     SmolVLM client          caption_worker.py
  rerank.py      llama.cpp reranker client
  mcp_server.py  MCP tools  (also the shared search surface)
  ui.py          local web UI
  evals.py       the harness
  cli.py         index / search / eval / info / ui / mcp
scripts/         testbed, probes, threshold sweeps, memory measurement
evals/           smoke.json, media.json
```

## Credits

The embedding worker is derived from [DigUp](https://github.com/ARahim3/DigUp) (MIT,
Copyright (c) 2026 Abdur Rahim) and kept close to upstream because it must match the float32
reference. DigUp's own evals and the `Searcher` comments were the starting point for several of
the design decisions recorded above.
