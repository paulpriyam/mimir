"""Constants and paths. One place to change the model, the prompts and the index location."""

from __future__ import annotations

import os
from pathlib import Path

# --- The embedding model -------------------------------------------------------------
# EmbeddingGemma 2: text, images, audio and video in one 768-d space. Apache 2.0, 740M params.
MODEL_ID = "google/embeddinggemma-2"
DIM = 768
DTYPE = "f16"  # how vectors are stored; cosine is done in float32

# Prompts are the caller's business (the worker takes text already prompted).
# Kept identical to DigUp so vectors from either are comparable.
QUERY_PROMPT = "task: search result | query: {text}"
DOC_PROMPT = "title: none | text: {text}"

# --- Index ---------------------------------------------------------------------------
INDEX_DIR = Path(os.environ.get("MIMIR_INDEX", Path.home() / ".mimir")).expanduser()
MAIN_DB = "index.sqlite"

# --- Retrieval -----------------------------------------------------------------------
# How far the best segment must stand out (in per-query standard deviations) to count as a
# meaning match. Also what makes "nothing found" honest. DigUp's default, kept.
Z_GATE = 2.0
# How much finding the query's words counts against meaning (Searcher.wordWeight in DigUp).
WORD_WEIGHT = 2.0
# Share of a query's words that must be present for a file to qualify on words alone.
COVERAGE_GATE = 0.5
# Candidates handed to the reranker. Prefill is the cost, so this stays small.
RERANK_TOP = 25
