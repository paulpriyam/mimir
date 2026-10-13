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

# --- Speech ---------------------------------------------------------------------------
# Timestamped transcripts for audio and video, so exact words inside speech are findable.
# Off with MIMIR_TRANSCRIBE=0. Turbo is 809M params and covers 99 languages including Hindi
# and Bengali; Parakeet is faster but only covers 25 European languages.
WHISPER_MODEL = os.environ.get("MIMIR_WHISPER", "mlx-community/whisper-large-v3-turbo")
TRANSCRIBE = os.environ.get("MIMIR_TRANSCRIBE", "1") != "0"

# --- Captions -------------------------------------------------------------------------
# A small vision-language model writes one short sentence per image. This gives a picture real
# text: it becomes findable by words it never contained, and it becomes a text candidate the
# reranker can score at all (a text-only reranker cannot touch a raw image).
#
# SmolVLM 2.2B, 4-bit, Apache-2.0. The MLX build of SmolVLM2 proper is not on the Hub, and the
# Qwen2.5-VL builds carry no license tag, so this is the plan's intent with a clear licence.
# Swap for a bigger model if captions prove too weak; the eval decides.
CAPTION_MODEL = os.environ.get("MIMIR_CAPTION_MODEL", "mlx-community/SmolVLM-Instruct-4bit")
CAPTION_PROMPT = ("Describe this image in one short factual sentence. "
                  "Name the main subject, and mention colours only if they are distinctive.")
CAPTION_MAX_TOKENS = 64
CAPTION = os.environ.get("MIMIR_CAPTION", "1") != "0"

# --- Retrieval -----------------------------------------------------------------------
# How far the best segment must stand out (in per-query standard deviations) to count as a
# meaning match. Also what makes "nothing found" honest. DigUp's default, kept.
Z_GATE = 2.0
# Centring every score on the query's own scale (Z_GATE) throws away the absolute level, and
# that level is the only thing that says "nothing here resembles this query at all". So a
# meaning match must clear both: stand out, and be close enough in absolute terms.
#
# 0.68 is not tuned to the eval set; it is the corpus's own typical self-similarity. Measured
# on the testbed, two unrelated images sit at cos 0.680 (sd 0.050). A query that scores below
# that against an image is not closer to it than the images are to each other, so calling it a
# match is unjustified. Ideally this is derived per index rather than hardcoded; for now it is
# a constant with the reasoning written down.
#
# The measured trade-off, so nobody re-litigates it blind. On evals/media.json the eight
# negatives top out at 0.663. Lowering the floor to 0.665 would win three paraphrases
# ("percussion kit you sit behind" 0.674, "bird of prey that hunts at night" 0.665, "what
# should I pick up at the store" 0.669), lifting paraphrase R@1 from 0.60 to 0.67 and R@5 from
# 0.67 to 0.87 - but it would sit 0.002 above the worst negative, which is overfitting to eight
# samples. Raising it costs real answers. Kept at 0.68 for the margin.
#
# What the floor does NOT fix: two paraphrase misses had the expected file below the strongest
# negative (0.648 and 0.595), which no gate can repair. Those need better retrieval, not a
# different threshold.
COS_FLOOR = 0.68
# How much finding the query's words counts against meaning (Searcher.wordWeight in DigUp).
WORD_WEIGHT = 2.0
# Share of a query's words that must be present for a file to qualify on words alone.
COVERAGE_GATE = 0.5
# Candidates handed to the reranker. Prefill is the cost, so this stays small.
RERANK_TOP = 25
