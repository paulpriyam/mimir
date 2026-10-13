"""Does EmbeddingGemma 2 actually work across modalities? Run this after any model or
mlx-vlm change, before trusting a single search result.

    python scripts/probe_modalities.py [testbed]     default: ~/mimir-testbed

Uses ground_truth.json from scripts/make_testbed.py: image queries are matched against file
names, audio queries against the known transcript. Prints R@1 per modality, so a regression in
the vision or audio tower shows up as a number rather than as "search feels worse".

Worth knowing: raw image-image cosines sit in a narrow band (mean ~0.68, sd ~0.05), and each
modality sits at a different level. That is exactly why search centres every score on the
query's own scale instead of thresholding raw cosine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mimir.embed import WorkerEmbedder

IMAGE_QUERIES = {
    "a penguin": "Animals-Penguin", "a zebra": "Animals-Zebra", "a violin": "Instruments-Violin",
    "a sunflower": "Flowers-Sunflower", "a drum kit": "Instruments-Drum", "an owl": "Animals-Owl",
}
AUDIO_QUERIES = {
    "buying groceries on the way home": "speech_groceries",
    "a flight delayed by a storm": "speech_flight_delay",
    "baking a chocolate cake": "speech_cake_recipe",
    "quarterly revenue grew in Europe": "speech_revenue",
}


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/mimir-testbed").expanduser()
    if not (root / "ground_truth.json").exists():
        print(f"no ground_truth.json in {root}; run scripts/make_testbed.py first", file=sys.stderr)
        return 1
    json.loads((root / "ground_truth.json").read_text())

    embedder = WorkerEmbedder()
    images = sorted((root / "img").glob("*.jpg"))
    audio = sorted((root / "audio").glob("*.wav"))
    image_vectors, image_errors = embedder.embed_media("image", [str(p) for p in images])
    audio_vectors, audio_errors = embedder.embed_media("audio", [str(p) for p in audio])
    print(f"images {image_vectors.shape}, audio {audio_vectors.shape}; "
          f"errors: {sum(e is not None for e in image_errors)} image, {sum(e is not None for e in audio_errors)} audio")

    queries = list(IMAGE_QUERIES) + list(AUDIO_QUERIES)
    query_vectors = unit(embedder.embed_text(queries))
    image_vectors, audio_vectors = unit(image_vectors), unit(audio_vectors)

    def report(label: str, matrix: np.ndarray, files: list[Path], table: dict[str, str], offset: int) -> int:
        print(f"\n=== text -> {label} (top 2 of {len(files)}) ===")
        correct = 0
        for i, (query, expected) in enumerate(table.items()):
            order = np.argsort(-matrix[offset + i])[:2]
            hit = files[order[0]].stem == expected
            correct += hit
            ranked = [(files[j].stem, round(float(matrix[offset + i][j]), 3)) for j in order]
            print(f"{'ok  ' if hit else 'MISS'} {query!r:38} -> {ranked}")
        print(f"{label} R@1: {correct}/{len(table)}")
        return correct

    image_ok = report("image", query_vectors @ image_vectors.T, images, IMAGE_QUERIES, 0)
    audio_ok = report("audio", query_vectors @ audio_vectors.T, audio, AUDIO_QUERIES, len(IMAGE_QUERIES))

    if len(image_vectors) > 1:
        similarity = image_vectors @ image_vectors.T
        off_diagonal = similarity[~np.eye(len(image_vectors), dtype=bool)]
        print(f"\nimage-image cosine: mean {off_diagonal.mean():.3f} sd {off_diagonal.std():.3f} "
              f"min {off_diagonal.min():.3f} max {off_diagonal.max():.3f}")
    embedder.close()
    return 0 if image_ok == len(IMAGE_QUERIES) and audio_ok == len(AUDIO_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
