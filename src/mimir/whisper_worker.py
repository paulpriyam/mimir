"""mimir speech worker: Whisper in its own process.

This exists because of a measured problem. mlx_whisper caches its model in a class attribute
(transcribe.py, ModelHolder.get_model), so running it inside the indexing process kept Whisper
resident for the whole run - overlapping the caption and embedding phases. Phased indexing was
true in code but not in memory: peak tree RSS measured 3846 MB, close to the sum of all three
models, when it should have been the largest single one.

In a process of its own, the model goes away when the phase ends, exactly like the caption and
embedding workers.

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.
  {"id": 1, "op": "transcribe", "inputs": ["/tmp/a.wav"]}
  {"id": 2, "op": "quit"}
Reply: {"id": 1, "ok": true, "results": [[{"start": 0.0, "end": 5.1, "text": "..."}]], "errors": [null]}
"""

import argparse
import json
import sys
import time

import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def log(*parts):
    print("[speech]", *parts, file=sys.stderr, flush=True)


class Whisper:
    def __init__(self, model):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model = model
        started = time.perf_counter()
        # A tenth of a second of silence loads the weights and warms the kernels, so `ready`
        # means ready and the first real file does not pay for it.
        mlx_whisper.transcribe(np.zeros(1600, np.float32), path_or_hf_repo=model)
        self.info = {"model": model, "load_seconds": round(time.perf_counter() - started, 2)}

    def transcribe(self, path):
        from mimir.extract import decode_audio

        samples, _ = decode_audio(path)
        if samples.size == 0:
            return []
        result = self.mlx_whisper.transcribe(samples.astype(np.float32) / 32768.0,
                                             path_or_hf_repo=self.model)
        segments = []
        for segment in result.get("segments") or []:
            text = (segment.get("text") or "").strip()
            if text:
                segments.append({"start": float(segment["start"]), "end": float(segment["end"]),
                                 "text": text})
        return segments


def reply(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        whisper = Whisper(args.model)
    except Exception as error:
        reply({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    log(f"ready in {whisper.info['load_seconds']} s")
    reply({"ready": True, "info": whisper.info})

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id, op = request.get("id"), request.get("op")
            if op == "quit":
                reply({"id": request_id, "ok": True})
                break
            if op == "info":
                reply({"id": request_id, "ok": True, "info": whisper.info})
                continue
            if op != "transcribe":
                raise ValueError(f"unknown op {op!r}")
            results, errors = [], []
            for path in request.get("inputs") or []:
                try:
                    results.append(whisper.transcribe(path))
                    errors.append(None)
                except Exception as item_error:
                    message = f"{type(item_error).__name__}: {item_error}"
                    # A silent clip is normal, not a failure.
                    if "no audio stream" in message:
                        results.append([])
                        errors.append(None)
                    else:
                        results.append([])
                        errors.append(message[:300])
            reply({"id": request_id, "ok": True, "results": results, "errors": errors})
        except Exception as error:
            reply({"id": request_id, "ok": False, "error": f"{type(error).__name__}: {error}"[:500]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
