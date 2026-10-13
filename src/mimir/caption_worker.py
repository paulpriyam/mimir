"""mimir caption worker: a small vision-language model writes one short factual sentence per
image.

Why captions at all: the embedding space puts every image in a narrow band (two unrelated
photos measure cos 0.680), so a text query against an image is a weak signal, and a picture
cannot be keyword-matched because it contains no words. A caption gives every image real text:
it becomes searchable by words it never contained, and - the point for the reranker - it
becomes a text candidate that a text-only reranker can actually rescore.

Runs in its own process, like the embedding worker, so the model is guaranteed to be released
when the caption phase ends rather than lingering in the indexing process.

Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.
  {"id": 1, "op": "caption", "inputs": ["/tmp/a.jpg"], "prompt": "..."}
  {"id": 2, "op": "quit"}
Reply: {"id": 1, "ok": true, "count": N, "captions": ["..."], "errors": [null]}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = ("Describe this image in one short factual sentence. "
                  "Name the main subject, and mention colours only if they are distinctive.")


def log(*parts):
    print("[caption]", *parts, file=sys.stderr, flush=True)


def resolve_model(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        log(f"downloading {model} (first run only)")
        return snapshot_download(model)


class Captioner:
    def __init__(self, model_id):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        started = time.perf_counter()
        path = resolve_model(model_id)
        self.model, self.processor = load(path)
        self.config = load_config(path)
        self.info = {
            "model": model_id,
            "revision": Path(path).name,
            "load_seconds": round(time.perf_counter() - started, 2),
        }

    def caption(self, path, prompt, max_tokens):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        text = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        out = generate(self.model, self.processor, text, image=[path], max_tokens=max_tokens, verbose=False)
        return (getattr(out, "text", "") or str(out)).strip()


def reply(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/SmolVLM-Instruct-4bit")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    try:
        captioner = Captioner(args.model)
    except Exception as error:
        reply({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    log(f"ready in {captioner.info['load_seconds']} s")
    reply({"ready": True, "info": captioner.info})

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
                reply({"id": request_id, "ok": True, "info": captioner.info})
                continue
            if op != "caption":
                raise ValueError(f"unknown op {op!r}")
            prompt = request.get("prompt") or args.prompt
            captions, errors = [], []
            for path in request.get("inputs") or []:
                try:
                    captions.append(captioner.caption(path, prompt, args.max_tokens))
                    errors.append(None)
                except Exception as item_error:
                    captions.append("")
                    errors.append(f"{type(item_error).__name__}: {item_error}"[:300])
            reply({"id": request_id, "ok": True, "count": len(captions),
                   "captions": captions, "errors": errors})
        except Exception as error:
            reply({"id": request_id, "ok": False, "error": f"{type(error).__name__}: {error}"[:500]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
