"""Download the reranker runtime: a prebuilt llama.cpp for macOS arm64, plus the model.

No Homebrew and no build step. llama.cpp publishes macOS arm64 binaries on every release, and
the reranker is ggml-org's own GGUF conversion of Qwen3-Reranker-0.6B (Apache-2.0, ~640 MB).

    python scripts/fetch_reranker.py

Installs into ~/.mimir:
    tools/llama/llama-server                 the server, run with --reranking
    models/qwen3-reranker-0.6b-q8_0.gguf     the model

Both paths can be overridden with MIMIR_LLAMA_SERVER and MIMIR_RERANK_MODEL.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("MIMIR_HOME", Path.home() / ".mimir")).expanduser()
TOOLS = HOME / "tools" / "llama"
MODELS = HOME / "models"
REPO = "ggml-org/llama.cpp"
ASSET_SUFFIX = "-bin-macos-arm64.tar.gz"
RERANK_REPO = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
RERANK_FILE = "qwen3-reranker-0.6b-q8_0.gguf"


def latest_macos_build() -> str:
    request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases?per_page=20",
                                     headers={"User-Agent": "mimir"})
    releases = json.load(urllib.request.urlopen(request, timeout=60))
    for release in releases:
        for asset in release["assets"]:
            if asset["name"].endswith(ASSET_SUFFIX):
                return release["tag_name"]
    raise SystemExit(f"no macOS arm64 build found in the last 20 {REPO} releases")


def fetch_llama() -> Path:
    binary = TOOLS / "llama-server"
    if binary.exists():
        print(f"llama-server already present: {binary}")
        return binary
    tag = latest_macos_build()
    url = f"https://github.com/{REPO}/releases/download/{tag}/llama-{tag}{ASSET_SUFFIX}"
    print(f"downloading {tag} ...")
    TOOLS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "llama.tar.gz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not m.name.startswith(".")]
            tar.extractall(temp, members=members, filter="data")
        extracted = next(Path(temp).glob("llama-*"))
        for item in extracted.iterdir():
            shutil.move(str(item), str(TOOLS / item.name))
    if not binary.exists():
        raise SystemExit(f"llama-server not found after extracting {tag}")
    binary.chmod(0o755)
    # The dylibs sit beside the binary; macOS needs the loader to look there first.
    for dylib in TOOLS.glob("*.dylib"):
        subprocess.run(["install_name_tool", "-id", f"@rpath/{dylib.name}", str(dylib)],
                       capture_output=True)
    print(f"llama-server ready: {binary}")
    return binary


def fetch_model() -> Path:
    target = MODELS / RERANK_FILE
    if target.exists():
        print(f"reranker model already present: {target}")
        return target
    from huggingface_hub import hf_hub_download

    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RERANK_REPO} ...")
    hf_hub_download(RERANK_REPO, RERANK_FILE, local_dir=MODELS)
    print(f"model ready: {target}  ({target.stat().st_size / 1e6:.0f} MB)")
    return target


def main() -> int:
    fetch_llama()
    fetch_model()
    print("\ndone. deep search is now available:")
    print('  mimir search --deep "percussion kit you sit behind"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
