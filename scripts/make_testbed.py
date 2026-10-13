"""Build a small testbed with known content, using only tools macOS ships.

No ffmpeg, no Homebrew, no downloads:
  images   /Library/User Pictures -> sips -> jpg   (what the picture shows is in the file name)
  audio    say (Samantha TTS)      -> afconvert    (the transcript is known exactly)
  shots    PIL                                     (text is known, for the OCR pipeline later)

Writes ground_truth.json next to them: {relative path: known content}. Evals reference these
paths, so every number the harness prints has something to check against.

    python scripts/make_testbed.py [dir]     default: ~/mimir-testbed
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PICTURE_DIRS = ["Animals", "Flowers", "Instruments", "Nature", "Sports", "Fun"]

SPEECH = {
    "speech_groceries": "Don't forget to buy milk, eggs and bread on the way home tomorrow.",
    "speech_revenue": "Our quarterly revenue grew by twenty percent, driven by strong sales in Europe.",
    "speech_flight_delay": "Attention passengers. The flight to Tokyo has been delayed by two hours because of a storm.",
    "speech_cake_recipe": "In this recipe we will bake a chocolate cake. First, preheat the oven to one hundred "
                          "eighty degrees, then mix the flour, sugar and cocoa powder in a large bowl. Add the "
                          "eggs, the melted butter and the milk, and stir until the batter is smooth. Pour it into "
                          "a greased pan and bake for thirty five minutes. When the cake has cooled down "
                          "completely, spread the frosting on top and decorate it with fresh strawberries.",
}

SCREENSHOTS = {
    "shot_invoice": ["ACME Supplies", "Invoice INV-20931", "Total due 4200.00", "Payment declined"],
    "shot_meeting": ["Team standup", "9:30 AM - 9:45 AM", "Notes: ship the retry fix", "Owner: Priya"],
    "shot_weather": ["Weather", "Bengaluru 28 C", "Light rain expected", "Humidity 74%"],
}


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_images(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    library = Path("/Library/User Pictures")
    if not library.exists():
        print("warning: /Library/User Pictures is missing; no sample images", file=sys.stderr)
        return truth
    for folder in PICTURE_DIRS:
        for source in sorted((library / folder).glob("*.heic")):
            name = f"{folder}-{source.stem}.jpg"
            run("sips", "-s", "format", "jpeg", "-Z", "768", str(source), "--out", str(out / name))
            truth[f"img/{name}"] = source.stem
    return truth


def make_audio(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    for name, sentence in SPEECH.items():
        aiff = out / f"{name}.aiff"
        run("say", "-v", "Samantha", "-o", str(aiff), sentence)
        run("afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(out / f"{name}.wav"))
        aiff.unlink(missing_ok=True)
        truth[f"audio/{name}.wav"] = sentence
    return truth


def make_screenshots(out: Path) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("note: pillow missing (uv sync --extra embed); no screenshots", file=sys.stderr)
        return {}
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    for name, lines in SCREENSHOTS.items():
        image = Image.new("RGB", (900, 600), "white")
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            draw.text((40, 40 + i * 70), line, fill="black")
        image.save(out / f"{name}.png")
        truth[f"screenshots/{name}.png"] = "\n".join(lines)
    return truth


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/mimir-testbed").expanduser()
    if root.exists():
        print(f"{root} already exists; delete it first (rm -rf {root})", file=sys.stderr)
        return 1
    root.mkdir(parents=True)
    truth = {}
    truth |= make_images(root / "img")
    truth |= make_audio(root / "audio")
    truth |= make_screenshots(root / "screenshots")
    (root / "ground_truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False))
    print(f"testbed in {root}: {len(truth)} files with known content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
