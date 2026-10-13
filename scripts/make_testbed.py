"""Build a small testbed with known content, using only tools macOS ships.

No ffmpeg, no Homebrew, no downloads:
  images       /Library/User Pictures -> sips -> jpg   (what the picture shows is in the file name)
  audio        say (Samantha TTS)      -> afconvert    (the transcript is known exactly)
  screenshots  PIL, large text                          (text is known, for the OCR path)
  pdf          cupsfilter -> PDFKit                     (page text, and which page holds a marker)

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
    "shot_invoice": ["ACME Supplies", "Invoice INV-20931", "Total due 4200.00", "Payment declined by the bank"],
    "shot_meeting": ["Team standup", "9:30 AM - 9:45 AM", "Notes: ship the retry fix today", "Owner: Priya"],
    "shot_weather": ["Weather", "Bengaluru 28 C", "Light rain expected", "Humidity 74 percent"],
}

DOCUMENTS = {
    "lease": ["LEASE AGREEMENT", "",
              "Section 7. The tenant may keep pets in the apartment with the landlord's written consent.",
              "Rent is due on the first day of each month and is payable by bank transfer.",
              "The deposit is held by the landlord's agent for the duration of the tenancy."],
    "handbook": None,  # generated long, with a marker on a later page
}

HANDBOOK_MARKER = "Section 42 covers the fire drill procedure and the assembly point"


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_font(size: int):
    from PIL import ImageFont

    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc",
                      "/System/Library/Fonts/SFNS.ttf"]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


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
        print("note: pillow missing; no screenshots", file=sys.stderr)
        return {}
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    font = find_font(46)
    for name, lines in SCREENSHOTS.items():
        image = Image.new("RGB", (1400, 900), "white")
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            draw.text((60, 60 + i * 110), line, fill="black", font=font)
        image.save(out / f"{name}.png")
        truth[f"screenshots/{name}.png"] = "\n".join(lines)
    return truth


def make_pdfs(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    for name, lines in DOCUMENTS.items():
        if lines is None:
            body = [f"Paragraph {i}. This handbook section is filler text, present so the document runs "
                    f"to several pages and page-level search has something to get wrong." for i in range(1, 240)]
            body.insert(170, HANDBOOK_MARKER + " in the north car park.")
            lines = ["STAFF HANDBOOK", ""] + body
        sources[name] = lines
        text_path = out / f"{name}.txt"
        text_path.write_text("\n".join(lines))
        with open(out / f"{name}.pdf", "wb") as handle:
            subprocess.run(["/usr/sbin/cupsfilter", str(text_path)], stdout=handle,
                           stderr=subprocess.DEVNULL, check=True)
        text_path.unlink()
        truth[f"pdf/{name}.pdf"] = " ".join(lines)[:200]

    # Which page the marker landed on: ground truth for page-level retrieval, not a guess.
    try:
        from Foundation import NSURL
        from Quartz import PDFDocument

        document = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(str(out / "handbook.pdf")))
        for index in range(document.pageCount()):
            text = document.pageAtIndex_(index).string() or ""
            if HANDBOOK_MARKER in text:
                truth["pdf/handbook.pdf"] = f"{HANDBOOK_MARKER} (page {index + 1})"
                truth["pdf/handbook.pdf@marker"] = str(index + 1)
                break
    except Exception as error:
        print(f"note: could not locate the handbook marker: {error}", file=sys.stderr)
    return truth


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/mimir-testbed").expanduser()
    if root.exists():
        print(f"{root} already exists; delete it first (rm -rf {root})", file=sys.stderr)
        return 1
    root.mkdir(parents=True)
    truth: dict[str, str] = {}
    truth |= make_images(root / "img")
    truth |= make_audio(root / "audio")
    truth |= make_screenshots(root / "screenshots")
    truth |= make_pdfs(root / "pdf")
    (root / "ground_truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False))
    print(f"testbed in {root}: {len(truth)} files with known content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
