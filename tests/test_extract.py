"""Extractor tests. The native paths (Vision, PDFKit, PyAV) are skipped when the media extra
is not installed, so a bare `uv sync --extra dev` still runs the rest."""

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from mimir.extract import audio_items, extract, kind_of, pdf_items, text_items, video_items


def test_kind_of_covers_every_supported_family():
    assert kind_of(Path("a.JPG")) == "image"
    assert kind_of(Path("a.heic")) == "image"
    assert kind_of(Path("a.m4a")) == "audio"
    assert kind_of(Path("a.mov")) == "video"
    assert kind_of(Path("a.pdf")) == "pdf"
    assert kind_of(Path("a.md")) == "doc"
    assert kind_of(Path("a.docx")) is None


def test_text_items_chunk_and_carry_an_offset():
    path = Path("/tmp/mimir-test-chunk.txt")
    path.write_text("word " * 400)
    items = text_items(path)
    path.unlink()
    assert len(items) > 1
    assert items[0].loc == 0.0
    assert items[1].loc > 0
    assert all(item.modality == "text" for item in items)


def test_audio_windows_overlap_and_place_themselves():
    pytest.importorskip("av")
    path = Path("/tmp/mimir-test-60s.wav")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000 * 60, np.int16).tobytes())
    items = audio_items(path, Path("/tmp"))
    for item in items:
        Path(item.media).unlink()
    path.unlink()
    assert [item.loc for item in items] == [0.0, 25.0, 50.0]   # 30 s windows, 5 s overlap
    assert all(item.loc_end <= 60.0 for item in items)
    assert all(item.modality == "audio" and item.media_kind == "audio" for item in items)


def test_ocr_reads_text_from_a_screenshot():
    pytest.importorskip("Vision")
    PIL = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw, ImageFont

    path = Path("/tmp/mimir-test-ocr.png")
    image = PIL.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 60)
    draw.text((40, 80), "Invoice INV-20931", fill="black", font=font)
    draw.text((40, 200), "Payment declined", fill="black", font=font)
    image.save(path)

    items = extract(path, "image", Path("/tmp"))
    path.unlink()
    text = " ".join(item.text or "" for item in items)
    assert "INV-20931" in text
    assert any(item.kind == "image" for item in items)      # the picture itself is still indexed


def test_video_frames_are_placed_at_their_moment():
    av = pytest.importorskip("av")
    path = Path("/tmp/mimir-test-video.mp4")

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=8)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    for colour in [(255, 0, 0), (0, 0, 255)]:      # red 0-4 s, blue 4-8 s
        frame = np.zeros((64, 64, 3), np.uint8)
        frame[:, :] = colour
        for _ in range(4 * 8):
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    items = video_items(path, Path("/tmp"))
    path.unlink()
    assert items, "a video with no audio track must still yield frames"
    assert all(item.modality == "image" and item.media_kind == "image" for item in items)
    assert [item.loc for item in items] == [0.0, 4.0]   # one frame every VIDEO_EVERY seconds
    for item in items:
        Path(item.media).unlink()


def test_pdf_pages_are_numbered():
    pytest.importorskip("Quartz")
    source = Path("/tmp/mimir-test-lease.txt")
    pdf = Path("/tmp/mimir-test-lease.pdf")
    source.write_text("LEASE AGREEMENT\n\nSection 7. The tenant may keep pets in the apartment.\n")
    with open(pdf, "wb") as handle:
        subprocess.run(["/usr/sbin/cupsfilter", str(source)], stdout=handle,
                       stderr=subprocess.DEVNULL, check=True)
    items = pdf_items(pdf)
    source.unlink()
    pdf.unlink()
    assert items and items[0].kind == "page" and items[0].loc == 1.0
    assert "pets" in items[0].text
