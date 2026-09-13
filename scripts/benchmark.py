"""Measure real CPU throughput of the Xovato backend.

Run from the repo root:

    python scripts/benchmark.py

Writes ``docs/assets/benchmark_data.json``. No GPU, no network, no models
required (background removal / face restoration are skipped when their ONNX
files are absent).
"""

from __future__ import annotations

import io
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from PIL import Image, ImageFilter

from app import audio_processing, image_processing
from app.services.core import VoiceStore

rng = np.random.default_rng(11)


def timeit(fn, repeat: int = 3) -> float:
    runs = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - started)
    return round(statistics.median(runs), 3)


def make_jpeg(megapixels: float) -> tuple[bytes, tuple[int, int]]:
    height = int((megapixels * 1e6 * 3 / 4) ** 0.5)
    width = int(height * 4 / 3)
    image = Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(2))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), (width, height)


def make_wav(seconds: float, sample_rate: int = 48000) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
    signal = 0.5 * envelope * np.sin(2 * np.pi * 200 * t) + 0.05 * rng.standard_normal(t.size)
    data = np.stack([signal, signal * 0.9], axis=1).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def main() -> None:
    results: dict[str, object] = {
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "image_enhance": {},
        "audio_enhance": {},
        "tts_preview": {},
    }

    for megapixels in (1, 3, 8):
        data, (width, height) = make_jpeg(megapixels)
        results["image_enhance"][f"{megapixels}MP ({width}x{height})"] = timeit(
            lambda: image_processing.enhance_image(data), 3
        )

    for seconds in (10, 60, 300):
        data = make_wav(seconds)
        results["audio_enhance"][f"{seconds}s stereo 48k"] = timeit(lambda: audio_processing.enhance_audio(data), 2)

    store = VoiceStore(data_dir=Path(tempfile.mkdtemp()))
    for label, text in (
        ("short (12 words)", "This is a short preview sentence for the tone synthesiser."),
        ("medium (40 words)", " ".join(["word"] * 40)),
    ):
        results["tts_preview"][label] = timeit(lambda: store.write_preview_wav(f"job_{time.time_ns()}", text), 3)

    out = ROOT / "docs" / "assets" / "benchmark_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
