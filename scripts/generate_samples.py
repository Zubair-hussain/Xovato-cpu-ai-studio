"""Generate real demo samples for every feature that can run without a GPU.

Outputs go to docs/assets/samples/. Everything here is produced by the actual
backend functions - there is no faking. Features that need a model file (u2netp,
yunet, gpen) run only when that file is present; otherwise the input sample is
still written and the step is reported as skipped.

Run from the repo root:

    python scripts/generate_samples.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app import audio_processing, image_processing

OUT = ROOT / "docs" / "assets" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)


def _font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# 1. Image enhancement - a degraded "photo" -> enhanced. Genuinely run.
# --------------------------------------------------------------------------- #
def make_degraded_photo(w=1280, h=853) -> Image.Image:
    """A synthetic but photo-like scene: sky gradient, ground, sun, hills."""
    ys = np.linspace(0, 1, h)[:, None]
    sky = np.stack([0.30 + 0.15 * ys, 0.40 + 0.20 * ys, 0.55 + 0.25 * ys], axis=-1) * np.ones((1, w, 1))
    img = (sky * 255).astype(np.uint8)
    image = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(image)
    # ground
    draw.rectangle([0, int(h * 0.68), w, h], fill=(70, 90, 55))
    # rolling hills
    for i, color in enumerate([(90, 110, 70), (60, 80, 50)]):
        cx = w * (0.3 + 0.4 * i)
        draw.ellipse([cx - 500, h * 0.6 + i * 40, cx + 500, h * 0.95], fill=color)
    # sun
    draw.ellipse([w * 0.72, h * 0.12, w * 0.72 + 120, h * 0.12 + 120], fill=(230, 210, 160))
    # a subject
    draw.rectangle([w * 0.44, h * 0.55, w * 0.5, h * 0.7], fill=(40, 40, 45))
    draw.ellipse([w * 0.44, h * 0.5, w * 0.5, h * 0.57], fill=(60, 55, 50))

    arr = np.asarray(image, dtype=np.float32)
    # Flatten contrast (the "flat/dull" look), add haze and sensor noise, soften.
    arr = arr * 0.55 + 90.0  # low contrast, lifted blacks (hazy)
    arr += rng.normal(0, 7, arr.shape)  # sensor noise
    degraded = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.4)  # soft focus
    )
    return degraded


def image_demo():
    before = make_degraded_photo()
    before_bytes = io.BytesIO()
    before.save(before_bytes, format="JPEG", quality=90)
    (OUT / "image_before.jpg").write_bytes(before_bytes.getvalue())

    after_bytes, analysis = image_processing.enhance_image(
        before_bytes.getvalue(), output_format="jpg", quality=92, strength=1.0
    )
    (OUT / "image_after.jpg").write_bytes(after_bytes)

    # Side-by-side board with labels.
    after = Image.open(io.BytesIO(after_bytes)).convert("RGB")
    before_rgb = before.convert("RGB")
    gap, pad, header = 16, 20, 44
    bw, bh = before_rgb.size
    board = Image.new("RGB", (bw * 2 + gap + pad * 2, bh + header + pad * 2), (245, 246, 248))
    board.paste(before_rgb, (pad, header + pad))
    board.paste(after, (pad + bw + gap, header + pad))
    d = ImageDraw.Draw(board)
    d.text((pad, 14), "BEFORE (degraded input)", font=_font(22), fill=(120, 40, 40))
    d.text((pad + bw + gap, 14), "AFTER (image enhancement)", font=_font(22), fill=(30, 110, 60))
    board.save(OUT / "image_before_after.png")
    print("image: wrote image_before.jpg, image_after.jpg, image_before_after.png")
    print("       ", analysis.summary)


# --------------------------------------------------------------------------- #
# 2. Audio enhancement - a clear 60s clip -> cleaned. Genuinely run.
# --------------------------------------------------------------------------- #
def make_noisy_speech(seconds=60, sr=44100) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    # Speech-like: a fundamental with formant-ish harmonics, amplitude-gated into
    # "syllables", so loudness normalisation and the limiter have real work.
    syllable = np.sin(2 * np.pi * 3.2 * t) ** 8  # ~3 syllables/sec bursts
    voice = (
        0.6 * np.sin(2 * np.pi * 130 * t) + 0.3 * np.sin(2 * np.pi * 260 * t) + 0.15 * np.sin(2 * np.pi * 520 * t)
    ) * syllable
    hiss = 0.08 * rng.standard_normal(t.size)  # broadband noise
    hum = 0.05 * np.sin(2 * np.pi * 50 * t)  # mains hum
    rumble = 0.06 * np.sin(2 * np.pi * 30 * t)  # low rumble
    mono = 0.35 * voice + hiss + hum + rumble
    return np.stack([mono, mono], axis=1).astype(np.float32), sr


def waveform_png(before, after, sr, path):
    W, H = 1000, 300
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)

    def draw(sig, y0, y1, color, label):
        mono = sig[:, 0]
        n = len(mono)
        step = max(1, n // W)
        mid = (y0 + y1) / 2
        amp = (y1 - y0) / 2 * 0.9
        d.line([(0, mid), (W, mid)], fill=(225, 228, 233))
        pts = []
        for x in range(W):
            seg = mono[x * step : (x + 1) * step]
            v = float(np.max(np.abs(seg))) if len(seg) else 0.0
            pts.append((x, mid - v * amp))
            pts.append((x, mid + v * amp))
        d.line(pts, fill=color, width=1)
        d.text((8, y0 + 4), label, font=_font(16), fill=color)

    draw(before, 10, 145, (170, 60, 60), "BEFORE - noisy, uneven level (60s)")
    draw(after, 155, 290, (40, 130, 80), "AFTER - denoised + loudness-normalised")
    img.save(path)


def audio_demo():
    samples, sr = make_noisy_speech(60)
    before = io.BytesIO()
    sf.write(before, samples, sr, format="WAV", subtype="PCM_16")
    (OUT / "audio_before.wav").write_bytes(before.getvalue())

    payload, fmt, analysis = audio_processing.enhance_audio(
        before.getvalue(), denoise_strength=0.7, target_lufs=-16.0, output_format="wav"
    )
    (OUT / "audio_after.wav").write_bytes(payload)
    after, _ = sf.read(io.BytesIO(payload), always_2d=True)

    waveform_png(samples, after, sr, OUT / "audio_before_after.png")
    dur = len(samples) / sr
    print(f"audio: wrote audio_before.wav / audio_after.wav ({dur:.0f}s each) + waveform png")
    print("       ", analysis.summary)


# --------------------------------------------------------------------------- #
# 3. Background removal - runs only if the u2netp model is present.
# --------------------------------------------------------------------------- #
def make_subject_image(w=800, h=800) -> Image.Image:
    """A salient subject on a busy background - what u2netp is meant to isolate."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:] = [120, 150, 180]
    for _ in range(400):  # clutter
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = int(rng.integers(6, 40))
        color = rng.integers(60, 210, 3)
        Image.fromarray(bg.astype(np.uint8))  # noop to keep types happy
        img = Image.fromarray(bg.astype(np.uint8))
        d = ImageDraw.Draw(img)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c) for c in color))
        bg = np.asarray(img, dtype=np.float32)
    image = Image.fromarray(bg.astype(np.uint8))
    d = ImageDraw.Draw(image)
    # A clear central "subject": a person-ish silhouette.
    d.ellipse([w * 0.4, h * 0.18, w * 0.6, h * 0.38], fill=(235, 205, 180))  # head
    d.rectangle([w * 0.36, h * 0.36, w * 0.64, h * 0.8], fill=(200, 70, 70))  # body
    return image


def checker(size, cell=20):
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2 == 0:
                d.rectangle([x, y, x + cell, y + cell], fill=(210, 214, 220))
    return img


def bg_demo():
    from app.config import get_settings

    settings = get_settings()
    model = settings.segmentation_model_path
    subject = make_subject_image()
    src = io.BytesIO()
    subject.save(src, format="JPEG", quality=92)
    (OUT / "bg_input.jpg").write_bytes(src.getvalue())

    if not image_processing.segmentation_model_available(model):
        print(f"bg: wrote bg_input.jpg; SKIPPED cutout - model missing at {model}")
        print("    download u2netp.onnx (see docs/models.md) then re-run to produce bg_cutout.png")
        return

    payload, analysis = image_processing.remove_background(src.getvalue(), model_path=model, output_format="png")
    (OUT / "bg_cutout.png").write_bytes(payload)
    cutout = Image.open(io.BytesIO(payload)).convert("RGBA")
    board = checker(cutout.size).convert("RGBA")
    board.alpha_composite(cutout)
    board.convert("RGB").save(OUT / "bg_cutout_on_checker.png")
    print("bg: wrote bg_input.jpg, bg_cutout.png, bg_cutout_on_checker.png")
    print("    ", analysis.summary)


# --------------------------------------------------------------------------- #
# 4. Text-to-speech / voice cloning - the CPU preview engine.
# --------------------------------------------------------------------------- #
def tts_demo():
    """Write the real CPU preview-path audio the backend produces without the
    neural model. On Windows this is the System.Speech (SAPI) voice; elsewhere it
    is the tonal placeholder. It is NOT the OmniVoice model - that needs the
    2.5 GB checkpoint (see docs/gpu.md / models/model_test.ipynb)."""
    import tempfile

    from app.services.core import VoiceStore

    store = VoiceStore(data_dir=Path(tempfile.mkdtemp()))
    tts_uri = store.write_preview_wav(
        "sample_tts", "Welcome to Xovato. This voice is generated locally on your own machine."
    )
    clone_uri = store.write_preview_wav("sample_clone", "This is the cloning workflow reading a brand new sentence.")
    for uri, name in ((tts_uri, "tts_preview.wav"), (clone_uri, "clone_preview.wav")):
        src = store.resolve_local_uri(uri)
        (OUT / name).write_bytes(Path(src).read_bytes())
    print("tts: wrote tts_preview.wav, clone_preview.wav (CPU preview engine - not the neural model)")


if __name__ == "__main__":
    image_demo()
    audio_demo()
    bg_demo()
    tts_demo()
    print("\nSamples written to", OUT)
