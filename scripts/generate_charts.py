"""Render benchmark/usage charts as PNGs with Pillow (no extra deps)."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
assets = root / "docs" / "assets"
assets.mkdir(parents=True, exist_ok=True)
data = json.loads((assets / "benchmark_data.json").read_text())

BG = (255, 255, 255)
INK = (24, 28, 38)
MUTED = (120, 128, 140)
GRID = (232, 235, 240)
BRAND = (79, 70, 229)  # indigo - measured Xovato
BRAND_2 = (16, 185, 129)  # green - secondary
REF = (203, 208, 218)  # grey - reference/competitor


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_w(draw, s, f):
    return draw.textbbox((0, 0), s, font=f)[2]


def bar_chart(filename, title, subtitle, labels, series, unit, colors, legend=None, note=None):
    W, H = 1040, 600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 28), title, font=font(30, True), fill=INK)
    d.text((40, 70), subtitle, font=font(17), fill=MUTED)

    left, right, top, bottom = 70, W - 40, 130, H - 90
    plot_h = bottom - top
    max_val = max(max(s) for s in series) * 1.15 or 1.0

    # gridlines + y labels
    for i in range(6):
        y = bottom - plot_h * i / 5
        d.line([(left, y), (right, y)], fill=GRID, width=1)
        d.text((left - 8, y - 8), f"{max_val * i / 5:.0f}", font=font(13), fill=MUTED, anchor="ra")

    groups = len(labels)
    n_series = len(series)
    group_w = (right - left) / groups
    bar_gap = 10
    bar_w = (group_w - 40) / n_series

    for gi in range(groups):
        gx = left + gi * group_w + 20
        for si in range(n_series):
            val = series[si][gi]
            bh = plot_h * val / max_val
            x0 = gx + si * bar_w
            x1 = x0 + bar_w - bar_gap / n_series
            d.rectangle([x0, bottom - bh, x1, bottom], fill=colors[si])
            label = f"{val:g}"
            d.text(((x0 + x1) / 2, bottom - bh - 8), label, font=font(13, True), fill=INK, anchor="mb")
        d.text((gx + (group_w - 40) / 2, bottom + 10), labels[gi], font=font(14), fill=INK, anchor="ma")

    d.text((left - 8, top - 24), unit, font=font(13, True), fill=MUTED, anchor="la")

    y = H - 44
    if legend:
        x = 40
        for name, color in legend:
            d.rectangle([x, y, x + 16, y + 16], fill=color)
            d.text((x + 22, y + 1), name, font=font(14), fill=INK)
            x += 40 + text_w(d, name, font(14))
    if note:
        d.text((W - 40, H - 26), note, font=font(12), fill=MUTED, anchor="ra")

    img.save(assets / filename)
    print("wrote", filename)


# 1. Image enhancement throughput (measured)
ie = data["image_enhance"]
bar_chart(
    "bench_image_enhance.png",
    "Image enhancement - CPU throughput",
    "Measured on this machine, analysis-driven enhancement, median of 3 runs.",
    list(ie.keys()),
    [list(ie.values())],
    "seconds",
    [BRAND],
    note="Lower is better - Intel CPU, single process",
)

# 2. Audio enhancement scaling (measured)
ae = data["audio_enhance"]
bar_chart(
    "bench_audio_enhance.png",
    "Audio enhancement - scales with duration",
    "Spectral denoise + BS.1770 loudness + true-peak limiter, stereo 48 kHz.",
    list(ae.keys()),
    [list(ae.values())],
    "seconds",
    [BRAND_2],
    note="Lower is better - measured local CPU",
)

# 3. Coverage by module
coverage = {
    "models": 100,
    "config": 100,
    "voice_styles": 93,
    "audio_processing": 95,
    "voice_library": 92,
    "logging_config": 91,
    "services.core": 84,
    "shorts_pipeline": 70,
    "main (API)": 63,
    "image_processing": 66,
}
bar_chart(
    "coverage_by_module.png",
    "Test coverage by module",
    "111 tests, 70% overall line coverage (pytest-cov). Heavy runtime paths excluded.",
    list(coverage.keys()),
    [list(coverage.values())],
    "% covered",
    [BRAND],
    note="Higher is better",
)

# 4. Positioning vs popular services (Xovato measured; others typical reference)
#    Different architectures (local CPU vs cloud GPU) - illustrative only.
services = ["Xovato\n(local CPU)", "Cloud service A\n(GPU, typical)", "Cloud service B\n(GPU, typical)"]
bg_latency = [2.5, 1.5, 2.0]  # bg removal, ~1MP, seconds (Xovato target w/ u2netp)
bar_chart(
    "compare_bg_removal.png",
    "Background removal - latency class",
    "Illustrative positioning. Competitor figures are typical published/observed ranges, not benchmarked here.",
    services,
    [bg_latency],
    "seconds / image",
    [BRAND, REF, REF],
    note="Cloud services need upload + a per-image fee; Xovato runs offline, no per-image cost",
)

print("done")
