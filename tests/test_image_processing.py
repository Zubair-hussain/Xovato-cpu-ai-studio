import io

import numpy as np
import pytest
from PIL import Image

from app import image_processing as ip


def _jpeg(width=200, height=150):
    rng = np.random.default_rng(2)
    image = Image.fromarray((rng.random((height, width, 3)) * 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_target_size_source_is_identity():
    assert ip.target_size(1000, 800, "source") == (1000, 800)


def test_target_size_fits_inside_box_preserving_aspect():
    w, h = ip.target_size(4000, 3000, "1080p")  # 4:3 into 1920x1080 box
    assert (w, h) == (1440, 1080)
    assert abs((w / h) - (4000 / 3000)) < 0.01


def test_aspect_label_common_ratios():
    assert ip._aspect_label(1920, 1080) == "16:9"
    assert ip._aspect_label(1000, 1000) == "1:1"


def test_resolve_format_promotes_jpeg_with_alpha():
    assert ip.resolve_format("jpg", keep_alpha=True) == "PNG"
    assert ip.resolve_format("jpg", keep_alpha=False) == "JPEG"
    assert ip.resolve_format("weird", keep_alpha=False) == "PNG"


def test_enhance_image_returns_valid_output_and_analysis():
    payload, analysis = ip.enhance_image(_jpeg(), output_format="jpg", quality=88)
    assert payload[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert analysis.width == 200 and analysis.height == 150
    assert analysis.actions
    assert "corrections" in analysis.summary


def test_enhance_image_png_output():
    payload, _ = ip.enhance_image(_jpeg(), output_format="png")
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_enhance_image_rejects_garbage():
    with pytest.raises(ip.ImageProcessingError):
        ip.enhance_image(b"not an image")


def test_segmentation_model_available_false(tmp_path):
    assert ip.segmentation_model_available(tmp_path / "missing.onnx") is False


def test_encode_image_webp_roundtrip():
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    payload = ip.encode_image(image, output_format="webp", quality=80, keep_alpha=False)
    assert Image.open(io.BytesIO(payload)).size == (32, 32)


def test_encode_rgba_to_jpeg_flattens_alpha():
    image = Image.new("RGBA", (16, 16), (200, 100, 50, 128))
    payload = ip.encode_image(image, output_format="jpg", quality=85, keep_alpha=False)
    assert Image.open(io.BytesIO(payload)).mode == "RGB"


def test_load_image_converts_palette_and_applies_exif():
    palette = Image.new("P", (10, 10))
    buffer = io.BytesIO()
    palette.save(buffer, format="PNG")
    loaded = ip.load_image(buffer.getvalue())
    assert loaded.mode in {"RGB", "RGBA"}


def test_box_filter_averages_uniform_region():
    src = np.ones((20, 20), dtype=np.float32)
    out = ip._box_filter(src, radius=3)
    assert np.allclose(out, 1.0, atol=1e-5)


def test_guided_filter_returns_same_shape():
    rng = np.random.default_rng(6)
    guide = rng.random((30, 30)).astype(np.float32)
    target = rng.random((30, 30)).astype(np.float32)
    out = ip._guided_filter(guide, target, radius=3, eps=1e-4)
    assert out.shape == (30, 30)


def test_tone_curve_lut_is_monotonic():
    lut = ip._tone_curve_lut(0.8, 0.3)[:256]
    assert lut[0] <= lut[128] <= lut[255]


def test_remove_background_reports_missing_model(tmp_path):
    with pytest.raises(ip.ImageProcessingError):
        ip.remove_background(_jpeg(), model_path=tmp_path / "missing.onnx")
