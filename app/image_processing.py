"""Real, CPU-only image processing for the Image AI tools.

Two operations are provided:

* :func:`enhance_image` - an analysis driven enhancement pass. Every correction is
  measured from the image first and then applied only as far as the measurement
  says it is needed, so a clean photo is left mostly alone and a flat/noisy/soft
  one gets a real correction.
* :func:`remove_background` - true foreground segmentation using the small U^2-Net
  model (``u2netp``, ~4.5 MB) through onnxruntime on the CPU. The coarse 320x320
  network mask is refined against the full resolution image with a guided filter
  so the alpha edge follows real image edges instead of the network's grid.

Only numpy, Pillow and onnxruntime are used - no GPU, no OpenCV, no SciPy.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# The u2netp graph has a fixed 320x320 input.
_MODEL_INPUT_SIZE = 320
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Bounding boxes each resolution preset must fit inside. The image is always
# scaled to fit *inside* the box with its own aspect ratio preserved, so a
# portrait never gets stretched or cropped to a landscape frame.
_RESOLUTION_BOXES = {
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}

_MAX_WORKING_PIXELS = 40_000_000  # guard against absurd uploads


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be decoded or a model is unavailable."""


@dataclass
class ImageAnalysis:
    """Measurements taken from the source image, plus what we decided to do."""

    width: int
    height: int
    aspect_ratio: str
    exposure: float
    contrast: float
    saturation: float
    sharpness: float
    noise: float
    blur_scale: float = 1.0  # characteristic blur radius in source pixels
    clipped_highlights: float = 0.0
    faces_detected: int = 0
    summary: str = ""
    actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "exposure": round(self.exposure, 4),
            "contrast": round(self.contrast, 4),
            "saturation": round(self.saturation, 4),
            "sharpness": round(self.sharpness, 4),
            "noise": round(self.noise, 4),
            "blur_scale": round(self.blur_scale, 2),
            "clipped_highlights": round(self.clipped_highlights, 4),
            "faces_detected": self.faces_detected,
            "summary": self.summary,
            "actions": self.actions,
        }


# --------------------------------------------------------------------------- #
# Segmentation session (lazy, process wide, thread safe)
# --------------------------------------------------------------------------- #

_session_lock = threading.Lock()
_session: object | None = None
_session_path: Path | None = None


def segmentation_model_available(model_path: Path) -> bool:
    return model_path.exists() and model_path.stat().st_size > 0


def _get_session(model_path: Path):
    """Load the ONNX session once and reuse it for every request."""
    global _session, _session_path

    with _session_lock:
        if _session is not None and _session_path == model_path:
            return _session

        if not segmentation_model_available(model_path):
            raise ImageProcessingError(
                f"Segmentation model missing at {model_path}. "
                "Download u2netp.onnx into that location to enable background removal."
            )

        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency guard
            raise ImageProcessingError(
                "onnxruntime is not installed; run pip install -r requirements.txt"
            ) from error

        options = ort.SessionOptions()
        # Keep the footprint small and predictable on a laptop CPU.
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        _session_path = model_path
        logger.info(
            "segmentation model loaded",
            extra={"module_name": "background-removal", "action": "model-load", "path": str(model_path)},
        )
        return _session


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def load_image(data: bytes) -> Image.Image:
    """Decode bytes into an RGB(A) image with EXIF rotation already applied."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:
        raise ImageProcessingError("Could not decode the uploaded image.") from error

    image = ImageOps.exif_transpose(image)

    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.mode or image.mode == "P" else "RGB")

    if image.width * image.height > _MAX_WORKING_PIXELS:
        image.thumbnail((8000, 8000), Image.LANCZOS)

    return image


def _aspect_label(width: int, height: int) -> str:
    from math import gcd

    divisor = gcd(width, height) or 1
    w, h = width // divisor, height // divisor
    # Collapse awkward ratios like 1234:987 into a readable approximation.
    if w > 32 or h > 32:
        ratio = width / height
        for label, value in (
            ("1:1", 1.0), ("4:3", 4 / 3), ("3:2", 3 / 2), ("16:9", 16 / 9), ("3:4", 3 / 4),
            ("2:3", 2 / 3), ("9:16", 9 / 16), ("5:4", 5 / 4), ("4:5", 4 / 5),
        ):
            if abs(ratio - value) < 0.02:
                return label
        return f"{ratio:.2f}:1"
    return f"{w}:{h}"


def target_size(width: int, height: int, resolution: str) -> tuple[int, int]:
    """Fit ``width x height`` inside the preset's box, preserving aspect ratio.

    ``source`` keeps the native pixel dimensions untouched. Every other preset
    scales the image so it fits *inside* the box - a 3:4 portrait at ``4k``
    becomes 1620x2160, never 3840x2160.
    """
    if resolution == "source" or resolution not in _RESOLUTION_BOXES:
        return width, height

    box_width, box_height = _RESOLUTION_BOXES[resolution]
    scale = min(box_width / width, box_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114


def _analysis_thumbnail(image: Image.Image, longest: int = 512) -> np.ndarray:
    """Small float32 RGB copy in 0..1, used for all measurements."""
    thumb = image.convert("RGB")
    thumb.thumbnail((longest, longest), Image.BILINEAR)
    return np.asarray(thumb, dtype=np.float32) / 255.0


def analyze_image(image: Image.Image) -> ImageAnalysis:
    """Measure exposure, contrast, saturation, sharpness and noise."""
    sample = _analysis_thumbnail(image)
    luma = _luminance(sample)

    exposure = float(luma.mean())
    contrast = float(luma.std())

    channel_max = sample.max(axis=2)
    channel_min = sample.min(axis=2)
    saturation = float(np.mean((channel_max - channel_min) / (channel_max + 1e-6)))

    # A 3x3 median removes sensor noise while keeping real edges, so it serves
    # as the reference for both the noise and the sharpness measurement.
    median = np.asarray(
        Image.fromarray((luma * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
        dtype=np.float32,
    ) / 255.0

    # Noise: the high frequency energy the median removed.
    noise = float(np.clip(np.median(np.abs(luma - median)) * 40.0, 0.0, 1.0))

    # Sharpness: variance of a 4-neighbour Laplacian, measured on the denoised
    # luma. Measuring it on the raw luma would let grain masquerade as detail
    # and leave genuinely soft photos under-sharpened.
    laplacian = (
        -4.0 * median[1:-1, 1:-1]
        + median[:-2, 1:-1]
        + median[2:, 1:-1]
        + median[1:-1, :-2]
        + median[1:-1, 2:]
    )
    sharpness = float(np.clip(laplacian.var() * 900.0, 0.0, 1.0))

    # Blur scale is measured on the thumbnail, then rescaled to source pixels so
    # the sharpening radius is correct for the image we actually deliver.
    luma_image = Image.fromarray((luma * 255).astype(np.uint8), mode="L")
    thumbnail_longest = max(luma_image.size)
    scale_to_source = max(image.width, image.height) / max(thumbnail_longest, 1)
    blur_scale = _detail_scale(luma_image) * scale_to_source

    return ImageAnalysis(
        width=image.width,
        height=image.height,
        aspect_ratio=_aspect_label(image.width, image.height),
        exposure=exposure,
        contrast=contrast,
        saturation=saturation,
        sharpness=sharpness,
        noise=noise,
        blur_scale=blur_scale,
        clipped_highlights=float((luma > 0.995).mean()),
    )


def _curve_to_lut(curve: np.ndarray) -> list[int]:
    """Turn a 256 entry 0..1 curve into a Pillow point() table for R, G and B."""
    table = (np.clip(curve, 0.0, 1.0) * 255.0).round().astype(np.uint8).tolist()
    return table * 3  # identical table per channel keeps hues intact


def _auto_levels_lut(image: Image.Image, low_pct: float, high_pct: float) -> list[int] | None:
    """Build a clipping-aware black/white point curve.

    Blown highlights are the trap here: on a photo with a large pure-white sky
    the high percentile sits at 1.0, so a naive stretch pushes even more pixels
    into clipping. When the highlights are already clipped we hold the white
    point at 1.0 and only recover the black point.
    """
    sample = _analysis_thumbnail(image)
    luma = _luminance(sample)

    clipped_high = float((luma > 0.995).mean())
    clipped_low = float((luma < 0.005).mean())

    black = float(np.percentile(luma, low_pct))
    white = float(np.percentile(luma, high_pct))

    # Do not drag the white point down into already blown highlights, and do not
    # lift the black point into already crushed shadows.
    if clipped_high > 0.02:
        white = 1.0
    if clipped_low > 0.02:
        black = 0.0

    if white - black < 0.06:  # already full range, or a flat/solid image
        return None

    black = max(0.0, black - 0.02)
    white = min(1.0, white + 0.02)

    ramp = np.arange(256, dtype=np.float32) / 255.0
    return _curve_to_lut((ramp - black) / (white - black))


def _tone_curve_lut(midtone_gamma: float, highlight_shoulder: float) -> list[int]:
    """Midtone gamma with a soft shoulder, instead of a linear brightness multiply.

    A multiply scales every pixel, so it drives highlights further into clipping
    and washes the picture out - that is the "it only added light" failure. A
    gamma curve pins 0 to 0 and 1 to 1, lifting shadows and midtones while
    leaving white where it is. The shoulder then eases the top end back down so
    bright areas keep their separation instead of fusing into a flat white.
    """
    ramp = np.arange(256, dtype=np.float32) / 255.0
    curve = np.power(ramp, midtone_gamma)

    if highlight_shoulder > 0.0:
        knee = 0.70
        strength = 1.0 + 2.5 * highlight_shoulder
        headroom = 1.0 - knee
        # Exponential shoulder: monotonic, maps knee->knee and 1->1 exactly.
        normalised = np.clip((curve - knee) / headroom, 0.0, 1.0)
        softened = knee + headroom * (1.0 - np.exp(-strength * normalised)) / (1.0 - np.exp(-strength))
        curve = np.where(curve > knee, softened, curve)

    return _curve_to_lut(curve)


def _detail_scale(luma_image: Image.Image) -> float:
    """Estimate the characteristic blur radius, in pixels of the given image.

    Sharpening only works when its radius matches the scale of the blur. A fixed
    1.1px unsharp mask does nothing at all to a 20px-wide soft edge no matter how
    high the percentage is driven, which is exactly why the previous pipeline
    appeared to do nothing but brighten. We find the radius at which additional
    blurring stops removing new detail, and sharpen at that scale instead.
    """
    base = np.asarray(luma_image, dtype=np.float32) / 255.0
    radii = (0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
    energies = [
        float(np.mean(np.abs(base - np.asarray(luma_image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0)))
        for radius in radii
    ]

    total = energies[-1]
    if total < 1e-6:
        return radii[-1]

    # Blurring by less than the blur already present removes almost nothing, so
    # detail energy stays near zero until the test radius reaches the true blur
    # scale. The first radius that lifts energy above ~6% of the full-scale total
    # is therefore the width of the softest edges. Calibrated against known
    # Gaussian blurs: sharp -> ~1px, blur 2 -> 2px, blur 4 -> 4px, blur 16 -> 8px.
    for radius, energy in zip(radii, energies):
        if energy >= 0.06 * total:
            return radius
    return radii[-1]


def _apply_lut(image: Image.Image, lut: list[int]) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb = image.convert("RGB").point(lut)
        rgb.putalpha(alpha)
        return rgb
    return image.point(lut)


# --------------------------------------------------------------------------- #
# Enhancement
# --------------------------------------------------------------------------- #


def run_enhancement(
    image: Image.Image,
    analysis: ImageAnalysis,
    *,
    resolution: str = "source",
    strength: float = 1.0,
) -> tuple[Image.Image, list[str]]:
    """Apply the measured corrections to an already loaded image.

    Pipeline order matters: noise is removed before tones are stretched (so the
    stretch does not amplify grain), the image is resized before sharpening (so
    sharpening is tuned to the delivered pixels), and sharpening runs last.

    Split out from :func:`enhance_image` so face restoration can reuse the exact
    same corrections after it has rebuilt the faces.
    """
    strength = float(np.clip(strength, 0.0, 2.0))
    actions: list[str] = []

    working = image

    # 1. Denoise, but only when there is measurable noise. A median blend keeps
    #    edges intact far better than a blur.
    noise_blend = float(np.clip((analysis.noise - 0.18) * 1.6, 0.0, 0.6)) * strength
    if noise_blend > 0.02:
        denoised = working.filter(ImageFilter.MedianFilter(3))
        working = Image.blend(working, denoised, noise_blend)
        actions.append(f"reduced sensor noise ({noise_blend * 100:.0f}% median blend)")
    else:
        actions.append("noise already low, left untouched")

    # 2. Black/white point recovery - the single biggest win on flat images.
    lut = _auto_levels_lut(working, low_pct=0.5, high_pct=99.5)
    if lut is not None:
        levelled = _apply_lut(working, lut)
        working = Image.blend(working, levelled, min(1.0, strength))
        actions.append("recovered black and white points")
    else:
        actions.append("tonal range already full")

    # 3. Exposure via a gamma tone curve, driven by the MEDIAN luminance.
    #    Median rather than mean, because a large blown sky drags the mean up and
    #    would tell us a dark photo is correctly exposed. Gamma rather than a
    #    brightness multiply, because a multiply pushes highlights into clipping.
    post_sample = _analysis_thumbnail(working)
    post_luma = _luminance(post_sample)
    median_luma = float(np.median(post_luma))
    clipped = float((post_luma > 0.995).mean())

    if abs(median_luma - 0.45) > 0.03 and 0.01 < median_luma < 0.99:
        gamma = float(np.clip(math.log(0.45) / math.log(median_luma), 0.55, 1.85))
        gamma = 1.0 + (gamma - 1.0) * strength
        # Add a shoulder proportional to how much of the frame is already blown.
        shoulder = float(np.clip(clipped * 6.0, 0.0, 1.0))
        working = _apply_lut(working, _tone_curve_lut(gamma, shoulder))
        direction = "lifted shadows and midtones" if gamma < 1.0 else "pulled midtones down"
        detail = f" with a highlight shoulder ({clipped * 100:.1f}% blown)" if shoulder > 0.05 else ""
        actions.append(f"{direction} (gamma {gamma:.2f}){detail}")
    else:
        actions.append("exposure already balanced")

    # 4. Local contrast, which is what actually gives a photo depth. A global
    #    contrast multiply just brightens and darkens; a wide-radius unsharp mask
    #    separates nearby tones, so foliage and texture stop looking like mush.
    clarity_radius = max(6.0, min(working.width, working.height) / 45.0)
    clarity_amount = int(np.clip((0.22 - post_luma.std()) * 320.0 + 28.0, 18.0, 70.0) * strength)
    if clarity_amount > 0:
        working = working.filter(
            ImageFilter.UnsharpMask(radius=clarity_radius, percent=clarity_amount, threshold=0)
        )
        actions.append(f"added local contrast ({clarity_amount}% at r={clarity_radius:.0f})")

    # 5. Saturation - lift dull images, and pull back ones already oversaturated.
    channel_max = post_sample.max(axis=2)
    channel_min = post_sample.min(axis=2)
    post_saturation = float(np.mean((channel_max - channel_min) / (channel_max + 1e-6)))

    if post_saturation < 0.30:
        factor = 1.0 + float(np.clip((0.30 - post_saturation) * 1.8, 0.0, 0.32)) * strength
        working = ImageEnhance.Color(working).enhance(factor)
        actions.append(f"revived colour ({factor:.2f}x)")
    elif post_saturation > 0.62:
        factor = 1.0 - float(np.clip((post_saturation - 0.62) * 0.6, 0.0, 0.14)) * strength
        working = ImageEnhance.Color(working).enhance(factor)
        actions.append(f"tamed oversaturation ({factor:.2f}x)")
    else:
        actions.append("colour balance already good")

    # 6. Resize with a proper resampling kernel, aspect ratio preserved.
    out_width, out_height = target_size(working.width, working.height, resolution)
    if (out_width, out_height) != (working.width, working.height):
        upscaling = out_width > working.width
        working = working.resize((out_width, out_height), Image.LANCZOS)
        actions.append(
            f"{'upscaled' if upscaling else 'resized'} to {out_width}x{out_height} "
            f"({analysis.aspect_ratio} preserved)"
        )
    else:
        actions.append(f"kept source size {working.width}x{working.height}")

    # 7. Sharpen last, at the radius the blur actually occupies.
    #    This is the fix that matters most: a fixed 1.1px unsharp mask leaves a
    #    20px-wide soft edge completely untouched no matter how high the percent
    #    is pushed, so the old pipeline's only visible effect was brightening.
    output_scale = max(working.size) / max(max(analysis.width, analysis.height), 1)
    radius = float(np.clip(analysis.blur_scale * output_scale * 0.5, 0.8, 12.0))
    # Wide radii need a gentler hand - big radius plus big amount is what creates
    # the classic halo. Amount therefore falls as radius grows.
    amount = int(np.clip(150.0 / (1.0 + radius * 0.45), 35.0, 150.0) * strength)
    threshold = 3 if analysis.noise > 0.25 else 2

    if amount > 0:
        working = working.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=threshold)
        )
        actions.append(f"unsharp mask matched to blur ({amount}% at r={radius:.1f})")

    # Be honest when the blur is past the point classical sharpening can help.
    beyond_recovery = analysis.blur_scale * output_scale > 5.0
    if beyond_recovery:
        actions.append("blur is too wide to fully recover without a deblur model")

    return working, actions


def _describe(analysis: ImageAnalysis, actions: list[str], prefix: str = "") -> None:
    """Attach the action list and a measurement summary to the analysis."""
    analysis.actions = actions
    analysis.summary = (
        prefix
        + f"Measured {analysis.width}x{analysis.height} ({analysis.aspect_ratio}): "
        f"exposure {analysis.exposure:.2f}, contrast {analysis.contrast:.2f}, "
        f"saturation {analysis.saturation:.2f}, noise {analysis.noise:.2f}, "
        f"blur scale {analysis.blur_scale:.1f}px"
        + (f", {analysis.clipped_highlights * 100:.1f}% blown highlights" if analysis.clipped_highlights > 0.005 else "")
        + f". Applied {len(actions)} measured corrections."
    )


def enhance_image(
    data: bytes,
    *,
    resolution: str = "source",
    output_format: str = "jpg",
    quality: int = 90,
    strength: float = 1.0,
) -> tuple[bytes, ImageAnalysis]:
    """Enhance an image based on what is actually wrong with it."""
    image = load_image(data)
    analysis = analyze_image(image)
    working, actions = run_enhancement(image, analysis, resolution=resolution, strength=strength)
    _describe(analysis, actions)

    payload = encode_image(working, output_format=output_format, quality=quality, keep_alpha=False)
    return payload, analysis


def restore_faces_image(
    data: bytes,
    *,
    detector_path: Path,
    restorer_path: Path,
    resolution: str = "source",
    output_format: str = "jpg",
    quality: int = 92,
    blend: float = 1.0,
    enhance_whole_image: bool = True,
) -> tuple[bytes, ImageAnalysis]:
    """Rebuild facial detail, then optionally clean up the rest of the photo.

    Faces are restored first, at source resolution, so the generative model sees
    the most detail available. The global corrections run afterwards at reduced
    strength, otherwise a freshly restored face gets over-sharpened.
    """
    from app.face_restoration import restore_faces

    image = load_image(data)
    analysis = analyze_image(image)

    restored, face_actions, face_count = restore_faces(
        image, detector_path=detector_path, restorer_path=restorer_path, blend=blend
    )
    analysis.faces_detected = face_count
    actions = list(face_actions)

    if enhance_whole_image:
        restored, enhancement_actions = run_enhancement(
            restored, analysis, resolution=resolution, strength=0.65
        )
        actions.extend(enhancement_actions)
    else:
        out_width, out_height = target_size(restored.width, restored.height, resolution)
        if (out_width, out_height) != (restored.width, restored.height):
            restored = restored.resize((out_width, out_height), Image.LANCZOS)
            actions.append(f"resized to {out_width}x{out_height} ({analysis.aspect_ratio} preserved)")

    prefix = (
        f"Restored {face_count} face{'s' if face_count != 1 else ''}. "
        if face_count
        else "No faces were found, so only the global corrections were applied. "
    )
    _describe(analysis, actions, prefix=prefix)

    payload = encode_image(restored, output_format=output_format, quality=quality, keep_alpha=False)
    return payload, analysis


# --------------------------------------------------------------------------- #
# Background removal
# --------------------------------------------------------------------------- #


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Normalised box blur via cumulative sums (border safe, no SciPy)."""
    height, width = src.shape
    radius = max(1, min(radius, (min(height, width) - 1) // 2))

    def _sum_axis0(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(values, axis=0)
        out = np.empty_like(values)
        out[: radius + 1] = cumulative[radius : 2 * radius + 1]
        out[radius + 1 : height - radius] = (
            cumulative[2 * radius + 1 :] - cumulative[: height - 2 * radius - 1]
        )
        out[height - radius :] = (
            cumulative[height - 1][None, :]
            - cumulative[height - 2 * radius - 1 : height - radius - 1]
        )
        return out

    def _sum_axis1(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(values, axis=1)
        out = np.empty_like(values)
        out[:, : radius + 1] = cumulative[:, radius : 2 * radius + 1]
        out[:, radius + 1 : width - radius] = (
            cumulative[:, 2 * radius + 1 :] - cumulative[:, : width - 2 * radius - 1]
        )
        out[:, width - radius :] = (
            cumulative[:, width - 1][:, None]
            - cumulative[:, width - 2 * radius - 1 : width - radius - 1]
        )
        return out

    totals = _sum_axis1(_sum_axis0(src))
    counts = _sum_axis1(_sum_axis0(np.ones_like(src)))
    return totals / counts


def _guided_filter(guide: np.ndarray, target: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge aware refinement of ``target`` using ``guide`` as the edge reference.

    This is what makes a 320x320 network mask usable at full resolution: the
    alpha is pulled onto the real luminance edges of the photo instead of the
    blurry upscaled boundary.
    """
    mean_guide = _box_filter(guide, radius)
    mean_target = _box_filter(target, radius)
    corr_guide = _box_filter(guide * guide, radius)
    corr_cross = _box_filter(guide * target, radius)

    var_guide = np.maximum(corr_guide - mean_guide * mean_guide, 0.0)
    cov_cross = corr_cross - mean_guide * mean_target

    a = cov_cross / (var_guide + eps)
    b = mean_target - a * mean_guide

    return _box_filter(a, radius) * guide + _box_filter(b, radius)


def _predict_mask(image: Image.Image, model_path: Path) -> np.ndarray:
    """Run u2netp and return a 0..1 float mask at the network's resolution."""
    session = _get_session(model_path)

    resized = image.convert("RGB").resize(
        (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS
    )
    tensor = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = tensor.transpose(2, 0, 1)[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name  # type: ignore[attr-defined]
    outputs = session.run(None, {input_name: tensor})  # type: ignore[attr-defined]

    mask = outputs[0][0, 0]
    lo, hi = float(mask.min()), float(mask.max())
    if hi - lo < 1e-6:
        raise ImageProcessingError("Segmentation produced an empty mask for this image.")
    return (mask - lo) / (hi - lo)


def remove_background(
    data: bytes,
    *,
    model_path: Path,
    resolution: str = "source",
    output_format: str = "png",
    quality: int = 95,
    edge_softness: float = 1.0,
) -> tuple[bytes, ImageAnalysis]:
    """Cut the subject out of its background and return an image with real alpha."""
    image = load_image(data)
    analysis = analyze_image(image)
    actions: list[str] = []

    rgb = image.convert("RGB")

    coarse = _predict_mask(rgb, model_path)
    actions.append("segmented subject with u2netp (CPU)")

    # Upscale the coarse mask, then snap it to the photo's own edges.
    mask_image = Image.fromarray((coarse * 255).astype(np.uint8), mode="L").resize(
        rgb.size, Image.BICUBIC
    )
    alpha = np.asarray(mask_image, dtype=np.float32) / 255.0

    guide = _luminance(np.asarray(rgb, dtype=np.float32) / 255.0)
    radius = max(2, int(round(max(rgb.size) / 220)))
    alpha = _guided_filter(guide, alpha, radius=radius, eps=1e-4)
    alpha = np.clip(alpha, 0.0, 1.0)
    actions.append(f"refined alpha against image edges (guided filter r={radius})")

    # Push confident areas to fully opaque/transparent while leaving a genuine
    # soft band for hair and motion edges.
    softness = float(np.clip(edge_softness, 0.1, 3.0))
    low, high = 0.5 - 0.16 * softness, 0.5 + 0.16 * softness
    alpha = np.clip((alpha - low) / max(high - low, 1e-4), 0.0, 1.0)
    actions.append("hardened interior, kept soft hair/edge band")

    coverage = float(alpha.mean())

    cutout = rgb.convert("RGBA")
    cutout.putalpha(Image.fromarray((alpha * 255).round().astype(np.uint8), mode="L"))

    out_width, out_height = target_size(cutout.width, cutout.height, resolution)
    if (out_width, out_height) != (cutout.width, cutout.height):
        cutout = cutout.resize((out_width, out_height), Image.LANCZOS)
        actions.append(f"resized to {out_width}x{out_height} ({analysis.aspect_ratio} preserved)")
    else:
        actions.append(f"kept source size {cutout.width}x{cutout.height}")

    analysis.actions = actions
    analysis.summary = (
        f"Isolated the subject from {analysis.width}x{analysis.height} ({analysis.aspect_ratio}). "
        f"Foreground covers {coverage * 100:.1f}% of the frame. "
        f"Transparency is per-pixel alpha, not a crop."
    )

    payload = encode_image(cutout, output_format=output_format, quality=quality, keep_alpha=True)
    return payload, analysis


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

_FORMAT_ALIASES = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}
MIME_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def resolve_format(output_format: str, *, keep_alpha: bool) -> str:
    pillow_format = _FORMAT_ALIASES.get(output_format.lower().removesuffix("-alpha"), "PNG")
    if keep_alpha and pillow_format == "JPEG":
        # JPEG cannot store transparency; silently falling back to a white matte
        # would look like the cutout failed, so promote to PNG instead.
        return "PNG"
    return pillow_format


def encode_image(image: Image.Image, *, output_format: str, quality: int, keep_alpha: bool) -> bytes:
    pillow_format = resolve_format(output_format, keep_alpha=keep_alpha)
    buffer = io.BytesIO()

    if pillow_format == "JPEG":
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
        # subsampling=0 keeps full chroma resolution, which is the point of an
        # enhancement pass. It must not be combined with progressive/optimize -
        # that pair overflows Pillow's encode buffer on a BytesIO target and
        # raises "broken data stream".
        image.save(
            buffer,
            format="JPEG",
            quality=int(np.clip(quality, 1, 100)),
            subsampling=0,
        )
    elif pillow_format == "WEBP":
        if not keep_alpha and image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffer, format="WEBP", quality=int(np.clip(quality, 1, 100)), method=4)
    else:
        if not keep_alpha and image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffer, format="PNG", optimize=True)

    return buffer.getvalue()
