"""CPU-only face restoration: detect, align, restore, blend back.

Two small ONNX models do the work:

* **YuNet** (~230 KB) locates faces and returns five landmarks each.
* **GPEN-BFR-256** (~72 MB) regenerates a 256x256 aligned face.

Each face is warped onto the standard FFHQ landmark template, restored, then
warped back and blended through a feathered mask so only the face changes and
the seam is invisible. Everything is numpy + Pillow + onnxruntime; there is no
OpenCV dependency and no GPU.

Note on what this actually does: face restoration is *generative*. It does not
recover the original pixels, it synthesises plausible detail. On badly damaged
input it can subtly alter a person's appearance, which is inherent to the
technique. ``blend`` exists so callers can dial that back.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger(__name__)

_DETECTOR_SIZE = 640
_RESTORE_SIZE = 256
_STRIDES = (8, 16, 32)

# Standard FFHQ five point template, normalised 0..1. Order matches YuNet's
# landmark order: viewer-left eye, viewer-right eye, nose, left mouth, right mouth.
_FFHQ_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)


class FaceRestorationError(RuntimeError):
    """Raised when a face model is missing or a face cannot be processed."""


@dataclass
class DetectedFace:
    box: tuple[float, float, float, float]
    landmarks: np.ndarray  # (5, 2) in source image coordinates
    score: float


_lock = threading.Lock()
_sessions: dict[str, object] = {}


def face_models_available(detector_path: Path, restorer_path: Path) -> bool:
    return detector_path.exists() and restorer_path.exists()


def _session(path: Path):
    """Load and cache an ONNX session, one per model path."""
    key = str(path)
    with _lock:
        if key in _sessions:
            return _sessions[key]

        if not path.exists():
            raise FaceRestorationError(f"Face model missing at {path}.")

        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency guard
            raise FaceRestorationError("onnxruntime is not installed.") from error

        options = ort.SessionOptions()
        # GPEN exposes its weights as graph inputs, which makes onnxruntime very
        # chatty on load; silence that without hiding real errors.
        options.log_severity_level = 3
        import os

        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))

        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        _sessions[key] = session
        logger.info(
            "face model loaded",
            extra={"module_name": "face-restoration", "action": "model-load", "path": str(path)},
        )
        return session


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Umeyama similarity transform (scale + rotation + translation) as a 2x3."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centred = source - source_mean
    target_centred = target - target_mean

    covariance = target_centred.T @ source_centred / len(source)
    u, singular, vt = np.linalg.svd(covariance)

    correction = np.ones(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1.0

    rotation = u @ np.diag(correction) @ vt
    variance = (source_centred**2).sum() / len(source)
    scale = float((singular * correction).sum() / max(variance, 1e-8))

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * rotation @ source_mean
    return matrix


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    inverse_linear = np.linalg.inv(linear)
    return np.hstack([inverse_linear, (-inverse_linear @ translation)[:, None]])


def _warp(image: Image.Image, matrix: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Affine warp. Pillow samples the destination through the given matrix,
    so callers pass the destination-to-source mapping."""
    coefficients = (
        matrix[0, 0], matrix[0, 1], matrix[0, 2],
        matrix[1, 0], matrix[1, 1], matrix[1, 2],
    )
    return image.transform(size, Image.AFFINE, coefficients, resample=Image.BICUBIC)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        overlap_w = np.maximum(0.0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest]))
        intersection = overlap_w * overlap_h
        union = areas[current] + areas[rest] - intersection
        order = rest[intersection / np.maximum(union, 1e-8) <= threshold]

    return keep


def detect_faces(
    image: Image.Image,
    detector_path: Path,
    *,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
) -> list[DetectedFace]:
    """Run YuNet and decode its anchor grid into face boxes plus 5 landmarks."""
    session = _session(detector_path)

    # Letterbox into the detector's fixed 640x640 input, preserving aspect ratio
    # so faces are not distorted before detection.
    scale = min(_DETECTOR_SIZE / image.width, _DETECTOR_SIZE / image.height)
    resized = image.convert("RGB").resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.BILINEAR,
    )
    canvas = Image.new("RGB", (_DETECTOR_SIZE, _DETECTOR_SIZE), (0, 0, 0))
    canvas.paste(resized, (0, 0))

    # YuNet was trained through OpenCV, so it expects BGR in raw 0..255.
    tensor = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    outputs = session.run(None, {session.get_inputs()[0].name: tensor})  # type: ignore[attr-defined]
    named = {output.name: outputs[index] for index, output in enumerate(session.get_outputs())}  # type: ignore[attr-defined]

    boxes: list[list[float]] = []
    landmark_sets: list[np.ndarray] = []
    scores: list[float] = []

    for stride in _STRIDES:
        cls = named[f"cls_{stride}"][0].reshape(-1)
        obj = named[f"obj_{stride}"][0].reshape(-1)
        bbox = named[f"bbox_{stride}"][0].reshape(-1, 4)
        kps = named[f"kps_{stride}"][0].reshape(-1, 10)

        grid = _DETECTOR_SIZE // stride
        columns = np.tile(np.arange(grid), grid).astype(np.float32)
        rows = np.repeat(np.arange(grid), grid).astype(np.float32)

        confidence = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
        selected = np.nonzero(confidence >= score_threshold)[0]

        for index in selected:
            column, row = columns[index], rows[index]
            centre_x = (column + bbox[index, 0]) * stride
            centre_y = (row + bbox[index, 1]) * stride
            width = np.exp(bbox[index, 2]) * stride
            height = np.exp(bbox[index, 3]) * stride

            points = np.empty((5, 2), dtype=np.float32)
            for point in range(5):
                points[point, 0] = (column + kps[index, point * 2]) * stride
                points[point, 1] = (row + kps[index, point * 2 + 1]) * stride

            boxes.append([centre_x - width / 2, centre_y - height / 2, width, height])
            landmark_sets.append(points)
            scores.append(float(confidence[index]))

    if not boxes:
        return []

    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    keep = _non_max_suppression(box_array, score_array, nms_threshold)

    faces: list[DetectedFace] = []
    for index in keep:
        # Undo the letterbox scaling to get source image coordinates.
        box = box_array[index] / scale
        points = landmark_sets[index] / scale
        faces.append(
            DetectedFace(
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                landmarks=points,
                score=float(score_array[index]),
            )
        )

    faces.sort(key=lambda face: face.box[2] * face.box[3], reverse=True)
    return faces


# --------------------------------------------------------------------------- #
# Restoration
# --------------------------------------------------------------------------- #


def _restore_aligned_face(aligned: Image.Image, restorer_path: Path) -> Image.Image:
    session = _session(restorer_path)

    tensor = np.asarray(aligned.convert("RGB"), dtype=np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5  # GPEN expects -1..1
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    input_name = next(i.name for i in session.get_inputs() if len(i.shape) == 4)  # type: ignore[attr-defined]
    output = session.run(None, {input_name: tensor})[0]  # type: ignore[attr-defined]

    result = output[0].transpose(1, 2, 0)
    result = np.clip((result + 1.0) / 2.0, 0.0, 1.0)
    return Image.fromarray((result * 255.0).round().astype(np.uint8), mode="RGB")


def _blend_mask(size: int, feather: float) -> Image.Image:
    """Soft oval mask so the restored face fades into the original photo."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    inset = size * 0.06
    draw.ellipse([inset, inset, size - inset, size - inset], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(2.0, size * 0.035 * feather)))


def restore_faces(
    image: Image.Image,
    *,
    detector_path: Path,
    restorer_path: Path,
    blend: float = 1.0,
    max_faces: int = 8,
    min_face_pixels: int = 24,
) -> tuple[Image.Image, list[str], int]:
    """Restore every detected face and composite the results back.

    Returns the new image, a list of human readable actions, and the face count.
    """
    faces = detect_faces(image, detector_path)
    actions: list[str] = []

    if not faces:
        return image, ["no faces detected; left the image untouched"], 0

    usable = [face for face in faces if min(face.box[2], face.box[3]) >= min_face_pixels][:max_faces]
    if not usable:
        return image, [f"faces found but all smaller than {min_face_pixels}px; skipped"], 0

    blend = float(np.clip(blend, 0.0, 1.0))
    result = image.convert("RGB")
    template = _FFHQ_TEMPLATE * _RESTORE_SIZE
    mask = _blend_mask(_RESTORE_SIZE, feather=1.0)

    for face in usable:
        matrix = _similarity_transform(face.landmarks.astype(np.float64), template.astype(np.float64))
        inverse = _invert_affine(matrix)

        # Destination is the aligned crop, so sample through the inverse.
        aligned = _warp(result, inverse, (_RESTORE_SIZE, _RESTORE_SIZE))
        restored = _restore_aligned_face(aligned, restorer_path)

        if blend < 1.0:
            restored = Image.blend(aligned, restored, blend)

        # Destination is now the full image, so sample through the forward matrix.
        warped_face = _warp(restored, matrix, result.size)
        warped_mask = _warp(mask, matrix, result.size)

        result = Image.composite(warped_face, result, warped_mask)

    actions.append(f"detected {len(faces)} face{'s' if len(faces) != 1 else ''} with YuNet")
    actions.append(f"restored {len(usable)} face{'s' if len(usable) != 1 else ''} with GPEN-BFR-256 at 256x256")
    if blend < 1.0:
        actions.append(f"blended restoration at {blend * 100:.0f}% to preserve original likeness")
    actions.append("composited through a feathered oval mask so only faces changed")

    return result, actions, len(usable)
