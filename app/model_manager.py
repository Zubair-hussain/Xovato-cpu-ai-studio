from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    feature: str
    path: Path
    required: bool = False

    @property
    def exists(self) -> bool:
        return self.path.exists()


MODEL_SPECS = [
    ModelSpec("u2netp", "background-removal", MODEL_ROOT / "segmentation" / "u2netp.onnx"),
    ModelSpec("yunet", "face-detection", MODEL_ROOT / "face" / "yunet.onnx"),
    ModelSpec("gpen_bfr_256", "face-restoration", MODEL_ROOT / "face" / "gpen_bfr_256.onnx"),
    ModelSpec("k2-fsa_OmniVoice", "text-to-speech", MODEL_ROOT / "omnivoice" / "k2-fsa_OmniVoice"),
    ModelSpec("whisper-cpu", "transcription-and-subtitles", MODEL_ROOT / "whisper-cpu"),
    ModelSpec("audio-cpu", "audio-enhancement", MODEL_ROOT / "audio-cpu"),
]


def model_status() -> list[dict[str, object]]:
    return [
        {
            "id": spec.id,
            "feature": spec.feature,
            "path": str(spec.path),
            "exists": spec.exists,
            "required": spec.required,
        }
        for spec in MODEL_SPECS
    ]
