from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import Settings


def model_ready(model_dir: Path) -> bool:
    required = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "audio_tokenizer" / "config.json",
    ]
    has_weights = any(model_dir.glob("*.safetensors")) or any((model_dir / "audio_tokenizer").glob("*.safetensors"))
    return all(path.exists() for path in required) and has_weights


def synthesize_cpu(
    *,
    settings: Settings,
    text: str,
    output_path: Path,
    language: str | None = None,
    ref_audio_path: Path | None = None,
    ref_text: str | None = None,
) -> None:
    if not model_ready(settings.omnivoice_model_dir):
        raise RuntimeError(f"OmniVoice model is not fully downloaded at {settings.omnivoice_model_dir}")
    if not (settings.omnivoice_source_dir / "omnivoice").exists():
        raise RuntimeError(f"OmniVoice source package was not found at {settings.omnivoice_source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    code = """
import sys
from pathlib import Path

import torch
import torchaudio

from omnivoice.models.omnivoice import OmniVoice

model_dir = sys.argv[1]
text = sys.argv[2]
language = sys.argv[3] or None
ref_audio = sys.argv[4] or None
ref_text = sys.argv[5] or None
output = sys.argv[6]

model = OmniVoice.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32)
audios = model.generate(
    text=text,
    language=language,
    ref_audio=ref_audio,
    ref_text=ref_text,
    num_step=16,
    guidance_scale=2.0,
    speed=1.0,
)
torchaudio.save(output, audios[0].cpu(), model.sampling_rate)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(settings.omnivoice_source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_HOME", str(settings.data_dir / "models" / "hf"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    command = [
        sys.executable,
        "-c",
        code,
        str(settings.omnivoice_model_dir),
        text,
        language or "",
        str(ref_audio_path) if ref_audio_path else "",
        ref_text or "",
        str(output_path),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "OmniVoice synthesis failed").strip()
        raise RuntimeError(detail[-2000:])
