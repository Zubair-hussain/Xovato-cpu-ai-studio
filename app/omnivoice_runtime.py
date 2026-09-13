"""OmniVoice model presence check.

The subprocess-per-request ``synthesize_cpu`` helper that used to live here has
been removed: the warm, in-process :mod:`app.voice_engine` replaced it, and
keeping a second, slower synthesis path around only invited drift. This module
now just reports whether a downloaded model looks complete.
"""

from __future__ import annotations

from pathlib import Path


def model_ready(model_dir: Path) -> bool:
    """True when the OmniVoice checkpoint at ``model_dir`` looks complete."""
    required = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "audio_tokenizer" / "config.json",
    ]
    has_weights = any(model_dir.glob("*.safetensors")) or any((model_dir / "audio_tokenizer").glob("*.safetensors"))
    return all(path.exists() for path in required) and has_weights
