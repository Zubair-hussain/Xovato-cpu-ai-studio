"""Warm, in-process OmniVoice engine.

The previous implementation spawned a subprocess per request and called
``OmniVoice.from_pretrained`` inside it, so a 2.45 GB checkpoint was read from
disk and re-initialised for **every sentence**. That load cost dwarfed the
actual synthesis.

This module loads the model once and keeps it resident, and caches the
per-speaker ``VoiceClonePrompt`` so reference audio is tokenised once per voice
profile rather than once per request. A lock serialises generation because the
CPU backend is already internally threaded and concurrent calls only thrash.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.voice_styles import VoiceStyle, instruct_string, shape_prosody

logger = logging.getLogger(__name__)


class VoiceEngineError(RuntimeError):
    """Raised when the engine cannot load or synthesis fails."""


def _load_reference_waveform(path: Path) -> tuple["object", int]:
    """Decode reference audio to a (waveform, sample_rate) tuple for OmniVoice.

    OmniVoice's own loader calls ``torchaudio.load``, which in torchaudio 2.11
    delegates to TorchCodec and raises ImportError unless TorchCodec *and*
    FFmpeg are installed. Neither is available here, so voice cloning failed
    before it ever reached the model.

    ``create_voice_clone_prompt`` also accepts a ``(waveform, sample_rate)``
    tuple, and handles mono-mixing and resampling itself, so decoding with
    soundfile sidesteps the whole problem with no extra dependency.
    """
    try:
        import soundfile as sf
        import torch
    except ImportError as error:  # pragma: no cover - dependency guard
        raise VoiceEngineError("soundfile and torch are required to read reference audio.") from error

    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as error:
        raise VoiceEngineError(
            f"Could not read the reference audio at {path.name}. "
            "Supported: WAV, MP3, FLAC, OGG, AIFF (M4A/AAC is not supported)."
        ) from error

    if samples.size == 0:
        raise VoiceEngineError(f"The reference clip {path.name} contains no audio.")

    # soundfile gives (frames, channels); OmniVoice wants (channels, frames).
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T, dtype=np.float32))
    return waveform, int(sample_rate)


@dataclass
class SynthesisResult:
    audio: np.ndarray  # (frames,) float32
    sample_rate: int
    load_seconds: float
    generate_seconds: float
    cloned: bool
    instruct: str | None
    speed: float


class VoiceEngine:
    """Holds the loaded model, the clone-prompt cache, and the generation lock."""

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._prompt_cache: dict[str, object] = {}
        self._load_seconds = 0.0

    # -- lifecycle ------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, object]:
        return {
            "loaded": self.is_loaded,
            "load_seconds": round(self._load_seconds, 2),
            "cached_voice_prompts": len(self._prompt_cache),
        }

    def unload(self) -> None:
        """Release the model and free its memory."""
        with self._model_lock:
            self._model = None
            self._prompt_cache.clear()
            self._load_seconds = 0.0

        try:
            import gc

            gc.collect()
        except Exception:  # pragma: no cover - best effort
            pass

        logger.info("voice engine unloaded", extra={"module_name": "text-to-speech", "action": "engine-unload"})

    def load(self, settings: Settings):
        """Load the model once. Subsequent calls return the resident instance."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_dir = settings.omnivoice_model_dir
            source_dir = settings.omnivoice_source_dir

            if not (source_dir / "omnivoice").exists():
                raise VoiceEngineError(f"OmniVoice source package was not found at {source_dir}")

            if str(source_dir) not in sys.path:
                sys.path.insert(0, str(source_dir))

            started = time.monotonic()
            try:
                import torch
                from omnivoice.models.omnivoice import OmniVoice

                # torch defaults to half the cores here (2 of 4), which simply
                # halves throughput on a CPU-only box. Use every core.
                cores = max(1, os.cpu_count() or 1)
                torch.set_num_threads(cores)

                model = OmniVoice.from_pretrained(str(model_dir), device_map="cpu", dtype=torch.float32)
                model.eval()

                if settings.omnivoice_quantize_int8:
                    # Dynamic int8 over the Linear layers. This is the single
                    # biggest CPU win for a transformer of this shape: weights
                    # are quantised once, activations per call, and matmuls run
                    # on integer kernels. Quality loss on TTS is minor next to
                    # the difference between "usable" and "several minutes".
                    try:
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        logger.info(
                            "voice engine quantised to int8",
                            extra={"module_name": "text-to-speech", "action": "engine-quantise"},
                        )
                    except Exception as error:  # pragma: no cover - optional path
                        logger.warning(
                            "int8 quantisation unavailable; continuing in float32",
                            extra={
                                "module_name": "text-to-speech",
                                "action": "engine-quantise-skip",
                                "warning": str(error),
                            },
                        )
            except Exception as error:
                raise VoiceEngineError(f"Could not load OmniVoice: {error}") from error

            self._load_seconds = time.monotonic() - started
            self._model = model
            logger.info(
                "voice engine loaded",
                extra={
                    "module_name": "text-to-speech",
                    "action": "engine-load",
                    "load_seconds": round(self._load_seconds, 2),
                },
            )
            return self._model

    # -- cloning --------------------------------------------------------- #

    def clone_prompt(self, settings: Settings, *, cache_key: str, ref_audio: Path, ref_text: str | None):
        """Build (and cache) a reusable voice clone prompt for one speaker.

        ``ref_text`` must be the real transcript of the reference clip. Omitting
        it makes OmniVoice auto-transcribe, which silently tries to download an
        ASR model from HuggingFace and fails outright when offline - so we ask
        for the transcript instead of hiding a network dependency behind a
        clone. An accurate transcript also clones noticeably better than a
        wrong one, which is what the old hardcoded placeholder produced.
        """
        if not (ref_text or "").strip():
            raise VoiceEngineError(
                "This voice profile has no reference transcript. Add the exact words spoken "
                "in the reference clip, then clone again - the transcript must match the audio "
                "for the clone to sound right."
            )

        key = f"{cache_key}:{ref_text}"
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        model = self.load(settings)
        try:
            with self._generate_lock:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=_load_reference_waveform(ref_audio), ref_text=ref_text or None
                )
        except Exception as error:
            raise VoiceEngineError(f"Could not build a voice prompt from the reference audio: {error}") from error

        self._prompt_cache[key] = prompt
        logger.info(
            "voice clone prompt cached",
            extra={"module_name": "voice-cloning", "action": "prompt-cache", "cache_key": cache_key},
        )
        return prompt

    def invalidate(self, cache_key: str) -> None:
        for key in [key for key in self._prompt_cache if key.startswith(f"{cache_key}:")]:
            self._prompt_cache.pop(key, None)

    # -- synthesis ------------------------------------------------------- #

    def synthesize(
        self,
        settings: Settings,
        *,
        text: str,
        style: VoiceStyle,
        language: str | None = None,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        cache_key: str | None = None,
        num_step: int | None = None,
        guidance_scale: float | None = None,
    ) -> SynthesisResult:
        if not text.strip():
            raise VoiceEngineError("There is no text to synthesise.")

        num_step = num_step if num_step is not None else settings.omnivoice_num_step
        guidance_scale = (
            guidance_scale if guidance_scale is not None else settings.omnivoice_guidance_scale
        )

        load_started = time.monotonic()
        model = self.load(settings)
        load_seconds = time.monotonic() - load_started

        prompt = None
        if ref_audio is not None and ref_audio.exists():
            prompt = self.clone_prompt(
                settings, cache_key=cache_key or str(ref_audio), ref_audio=ref_audio, ref_text=ref_text
            )

        import torch

        instruct = instruct_string(style)
        generate_started = time.monotonic()
        try:
            # inference_mode disables autograd bookkeeping entirely. The model
            # calls eval(), but that only switches layer behaviour - without
            # this, every generation step still builds a graph it never uses.
            with self._generate_lock, torch.inference_mode():
                audios = model.generate(
                    text=text,
                    language=language or None,
                    voice_clone_prompt=prompt,
                    instruct=instruct,
                    speed=style.speed,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                )
        except ValueError as error:
            # Most often an unsupported instruct item slipping past validation.
            raise VoiceEngineError(str(error)) from error
        except Exception as error:
            raise VoiceEngineError(f"OmniVoice synthesis failed: {error}") from error

        generate_seconds = time.monotonic() - generate_started

        audio = audios[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        sample_rate = int(getattr(model, "sampling_rate", 24000))
        shaped = shape_prosody(audio, sample_rate, style)
        if shaped.ndim > 1:
            shaped = shaped[:, 0]

        return SynthesisResult(
            audio=shaped.astype(np.float32),
            sample_rate=sample_rate,
            load_seconds=load_seconds,
            generate_seconds=generate_seconds,
            cloned=prompt is not None,
            instruct=instruct,
            speed=style.speed,
        )


voice_engine = VoiceEngine()
