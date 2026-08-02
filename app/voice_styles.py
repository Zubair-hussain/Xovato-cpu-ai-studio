"""Emotion and style presets for speech synthesis.

OmniVoice has no emotion tokens. Its ``instruct`` vocabulary is validated against
a fixed list of 23 items covering gender, age, pitch, ``whisper`` and accent -
there is no "sad", "happy" or "angry", and passing one raises ``ValueError``.

So emotion here is built from things the model genuinely supports, plus a light
post-processing pass:

* ``instruct`` items the model accepts (mainly pitch and whisper),
* ``speed``, which the model applies during generation,
* a prosody pass over the rendered audio - spectral tilt, dynamics and level.

The prosody pass deliberately does **not** pitch-shift. Doing that without
formant correction produces obvious chipmunk/monster artefacts, which would cost
more in quality than the emotion gains. Tilt and dynamics are artefact-free and
carry a surprising amount of the perceived affect: sad speech is darker, quieter
and flatter; excited speech is brighter and more dynamic.

Presets are defaults, not limits - every field can be overridden per request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# Exactly the items OmniVoice._resolve_instruct accepts. Anything else raises.
VALID_INSTRUCT_ITEMS = (
    "male", "female",
    "child", "teenager", "young adult", "middle-aged", "elderly",
    "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
    "whisper",
    "american accent", "australian accent", "british accent", "canadian accent",
    "chinese accent", "indian accent", "japanese accent", "korean accent",
    "portuguese accent", "russian accent",
)

# Categories that cannot be combined with each other.
_EXCLUSIVE_GROUPS = (
    ("male", "female"),
    ("child", "teenager", "young adult", "middle-aged", "elderly"),
    ("very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"),
    tuple(item for item in VALID_INSTRUCT_ITEMS if item.endswith(" accent")),
)


@dataclass
class VoiceStyle:
    """A resolved style: what to ask the model for, and how to shape the result."""

    emotion: str = "neutral"
    instruct_items: list[str] = field(default_factory=list)
    speed: float = 1.0
    # Post-processing. tilt_db > 0 brightens, < 0 darkens.
    tilt_db: float = 0.0
    # dynamics < 1 flattens delivery, > 1 exaggerates it.
    dynamics: float = 1.0
    gain_db: float = 0.0
    description: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Tuned so each preset is audibly distinct without sounding processed.
EMOTION_PRESETS: dict[str, VoiceStyle] = {
    "neutral": VoiceStyle(
        emotion="neutral",
        description="Unaltered delivery, exactly as the model renders it.",
    ),
    "sad": VoiceStyle(
        emotion="sad",
        instruct_items=["low pitch"],
        speed=0.86,
        tilt_db=-3.5,
        dynamics=0.72,
        gain_db=-1.5,
        description="Slower, lower, darker and flatter - the shape of low-energy speech.",
    ),
    "happy": VoiceStyle(
        emotion="happy",
        instruct_items=["high pitch"],
        speed=1.10,
        tilt_db=2.5,
        dynamics=1.18,
        gain_db=0.0,
        description="Brighter and a little quicker, with more lively dynamics.",
    ),
    "excited": VoiceStyle(
        emotion="excited",
        instruct_items=["high pitch"],
        speed=1.18,
        tilt_db=3.5,
        dynamics=1.30,
        gain_db=1.0,
        description="Fast, bright and highly dynamic.",
    ),
    "calm": VoiceStyle(
        emotion="calm",
        speed=0.93,
        tilt_db=-1.5,
        dynamics=0.82,
        gain_db=-0.5,
        description="Measured and even, with the edges taken off.",
    ),
    "serious": VoiceStyle(
        emotion="serious",
        instruct_items=["low pitch"],
        speed=0.95,
        tilt_db=-1.0,
        dynamics=0.90,
        description="Lower and deliberate, without sounding downbeat.",
    ),
    "dramatic": VoiceStyle(
        emotion="dramatic",
        speed=0.94,
        tilt_db=1.0,
        dynamics=1.40,
        gain_db=0.5,
        description="Wide dynamic swings for emphasis and weight.",
    ),
    "intimate": VoiceStyle(
        emotion="intimate",
        instruct_items=["whisper"],
        speed=0.90,
        tilt_db=-2.0,
        dynamics=0.70,
        gain_db=-2.0,
        description="Close and hushed, using the model's real whisper style.",
    ),
}


class VoiceStyleError(ValueError):
    """Raised when a style cannot be resolved into something the model accepts."""


def validate_instruct_items(items: list[str]) -> list[str]:
    """Check items against the model's vocabulary before it raises on us.

    Failing here gives a useful message naming the offending item, instead of
    surfacing OmniVoice's internal ValueError from deep in a worker.
    """
    normalised: list[str] = []
    for raw in items:
        item = raw.strip().lower()
        if not item:
            continue
        if item not in VALID_INSTRUCT_ITEMS:
            raise VoiceStyleError(
                f"'{raw}' is not a supported voice attribute. "
                f"Supported: {', '.join(VALID_INSTRUCT_ITEMS)}."
            )
        normalised.append(item)

    for group in _EXCLUSIVE_GROUPS:
        hits = [item for item in normalised if item in group]
        if len(hits) > 1:
            raise VoiceStyleError(
                f"Conflicting attributes in the same category: {', '.join(hits)}. Pick one."
            )

    return normalised


def resolve_style(
    emotion: str = "neutral",
    *,
    base_instruct: list[str] | None = None,
    extra_instruct: list[str] | None = None,
    speed: float | None = None,
    speed_multiplier: float = 1.0,
    tilt_db: float | None = None,
    dynamics: float | None = None,
    gain_db: float | None = None,
) -> VoiceStyle:
    """Look up a preset and apply any per-request overrides.

    Attributes are layered by precedence, later layers replacing earlier ones
    within the same category (you cannot be both ``male`` and ``female``):

    1. ``base_instruct`` - the chosen stock voice's identity,
    2. the emotion preset - e.g. ``sad`` lowers pitch,
    3. ``extra_instruct`` - whatever the user set by hand.
    """
    key = (emotion or "neutral").strip().lower()
    preset = EMOTION_PRESETS.get(key)
    if preset is None:
        raise VoiceStyleError(
            f"Unknown emotion '{emotion}'. Available: {', '.join(sorted(EMOTION_PRESETS))}."
        )

    items: list[str] = []
    for layer in (base_instruct or [], preset.instruct_items, extra_instruct or []):
        for item in layer:
            cleaned = item.strip().lower()
            if not cleaned:
                continue
            for group in _EXCLUSIVE_GROUPS:
                if cleaned in group:
                    items = [existing for existing in items if existing not in group]
            items.append(cleaned)

    return VoiceStyle(
        emotion=key,
        instruct_items=validate_instruct_items(items),
        # A stock voice has its own natural pace and the emotion has its own
        # pacing; compose them rather than letting either silently win, so
        # "Atlas, sad" is slower than Atlas alone. An explicit request speed
        # still overrides both.
        speed=float(
            np.clip(
                speed if speed is not None else preset.speed * speed_multiplier,
                0.5,
                2.0,
            )
        ),
        tilt_db=float(np.clip(tilt_db if tilt_db is not None else preset.tilt_db, -12.0, 12.0)),
        dynamics=float(np.clip(dynamics if dynamics is not None else preset.dynamics, 0.3, 2.0)),
        gain_db=float(np.clip(gain_db if gain_db is not None else preset.gain_db, -12.0, 12.0)),
        description=preset.description,
    )


def instruct_string(style: VoiceStyle) -> str | None:
    """Render the instruct list the way OmniVoice expects it."""
    return ", ".join(style.instruct_items) if style.instruct_items else None


# --------------------------------------------------------------------------- #
# Prosody shaping
# --------------------------------------------------------------------------- #


def _tilt_response(frequencies: np.ndarray, tilt_db: float) -> np.ndarray:
    """A gentle spectral tilt, hinged around 1 kHz on a log frequency scale."""
    safe = np.maximum(frequencies, 20.0)
    weight = np.clip(np.log2(safe / 1000.0) / 4.0, -1.0, 1.0)
    return (10.0 ** (tilt_db * weight / 20.0)).astype(np.float64)


def shape_prosody(samples: np.ndarray, sample_rate: int, style: VoiceStyle) -> np.ndarray:
    """Apply tilt, dynamics and level to rendered speech.

    ``samples`` is (frames,) or (frames, channels), float32 in -1..1.
    """
    if samples.ndim == 1:
        samples = samples[:, None]

    audio = samples.astype(np.float32, copy=True)

    # 1. Spectral tilt - darker for withdrawn emotions, brighter for lively ones.
    if abs(style.tilt_db) > 0.05:
        spectrum = np.fft.rfft(audio, axis=0)
        frequencies = np.fft.rfftfreq(audio.shape[0], 1.0 / sample_rate)
        response = _tilt_response(frequencies, style.tilt_db)
        audio = np.fft.irfft(spectrum * response[:, None], n=audio.shape[0], axis=0).astype(np.float32)

    # 2. Dynamics - flatten or exaggerate the loudness contour. Working on a
    #    smoothed envelope keeps this from acting like a fast compressor and
    #    pumping on individual syllables.
    if abs(style.dynamics - 1.0) > 0.02:
        envelope = np.abs(audio).mean(axis=1)
        window = max(1, int(0.05 * sample_rate))
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(envelope, kernel, mode="same")

        reference = float(np.percentile(smoothed, 85.0)) or 1.0
        ratio = np.maximum(smoothed / reference, 1e-4)
        gain = ratio ** (style.dynamics - 1.0)
        # Keep it sane; without a cap, quiet passages can be lifted enormously.
        gain = np.clip(gain, 0.35, 2.6)
        audio *= gain[:, None]

    # 3. Level.
    if abs(style.gain_db) > 0.01:
        audio *= 10.0 ** (style.gain_db / 20.0)

    # 4. Guard the ceiling so styling can never clip the render.
    peak = float(np.max(np.abs(audio), initial=0.0))
    if peak > 0.99:
        audio *= 0.99 / peak

    return audio.astype(np.float32)
