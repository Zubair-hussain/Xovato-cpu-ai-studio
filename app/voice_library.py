"""Ready-made stock voices for text to speech.

These are original personas, not impersonations of real people. Each is defined
purely by attributes OmniVoice accepts natively, so they run in the model's
"voice design" mode - no reference audio and no cloning step, which also makes
them the fastest path to speech.

Every ``instruct_items`` entry must exist in
:data:`app.voice_styles.VALID_INSTRUCT_ITEMS`; anything else is rejected by the
model at generation time. :func:`validate_library` checks that at import time so
a typo fails immediately rather than on a user's first request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.voice_styles import VoiceStyleError, validate_instruct_items


@dataclass(frozen=True)
class StockVoice:
    id: str
    name: str
    tagline: str
    instruct_items: list[str] = field(default_factory=list)
    default_speed: float = 1.0
    best_for: str = ""


STOCK_VOICES: tuple[StockVoice, ...] = (
    StockVoice(
        id="aria",
        name="Aria",
        tagline="Warm, even female narrator with an American accent.",
        instruct_items=["female", "young adult", "moderate pitch", "american accent"],
        best_for="Explainers, e-learning, product walkthroughs",
    ),
    StockVoice(
        id="atlas",
        name="Atlas",
        tagline="Deep, authoritative male voice built for trailers.",
        instruct_items=["male", "middle-aged", "very low pitch", "american accent"],
        default_speed=0.94,
        best_for="Trailers, documentaries, announcements",
    ),
    StockVoice(
        id="bramble",
        name="Bramble",
        tagline="Older British gentleman with an unhurried, storybook delivery.",
        instruct_items=["male", "elderly", "low pitch", "british accent"],
        default_speed=0.9,
        best_for="Audiobooks, folklore, bedtime stories",
    ),
    StockVoice(
        id="clara",
        name="Clara",
        tagline="Composed British female voice with a corporate polish.",
        instruct_items=["female", "middle-aged", "moderate pitch", "british accent"],
        best_for="Corporate video, training, IVR prompts",
    ),
    StockVoice(
        id="rohan",
        name="Rohan",
        tagline="Clear, friendly male voice with an Indian accent.",
        instruct_items=["male", "young adult", "moderate pitch", "indian accent"],
        best_for="Tech explainers, tutorials, support content",
    ),
    StockVoice(
        id="meera",
        name="Meera",
        tagline="Bright, upbeat female presenter with an Indian accent.",
        instruct_items=["female", "young adult", "high pitch", "indian accent"],
        default_speed=1.05,
        best_for="Promos, social clips, announcements",
    ),
    StockVoice(
        id="kai",
        name="Kai",
        tagline="Energetic young Australian voice.",
        instruct_items=["male", "teenager", "high pitch", "australian accent"],
        default_speed=1.08,
        best_for="Gaming, youth brands, short-form video",
    ),
    StockVoice(
        id="nova",
        name="Nova",
        tagline="Animated, high-energy character voice.",
        instruct_items=["female", "teenager", "very high pitch", "american accent"],
        default_speed=1.1,
        best_for="Animation, characters, mascots",
    ),
    StockVoice(
        id="sable",
        name="Sable",
        tagline="Low, smoky female voice with a noir edge.",
        instruct_items=["female", "middle-aged", "very low pitch", "american accent"],
        default_speed=0.92,
        best_for="Drama, noir narration, luxury brands",
    ),
    StockVoice(
        id="yuki",
        name="Yuki",
        tagline="Soft-spoken female voice with a Japanese accent.",
        instruct_items=["female", "young adult", "moderate pitch", "japanese accent"],
        default_speed=0.96,
        best_for="Calm narration, meditation, soft branding",
    ),
    StockVoice(
        id="hush",
        name="Hush",
        tagline="Close, whispered voice using the model's real whisper style.",
        instruct_items=["female", "young adult", "whisper"],
        default_speed=0.9,
        best_for="ASMR, intimate narration, suspense",
    ),
    StockVoice(
        id="orion",
        name="Orion",
        tagline="Measured Canadian male voice, neutral and broadcast-ready.",
        instruct_items=["male", "middle-aged", "low pitch", "canadian accent"],
        best_for="News reads, podcasts, neutral narration",
    ),
)

STOCK_VOICES_BY_ID: dict[str, StockVoice] = {voice.id: voice for voice in STOCK_VOICES}


def validate_library() -> None:
    """Fail loudly at import time if any preset uses an unsupported attribute."""
    for voice in STOCK_VOICES:
        try:
            validate_instruct_items(list(voice.instruct_items))
        except VoiceStyleError as error:
            raise VoiceStyleError(f"Stock voice '{voice.id}' is invalid: {error}") from error


def get_stock_voice(voice_id: str | None) -> StockVoice | None:
    if not voice_id:
        return None
    return STOCK_VOICES_BY_ID.get(voice_id.strip().lower())


def library_payload() -> list[dict[str, object]]:
    return [
        {
            "id": voice.id,
            "name": voice.name,
            "tagline": voice.tagline,
            "instruct_items": list(voice.instruct_items),
            "default_speed": voice.default_speed,
            "best_for": voice.best_for,
        }
        for voice in STOCK_VOICES
    ]


validate_library()
