import numpy as np
import pytest

from app.voice_styles import (
    VoiceStyle,
    VoiceStyleError,
    design_to_instruct,
    instruct_string,
    resolve_style,
    shape_prosody,
    validate_instruct_items,
)


def test_validate_rejects_unknown_item():
    with pytest.raises(VoiceStyleError):
        validate_instruct_items(["happy"])


def test_validate_rejects_conflicting_category():
    with pytest.raises(VoiceStyleError):
        validate_instruct_items(["male", "female"])


def test_resolve_unknown_emotion():
    with pytest.raises(VoiceStyleError):
        resolve_style("ecstatic")


def test_resolve_layers_and_clamps_speed():
    style = resolve_style("sad", base_instruct=["female"], speed=5.0)
    assert style.emotion == "sad"
    assert "female" in style.instruct_items
    assert style.speed == 2.0  # clamped to the 0.5..2.0 range


def test_extra_instruct_overrides_category():
    style = resolve_style("neutral", base_instruct=["low pitch"], extra_instruct=["high pitch"])
    pitches = [item for item in style.instruct_items if item.endswith("pitch")]
    assert pitches == ["high pitch"]


def test_instruct_string_none_when_empty():
    assert instruct_string(VoiceStyle()) is None


def test_design_to_instruct_maps_fields():
    items = design_to_instruct(gender="female", age="young", accent="british", pitch=1.5)
    assert "female" in items
    assert "young adult" in items
    assert "british accent" in items
    assert "very high pitch" in items


def test_design_to_instruct_drops_unknowns():
    items = design_to_instruct(gender="robot", age="ancient", accent="martian", pitch=1.0)
    assert items == []


def test_shape_prosody_keeps_shape_and_ceiling():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(4000) * 0.5).astype(np.float32)
    style = resolve_style("excited")
    shaped = shape_prosody(audio, 24000, style)
    assert shaped.shape[0] == 4000
    assert float(np.max(np.abs(shaped))) <= 1.0
