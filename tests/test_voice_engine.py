"""Guard-path tests for the voice engine that do not require torch/omnivoice."""

import pytest

from app.config import get_settings
from app.voice_engine import VoiceEngine, VoiceEngineError, voice_engine
from app.voice_styles import resolve_style


def test_status_unloaded():
    status = voice_engine.status()
    assert status["loaded"] is False
    assert "cached_voice_prompts" in status


def test_synthesize_empty_text_raises():
    engine = VoiceEngine()
    with pytest.raises(VoiceEngineError):
        engine.synthesize(get_settings(), text="   ", style=resolve_style("neutral"))


def test_clone_prompt_requires_transcript(tmp_path):
    engine = VoiceEngine()
    with pytest.raises(VoiceEngineError):
        engine.clone_prompt(get_settings(), cache_key="k", ref_audio=tmp_path / "r.wav", ref_text="")


def test_invalidate_is_safe_on_empty_cache():
    engine = VoiceEngine()
    engine.invalidate("nothing")  # must not raise
