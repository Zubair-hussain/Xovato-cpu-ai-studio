import pytest

from app.shorts_pipeline import (
    Candidate,
    Segment,
    ShortsPipelineError,
    Transcript,
    Word,
    _parse_llm_candidates,
    _suppress_overlaps,
    rank_heuristic,
    rank_moments,
)


def _transcript():
    segments = [
        Segment(0.0, 6.0, "Why does nobody talk about this? It is the biggest mistake creators make."),
        Segment(6.0, 14.0, "Here is the secret: you must hook the viewer in the first three seconds."),
        Segment(14.0, 26.0, "The reason is simple. Attention is the only currency that matters online today."),
        Segment(26.0, 40.0, "So that is why you always start with a question that the viewer needs answered."),
    ]
    words = [Word(float(i), float(i) + 0.4, f"w{i}") for i in range(40)]
    return Transcript(engine="test", language="en", segments=segments, words=words)


def test_rank_heuristic_returns_ranked_candidates():
    candidates = rank_heuristic(_transcript(), target_count=3, min_len=8.0, max_len=30.0)
    assert candidates
    assert candidates[0].rank == 1
    assert all(c.duration >= 8.0 for c in candidates)


def test_rank_heuristic_empty_transcript():
    empty = Transcript(engine="t", language="en", segments=[], words=[])
    assert rank_heuristic(empty, target_count=3, min_len=5.0, max_len=20.0) == []


def test_suppress_overlaps_drops_overlapping():
    a = Candidate(0.0, 20.0, 0.9, "a", "")
    b = Candidate(5.0, 25.0, 0.5, "b", "")  # overlaps a, lower score
    c = Candidate(30.0, 45.0, 0.7, "c", "")
    kept = _suppress_overlaps([a, b, c], target_count=5)
    hooks = {k.hook for k in kept}
    assert hooks == {"a", "c"}


def test_rank_moments_falls_back_to_heuristic(monkeypatch):
    # No API key set -> LLM path raises and rank_moments degrades to heuristic.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidates, engine = rank_moments(_transcript(), provider="openai", min_len=8.0, max_len=30.0)
    assert engine == "heuristic"
    assert candidates


def test_parse_llm_candidates_valid_json():
    content = 'noise [{"start_sec": 1, "end_sec": 20, "score": 0.8, "hook": "hi", "rationale": "r"}] trailing'
    parsed = _parse_llm_candidates(content, target_count=5)
    assert parsed[0].hook == "hi"


def test_parse_llm_candidates_skips_bad_rows():
    content = '[{"start_sec": 1, "end_sec": 5}, {"bad": true}]'
    parsed = _parse_llm_candidates(content, target_count=5)
    assert len(parsed) == 1


def test_parse_llm_candidates_no_json_raises():
    with pytest.raises(ShortsPipelineError):
        _parse_llm_candidates("there is no array here", target_count=5)


def test_group_words_into_segments():
    from app.shorts_pipeline import _group_words_into_segments

    words = [Word(0.0, 0.4, "a"), Word(0.5, 0.9, "b"), Word(3.0, 3.4, "c")]
    segments = _group_words_into_segments(words, max_gap=0.6)
    assert len(segments) == 2  # the 2.1s gap splits into two segments


def test_transcribe_audio_deepgram_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    from app.shorts_pipeline import transcribe_audio

    with pytest.raises(ShortsPipelineError):
        transcribe_audio(tmp_path / "a.wav", provider="deepgram")


def test_rank_moments_unknown_provider_falls_back():
    candidates, engine = rank_moments(_transcript(), provider="totally-unknown", min_len=8.0, max_len=30.0)
    assert engine == "heuristic"


def test_transcript_full_text_and_as_dict():
    transcript = _transcript()
    assert "secret" in transcript.full_text
    payload = transcript.as_dict()
    assert payload["word_count"] == 40 and "segments" in payload


def test_candidate_as_dict_rounds():
    candidate = Candidate(1.234, 9.876, 0.55555, "hook", "why")
    payload = candidate.as_dict()
    assert payload["duration_sec"] == round(9.876 - 1.234, 2)
    assert payload["score"] == 0.556
