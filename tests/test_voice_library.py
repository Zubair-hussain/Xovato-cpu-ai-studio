from app.voice_library import STOCK_VOICES, get_stock_voice, library_payload, validate_library


def test_library_validates_at_import():
    # Should not raise: every stock voice uses only supported instruct items.
    validate_library()


def test_get_stock_voice_case_insensitive():
    assert get_stock_voice("ARIA").id == "aria"
    assert get_stock_voice("  atlas ").id == "atlas"


def test_get_stock_voice_unknown_returns_none():
    assert get_stock_voice("nope") is None
    assert get_stock_voice(None) is None


def test_library_payload_shape():
    payload = library_payload()
    assert len(payload) == len(STOCK_VOICES)
    first = payload[0]
    assert {"id", "name", "tagline", "instruct_items", "default_speed", "best_for"} <= set(first)
