from app.shorts_media import _escape_drawtext, _group_caption_phrases, build_caption_filter


def _words():
    return [
        {"start": 0.0, "end": 0.4, "text": "one"},
        {"start": 0.4, "end": 0.8, "text": "two"},
        {"start": 0.8, "end": 1.2, "text": "three"},
        {"start": 1.3, "end": 1.7, "text": "four"},
        {"start": 5.0, "end": 5.4, "text": "later"},  # big gap -> new phrase
    ]


def test_group_caption_phrases_limits_words_per_phrase():
    phrases = _group_caption_phrases(_words(), 0.0, 10.0, max_words=4, max_gap=0.7)
    # First four within a phrase, "later" split off by the gap.
    assert len(phrases) == 2
    assert phrases[0][2] == "one two three four"
    assert phrases[1][2] == "later"


def test_group_caption_phrases_rebases_times_to_clip():
    phrases = _group_caption_phrases(_words(), 0.5, 10.0)
    assert phrases[0][0] >= 0.0  # start rebased so clip begins at 0


def test_group_caption_phrases_skips_out_of_window():
    phrases = _group_caption_phrases(_words(), 2.0, 3.0)
    assert phrases == []  # nothing between 2s and 3s


def test_build_caption_filter_fewer_filters_than_words():
    words = [{"start": i * 0.4, "end": i * 0.4 + 0.35, "text": f"w{i}"} for i in range(20)]
    result = build_caption_filter(words, 0.0, 20.0)
    assert result.count("drawtext=") < len(words)  # grouped, not one per word


def test_escape_drawtext_escapes_specials():
    escaped = _escape_drawtext("a:b,c[d]")
    assert "\\:" in escaped and "\\," in escaped and "\\[" in escaped
