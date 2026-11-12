from src.preprocessing import clean_text


def test_removes_urls_and_mentions():
    out = clean_text("@apple check https://t.co/abc now")
    assert "http" not in out and "apple" not in out


def test_hashtag_text_is_kept():
    assert "winning" in clean_text("#Winning today")


def test_negation_scope_marked():
    out = clean_text("this is not good")
    assert "NOT_good" in out


def test_negation_stops_at_punctuation():
    assert "NOT_great" not in clean_text("not sure, great phone")


def test_elongation_and_emoji():
    out = clean_text("soooo happy 😍")
    assert "soo" in out and "emojipos" in out


def test_non_string_is_safe():
    assert clean_text(None) == ""
