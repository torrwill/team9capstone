"""Tests for LRS2 text normalization (refactored from notebook cell 49)."""
from lsn.data.normalize import normalize_lrs2


def test_normalize_returns_str():
    assert isinstance(normalize_lrs2("hello world"), str)


def test_normalize_invariant_on_clean_input():
    """Already-lowercase, no-punct, no-digit, single-spaced input round-trips."""
    s = "hello world"
    assert normalize_lrs2(s) == s


# ------------------------------------------------------------------
# Per-rule tests, one per transformation listed in the cell docstring
# ------------------------------------------------------------------

# Rule 1: lowercase
def test_normalize_lowercases():
    assert normalize_lrs2("HELLO") == "hello"


def test_normalize_lowercase_idempotent():
    assert normalize_lrs2("HELLO World") == normalize_lrs2("hello world")


# Rule 2: remove apostrophes (no space inserted) — straight + curly + backtick
def test_normalize_strips_straight_apostrophe():
    # "they're" -> "theyre" (no space inserted)
    assert normalize_lrs2("they're moving around") == "theyre moving around"


def test_normalize_strips_apostrophe_isnt():
    assert normalize_lrs2("when there isn't much else in the garden") == \
        "when there isnt much else in the garden"


def test_normalize_strips_apostrophe_its():
    assert normalize_lrs2("it's not all about size") == "its not all about size"


def test_normalize_strips_backtick():
    # The cell's apostrophe class is r"[''`]" — two ASCII apostrophes and a
    # backtick. (The comment in the cell calls these "straight and curly"
    # but the actual codepoints are 0x27, 0x27, 0x60 — no U+2019.)
    assert normalize_lrs2("they`re here") == "theyre here"


# Rule 3: remove digits, but DO insert a space (avoids word merging)
def test_normalize_digit_replaced_with_space():
    # Cell smoke test: "year 2019 was good" -> "year was good"
    assert normalize_lrs2("year 2019 was good") == "year was good"


def test_normalize_digit_does_not_merge_words():
    # Without the space substitution, "abc1def" would become "abcdef".
    # The cell substitutes digits with a space, so we get "abc def".
    assert normalize_lrs2("abc1def") == "abc def"


def test_normalize_strips_all_digits():
    assert normalize_lrs2("0123456789") == ""


# Rule 4: collapse whitespace runs into a single space
def test_normalize_collapses_internal_whitespace():
    assert normalize_lrs2("a    b") == "a b"


def test_normalize_collapses_tabs_and_newlines():
    assert normalize_lrs2("a\t\nb") == "a b"


# Rule 5: strip leading/trailing whitespace
def test_normalize_strips_leading_trailing():
    # Cell smoke test: "  extra   spaces  " -> "extra spaces"
    assert normalize_lrs2("  extra   spaces  ") == "extra spaces"


# ------------------------------------------------------------------
# Cell smoke-test cases (pasted from cell 49 _test_cases) — paranoid check
# ------------------------------------------------------------------
def test_cell_smoke_case_theyre():
    assert normalize_lrs2("they're moving around") == "theyre moving around"


def test_cell_smoke_case_isnt():
    assert normalize_lrs2("when there isn't much else in the garden") == \
        "when there isnt much else in the garden"


def test_cell_smoke_case_its():
    assert normalize_lrs2("it's not all about size") == "its not all about size"


def test_cell_smoke_case_hello_world():
    assert normalize_lrs2("HELLO World") == "hello world"


def test_cell_smoke_case_extra_spaces():
    assert normalize_lrs2("  extra   spaces  ") == "extra spaces"


def test_cell_smoke_case_year_2019():
    assert normalize_lrs2("year 2019 was good") == "year was good"
