"""LRS2-specific text normalization.

Applied at metric-compute time inside lsn.evaluation.report (spec §7),
not at JSON-write time. JSON predictions contain raw decoder output.

Refactored verbatim from notebook Stage H (cell 49). The notebook
exposed this function as ``normalize_text_lrs2``; the refactor renames
it to ``normalize_lrs2`` per the plan but the body is unchanged.

Normalizations applied (in order):
  1. lowercase               (safety — LRS2 is already lowercase, GRID too)
  2. remove apostrophes      (don't introduce a space; "they're" -> "theyre")
  3. remove digits           (but DO introduce a space to avoid word merging)
  4. collapse whitespace     (any remaining runs of spaces become single)
  5. strip                   (leading/trailing whitespace)
"""
from __future__ import annotations

import re

# Module-scope helper regexes (verbatim from cell 49).
_APOSTROPHE_RE = re.compile(r"[''`]")          # straight and curly apostrophes
_DIGIT_RE      = re.compile(r"\d")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_lrs2(s: str) -> str:
    """Refactored verbatim from notebook Stage H (cell 49).

    Lowercase + strip apostrophes + digits-to-space + collapse whitespace
    + strip. Applied to BOTH predictions and references on LRS2 before
    metric computation.
    """
    # Body pasted verbatim from cell 49 `normalize_text_lrs2`.
    s = s.lower()
    s = _APOSTROPHE_RE.sub("", s)     # "they're" -> "theyre" (no space inserted)
    s = _DIGIT_RE.sub(" ", s)         # "year 2019" -> "year      "
    s = _WHITESPACE_RE.sub(" ", s)    # collapse any whitespace runs
    return s.strip()
