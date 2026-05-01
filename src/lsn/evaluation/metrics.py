"""CTC metrics — refactored from notebook Stage E (cell 39)."""
from __future__ import annotations


def edit_distance(ref: list, hyp: list) -> int:
    """Standard Levenshtein. Works for character lists or word lists."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def cer(refs: list[str], hyps: list[str]) -> float:
    total_d, total_n = 0, 0
    for r, h in zip(refs, hyps):
        total_d += edit_distance(list(r), list(h))
        total_n += len(r)
    return total_d / max(total_n, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    total_d, total_n = 0, 0
    for r, h in zip(refs, hyps):
        total_d += edit_distance(r.split(), h.split())
        total_n += len(r.split())
    return total_d / max(total_n, 1)


def word_acc(refs: list[str], hyps: list[str]) -> float:
    """1 - WER (paper convention)."""
    return 1.0 - wer(refs, hyps)


def sentence_acc(refs: list[str], hyps: list[str]) -> float:
    """Exact-match rate over (ref, hyp) pairs after strip+lower normalize."""
    n_match = sum(1 for r, h in zip(refs, hyps)
                  if r.strip().lower() == h.strip().lower())
    return n_match / max(len(refs), 1)
