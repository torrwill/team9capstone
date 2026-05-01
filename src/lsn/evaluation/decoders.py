"""CTC decoders — greedy + beam=100 (paper setting).

Refactored from notebook Stage C cells 33 and 35.
beam_decode requires input_lengths (torchaudio API, spec §4.6).
Beam decoder uses an in-memory tokens list instead of a tokens file
(spec §9.3 — no runtime filesystem writes).
"""
from __future__ import annotations

import torch
from torch import Tensor

from lsn.data.vocab import BLANK_INDEX, CHARS, decode_ids


def greedy_decode(log_probs: Tensor,
                  input_lengths: Tensor | None = None) -> list[str]:
    """Greedy CTC decoding. Input shape (T, B, C); returns list of B strings.

    Refactored from notebook cell 33. input_lengths defaults to T (full
    sequence) when omitted — correct for GRID's fixed T=75 clips.
    """
    T = log_probs.shape[0]
    preds = log_probs.argmax(dim=-1).permute(1, 0).cpu().numpy()

    if input_lengths is not None:
        lens = input_lengths.cpu().numpy()
    else:
        lens = [T] * preds.shape[0]

    decoded = []
    for b in range(preds.shape[0]):
        seq = preds[b, :lens[b]]
        # CTC collapse: merge repeats, then strip blanks and out-of-range tokens.
        # Indices > len(CHARS) are unused classifier outputs (vocab_size=40 > 27
        # real characters) — skip them just as the beam decoder does.
        collapsed = []
        prev = -1
        for tok in seq:
            tok = int(tok)
            if tok != prev:
                if tok != BLANK_INDEX and 1 <= tok <= len(CHARS):
                    collapsed.append(tok)
                prev = tok
        decoded.append(decode_ids(collapsed).strip())

    return decoded


# In-memory token list matching the tokens file written in notebook cell 35.
# idx 0 = blank ("-"), idx 1 = space ("|"), idx 2..27 = "a".."z"
# Using "|" as the word-separator token per torchaudio convention.
_TOKENS: list[str] = ["-"] + [("|" if ch == " " else ch) for ch in CHARS]

_BEAM_DECODER_CACHE: dict[int, object] = {}


def _get_beam_decoder(beam_width: int):
    """Lazy-construct + cache the torchaudio ctc_decoder by beam width."""
    if beam_width not in _BEAM_DECODER_CACHE:
        from torchaudio.models.decoder import ctc_decoder  # lazy — GPU env only
        _BEAM_DECODER_CACHE[beam_width] = ctc_decoder(
            lexicon=None,
            tokens=_TOKENS,
            lm=None,
            beam_size=beam_width,
            blank_token="-",
            sil_token="|",
            unk_word="<unk>",
        )
    return _BEAM_DECODER_CACHE[beam_width]


def beam_decode(log_probs: Tensor, input_lengths: Tensor,
                beam_width: int = 100) -> list[str]:
    """Beam-search CTC decode.

    log_probs: (T, B, C) — same convention as model.forward output.
    input_lengths: (B,) long — actual T per sample (75 for fixed-T GRID).

    Refactored from notebook cell 35.
    """
    decoder = _get_beam_decoder(beam_width)
    # torchaudio expects (B, T, C) — transpose from training convention
    emissions = log_probs.permute(1, 0, 2).cpu()
    lens = input_lengths.cpu()

    hypotheses = decoder(emissions, lens)

    out = []
    for beams in hypotheses:
        top = beams[0]
        chars = []
        for tid in top.tokens.tolist():
            if tid == BLANK_INDEX:
                continue
            if tid < 1 or tid > len(CHARS):
                continue
            ch = CHARS[tid - 1]
            chars.append(ch)
        out.append("".join(chars).strip())

    return out
