"""Character-level tokenizer + CTC decoder for the ASR-style track.

CTC operates over (T, V+1) log-probabilities where V is the size of the
output alphabet and the extra slot is the special "blank" token. We use
character-level output because:

  1. The vocabulary is small (~50 entries), so the CTC head is tiny and
     trains fast — and gives much more headroom on the InfoNCE-style
     "is-the-encoder-doing-anything" diagnostic than the 262 144-token
     Gemma vocab.
  2. Per Jo et al. §4.3, character-level CER on noise vs EEG is the
     strongest test of "is the model decoding from EEG content?" — the
     LM-prior trap collapses to a constant for noise (whatever single
     char is most common at each greedy step), which is trivially
     distinguishable from "EEG-driven varying CTC outputs".
  3. CTC frees us from any LM whatsoever during training. The Gemma
     prior — which the matched-pair test in results.md shows is doing
     all of the lifting in the soft-prompt cells — is out of the loop.
     If CTC works, we *know* the encoder/bridge are using EEG content.

By convention the blank token is id 0 (this is what ``F.ctc_loss``
expects when ``blank=0``). UNK is id 1. Then CHARS occupies ids 2..V-1.
"""

from __future__ import annotations

from typing import Iterable

import torch


# ============================================================================
# Vocabulary
# ============================================================================
#
# Lowercased English letters + space + the punctuation that actually shows up
# in ZuCo's references (movie reviews + biographies). Keep it small: every
# extra char is one more output channel × T_seq × B that the CTC matrix
# allocates.

CHARS = "abcdefghijklmnopqrstuvwxyz '.,?!-:;\"()0123456789"

BLANK_ID = 0
UNK_ID = 1
_FIRST_CHAR_ID = 2

CHAR_TO_ID = {c: i + _FIRST_CHAR_ID for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHARS) + _FIRST_CHAR_ID  # 50 ish


def encode_text(text: str) -> list[int]:
    """Lowercase the input and map each char to its id; UNK for unknowns."""
    return [CHAR_TO_ID.get(c, UNK_ID) for c in text.lower()]


def decode_ids(ids: Iterable[int]) -> str:
    """Map ids back to chars; drop BLANK and UNK in the rendered output."""
    return "".join(ID_TO_CHAR[i] for i in ids if i in ID_TO_CHAR)


def ctc_greedy_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """Standard CTC greedy decoder.

    log_probs: ``(B, T, V)`` — log-probabilities per (batch, frame, token).
    Returns a list of length B where each element is a list of ids with
    repeats collapsed and BLANKs removed. Pass through ``decode_ids`` to
    render as text.
    """
    pred = log_probs.argmax(dim=-1).cpu().tolist()
    out: list[list[int]] = []
    for seq in pred:
        collapsed: list[int] = []
        prev: int | None = None
        for p in seq:
            if p == BLANK_ID:
                prev = p
                continue
            if p != prev:
                collapsed.append(p)
            prev = p
        out.append(collapsed)
    return out


def encode_batch(texts: list[str], *, device: torch.device | str = "cpu"
                 ) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Pack a batch of strings into ``(targets_concat, lengths)`` as required
    by ``torch.nn.functional.ctc_loss``. ``targets_concat`` is the
    concatenation of all encoded sequences (1-D LongTensor); ``lengths``
    holds each sequence's length.
    """
    encoded = [encode_text(t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long, device=device)
    flat = torch.tensor([t for e in encoded for t in e], dtype=torch.long, device=device)
    return flat, lengths


# ============================================================================
# Pretty-print helper for the train logger
# ============================================================================


def render(ids: Iterable[int]) -> str:
    """Same as ``decode_ids`` but explicit for readability in the trainer."""
    return decode_ids(ids)
