"""CTC vocabulary loaders — char (50 ids) or BPE-1k (~1024 ids).

Both vocabularies follow the standard CTC convention::

    id 0      BLANK token (required by torch.nn.functional.ctc_loss)
    id 1      UNK token   (out-of-vocab character / piece)
    id 2..    real tokens (chars or BPE pieces)

The two vocabularies live behind a common :class:`Vocab` interface so the
model and trainer don't have to special-case them.

BPE is built once via ``exp02 build-bpe`` (calls
:func:`exp02.tokenizer_build.train_bpe_tokenizer`) and lives at
``$EXP02_DATA_ROOT/bpe/spm.model``.
"""

from __future__ import annotations

from typing import Iterable, Protocol

import torch


# Reserved IDs (must match the CTC contract).
BLANK_ID = 0
UNK_ID = 1
_FIRST_REAL_ID = 2


class Vocab(Protocol):
    """Common interface for char and BPE vocabularies."""

    name: str
    size: int  # total IDs including BLANK + UNK

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Iterable[int]) -> str: ...

    def encode_batch(
        self, texts: list[str], *, device: torch.device | str = "cpu",
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Pack a batch of strings into ``(targets_concat, lengths)`` as
        required by ``torch.nn.functional.ctc_loss``.
        """
        ...


# ============================================================================
# Char vocabulary (50 ids — exp01 default)
# ============================================================================


_CHARS = "abcdefghijklmnopqrstuvwxyz '.,?!-:;\"()0123456789"
_CHAR_TO_ID = {c: i + _FIRST_REAL_ID for i, c in enumerate(_CHARS)}
_ID_TO_CHAR = {i: c for c, i in _CHAR_TO_ID.items()}


class CharVocab:
    name = "char"
    size = len(_CHARS) + _FIRST_REAL_ID  # 50

    def encode(self, text: str) -> list[int]:
        return [_CHAR_TO_ID.get(c, UNK_ID) for c in text.lower()]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(_ID_TO_CHAR[i] for i in ids if i in _ID_TO_CHAR)

    def encode_batch(self, texts, *, device="cpu"):
        encoded = [self.encode(t) for t in texts]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long, device=device)
        flat = torch.tensor([t for e in encoded for t in e], dtype=torch.long, device=device)
        return flat, lengths


# ============================================================================
# BPE-1k vocabulary (sentencepiece-trained on ZuCo + Wikipedia)
# ============================================================================


class BPEVocab:
    """Sentencepiece BPE wrapper.

    Sentencepiece's own ID 0 is reserved for ``<unk>``; we re-shift the
    sentencepiece IDs by 2 so our vocabulary has BLANK_ID=0 and UNK_ID=1
    at the front, matching the char vocab contract. This means our
    effective vocab size is ``sp.vocab_size() + 2``.
    """

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.name = "bpe1k"
        # +2 for BLANK + UNK.
        self.size = self.sp.GetPieceSize() + _FIRST_REAL_ID
        # Internal sentencepiece UNK ID (usually 0 in sp's space, mapped to
        # our UNK_ID=1).
        self._sp_unk = self.sp.unk_id()

    def encode(self, text: str) -> list[int]:
        sp_ids = self.sp.EncodeAsIds(text.lower())
        # Sentencepiece-internal IDs are non-negative; shift by +2. Map
        # sp's <unk> to our UNK_ID=1.
        return [
            UNK_ID if sid == self._sp_unk else (sid + _FIRST_REAL_ID)
            for sid in sp_ids
        ]

    def decode(self, ids: Iterable[int]) -> str:
        sp_ids: list[int] = []
        for i in ids:
            if i == BLANK_ID or i == UNK_ID:
                continue
            sp_ids.append(i - _FIRST_REAL_ID)
        return self.sp.DecodeIds(sp_ids)

    def encode_batch(self, texts, *, device="cpu"):
        encoded = [self.encode(t) for t in texts]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long, device=device)
        flat = torch.tensor([t for e in encoded for t in e], dtype=torch.long, device=device)
        return flat, lengths


# ============================================================================
# Factory
# ============================================================================


def load_vocab(name: str, *, bpe_model_path: str | None = None) -> Vocab:
    """Load a vocabulary by name.

    Args:
        name: ``"char"`` or ``"bpe1k"``.
        bpe_model_path: required when ``name="bpe1k"``; path to the trained
            sentencepiece ``.model`` file.
    """
    if name == "char":
        return CharVocab()
    if name == "bpe1k":
        if not bpe_model_path:
            raise ValueError(
                "BPE-1k vocab requires a model path. Build one with "
                "`exp02 build-bpe` first."
            )
        return BPEVocab(bpe_model_path)
    raise ValueError(f"unknown vocab: {name}")


# ============================================================================
# CTC greedy decoder — common to both vocabs (it operates on log-probs
# directly; vocab translation is the caller's responsibility).
# ============================================================================


def ctc_greedy_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """Standard CTC greedy decoder.

    log_probs: ``(B, T, V)`` — log-probabilities per (batch, frame, token).
    Returns a list of length B where each element is a list of ids with
    repeats collapsed and BLANKs removed. Pass through the vocab's
    ``decode(ids)`` to render as text.
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
