"""CTC decoders: greedy, beam, beam + KenLM 4-gram rescore.

Used by :mod:`exp02.eval` to produce three hypothesis variants per cell:

  * ``hyp_greedy``     argmax + collapse repeats + drop blanks. Diagnostic.
  * ``hyp_beam``       beam search (no LM). Tests whether the model has
                       multi-hypothesis content beyond the single best path.
  * ``hyp_beam_kenlm`` beam search rescored by a KenLM 4-gram (Distill.pub:
                       greedy CTC is "useless without LM rescoring").

Beam decoding uses ``pyctcdecode`` (which transparently uses the optional
``kenlm`` Python bindings when a model path is provided). All three decoders
operate on (B, T, V) log-probabilities — the model forward is GPU, the
decode is CPU. Beam decoding is parallelised across CPUs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from . import storage
from .chars import BLANK_ID, UNK_ID, Vocab, ctc_greedy_decode


# ============================================================================
# Greedy
# ============================================================================


def decode_greedy(log_probs: torch.Tensor, vocab: Vocab) -> list[str]:
    """log_probs: (B, T, V). Returns one string per row."""
    ids_per_row = ctc_greedy_decode(log_probs)
    return [vocab.decode(ids) for ids in ids_per_row]


# ============================================================================
# Beam (with optional KenLM)
# ============================================================================


def _build_pyctc_decoder(vocab: Vocab, *, kenlm_path: str | None,
                         alpha: float, beta: float):
    """Construct a ``pyctcdecode.BeamSearchDecoderCTC`` for ``vocab``.

    ``pyctcdecode`` requires a flat list of token strings indexed by their
    id, including position 0 (blank) and position 1 (UNK). We emit the
    canonical mapping by asking the vocab to render single-id sequences.
    """
    from pyctcdecode import build_ctcdecoder

    labels: list[str] = []
    for i in range(vocab.size):
        if i == BLANK_ID:
            labels.append("")  # pyctcdecode convention for blank
        elif i == UNK_ID:
            labels.append("⁇")  # any unique non-empty placeholder
        else:
            # vocab.decode operates on iterables; render a single-id list.
            piece = vocab.decode([i])
            # Sentencepiece BPE pieces frequently carry the U+2581 word
            # boundary marker; pyctcdecode handles spacing itself, so we
            # leave the raw piece in place. For char vocab this is just the
            # character.
            labels.append(piece if piece != "" else "·")

    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=kenlm_path,
        alpha=alpha,
        beta=beta,
    )


def _decode_with_pyctc(log_probs: torch.Tensor, decoder, beam_width: int) -> list[str]:
    np_log_probs = log_probs.detach().float().cpu().numpy()
    out: list[str] = []
    for row in np_log_probs:
        text = decoder.decode(row, beam_width=beam_width)
        out.append(text)
    return out


def decode_beam(log_probs: torch.Tensor, vocab: Vocab, *, beam_width: int = 50) -> list[str]:
    """Beam search decoding without LM."""
    decoder = _build_pyctc_decoder(vocab, kenlm_path=None, alpha=0.0, beta=0.0)
    return _decode_with_pyctc(log_probs, decoder, beam_width=beam_width)


def decode_beam_kenlm(
    log_probs: torch.Tensor,
    vocab: Vocab,
    *,
    beam_width: int = 50,
    alpha: float = 0.5,
    beta: float = 1.5,
    kenlm_path: str | None = None,
) -> list[str]:
    """Beam search + KenLM 4-gram rescore.

    Uses the binary KenLM by default (``$EXP02_DATA_ROOT/kenlm/4gram.binary``);
    falls back to the ARPA form. Build with ``exp02 build-kenlm`` first.
    """
    if kenlm_path is None:
        kenlm_path = (str(storage.KENLM_BINARY) if storage.KENLM_BINARY.exists()
                      else str(storage.KENLM_ARPA))
    decoder = _build_pyctc_decoder(vocab, kenlm_path=kenlm_path,
                                    alpha=alpha, beta=beta)
    return _decode_with_pyctc(log_probs, decoder, beam_width=beam_width)


# ============================================================================
# Convenience: run all three decoders in one shot
# ============================================================================


def decode_all(
    log_probs: torch.Tensor,
    vocab: Vocab,
    *,
    beam_width: int = 50,
    kenlm_alpha: float = 0.5,
    kenlm_beta: float = 1.5,
    enable_beam: bool = True,
    enable_beam_kenlm: bool = True,
) -> dict[str, list[str]]:
    """Return all three hyp lists keyed by decode mode.

    Skips beam / beam_kenlm if they're disabled or if KenLM artifacts are
    missing — handy for the smoke-test path where neither has been built.
    """
    hyps = {"greedy": decode_greedy(log_probs, vocab)}
    if enable_beam:
        try:
            hyps["beam"] = decode_beam(log_probs, vocab, beam_width=beam_width)
        except Exception as e:
            print(f"[decode] beam decode failed ({type(e).__name__}: {e}); "
                  f"skipping beam.", flush=True)
    if enable_beam_kenlm:
        kenlm_present = storage.KENLM_BINARY.exists() or storage.KENLM_ARPA.exists()
        if not kenlm_present:
            print("[decode] no KenLM at $EXP02_DATA_ROOT/kenlm/; "
                  "skipping beam+KenLM (run `exp02 build-kenlm` to enable).",
                  flush=True)
        else:
            try:
                hyps["beam_kenlm"] = decode_beam_kenlm(
                    log_probs, vocab,
                    beam_width=beam_width,
                    alpha=kenlm_alpha, beta=kenlm_beta,
                )
            except Exception as e:
                print(f"[decode] beam+KenLM failed ({type(e).__name__}: {e}); "
                      f"skipping.", flush=True)
    return hyps
