"""Bridges: encoder features -> decoder input embeddings (or token ids).

All bridges expose a single ``forward(features) -> bridge_output`` where
``bridge_output`` is one of:

  ``("embed", Tensor[B, K, d_lm])``   for soft-prompt bridges (Linear, Q-Former)
  ``("ids",   LongTensor[B, K])``     for vocab-extension bridges

The model wrapper (``model.py``) consumes the tagged tuple and routes to
either ``inputs_embeds=`` (soft-prompt) or ``input_ids=`` (vocab) on the
Gemma decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# 1. Linear projector + soft prompt  (LLaVA / Gemma-native)
# ============================================================================


class LinearBridge(nn.Module):
    """Attention-pool tokens over time, RMSNorm + Linear into d_lm.
    Byte-identical to ``Gemma4MultimodalEmbedder``: RMSNorm -> Linear.
    """

    def __init__(self, d_in: int, d_lm: int, n_soft_tokens: int = 32):
        super().__init__()
        self.n_soft = n_soft_tokens
        self.norm = nn.RMSNorm(d_in)
        self.proj = nn.Linear(d_in, d_lm, bias=False)
        # Learnable query-bank for fixed-K time pooling.
        self.queries = nn.Parameter(torch.randn(n_soft_tokens, d_in) * 0.02)

    def forward(self, features: torch.Tensor) -> tuple[str, torch.Tensor]:
        # features: (B, T_seq, d_in). Cross-attend learnable queries to features.
        B, T, D = features.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # (B, K, D)
        # Lightweight scaled-dot-product attention (no params).
        attn = torch.softmax(q @ features.transpose(-1, -2) / (D ** 0.5), dim=-1)
        pooled = attn @ features                                    # (B, K, D)
        out = self.proj(self.norm(pooled))                          # (B, K, d_lm)
        return ("embed", out)


# ============================================================================
# 2. Q-Former  (BLIP-2 / BELT-2)
# ============================================================================


class QFormerBridge(nn.Module):
    """K learnable queries cross-attend to encoder features through L blocks."""

    def __init__(self, d_in: int, d_lm: int, n_queries: int = 32, n_layers: int = 6, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_in) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_in,
            nhead=n_heads,
            dim_feedforward=4 * d_in,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_in, d_lm, bias=False)
        self.norm = nn.RMSNorm(d_lm)

    def forward(self, features: torch.Tensor) -> tuple[str, torch.Tensor]:
        B = features.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # (B, K, d_in)
        out = self.blocks(tgt=q, memory=features)                # (B, K, d_in)
        out = self.norm(self.proj(out))                          # (B, K, d_lm)
        return ("embed", out)


# ============================================================================
# 3. Vocabulary extension  (NeuroLM-style)
# ============================================================================


class VocabBridge(nn.Module):
    """Pass-through for already-discrete encoders (TFM).

    The decoder's embedding table must be extended by ``codebook_size`` rows
    *before* training; ``decoder.py::extend_vocab`` does this once.
    The bridge then just rebases token ids by the offset so they index the
    new rows. ``forward`` returns the offset id stream directly.
    """

    def __init__(self, codebook_size: int, vocab_offset: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.vocab_offset = vocab_offset

    def forward(self, tokens: torch.LongTensor) -> tuple[str, torch.LongTensor]:
        return ("ids", tokens + self.vocab_offset)


# ============================================================================
# 4. RVQ head — turns continuous encoder features into discrete codes for the
#    off-diagonal (REVE × vocab) and (DIVER-1 × vocab) cells.
# ============================================================================


class RVQHead(nn.Module):
    """Tiny single-stage VQ over encoder features. Sufficient for the §2.3.2
    construction that just needs a discrete codebook of comparable size.

    Returns a tuple ``(ids, commit_loss)``. ``ids`` is the discrete code
    sequence (LongTensor (B, T)). ``commit_loss`` is the standard VQ-VAE
    commitment loss — added to the LM loss in trainables_stage1 so that the
    codebook actually moves toward the encoder distribution. Without this,
    the codebook stays at its random initialisation forever (because integer
    ids cut the gradient to the LM).
    """

    def __init__(self, d_in: int, codebook_size: int = 8192,
                 commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_in) * 0.02)
        self.commitment_weight = commitment_weight
        self.last_commit_loss: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.LongTensor:
        B, T, D = features.shape
        f = features.reshape(-1, D)
        # ||f-c||^2 = ||f||^2 - 2 f.c + ||c||^2 — argmin over codes.
        d = (f.pow(2).sum(-1, keepdim=True)
             - 2 * f @ self.codebook.t()
             + self.codebook.pow(2).sum(-1))
        ids_flat = d.argmin(dim=-1)
        # Commitment loss: pull codebook entries toward the encoder
        # features that select them, and (with a smaller coefficient) pull
        # encoder features toward their selected code (the latter only
        # matters if the encoder is trainable; harmless when it's frozen).
        codes = self.codebook[ids_flat]                                     # (B*T, D)
        commit = ((codes - f.detach()).pow(2).mean()
                  + self.commitment_weight * (codes.detach() - f).pow(2).mean())
        self.last_commit_loss = commit
        return ids_flat.reshape(B, T)


# ============================================================================
# 5. CTC head — ASR-style direct EEG -> text-token decoding
# ============================================================================


class CTCBridge(nn.Module):
    """ASR-style CTC head over encoder features.

    Maps encoder features ``(B, T_seq, D)`` to per-frame log-probabilities
    over a small character vocabulary (see ``chars.py``), then training
    minimises ``F.ctc_loss``. There is **no decoder LM in the loop during
    training** — so the LM-prior trap that dominates the soft-prompt
    cells (matched-pair §4.3 result in results.md) is impossible by
    construction. If the EEG carries any information about the text, the
    CTC matrix will reflect it.

    Architecture:
      RMSNorm -> Linear(D -> hidden) -> n_layers x TransformerEncoderLayer
      -> Linear(hidden, vocab_size)

    Returns ``("ctc_logits", logits)`` where logits has shape
    ``(B, T_seq, vocab_size)`` (not log-softmaxed; trainer applies
    ``log_softmax`` and feeds to ``F.ctc_loss``).
    """

    def __init__(self, d_in: int, vocab_size: int, *,
                 hidden: int = 512, n_layers: int = 2, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.RMSNorm(d_in)
        self.proj_in = nn.Linear(d_in, hidden, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=4 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, features: torch.Tensor) -> tuple[str, torch.Tensor]:
        # features: (B, T_seq, D)
        x = self.norm(features)
        x = self.proj_in(x)
        x = self.encoder(x)
        logits = self.head(x)
        return ("ctc_logits", logits)


# ============================================================================
# Factory
# ============================================================================


def build_bridge(*, kind: str, d_in: int, d_lm: int, n_queries: int,
                 codebook_size: int, vocab_offset: int,
                 ctc_vocab_size: int = 0):
    if kind == "linear":
        return LinearBridge(d_in=d_in, d_lm=d_lm, n_soft_tokens=n_queries)
    if kind == "qformer":
        return QFormerBridge(d_in=d_in, d_lm=d_lm, n_queries=n_queries)
    if kind == "vocab":
        return VocabBridge(codebook_size=codebook_size, vocab_offset=vocab_offset)
    if kind == "ctc":
        if ctc_vocab_size <= 0:
            raise ValueError("CTC bridge requires ctc_vocab_size > 0")
        return CTCBridge(d_in=d_in, vocab_size=ctc_vocab_size)
    raise ValueError(f"unknown bridge: {kind}")
