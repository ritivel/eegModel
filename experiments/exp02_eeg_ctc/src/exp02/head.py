"""CTC head + optional CR-CTC, intermediate-CTC, and AED branches.

Architecture::

    encoder features (B, T_seq, D)
      -> RMSNorm -> Linear(D -> hidden)
      -> N x TransformerEncoderLayer(d_model=hidden, ...)
         (intermediate predictions taken from selected layers)
      -> Linear(hidden, vocab_size)
      -> log_softmax (in trainer)

For ``ctcaed`` cells we additionally attach an attention decoder branch
(:class:`AEDHead`) that takes the encoder features and target token IDs
(teacher-forced during training) and produces cross-entropy. Trainer
combines: ``loss = (1-λ) * L_CTC + λ * L_AED``.

For ``crctc`` cells the head is run twice on two SpecAugmented views; the
trainer computes the symmetric KL between the two log-probability streams.

For ``intctc`` cells the trainer also adds a CTC loss on each intermediate
layer's projection through the same final ``Linear`` head, weighted by
``intermediate_ctc_weight``. (See Komatsu et al. 2022.)

The label-prior trick (Zeyer 2021 §7) is applied at loss-computation time:
the trainer subtracts a learned EMA of the per-token log-probabilities from
the logits before passing through ``F.ctc_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadOutput:
    """All tensors the trainer needs for one forward pass.

    - ``logits``: (B, T, V) raw logits. Trainer applies log_softmax + ctc_loss.
    - ``intermediate_logits``: optional list of (B, T, V) tensors, one per
      configured intermediate layer (intctc variant only).
    - ``aed_logits``: optional (B, L, V) tensor when AED branch is attached
      and target tokens were provided.
    """

    logits: torch.Tensor
    intermediate_logits: list[torch.Tensor]
    aed_logits: torch.Tensor | None


class CTCHead(nn.Module):
    """Transformer encoder + CTC projection.

    Args:
        d_in: encoder feature dim.
        vocab_size: includes BLANK + UNK + real tokens.
        hidden: transformer width.
        n_layers: transformer depth.
        n_heads: attention heads.
        dropout: dropout in transformer + post-projection.
        intermediate_layers: 0-based indices (within ``range(n_layers)``) at
            which to expose intermediate predictions (used by ``intctc``).
        attach_aed: if True, attach an AEDHead branch (used by ``ctcaed``).
        aed_layers / aed_heads / aed_dropout / aed_max_target_len: AED branch knobs.
    """

    def __init__(
        self,
        *,
        d_in: int,
        vocab_size: int,
        hidden: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        intermediate_layers: tuple[int, ...] = (),
        attach_aed: bool = False,
        aed_layers: int = 4,
        aed_heads: int = 8,
        aed_dropout: float = 0.1,
        aed_max_target_len: int = 96,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.intermediate_layers = tuple(sorted(set(intermediate_layers)))

        self.norm_in = nn.RMSNorm(d_in)
        self.proj_in = nn.Linear(d_in, hidden, bias=False)

        # Build encoder layers individually so we can pull intermediate
        # activations from arbitrary depths.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=4 * hidden,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Linear(hidden, vocab_size)

        self.aed: AEDHead | None = None
        if attach_aed:
            self.aed = AEDHead(
                d_in=hidden,
                vocab_size=vocab_size,
                n_layers=aed_layers,
                n_heads=aed_heads,
                dropout=aed_dropout,
                max_target_len=aed_max_target_len,
            )

    def forward(
        self,
        features: torch.Tensor,
        *,
        aed_target_ids: torch.LongTensor | None = None,
    ) -> HeadOutput:
        """features: (B, T_seq, D). Returns :class:`HeadOutput`."""
        x = self.proj_in(self.norm_in(features))

        intermediates: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.intermediate_layers:
                intermediates.append(self.head(x))

        logits = self.head(x)

        aed_logits = None
        if self.aed is not None and aed_target_ids is not None:
            aed_logits = self.aed(memory=x, target_ids=aed_target_ids)

        return HeadOutput(
            logits=logits,
            intermediate_logits=intermediates,
            aed_logits=aed_logits,
        )

    @torch.no_grad()
    def aed_generate(self, features: torch.Tensor, *, max_len: int,
                     bos_id: int, eos_id: int) -> torch.LongTensor:
        """Greedy decode from the AED branch (used at eval time for the
        ``ctcaed`` cell when reporting AED-only hypotheses for completeness).
        """
        if self.aed is None:
            raise ValueError("AED branch not attached.")
        x = self.proj_in(self.norm_in(features))
        for layer in self.layers:
            x = layer(x)
        return self.aed.greedy_generate(memory=x, max_len=max_len,
                                        bos_id=bos_id, eos_id=eos_id)


# ============================================================================
# AED branch (used only by the ``ctcaed`` variant)
# ============================================================================
#
# Standard transformer decoder. Cross-entropy on shifted target tokens; trainer
# wraps it in label-smoothing if desired. Operates on the *same* vocab as
# CTC so beam-search can later combine CTC + AED log-probs at decode time.


class AEDHead(nn.Module):
    """Causal transformer decoder over the CTC vocabulary.

    Inputs at training time:
      - memory: (B, T_seq, hidden) — the post-encoder activations from the
        CTC head's transformer stack.
      - target_ids: (B, L) — target token sequence including BOS at position 0
        (so the decoder predicts target_ids[:, 1:]).
    Returns:
      - logits: (B, L, vocab_size).
    """

    def __init__(
        self,
        *,
        d_in: int,
        vocab_size: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_target_len: int = 96,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_in)
        self.pos = nn.Embedding(max_target_len, d_in)
        layer = nn.TransformerDecoderLayer(
            d_model=d_in,
            nhead=n_heads,
            dim_feedforward=4 * d_in,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_in, vocab_size)
        self.max_target_len = max_target_len

    def forward(self, memory: torch.Tensor, target_ids: torch.LongTensor) -> torch.Tensor:
        B, L = target_ids.shape
        pos = torch.arange(L, device=target_ids.device).unsqueeze(0).expand(B, -1)
        tgt = self.embed(target_ids) + self.pos(pos)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"),
                                            device=target_ids.device), diagonal=1)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        return self.head(out)

    @torch.no_grad()
    def greedy_generate(self, memory: torch.Tensor, *, max_len: int,
                        bos_id: int, eos_id: int) -> torch.LongTensor:
        B = memory.size(0)
        device = memory.device
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            out = self.forward(memory, ids)
            nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
            nxt = torch.where(finished.unsqueeze(-1), torch.full_like(nxt, eos_id), nxt)
            ids = torch.cat([ids, nxt], dim=1)
            finished = finished | (nxt.squeeze(-1) == eos_id)
            if bool(finished.all()):
                break
        return ids
