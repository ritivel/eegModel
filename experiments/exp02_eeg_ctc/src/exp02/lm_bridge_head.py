"""LM-bridge CTC head — pretrained DistilBERT as the bridge transformer.

The findings from wave-1 (May 1) make the case that the standard
randomly-initialised TransformerEncoder head can't learn English token
transitions, word boundaries, and syntax from the 1,237 unique training
sentences ZuCo provides. This module replaces that head with a pretrained
``distilbert-base-uncased`` Transformer, so the head starts already knowing
English.

Architecture::

    encoder features (B, T_seq, D)
      -> RMSNorm -> Linear(D -> 768)            # project EEG features into BERT's hidden dim
      -> + sinusoidal_pos_encoding(T_seq, 768)  # fresh positional info (BERT's positions are token-id-tied)
      -> DistilBERT.transformer (6 BERT layers, full fine-tune)
      -> Linear(768, vocab_size)                # CTC logits

The DistilBERT word embeddings (vocab × 768) are **deleted** — we feed
continuous features, not token IDs. The DistilBERT position embeddings
(0..511 × 768) are also discarded; we use a longer-capacity sinusoidal
encoding because EEG sequence lengths can exceed 512.

References:
  - DistilBERT (Sanh et al. 2019): https://arxiv.org/abs/1910.01108
  - Wav2Vec2 + BERT-init for ASR ([Tsai 2024](https://arxiv.org/abs/2401.16985)):
    initialising a CTC head's transformer from a pretrained text Transformer
    transfers English priors that take 100M+ training tokens to learn from
    scratch.

Memory: DistilBERT is 66M params, hidden=768, 6 layers, FFN-dim 3072.
With REVE (~70M) + this bridge + a moderately-sized AED branch we comfortably
fit on a single H100-80GB at batch 16.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .head import HeadOutput, AEDHead


class LMBridgeHead(nn.Module):
    """CTC head whose transformer block is a pretrained ``distilbert-base-uncased``.

    Args:
        d_in: encoder feature dim (e.g. REVE's 512).
        vocab_size: includes BLANK + UNK + real tokens.
        model_id: HuggingFace model id of the pretrained Transformer to use as
            the bridge. Default ``distilbert-base-uncased`` (66M params,
            6 layers, hidden=768).
        dropout: dropout applied to the input projection.
        max_seq_len: capacity of the learnable position encoding. Default
            2048 — covers REVE's typical post-encoder sequence length even at
            12-second 200-Hz inputs.
        intermediate_layers: 0-based indices (within ``range(n_bert_layers)``)
            at which to expose intermediate predictions (used by ``intctc``).
        attach_aed: if True, attach an :class:`AEDHead` branch.
        aed_layers / aed_heads / aed_dropout / aed_max_target_len: AED knobs.
        cache_dir: HF cache dir (set in trainer wiring; default None).
    """

    def __init__(
        self,
        *,
        d_in: int,
        vocab_size: int,
        model_id: str = "distilbert-base-uncased",
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        intermediate_layers: tuple[int, ...] = (),
        attach_aed: bool = False,
        aed_layers: int = 4,
        aed_heads: int = 8,
        aed_dropout: float = 0.1,
        aed_max_target_len: int = 96,
        cache_dir: str | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.vocab_size = vocab_size
        self.model_id = model_id
        self.intermediate_layers = tuple(sorted(set(intermediate_layers)))

        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        bridge = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)

        # The hidden size BERT/DistilBERT use depends on the model; pull it
        # from the config so we work for any DistilBERT/BERT variant.
        self.hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "dim", 768)))

        # Bridge transformer stack. We delete the embedding/pooler heads
        # because we feed continuous features, not token IDs.
        # DistilBERT layout: ``bridge.embeddings`` + ``bridge.transformer``
        # BERT layout:        ``bridge.embeddings`` + ``bridge.encoder`` + ``bridge.pooler``
        if hasattr(bridge, "transformer"):
            self.transformer = bridge.transformer
            self._bridge_kind = "distilbert"
        elif hasattr(bridge, "encoder"):
            self.transformer = bridge.encoder
            self._bridge_kind = "bert"
        else:
            raise ValueError(f"Unsupported bridge model layout: {type(bridge).__name__}")

        # Drop the embeddings/pooler — neither is useful when feeding
        # projected EEG features. We keep the registered submodules detached
        # by NOT assigning them, so they won't be included in
        # self.parameters().
        del bridge

        # Input projection: EEG feature_dim -> bridge hidden_dim, plus a
        # LayerNorm to keep activations on the scale BERT expects.
        self.norm_in = nn.RMSNorm(d_in)
        self.proj_in = nn.Linear(d_in, self.hidden, bias=False)
        self.proj_dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding (no learnable parameters; supports
        # any sequence length up to ``max_seq_len``).
        self.register_buffer(
            "pos_enc",
            _sinusoidal_position_encoding(max_seq_len, self.hidden),
            persistent=False,
        )
        self.max_seq_len = max_seq_len

        # CTC logits projection.
        self.head = nn.Linear(self.hidden, vocab_size)

        # Optional AED branch (used by ``ctcaed`` cells).
        self.aed: AEDHead | None = None
        if attach_aed:
            self.aed = AEDHead(
                d_in=self.hidden,
                vocab_size=vocab_size,
                n_layers=aed_layers,
                n_heads=aed_heads,
                dropout=aed_dropout,
                max_target_len=aed_max_target_len,
            )

        # ---- Diagnostics ----
        n_bridge = sum(p.numel() for p in self.transformer.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[LMBridgeHead] {model_id} ({self._bridge_kind}, hidden={self.hidden}); "
              f"bridge={n_bridge/1e6:.1f}M params; total head={n_total/1e6:.1f}M params; "
              f"vocab_size={vocab_size}", flush=True)

    def _bridge_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the bridge transformer, optionally collecting per-layer outputs.

        Returns ``(final_hidden, intermediates_list)``.
        """
        # We iterate over the bridge's layer list directly rather than calling
        # the wrapper ``Transformer`` / ``BertEncoder`` ``forward`` — those
        # wrappers' kwargs vary across transformers versions (DistilBERT
        # 4.40+ renames ``x`` to ``hidden_states`` and drops most of the
        # output_* knobs into ``**kwargs``). The per-layer call is stable.
        h = x
        layers = self.transformer.layer  # nn.ModuleList of TransformerBlock
        intermediate_set = set(self.intermediate_layers)
        inters: list[torch.Tensor] = []

        if self._bridge_kind == "distilbert":
            # TransformerBlock.forward(hidden_states, attention_mask=None, **kwargs).
            # We don't pad-mask at this level (input is dense EEG features).
            for li, block in enumerate(layers):
                out = block(h)
                # block returns a tuple: (hidden_states,) or (hidden_states, attn).
                h = out[0] if isinstance(out, tuple) else out
                if li in intermediate_set:
                    inters.append(h)
        else:  # bert layout
            # BertLayer.forward(hidden_states, attention_mask=None, head_mask=None,
            # encoder_hidden_states=None, encoder_attention_mask=None,
            # past_key_value=None, output_attentions=False).
            B, T, _ = x.shape
            extended_attn_mask = torch.zeros(
                B, 1, 1, T, device=x.device, dtype=x.dtype)
            for li, block in enumerate(layers):
                out = block(h, attention_mask=extended_attn_mask)
                h = out[0] if isinstance(out, tuple) else out
                if li in intermediate_set:
                    inters.append(h)
        return h, inters

    def forward(
        self,
        features: torch.Tensor,
        *,
        aed_target_ids: torch.LongTensor | None = None,
    ) -> HeadOutput:
        """features: (B, T_seq, D). Returns :class:`HeadOutput`."""
        x = self.proj_dropout(self.proj_in(self.norm_in(features)))

        # Add sinusoidal positional encoding (truncate to T or capacity).
        T = x.size(1)
        if T > self.max_seq_len:
            # Should never happen with REVE @ 200Hz × 12s, but defend anyway.
            x = x[:, : self.max_seq_len]
            T = self.max_seq_len
        x = x + self.pos_enc[:T].unsqueeze(0).to(x.dtype)

        final, inters = self._bridge_forward(x)
        logits = self.head(final)

        intermediate_logits = [self.head(h) for h in inters]

        aed_logits = None
        if self.aed is not None and aed_target_ids is not None:
            aed_logits = self.aed(memory=final, target_ids=aed_target_ids)

        return HeadOutput(
            logits=logits,
            intermediate_logits=intermediate_logits,
            aed_logits=aed_logits,
        )


def _sinusoidal_position_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Standard ``Attention is all you need`` sinusoidal encoding."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                         * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
