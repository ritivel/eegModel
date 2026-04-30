"""End-to-end EEG2Text wrapper. Composes encoder + bridge + decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import bridges, decoder, encoders
from .config import CellConfig


class EEG2Text(nn.Module):
    def __init__(self, cfg: CellConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder. Frozen for stages 1+2; we never unfreeze it in this design.
        self.encoder = encoders.load_encoder(cfg.encoder)
        decoder.freeze(self.encoder)

        # Decoder. Frozen at construction; LoRA applied in stage 3 by the trainer.
        self.dec = decoder.load_decoder(cfg.decoder)
        decoder.freeze(self.dec.model)

        # Vocab extension is required for vocab bridges.
        self.vocab_offset = 0
        if cfg.bridge == "vocab":
            n_new = self._codebook_size()
            self.vocab_offset = decoder.extend_vocab(self.dec, n_new)

        # Optional RVQ for off-diagonal vocab cells (REVE × vocab, DIVER-1 × vocab).
        self.rvq = None
        if cfg.bridge == "vocab" and not self.encoder.spec.discrete:
            self.rvq = bridges.RVQHead(d_in=self.encoder.spec.feature_dim, codebook_size=cfg.rvq_codebook)

        self.bridge = bridges.build_bridge(
            kind=cfg.bridge,
            d_in=self.encoder.spec.feature_dim,
            d_lm=self.dec.embed_dim,
            n_queries=cfg.qformer_queries,
            codebook_size=self._codebook_size(),
            vocab_offset=self.vocab_offset,
        )

    def _codebook_size(self) -> int:
        if self.encoder.spec.discrete:
            return int(self.encoder.spec.codebook_size or 8192)
        return int(self.cfg.rvq_codebook)

    # ------------------------------------------------------------------
    # Forward (training): expects an already-collated batch.
    # ------------------------------------------------------------------

    def forward(
        self,
        eeg: torch.Tensor,         # (B, C, T) float
        sr: float,                 # batch-wise sampling rate
        channels: list[str],       # batch-wise channel names
        text_input_ids: torch.LongTensor,   # (B, L)
        text_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,           # (B, L) with -100 on prompt tokens
    ):
        # 1) Encoder features.
        with torch.no_grad():
            feats = self.encoder.encode(eeg, sr, channels)   # (B, T_seq, d_in)

        # 2) Bridge.
        if self.cfg.bridge == "vocab":
            if self.encoder.spec.discrete:
                ids = self.encoder.tokenize(eeg, sr, channels)
            else:
                ids = self.rvq(feats)
            tag, eeg_ids = self.bridge(ids)
            return self._forward_with_ids(eeg_ids, text_input_ids, text_attention_mask, labels)

        tag, eeg_embeds = self.bridge(feats)
        return self._forward_with_embeds(eeg_embeds, text_input_ids, text_attention_mask, labels)

    # ------------------------------------------------------------------
    # Soft-prompt path (Linear / Q-Former)
    # ------------------------------------------------------------------
    # Gemma 4 (E2B / E4B) uses Per-Layer Embeddings: the model needs
    # ``input_ids`` so it can look up a separate per-layer embedding table.
    # Passing ``inputs_embeds`` directly raises a runtime error if the
    # embeddings don't reverse-map to known token ids. We work around this
    # by passing ``input_ids`` (with a placeholder token for each EEG slot)
    # and registering a forward hook on the embedding layer that overwrites
    # the K placeholder positions with our bridge output. PLE then uses
    # the placeholder's per-layer embedding, which is a learned constant —
    # perfectly fine as a tied "EEG slot" per-layer embedding.

    def _forward_with_embeds(
        self,
        eeg_embeds: torch.Tensor,             # (B, K, d_lm)
        text_input_ids: torch.LongTensor,     # (B, L)
        text_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        B, K, _ = eeg_embeds.shape
        device = text_input_ids.device
        pad_id = self._eeg_placeholder_id()

        prefix_ids = torch.full((B, K), pad_id, dtype=text_input_ids.dtype, device=device)
        full_ids = torch.cat([prefix_ids, text_input_ids], dim=1)
        attn = torch.cat(
            [torch.ones(B, K, dtype=text_attention_mask.dtype, device=device),
             text_attention_mask], dim=1)
        prompt_pad = torch.full((B, K), -100, dtype=labels.dtype, device=device)
        full_labels = torch.cat([prompt_pad, labels], dim=1)

        embed_layer = self.dec.model.get_input_embeddings()
        eeg_cast = eeg_embeds.to(embed_layer.weight.dtype)

        def _hook(_module, _inputs, output):
            output = output.clone()
            output[:, :K, :] = eeg_cast
            return output

        handle = embed_layer.register_forward_hook(_hook)
        try:
            return self.dec.model(input_ids=full_ids, attention_mask=attn, labels=full_labels)
        finally:
            handle.remove()

    def _eeg_placeholder_id(self) -> int:
        tok = self.dec.tokenizer
        for cand in (tok.pad_token_id, tok.unk_token_id, tok.eos_token_id, 0):
            if cand is not None:
                return int(cand)
        return 0

    # ------------------------------------------------------------------
    # Vocab-extension path
    # ------------------------------------------------------------------

    def _forward_with_ids(
        self,
        eeg_ids: torch.LongTensor,            # (B, K) — already offset
        text_input_ids: torch.LongTensor,     # (B, L)
        text_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        full_ids = torch.cat([eeg_ids, text_input_ids], dim=1)
        K = eeg_ids.size(1)
        attn = torch.cat(
            [torch.ones(text_input_ids.size(0), K, dtype=text_attention_mask.dtype, device=text_attention_mask.device),
             text_attention_mask], dim=1)
        prompt_pad = torch.full((labels.size(0), K), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prompt_pad, labels], dim=1)
        return self.dec.model(input_ids=full_ids, attention_mask=attn, labels=full_labels)

    # ------------------------------------------------------------------
    # Generation (eval): no teacher forcing (per Jo et al.).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, eeg, sr, channels, *, max_new_tokens: int = 64) -> list[str]:
        feats = self.encoder.encode(eeg, sr, channels)
        if self.cfg.bridge == "vocab":
            if self.encoder.spec.discrete:
                ids = self.encoder.tokenize(eeg, sr, channels)
            else:
                ids = self.rvq(feats)
            _, eeg_ids = self.bridge(ids)
            out = self.dec.model.generate(
                input_ids=eeg_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            return self.dec.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Soft-prompt path: same hook trick as training so Gemma 4's PLE works.
        _, eeg_embeds = self.bridge(feats)
        B, K, _ = eeg_embeds.shape
        pad_id = self._eeg_placeholder_id()
        prefix_ids = torch.full((B, K), pad_id, dtype=torch.long, device=eeg.device)
        attn = torch.ones(B, K, dtype=torch.long, device=eeg.device)
        embed_layer = self.dec.model.get_input_embeddings()
        eeg_cast = eeg_embeds.to(embed_layer.weight.dtype)

        def _hook(_module, _inputs, output):
            output = output.clone()
            output[:, :K, :] = eeg_cast
            return output

        handle = embed_layer.register_forward_hook(_hook)
        try:
            out = self.dec.model.generate(
                input_ids=prefix_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        finally:
            handle.remove()
        # ``out`` includes the K placeholder tokens at the start; strip them.
        return self.dec.tokenizer.batch_decode(out[:, K:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Trainable-parameter bookkeeping per stage.
    # ------------------------------------------------------------------

    def trainables_stage1(self):
        """Stage 1 — modality alignment.

        - Linear / Q-Former: train the bridge.
        - Vocab: bridge has no params (offset-add only), so train the new
          embedding rows directly. They're the only thing connecting EEG
          codes to Gemma's hidden space.
        - Off-diagonal vocab (REVE/DIVER-1 × vocab): also train the RVQ head.
        """
        ps = list(self.bridge.parameters())
        if self.rvq is not None:
            ps += list(self.rvq.parameters())
        if self.cfg.bridge == "vocab":
            embed = self.dec.model.get_input_embeddings()
            embed.weight.requires_grad_(True)
            ps.append(embed.weight)
        return ps

    def trainables_stage2(self):
        """Stage 2 — frozen-LM SFT: same as Stage 1 (bridge / RVQ / new vocab
        rows). The "frozen-LM" name refers to the *pretrained* Gemma weights
        staying frozen; new vocab rows for vocab cells are still trained.
        """
        return self.trainables_stage1()

    def trainables_stage3(self):
        """Stage 3 — LoRA SFT: PEFT-injected LoRA params + bridge."""
        return [p for p in self.parameters() if p.requires_grad]
