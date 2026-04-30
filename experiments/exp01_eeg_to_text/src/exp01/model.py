"""End-to-end EEG2Text wrapper. Composes encoder + bridge + decoder.

Three forward paths:
  * Soft-prompt (linear / qformer): encoder -> bridge -> Gemma `inputs_embeds`
    via an embed-layer hook (PLE-safe). Trains LM cross-entropy on the text.
  * Vocab (vocab / RVQ + vocab): encoder -> discrete ids -> Gemma input_ids
    using an extended embedding table.
  * **CTC** (ctc): encoder -> CTC head over a small character vocabulary.
    No decoder LM in the loop during training. Trains ``F.ctc_loss``;
    decode at eval time with greedy CTC. This is the ASR-style track —
    by construction it can't lapse onto an LM prior because there is no
    LM. See chars.py and bridges.CTCBridge for details.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import bridges, chars, decoder, encoders
from .config import CellConfig


class EEG2Text(nn.Module):
    def __init__(self, cfg: CellConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder. Frozen for stages 1+2; we never unfreeze it in this design.
        self.encoder = encoders.load_encoder(cfg.encoder)
        decoder.freeze(self.encoder)

        # Decoder loading is conditional. CTC cells don't need an LM at all,
        # so we save ~5 GB of weights + the ~2 s tokenizer load.
        self.is_ctc = cfg.bridge == "ctc"
        self.dec = None
        if not self.is_ctc:
            self.dec = decoder.load_decoder(cfg.decoder)
            decoder.freeze(self.dec.model)
            if cfg.use_gradient_checkpointing and hasattr(self.dec.model, "gradient_checkpointing_enable"):
                try:
                    self.dec.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.dec.model.gradient_checkpointing_enable()

        # Vocab extension is required for vocab bridges.
        self.vocab_offset = 0
        if cfg.bridge == "vocab":
            n_new = self._codebook_size()
            self.vocab_offset = decoder.extend_vocab(self.dec, n_new)

        # Optional RVQ for off-diagonal vocab cells (REVE × vocab, DIVER-1 × vocab).
        self.rvq = None
        if cfg.bridge == "vocab" and not self.encoder.spec.discrete:
            self.rvq = bridges.RVQHead(d_in=self.encoder.spec.feature_dim, codebook_size=cfg.rvq_codebook)

        # ``d_lm`` is irrelevant for CTC (no LM); set to 0 so the bridge
        # factory doesn't waste params on a non-existent projection.
        d_lm = self.dec.embed_dim if self.dec is not None else 0
        self.bridge = bridges.build_bridge(
            kind=cfg.bridge,
            d_in=self.encoder.spec.feature_dim,
            d_lm=d_lm,
            n_queries=cfg.qformer_queries,
            codebook_size=self._codebook_size(),
            vocab_offset=self.vocab_offset,
            ctc_vocab_size=chars.VOCAB_SIZE if self.is_ctc else 0,
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
        text_input_ids: torch.LongTensor,   # (B, L)   ignored for CTC
        text_attention_mask: torch.LongTensor,        # ignored for CTC
        labels: torch.LongTensor,                     # ignored for CTC
        text: list[str] | None = None,                # required for CTC
    ):
        # 1) Encoder features.
        with torch.no_grad():
            feats = self.encoder.encode(eeg, sr, channels)   # (B, T_seq, d_in)

        # 2) Bridge / loss path.
        if self.cfg.bridge == "ctc":
            tag, logits = self.bridge(feats)              # (B, T_seq, V)
            self._stash_aux_ctc(logits=logits, text=text)
            return _CTCForwardOutput(logits=logits)

        if self.cfg.bridge == "vocab":
            if self.encoder.spec.discrete:
                ids = self.encoder.tokenize(eeg, sr, channels)
            else:
                ids = self.rvq(feats)
            tag, eeg_ids = self.bridge(ids)
            out = self._forward_with_ids(eeg_ids, text_input_ids, text_attention_mask, labels)
            self._stash_aux(eeg_ids=eeg_ids, eeg_embeds=None,
                            text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask)
            return out

        tag, eeg_embeds = self.bridge(feats)
        out = self._forward_with_embeds(eeg_embeds, text_input_ids, text_attention_mask, labels)
        self._stash_aux(eeg_ids=None, eeg_embeds=eeg_embeds,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask)
        return out

    # ------------------------------------------------------------------
    # Auxiliary tensors for contrastive alignment loss + RVQ commitment.
    # Stashed on ``self._last_aux`` after each forward so the trainer can
    # read them without changing the HF-style forward return type.
    # ``bridge_pooled``: (B, d_lm), sentence-pooled bridge output (in d_lm).
    # ``text_pooled``  : (B, d_lm), sentence-pooled frozen text embeddings.
    # ``commit_loss``  : scalar tensor or 0.0 (only nonzero for off-diagonal
    #                    vocab cells with a continuous encoder + RVQ).
    # ------------------------------------------------------------------

    def _stash_aux(
        self,
        *,
        eeg_ids: torch.LongTensor | None,
        eeg_embeds: torch.Tensor | None,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.LongTensor,
    ) -> None:
        embed_layer = self.dec.model.get_input_embeddings()
        with torch.no_grad():
            text_emb = embed_layer(text_input_ids)                # (B, L, d_lm)
            tmask = text_attention_mask.to(text_emb.dtype).unsqueeze(-1)
            text_pooled = (text_emb * tmask).sum(dim=1) / tmask.sum(dim=1).clamp(min=1)

        if eeg_embeds is not None:
            bridge_pooled = eeg_embeds.to(text_emb.dtype).mean(dim=1)
        else:
            assert eeg_ids is not None
            # NOTE: deliberately NOT inside no_grad — for vocab cells in
            # stages 1+2 the new vocab rows of the embed table are trainable,
            # and we want the alignment loss to push them toward text-token
            # geometry. The backward hook installed in trainables_stage1 zeros
            # gradients on the old (frozen) rows, so only the new EEG-vocab
            # rows actually learn from this signal.
            eeg_emb = embed_layer(eeg_ids)                       # (B, K, d_lm)
            bridge_pooled = eeg_emb.mean(dim=1)

        commit_loss: torch.Tensor | float = 0.0
        if self.rvq is not None and self.rvq.last_commit_loss is not None:
            commit_loss = self.rvq.last_commit_loss

        self._last_aux = {
            "bridge_pooled": bridge_pooled,
            "text_pooled": text_pooled,
            "commit_loss": commit_loss,
        }

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
        if self.cfg.bridge == "ctc":
            tag, logits = self.bridge(feats)                  # (B, T, V)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ids_per_row = chars.ctc_greedy_decode(log_probs)
            return [chars.decode_ids(ids) for ids in ids_per_row]

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
            # During incremental decoding the model only re-embeds the latest
            # token (output.shape[1] == 1) and uses the KV cache for the
            # prefix — the EEG embeds are already baked into the cache, so
            # we only inject on the FIRST forward call (the full prefix).
            if output.shape[1] < K:
                return output
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
          codes to Gemma's hidden space. We mark the whole embed table as
          ``requires_grad=True`` and register a backward hook that zeros out
          gradients for the *original* (frozen) rows — the new rows learn,
          the old rows don't (no AdamW updates because grad is identically 0).
        - Off-diagonal vocab (REVE/DIVER-1 × vocab): also train the RVQ head.
        - **CTC**: just the bridge (Transformer + head). No LM / no RVQ.
          The same param set is used for stages 1, 2, 3 — the 3-stage
          schedule is meaningless for CTC, but we honour the contract so
          the trainer doesn't have to special-case the cell.
        """
        ps = list(self.bridge.parameters())
        if self.rvq is not None:
            ps += list(self.rvq.parameters())
        if self.cfg.bridge == "vocab":
            embed = self.dec.model.get_input_embeddings()
            embed.weight.requires_grad_(True)
            offset = self.vocab_offset
            if not getattr(self, "_grad_mask_installed", False):
                def _mask_old_rows(grad):
                    g = grad.clone()
                    g[:offset] = 0
                    return g
                embed.weight.register_hook(_mask_old_rows)
                self._grad_mask_installed = True
            ps.append(embed.weight)
        return ps

    def trainables_stage2(self):
        """Stage 2 — frozen-LM SFT: same as Stage 1 (bridge / RVQ / new vocab
        rows). The "frozen-LM" name refers to the *pretrained* Gemma weights
        staying frozen; new vocab rows for vocab cells are still trained.
        For CTC: same as Stage 1 (no LM in the loop anyway).
        """
        return self.trainables_stage1()

    def trainables_stage3(self):
        """Stage 3 — LoRA SFT: PEFT-injected LoRA params + bridge.

        For CTC there is no LM and therefore no LoRA — return the same
        bridge params as stages 1/2 so AdamW keeps refining them. We
        rely on ``cfg.use_lora_in_stage3=False`` (set automatically for
        CTC cells in CellConfig) to skip the LoRA-attach step in train.py.
        """
        if self.cfg.bridge == "ctc":
            return self.trainables_stage1()
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # CTC-specific aux stash: keep the latest logits + targets so the
    # trainer can compute F.ctc_loss without changing the public forward
    # signature.
    # ------------------------------------------------------------------

    def _stash_aux_ctc(self, *, logits: torch.Tensor, text: list[str] | None) -> None:
        self._last_aux = {
            "ctc_logits": logits,
            "ctc_text": text,
            "bridge_pooled": None,
            "text_pooled": None,
            "commit_loss": 0.0,
        }


class _CTCForwardOutput:
    """Drop-in replacement for HF's CausalLMOutput in the CTC path.

    The trainer expects ``out.loss`` on the return value, but for CTC
    we compute the loss in the trainer itself (it needs target lengths
    that aren't known here). So this container just carries the logits
    and exposes a sentinel ``loss=None`` — the trainer detects ``None``
    and uses ``F.ctc_loss(...)`` instead.
    """
    __slots__ = ("loss", "logits")

    def __init__(self, logits: torch.Tensor):
        self.loss = None
        self.logits = logits
