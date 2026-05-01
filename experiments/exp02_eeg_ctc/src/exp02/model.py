"""End-to-end EEG2CTC model.

Composes a pretrained EEG encoder (REVE / TFM / DIVER-1) with a :class:`CTCHead`.
Encoder fine-tuning is configurable per cell:

  - ``full``    every encoder param is trainable (Wav2Vec2 standard).
  - ``lora``    PEFT LoRA adapters on encoder attention projections.
  - ``frozen``  encoder permanently frozen (head-only training; GROUP E ablation).

The trainer can additionally hold the encoder frozen for the first
``encoder_warmup_freeze_steps`` steps before flipping to whatever the
``encoder_finetune`` mode dictates.

Public surface:

  - ``encoder_features(eeg, sr, channels) -> (B, T_seq, D)``: forward through
    the encoder only (used by the trainer when running two SpecAugmented
    views for CR-CTC).
  - ``head_forward(features, aed_target_ids=None) -> HeadOutput``: forward
    through the head only.
  - ``forward(eeg, sr, channels, aed_target_ids=None) -> HeadOutput``: end-to-end.

Notes on gradient flow:

  - When ``encoder_finetune == "frozen"``, the encoder is wrapped in
    ``torch.no_grad()`` inside ``encoder_features`` to save activation memory.
  - When ``encoder_finetune == "lora"``, only the LoRA adapters get gradients;
    the base encoder weights stay frozen but participate in the forward graph.
  - When ``encoder_finetune == "full"``, every encoder parameter is trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eeg_common.encoders import load_encoder

from . import storage
from .chars import Vocab
from .config import CTCConfig
from .head import CTCHead, HeadOutput
from .lm_bridge_head import LMBridgeHead


class EEG2CTC(nn.Module):
    def __init__(self, cfg: CTCConfig, vocab: Vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab

        self.encoder = load_encoder(cfg.encoder, storage.STORAGE)

        # Apply the requested encoder fine-tune mode at construction time.
        # The trainer can additionally freeze for the first warmup-freeze
        # steps via ``set_encoder_trainable(False)``; after warmup it calls
        # ``set_encoder_trainable(True)`` and the underlying parameter
        # requires_grad flags decide which params actually move.
        self._configure_encoder_trainability()

        head_kwargs = dict(
            d_in=self.encoder.spec.feature_dim,
            vocab_size=vocab.size,
            dropout=cfg.head_dropout,
            intermediate_layers=cfg.intermediate_ctc_layers if cfg.variant == "intctc" else (),
            attach_aed=(cfg.variant == "ctcaed"),
            aed_layers=cfg.aed_layers,
            aed_heads=cfg.aed_heads,
            aed_dropout=cfg.aed_dropout,
            aed_max_target_len=cfg.aed_max_target_len,
        )
        if cfg.head_type == "lm_bridge":
            self.head = LMBridgeHead(
                **head_kwargs,
                model_id=cfg.head_lm_model_id,
                max_seq_len=cfg.head_lm_max_seq_len,
                cache_dir=str(storage.HF_CACHE),
            )
        else:
            self.head = CTCHead(
                **head_kwargs,
                hidden=cfg.head_hidden,
                n_layers=cfg.head_layers,
                n_heads=cfg.head_heads,
            )

        # Trainer toggles this to gate the encoder during the warmup-freeze
        # window. Distinct from the per-parameter requires_grad flags so we
        # can switch back and forth without losing the underlying mode.
        self._encoder_active = (cfg.encoder_finetune != "frozen")

    # ------------------------------------------------------------------
    # Encoder gating
    # ------------------------------------------------------------------

    def _configure_encoder_trainability(self) -> None:
        cfg = self.cfg
        if cfg.encoder_finetune == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif cfg.encoder_finetune == "lora":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self._lora_params = self.encoder.attach_lora(
                r=cfg.encoder_lora_r,
                alpha=cfg.encoder_lora_alpha,
                dropout=cfg.encoder_lora_dropout,
            )
        elif cfg.encoder_finetune == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown encoder_finetune: {cfg.encoder_finetune}")

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Trainer hook for the warmup-freeze window. ``trainable=False`` runs
        the encoder under ``torch.no_grad()``; ``trainable=True`` re-enables
        gradient flow according to the ``encoder_finetune`` mode.
        """
        if self.cfg.encoder_finetune == "frozen":
            self._encoder_active = False
            return
        self._encoder_active = trainable

    def encoder_trainable_parameters(self) -> list[nn.Parameter]:
        """The set of encoder parameters that the optimizer should hold.

        - ``frozen``: empty list.
        - ``lora``: the LoRA adapter params.
        - ``full``: every encoder parameter.
        """
        if self.cfg.encoder_finetune == "frozen":
            return []
        if self.cfg.encoder_finetune == "lora":
            return list(getattr(self, "_lora_params", []))
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.head.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encoder_features(self, eeg: torch.Tensor, sr: float,
                         channels: list[str]) -> torch.Tensor:
        if self._encoder_active:
            return self.encoder.encode(eeg, sr, channels)
        with torch.no_grad():
            return self.encoder.encode(eeg, sr, channels)

    def head_forward(self, features: torch.Tensor, *,
                     aed_target_ids: torch.LongTensor | None = None) -> HeadOutput:
        return self.head(features, aed_target_ids=aed_target_ids)

    def forward(
        self,
        eeg: torch.Tensor,
        sr: float,
        channels: list[str],
        *,
        aed_target_ids: torch.LongTensor | None = None,
    ) -> HeadOutput:
        feats = self.encoder_features(eeg, sr, channels)
        return self.head_forward(feats, aed_target_ids=aed_target_ids)
