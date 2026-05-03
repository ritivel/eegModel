"""Loss functions for exp03 SSL pretraining.

Each loss takes the dict returned by `EEGSSLModel.forward(x)` (see
`model.py`) and returns a scalar tensor + a dict of components for
logging. This contract lets the trainer / sanity-check code treat all
losses uniformly (Check A measures all of them on the same batch).

Implemented (concrete):
    - L1RawLoss:               L1 on raw signal at masked positions only
    - L2RawLoss:               L2 on raw signal at masked positions only
    - MRSTFTLogMagLoss:        multi-resolution STFT log-magnitude L1
    - L1PlusMRSTFTLoss:        the §4.2 default composite (L1 + 0.3·MR-STFT)
    - InfoNCELoss:             contrastive baseline (used in Check A only)

Stubbed (NotImplementedError until the relevant mini-experiment):
    - FSQCELoss:               cross-entropy on FSQ-quantised codec tokens (exp18)
    - JEPALatentLoss:          MSE on EMA-target encoder features (exp18)
    - HuBERTKMeansLoss:        cross-entropy on iterative k-means clusters (exp18)

Theoretical loss-at-init values (Check A pass criterion: within 20%):

    | Loss               | Expected at init                        |
    |--------------------|-----------------------------------------|
    | L2 raw, masked     | ≈ Var(target) ≈ 1.0  (z-scored input)   |
    | L1 raw, masked     | ≈ E|N(0,1)| = √(2/π) ≈ 0.7979           |
    | MR-STFT log-mag    | ≈ 1–3, scale-dependent — measured pre-train and used as the reference |
    | FSQ masked CE      | ≈ log(36000) ≈ 10.49                    |
    | InfoNCE            | ≈ log(B) for batch size B               |

Note: the §4.2-default Mamba-2 + MAE composite loss is `L1PlusMRSTFTLoss`
(L1 raw + 0.3 × MR-STFT log-magnitude); per the BioCodec recipe and exp08
denoising-target ablation. Check A measures every loss above on the same
fresh batch so we can compare measured vs. theoretical, *not* tune any
of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Common interface
# =============================================================================
#
# A loss is `Callable[[ModelOutput], (torch.Tensor scalar, dict[str, float] components)]`
# where ModelOutput is the dict returned by EEGSSLModel.forward.
# We don't enforce this via a Protocol — keeping it duck-typed keeps the
# Check A code (which iterates over a list of arbitrary losses) trivial.


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean of x over positions where mask=1, summed over all batch+sample dims."""
    # x: (B, T_samples), mask: (B, T_samples) with 0/1
    return (x * mask).sum() / (mask.sum() + eps)


# =============================================================================
# Raw-signal reconstruction losses
# =============================================================================


class L1RawLoss(nn.Module):
    """L1 reconstruction loss on raw signal at masked positions only.

    This matches the MAE 2022 recipe: reconstruction loss is computed only
    over the masked tokens (otherwise the model trivially copies the
    visible tokens through to the output).

    Theoretical loss-at-init for a z-scored target (channel std=1) and a
    randomly-initialised model whose output is approximately N(0, σ_recon)
    with σ_recon roughly matching σ_target after a random Linear's variance
    propagation: E|target - recon| ≈ √(2/π) × √(σ_t² + σ_r²) ≈ √(2/π) × √2
    ≈ 1.1284. In the limit where the model output is near zero (very
    small recon-head init), it reduces to E|target| = √(2/π) ≈ 0.7979.
    The Check A pass band of ±20% should contain both regimes.
    """

    def __init__(self, only_masked: bool = True):
        super().__init__()
        self.only_masked = only_masked

    def forward(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        recon = out["reconstruction"]
        target = out["target"]
        diff = (recon - target).abs()
        if self.only_masked:
            loss = _masked_mean(diff, out["mask"])
        else:
            loss = diff.mean()
        return loss, {"l1_raw": loss.item()}


class L2RawLoss(nn.Module):
    """L2 (MSE) reconstruction loss on raw signal at masked positions only.

    Theoretical loss-at-init: for z-scored target (Var=1) and ~zero model
    output, E[(target - 0)²] = Var(target) = 1.0. With a non-zero model
    output of variance σ_r² the value is 1.0 + σ_r², so the Check A pass
    band of [0.8, 1.2] is achievable only if the recon-head init is
    sufficiently small (default Linear init in PyTorch produces
    σ_r ≈ √(1/d_model) so σ_r² ≈ 1/256 ≈ 0.004; well within band).
    """

    def __init__(self, only_masked: bool = True):
        super().__init__()
        self.only_masked = only_masked

    def forward(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        recon = out["reconstruction"]
        target = out["target"]
        sq = (recon - target).pow(2)
        if self.only_masked:
            loss = _masked_mean(sq, out["mask"])
        else:
            loss = sq.mean()
        return loss, {"l2_raw": loss.item()}


# =============================================================================
# Multi-resolution STFT log-magnitude loss
# =============================================================================


class MRSTFTLogMagLoss(nn.Module):
    """Multi-resolution STFT log-magnitude L1.

    Three different STFT resolutions; for each, compute the log-magnitude
    spectrogram of (recon, target) and the L1 distance between them. This
    captures spectral content the L1-on-raw loss can't penalise (a
    reconstruction can match the average waveform shape but completely
    miss the band content). Standard recipe in BioCodec (arXiv 2510.09095)
    and in audio codecs more broadly.

    Default resolutions chosen for 500 Hz EEG:
        (n_fft, hop, win) = (256, 64, 256), (128, 32, 128), (64, 16, 64)
    These cover spectral resolution from ~2 Hz (n_fft=256 → ≈ 2 Hz/bin)
    down to ~8 Hz (n_fft=64 → 8 Hz/bin). The two finer-resolution ones
    are what catches the alpha/beta distinction (8–30 Hz).

    Implementation uses `torchaudio.transforms.Spectrogram`, which is a
    nn.Module so it serializes correctly.

    Theoretical loss-at-init: scale-dependent — we measure on a held-out
    batch first and use *that* as the reference. There is no closed-form
    target for it. Check A records the value but does not gate on a
    specific number.
    """

    DEFAULT_RESOLUTIONS = (
        (256, 64, 256),
        (128, 32, 128),
        (64, 16, 64),
    )

    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_RESOLUTIONS,
        only_masked: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        # Lazy-import torchaudio so this module's import doesn't fail on
        # pure-CPU dev boxes that don't have it installed.
        import torchaudio.transforms as TAT

        self.specs = nn.ModuleList([
            TAT.Spectrogram(
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                power=1.0,                      # magnitude (not power)
                center=True,
                pad_mode="reflect",
            )
            for n_fft, hop, win in resolutions
        ])
        self.resolutions = resolutions
        self.only_masked = only_masked
        self.eps = eps

    def _per_resolution_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        spec: nn.Module,
        sample_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # recon, target: (B, T_samples)
        # spec(...) -> (B, F, T_frames) magnitude
        mag_r = spec(recon)
        mag_t = spec(target)
        log_r = torch.log(mag_r + self.eps)
        log_t = torch.log(mag_t + self.eps)
        diff = (log_r - log_t).abs()       # (B, F, T_frames)

        if sample_mask is None or not self.only_masked:
            return diff.mean()

        # Down-project the sample-mask (B, T_samples) to a frame-mask
        # (B, T_frames) via average-pooling at the same hop. Frames that
        # contain *any* masked sample contribute fully; this is a coarse
        # but standard approximation (avoids per-frame zero-masking which
        # is more expensive and gives nearly identical numbers).
        n_fft = spec.n_fft
        hop = spec.hop_length
        # F.avg_pool1d expects (B, C, T); add and remove the channel dim.
        fm = F.avg_pool1d(
            sample_mask.unsqueeze(1).float(),
            kernel_size=n_fft,
            stride=hop,
            padding=n_fft // 2,
        ).squeeze(1)
        # frame is "masked" if any sample under it is masked → avg > 0
        frame_mask = (fm > 0).float()
        # Align frame count: torchaudio's center-padded spectrogram has
        # ceil(T_samples / hop) frames; our pooled mask should match.
        T_frames = diff.size(-1)
        if frame_mask.size(-1) > T_frames:
            frame_mask = frame_mask[..., :T_frames]
        elif frame_mask.size(-1) < T_frames:
            frame_mask = F.pad(frame_mask, (0, T_frames - frame_mask.size(-1)))
        # broadcast to (B, F, T_frames)
        loss = (diff * frame_mask.unsqueeze(1)).sum() / (
            (frame_mask.unsqueeze(1) * torch.ones_like(diff[:, :1])).sum() + 1e-8
        )
        return loss

    def forward(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        recon = out["reconstruction"]
        target = out["target"]
        sample_mask = out.get("mask") if self.only_masked else None
        per = []
        for spec in self.specs:
            per.append(self._per_resolution_loss(recon, target, spec, sample_mask))
        loss = torch.stack(per).mean()
        comps = {f"mrstft_res{i}": v.item() for i, v in enumerate(per)}
        comps["mrstft_logmag"] = loss.item()
        return loss, comps


# =============================================================================
# §4.2 default composite loss
# =============================================================================


class L1PlusMRSTFTLoss(nn.Module):
    """The §4.2 default reconstruction loss: L1 on raw + 0.3·MR-STFT log-mag.

    This is the headline SSL objective for Checks B and C, and the Check A
    "primary" composite (Check A also measures the components separately
    against their theoretical values).
    """

    def __init__(self, mrstft_weight: float = 0.3, only_masked: bool = True):
        super().__init__()
        self.l1 = L1RawLoss(only_masked=only_masked)
        self.mrstft = MRSTFTLogMagLoss(only_masked=only_masked)
        self.mrstft_weight = mrstft_weight

    def forward(self, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        l1, comps_l1 = self.l1(out)
        ms, comps_ms = self.mrstft(out)
        loss = l1 + self.mrstft_weight * ms
        comps = {**comps_l1, **comps_ms, "composite": loss.item(),
                 "mrstft_weight": self.mrstft_weight}
        return loss, comps


# =============================================================================
# Contrastive baseline — InfoNCE
# =============================================================================


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive baseline.

    Uses the encoder's mean-pooled features as "anchor" and "positive" via
    two independently-masked forward passes. Negatives are the other
    samples in the batch. This is *not* the §4.2 default loss (we use MAE
    reconstruction); it's measured by Check A so we have the InfoNCE
    loss-at-init number ≈ log(B) on file when exp04's framework comparison
    runs.

    Note: requires two forward passes per batch when used as a training
    objective. Check A only measures the loss-at-init on a single batch
    where we use the encoder twice with different masks.

    Theoretical loss-at-init: log(B). For B=64, log(64) ≈ 4.158.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def compute_from_features(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # z_a, z_b: (B, D) — anchor and positive features for each batch elem.
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = z_a @ z_b.T / self.temperature        # (B, B)
        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss = F.cross_entropy(logits, labels)
        return loss, {"infonce": loss.item(), "infonce_log_B": math.log(z_a.size(0))}

    def forward(
        self,
        out_a: dict[str, torch.Tensor],
        out_b: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Pull mean-pooled encoder features. For Check A we accept that
        # out_b may be missing (use the same out twice — gives ~log(B) by
        # construction since z_a == z_b).
        z_a = out_a["encoder_features"].mean(dim=1)
        z_b = (out_b or out_a)["encoder_features"].mean(dim=1)
        return self.compute_from_features(z_a, z_b)


# =============================================================================
# Stubs for losses owned by later mini-experiments
# =============================================================================


class _NotImplementedLoss(nn.Module):
    def __init__(self, name: str, mini_exp: str):
        super().__init__()
        self.name = name
        self.mini_exp = mini_exp

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{self.mini_exp}/README.md"
        )


def FSQCELoss() -> nn.Module:
    """Cross-entropy on FSQ-quantised codec tokens (exp18 reconstruction-target).

    Theoretical loss-at-init ≈ log(vocab_size). For 36000 (BioCodec default)
    this is ≈ 10.49.

    Stubbed until exp18 picks the FSQ vocab + the codec to use.
    """
    return _NotImplementedLoss("FSQCELoss", "18_reconstruction_target")


def JEPALatentLoss() -> nn.Module:
    """MSE on EMA-target encoder features (I-JEPA / EEG2Rep style; exp18)."""
    return _NotImplementedLoss("JEPALatentLoss", "18_reconstruction_target")


def HuBERTKMeansLoss() -> nn.Module:
    """Cross-entropy on iterative k-means cluster IDs (HuBERT style; exp18)."""
    return _NotImplementedLoss("HuBERTKMeansLoss", "18_reconstruction_target")


# =============================================================================
# Loss registry
# =============================================================================


LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "l1_raw":                lambda: L1RawLoss(),
    "l2_raw":                lambda: L2RawLoss(),
    "mrstft_logmag":         lambda: MRSTFTLogMagLoss(),
    "l1_plus_mrstft":        lambda: L1PlusMRSTFTLoss(),
    "infonce":               lambda: InfoNCELoss(),
    "fsq_ce":                FSQCELoss,
    "jepa_latent":           JEPALatentLoss,
    "hubert_kmeans":         HuBERTKMeansLoss,
}


def build_loss(kind: str) -> nn.Module:
    if kind not in LOSS_REGISTRY:
        raise ValueError(
            f"unknown loss kind {kind!r}; valid: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[kind]()
