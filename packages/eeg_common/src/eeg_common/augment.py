"""GPU-side signal augmentations for EEG training.

Each function takes a ``(B, C, T)`` torch tensor and returns one of the same
shape. They're designed to compose: the trainer can call several in sequence
behind a single ``--signal-aug`` umbrella flag.

The dataset already applies a per-row, row-deterministic SpecAugment
(time + channel masking) at load time (:func:`eeg_common.preprocessing.specaugment`,
numpy-side). The augmentations here run **on the GPU per training step** and
use a step-dependent random seed so the two views in CR-CTC actually differ.

Validated for EEG by:

* **Time shifts** — Brain Transformer (`Sci. Reports 2025
  <https://www.nature.com/articles/s41598-025-86294-3>`_): the *single most
  effective* augmentation in their study; only one that beat chance on
  BCI-IV-2a.
* **Channel dropout** — Strumiłło 2026 (`MDPI Sensors
  <https://www.mdpi.com/1424-8220/26/4/1258>`_): "Channels Dropout" with
  10% of channels at p=0.5, consistently in the top-3 single augmentations.
* **Time warp** — Xu 2026 (`MDPI Sensors
  <https://www.mdpi.com/1424-8220/26/2/399>`_): segment-stretch / telescope
  in the time dimension, part of the "selective augmentation" winning combo.
* **Fourier surrogate** — Strumiłło 2026: phase-randomisation preserving
  amplitude spectrum; one of the most reliable single methods.
* **Mixup** — Alwasiti & Yusoff (CNN motor imagery), Chen et al. (frequency-
  domain), Zoumpourlis & Patras (CovMix) — well-studied for EEG
  classification, less so for CTC sequence loss; we provide a
  feature-space mixup that mixes two encoder outputs and combines the two
  CTC losses with the same lambda.
* **Time-domain concatenation of variants** — Guhdar et al. 2025
  (`arXiv 2507.12645 <https://www.arxiv.org/pdf/2507.12645>`_): novel,
  presents original + augmented in the same training example; provides
  much stronger regularisation than per-batch random selection.

NOT included (deliberately): GAN-based generation (overkill, requires
training a GAN), DWT-based augmentation (we don't have a wavelet pipeline
in eeg_common), VAE-based augmentation (`arXiv 2501.04359`; nice idea, but
needs a separately-trained VAE — out of scope for this iteration).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignalAugmentConfig:
    """Per-step signal augmentation knobs. All default OFF; the trainer enables
    each independently via ``--signal-aug-*`` flags. The recommended preset
    for EEG-to-text is ``time_shift + channel_dropout + freq_mask`` (a
    compositionally-validated triplet from the literature).
    """

    # Time-shift: random circular shift of the time axis, up to ±N samples.
    # Set to 0 to disable. Recommended: 0.05 * T_seq (5% of sequence length).
    time_shift_max_frac: float = 0.0

    # Channel dropout: with probability ``channel_dropout_p``, zero out
    # ``channel_dropout_frac`` of channels (Strumiłło 2026 default: p=0.5,
    # frac=0.1).
    channel_dropout_p: float = 0.0
    channel_dropout_frac: float = 0.1

    # Frequency masking: with probability ``freq_mask_p``, zero out ``n_masks``
    # contiguous frequency bands of width up to ``freq_mask_max_hz`` Hz each
    # (FFT-based; only applied when sr is set on the input).
    freq_mask_p: float = 0.0
    freq_mask_n: int = 2
    freq_mask_max_hz: float = 8.0

    # Time warp: with probability ``time_warp_p``, divide the signal into
    # ``time_warp_segments`` equal-length segments, randomly stretch half of
    # them and telescope the other half (Xu 2026 defaults: stretch 1.5-2x,
    # telescope 0.5-0.67x).
    time_warp_p: float = 0.0
    time_warp_segments: int = 10
    time_warp_factor_low: float = 0.6
    time_warp_factor_high: float = 1.7

    # Gaussian additive noise on the post-z-score signal.
    gaussian_noise_sigma: float = 0.0

    # Fourier surrogate: with probability ``fourier_surrogate_p``, randomise
    # phases while preserving the per-channel amplitude spectrum.
    fourier_surrogate_p: float = 0.0

    # Feature-space mixup (handled in the trainer, not here, because it
    # requires the post-encoder features and the matching loss combination).
    mixup_alpha: float = 0.0


# ============================================================================
# Per-augmentation functions
# ============================================================================


def time_shift(eeg: torch.Tensor, max_frac: float, *,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """Random circular shift of the time axis by ±max_frac * T samples.

    Distinct shift per batch element. ``max_frac=0`` is a no-op.
    """
    if max_frac <= 0:
        return eeg
    B, C, T = eeg.shape
    max_shift = max(1, int(T * max_frac))
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), generator=generator,
                           device=generator.device if generator else eeg.device)
    out = torch.empty_like(eeg)
    for i in range(B):
        out[i] = torch.roll(eeg[i], shifts=int(shifts[i].item()), dims=-1)
    return out


def channel_dropout(eeg: torch.Tensor, *, p: float, frac: float,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """With probability ``p``, zero out ``frac`` of channels per batch element.

    ``frac=0.1, p=0.5`` is the Strumiłło 2026 default.
    """
    if p <= 0 or frac <= 0:
        return eeg
    B, C, T = eeg.shape
    n_drop = max(1, int(C * frac))
    coin = torch.rand(B, generator=generator, device=eeg.device)
    out = eeg.clone()
    for i in range(B):
        if coin[i].item() >= p:
            continue
        idx = torch.randperm(C, generator=generator, device=eeg.device)[:n_drop]
        out[i, idx, :] = 0.0
    return out


def gaussian_noise(eeg: torch.Tensor, sigma: float,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Add per-element Gaussian noise. ``sigma=0`` is a no-op.

    For z-scored inputs (V2 preprocessing, std≈1) sigma=0.05–0.1 is mild;
    sigma=0.3 is aggressive.
    """
    if sigma <= 0:
        return eeg
    return eeg + torch.randn(eeg.shape, generator=generator,
                             device=eeg.device, dtype=eeg.dtype) * sigma


def fourier_surrogate(eeg: torch.Tensor, *, p: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Phase-randomise the per-channel FFT, preserve amplitude.

    The result has identical power spectrum to the input but randomised
    temporal structure. With probability ``p`` per batch element.

    Strumiłło 2026 lists this as one of the top-3 robust single
    augmentations for EEG classification.
    """
    if p <= 0:
        return eeg
    B, C, T = eeg.shape
    coin = torch.rand(B, generator=generator, device=eeg.device)
    if not torch.any(coin < p):
        return eeg

    out = eeg.clone()
    # rfft along time axis -> (B, C, T_freq) complex
    spec = torch.fft.rfft(out, dim=-1)
    n_freq = spec.shape[-1]
    # Randomise phase only for selected batch elements.
    rand_phase = torch.rand(B, C, n_freq - 1, generator=generator,
                            device=eeg.device, dtype=eeg.dtype) * 2 * torch.pi
    new_spec = spec.clone()
    # Keep DC bin (index 0) intact; only randomise positive-frequency phases.
    amp = spec[..., 1:].abs()
    new_phase = torch.complex(torch.cos(rand_phase), torch.sin(rand_phase))
    new_spec[..., 1:] = amp * new_phase

    selected = (coin < p).view(B, 1, 1).expand_as(spec)
    spec = torch.where(selected, new_spec, spec)
    out = torch.fft.irfft(spec, n=T, dim=-1)
    return out


def freq_mask(eeg: torch.Tensor, sr: float, *, p: float, n_masks: int,
              max_hz: float, generator: torch.Generator | None = None
              ) -> torch.Tensor:
    """FFT-based frequency band masking. ``sr`` is the sampling rate of ``eeg``.

    With probability ``p`` per batch element, zero out ``n_masks`` random
    frequency bands of width up to ``max_hz`` each.
    """
    if p <= 0 or n_masks <= 0 or max_hz <= 0:
        return eeg
    B, C, T = eeg.shape
    coin = torch.rand(B, generator=generator, device=eeg.device)
    if not torch.any(coin < p):
        return eeg

    spec = torch.fft.rfft(eeg, dim=-1)            # (B, C, T_freq)
    freq_bins = torch.linspace(0, sr / 2, spec.shape[-1], device=eeg.device)
    # Convert max_hz to bin width
    bin_width = float(sr / 2 / spec.shape[-1])
    max_bins = max(1, int(round(max_hz / bin_width)))

    out_spec = spec.clone()
    for i in range(B):
        if coin[i].item() >= p:
            continue
        for _ in range(n_masks):
            w = int(torch.randint(1, max_bins + 1, (1,),
                                  generator=generator, device=eeg.device).item())
            start = int(torch.randint(0, max(1, spec.shape[-1] - w), (1,),
                                       generator=generator, device=eeg.device).item())
            out_spec[i, :, start:start + w] = 0.0
    return torch.fft.irfft(out_spec, n=T, dim=-1)


def time_warp(eeg: torch.Tensor, *, p: float, segments: int = 10,
              factor_low: float = 0.6, factor_high: float = 1.7,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Per-segment time warping: divide time into ``segments`` equal pieces,
    randomly stretch / telescope each, then resample back to T.

    With probability ``p`` per batch element. Uses 1-D linear interpolation
    so it composes with any input shape.
    """
    if p <= 0 or segments <= 1:
        return eeg
    import torch.nn.functional as F

    B, C, T = eeg.shape
    coin = torch.rand(B, generator=generator, device=eeg.device)
    if not torch.any(coin < p):
        return eeg

    out = eeg.clone()
    seg_len = T // segments
    if seg_len < 2:
        return eeg
    for i in range(B):
        if coin[i].item() >= p:
            continue
        # Per-segment factor: randomly in [factor_low, factor_high]
        factors = (factor_low
                   + (factor_high - factor_low) * torch.rand(segments, generator=generator,
                                                              device=eeg.device))
        warped_pieces: list[torch.Tensor] = []
        for s in range(segments):
            start = s * seg_len
            end = min(T, (s + 1) * seg_len) if s < segments - 1 else T
            piece = eeg[i, :, start:end]               # (C, seg_T)
            new_seg_T = max(1, int(round(piece.shape[-1] * factors[s].item())))
            piece4 = piece.unsqueeze(0)                # (1, C, seg_T)
            piece4 = F.interpolate(piece4, size=new_seg_T, mode="linear",
                                    align_corners=False)
            warped_pieces.append(piece4.squeeze(0))
        warped = torch.cat(warped_pieces, dim=-1)      # (C, sum_T)
        # Resample back to original T so downstream tensor shapes are stable.
        warped = F.interpolate(warped.unsqueeze(0), size=T, mode="linear",
                               align_corners=False).squeeze(0)
        out[i] = warped
    return out


# ============================================================================
# Composite pipeline
# ============================================================================


def apply(eeg: torch.Tensor, sr: float, cfg: SignalAugmentConfig, *,
          generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply each enabled augmentation in a fixed order.

    Order matters: time_shift → time_warp → freq_mask → fourier_surrogate
    → channel_dropout → gaussian_noise. Earlier steps are coarser, later
    steps add finer-grained perturbation.
    """
    out = eeg
    if cfg.time_shift_max_frac > 0:
        out = time_shift(out, cfg.time_shift_max_frac, generator=generator)
    if cfg.time_warp_p > 0:
        out = time_warp(out, p=cfg.time_warp_p,
                        segments=cfg.time_warp_segments,
                        factor_low=cfg.time_warp_factor_low,
                        factor_high=cfg.time_warp_factor_high,
                        generator=generator)
    if cfg.freq_mask_p > 0:
        out = freq_mask(out, sr, p=cfg.freq_mask_p,
                        n_masks=cfg.freq_mask_n,
                        max_hz=cfg.freq_mask_max_hz,
                        generator=generator)
    if cfg.fourier_surrogate_p > 0:
        out = fourier_surrogate(out, p=cfg.fourier_surrogate_p,
                                generator=generator)
    if cfg.channel_dropout_p > 0:
        out = channel_dropout(out, p=cfg.channel_dropout_p,
                              frac=cfg.channel_dropout_frac,
                              generator=generator)
    if cfg.gaussian_noise_sigma > 0:
        out = gaussian_noise(out, cfg.gaussian_noise_sigma,
                             generator=generator)
    return out


# ============================================================================
# Feature-space mixup (composed by the trainer; we just provide the mixer)
# ============================================================================


def feature_mixup(features_a: torch.Tensor, alpha: float,
                  generator: torch.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixup over the batch dimension of post-encoder features.

    Returns ``(mixed_features, perm, lam)`` where:
      - ``mixed_features = lam * features_a + (1 - lam) * features_a[perm]``
      - ``perm`` is the permutation index used to pair samples
      - ``lam`` is sampled from Beta(alpha, alpha)

    The trainer is responsible for mixing the loss as
    ``L = lam * ctc_loss(mixed, targets) + (1 - lam) * ctc_loss(mixed, targets[perm])``.
    """
    if alpha <= 0:
        B = features_a.size(0)
        perm = torch.arange(B, device=features_a.device)
        lam = torch.tensor(1.0, device=features_a.device, dtype=features_a.dtype)
        return features_a, perm, lam
    B = features_a.size(0)
    # Sample lam from Beta(alpha, alpha); torch.distributions doesn't support
    # generator easily, fall back to ``rand`` + inverse CDF for alpha=1 (uniform).
    # For alpha=1 (default mixup), Beta(1,1) == Uniform(0,1).
    if abs(alpha - 1.0) < 1e-6:
        lam = torch.rand((), generator=generator, device=features_a.device,
                         dtype=features_a.dtype)
    else:
        # General-case sampling via gamma ratios.
        g_alpha = torch.distributions.Gamma(
            torch.tensor(alpha, device=features_a.device, dtype=features_a.dtype),
            torch.tensor(1.0, device=features_a.device, dtype=features_a.dtype),
        )
        x = g_alpha.sample()
        y = g_alpha.sample()
        lam = x / (x + y)
    # Symmetrise: lam <-> 1-lam isn't important, but bias toward larger lam
    # so the original sample dominates (standard mixup).
    lam = torch.maximum(lam, 1.0 - lam)
    perm = torch.randperm(B, generator=generator, device=features_a.device)
    mixed = lam * features_a + (1.0 - lam) * features_a[perm]
    return mixed, perm, lam
