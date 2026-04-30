"""Per-row EEG preprocessing pipelines.

The Apr-30 pilot showed that the §4.3 gap goes the wrong way (noise > eeg)
because we feed REVE / TFM raw, unfiltered, unnormalised EEG — both encoders
were pretrained on data that had been bandpass-filtered, line-noise-notched,
and per-recording z-scored + clipped. With that mismatch, the encoder outputs
essentially-uninformative features and the bridge cannot align them to text in
any number of contrastive steps.

Recipe sources (each verified against the corresponding paper / repo):

  - REVE (NeurIPS 2025, `arXiv 2510.21585 <https://arxiv.org/abs/2510.21585>`_, §3.1.1):
      bandpass 0.5–99.5 Hz, resample 200 Hz, per-recording z-score, ±15-σ clip.
  - TFM-Tokenizer (ICLR 2026, `arXiv 2502.16060 <https://arxiv.org/abs/2502.16060>`_, §B.2 / B.6):
      bandpass 0.1–75 Hz, notch 50 Hz, resample 200 Hz, STFT(n_fft=200, hop=100).
  - Défossez & King 2025 (`Nat. Commun. 16, 10521 <https://www.nature.com/articles/s41467-025-65499-0>`_):
      bandpass 0.1–40 Hz, RobustScaler, clip ±5, resample 50 Hz.

Two design choices that depart from a literal copy of any one paper:

1. We z-score per *row* (sentence) rather than per recording session, because
   the parquet schema doesn't carry session-level groupings. This is a
   reasonable proxy when sentences are tens of seconds long; we re-evaluate
   if it leaves any visible amplitude drift.
2. We expose two named pipelines (``v2_reve`` and ``v2_tfm``) keyed off the
   encoder name so each encoder sees the cutoffs *it* was pretrained on.

Everything in here runs on numpy arrays so it can execute inside DataLoader
worker processes without holding the GIL on the main thread. Filters use
``scipy.signal``; resampling uses polyphase ``resample_poly`` for proper
anti-aliasing (replacing the previous ``F.interpolate(mode="linear")``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================================
# Building blocks
# ============================================================================


def _design_bandpass(lo: float, hi: float, sr: float, order: int = 4):
    """Return a (sos) Butterworth bandpass filter object usable by ``sosfiltfilt``.

    Falls back to a high-pass if ``hi`` is at or above Nyquist (which can
    happen for low-SR sources where 0.4*sr < 75 Hz). Falls back to a low-pass
    if ``lo <= 0`` (degenerate config).
    """
    from scipy import signal

    nyq = 0.5 * sr
    lo_w = lo / nyq if lo > 0 else None
    hi_w = hi / nyq if hi < nyq else None

    if hi_w is not None and hi_w >= 0.99:
        hi_w = 0.95

    if lo_w is None and hi_w is None:
        return None
    if lo_w is None:
        return signal.butter(order, hi_w, btype="lowpass", output="sos")
    if hi_w is None:
        return signal.butter(order, lo_w, btype="highpass", output="sos")
    return signal.butter(order, [lo_w, hi_w], btype="bandpass", output="sos")


def bandpass(eeg: np.ndarray, sr: float, lo: float, hi: float, *, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass on the time axis (last axis)."""
    from scipy import signal

    sos = _design_bandpass(lo, hi, sr, order=order)
    if sos is None:
        return eeg
    padlen = min(3 * (sos.shape[0] * 2), max(0, eeg.shape[-1] - 1))
    if padlen <= 0:
        return eeg
    return signal.sosfiltfilt(sos, eeg, axis=-1, padlen=padlen).astype("float32", copy=False)


def notch(eeg: np.ndarray, sr: float, hz: float = 50.0, *, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch filter to remove power-line interference.

    Skipped if ``2*hz >= sr`` (Nyquist violation; e.g. 50 Hz notch with 100 Hz
    sampling).
    """
    from scipy import signal

    if 2.0 * hz >= sr:
        return eeg
    b, a = signal.iirnotch(w0=hz, Q=q, fs=sr)
    padlen = min(3 * max(len(a), len(b)), max(0, eeg.shape[-1] - 1))
    if padlen <= 0:
        return eeg
    return signal.filtfilt(b, a, eeg, axis=-1, padlen=padlen).astype("float32", copy=False)


def resample_polyphase(eeg: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    """Anti-aliased polyphase resample on the time axis."""
    from math import gcd
    from scipy import signal

    if abs(src_sr - dst_sr) < 0.5:
        return eeg
    a, b = int(round(dst_sr)), int(round(src_sr))
    g = gcd(a, b)
    up, down = a // g, b // g
    return signal.resample_poly(eeg, up=up, down=down, axis=-1).astype("float32", copy=False)


def zscore_per_recording(eeg: np.ndarray, *, eps: float = 1e-6,
                         clip_sigma: float | None = 15.0) -> np.ndarray:
    """Z-score with statistics computed across all (channel, time) samples of
    the row. Optionally clip values exceeding ``±clip_sigma`` standard
    deviations.

    REVE pretraining used statistics across the recording session and a 15-σ
    clip; per-row stats are a defensible proxy when sentences are tens of
    seconds long.
    """
    mu = float(eeg.mean())
    sd = float(eeg.std()) + eps
    out = (eeg - mu) / sd
    if clip_sigma is not None:
        out = np.clip(out, -clip_sigma, clip_sigma)
    return out.astype("float32", copy=False)


def zscore_per_channel(eeg: np.ndarray, *, eps: float = 1e-6,
                       clip_sigma: float | None = 5.0) -> np.ndarray:
    """Per-channel z-score (Brain4FMs / Graph-Enhanced ICLR 2026 recipe)."""
    mu = eeg.mean(axis=-1, keepdims=True)
    sd = eeg.std(axis=-1, keepdims=True) + eps
    out = (eeg - mu) / sd
    if clip_sigma is not None:
        out = np.clip(out, -clip_sigma, clip_sigma)
    return out.astype("float32", copy=False)


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels (CAR). Standard EEG referencing when
    the original recording reference is unknown. Acts as a coarse spatial
    high-pass.
    """
    return (eeg - eeg.mean(axis=0, keepdims=True)).astype("float32", copy=False)


# ============================================================================
# Named pipelines
# ============================================================================


@dataclass(frozen=True)
class PreprocessSpec:
    """Resolves a (preprocess, encoder) pair into the concrete pipeline that
    should be applied to a single EEG row.

    A row is a (channels, time) numpy array at ``src_sr``. The pipeline
    applies in order: bandpass → notch → resample → z-score → clip.
    """

    name: str
    bandpass_lo: float | None
    bandpass_hi: float | None
    notch_hz: float | None
    target_sr: int | None
    z_score: str
    clip_sigma: float | None
    car: bool

    def apply(self, eeg: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
        eeg = np.nan_to_num(eeg, copy=False).astype("float32", copy=False)

        if self.car:
            eeg = common_average_reference(eeg)

        if self.bandpass_lo is not None or self.bandpass_hi is not None:
            lo = self.bandpass_lo if self.bandpass_lo is not None else 0.0
            hi = self.bandpass_hi if self.bandpass_hi is not None else (0.5 * sr)
            eeg = bandpass(eeg, sr, lo, hi)

        if self.notch_hz is not None:
            eeg = notch(eeg, sr, self.notch_hz)

        if self.target_sr is not None and abs(sr - self.target_sr) > 0.5:
            eeg = resample_polyphase(eeg, sr, float(self.target_sr))
            sr = float(self.target_sr)

        if self.z_score == "row":
            eeg = zscore_per_recording(eeg, clip_sigma=self.clip_sigma)
        elif self.z_score == "channel":
            eeg = zscore_per_channel(eeg, clip_sigma=self.clip_sigma)
        elif self.z_score == "none":
            if self.clip_sigma is not None:
                eeg = np.clip(eeg, -self.clip_sigma, self.clip_sigma)
        else:
            raise ValueError(f"unknown z_score: {self.z_score}")

        return eeg.astype("float32", copy=False), sr


V1_NOOP = PreprocessSpec(
    name="v1",
    bandpass_lo=None, bandpass_hi=None,
    notch_hz=None,
    target_sr=None,
    z_score="none",
    clip_sigma=None,
    car=False,
)

V2_REVE = PreprocessSpec(
    name="v2_reve",
    bandpass_lo=0.5, bandpass_hi=99.5,
    notch_hz=50.0,
    target_sr=200,
    z_score="row",
    clip_sigma=15.0,
    car=False,
)

V2_TFM = PreprocessSpec(
    name="v2_tfm",
    bandpass_lo=0.1, bandpass_hi=75.0,
    notch_hz=50.0,
    target_sr=200,
    z_score="row",
    clip_sigma=15.0,
    car=False,
)

V2_DK25 = PreprocessSpec(
    name="v2_dk25",
    bandpass_lo=0.1, bandpass_hi=40.0,
    notch_hz=50.0,
    target_sr=50,
    z_score="row",
    clip_sigma=5.0,
    car=True,
)


PIPELINES: dict[str, PreprocessSpec] = {
    "v1": V1_NOOP,
    "v2_reve": V2_REVE,
    "v2_tfm": V2_TFM,
    "v2_dk25": V2_DK25,
}


# ============================================================================
# SpecAugment (Park et al. 2019, arXiv 1904.08779) — time + channel masking
# ============================================================================
#
# The standard ASR regulariser. For our setting:
#   * "Frequency" axis becomes the *channel* axis (we don't have an STFT in
#     the model input — encoders compute STFT internally for TFM, raw for
#     REVE; masking channels still produces the same regularisation effect).
#   * Time-mask width is parameterised in milliseconds so it scales with
#     whatever sampling rate the row arrives at.


def specaugment(eeg: np.ndarray, sr: float, *,
                n_time_masks: int = 2, time_mask_ms: int = 200,
                n_chan_masks: int = 2, chan_mask_max: int = 8,
                rng: "np.random.Generator | None" = None,
                ) -> np.ndarray:
    """Apply SpecAugment-style time + channel masking to a (C, T) EEG row.

    Use a row-deterministic RNG (seed by ``hash((sub, text))``) at training
    time to keep augmentations stable across epochs of the same sample —
    this matches the SpecAugment paper's "single-shot" augmentation
    convention rather than re-sampling per epoch.

    For exp02's CR-CTC track the trainer overrides this convention and seeds
    by ``hash(("specaug", sub, text, view_idx, step))`` so the two
    "augmented views" of the same row differ from each other.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = eeg.copy()
    C, T = out.shape

    max_t = max(1, int(round(time_mask_ms * sr / 1000.0)))
    for _ in range(n_time_masks):
        if T <= 1:
            break
        w = int(rng.integers(0, max(1, min(max_t, T))))
        if w <= 0:
            continue
        start = int(rng.integers(0, max(1, T - w)))
        out[:, start: start + w] = 0.0

    for _ in range(n_chan_masks):
        if C <= 1:
            break
        w = int(rng.integers(0, max(1, min(chan_mask_max, C))))
        if w <= 0:
            continue
        start = int(rng.integers(0, max(1, C - w)))
        out[start: start + w, :] = 0.0

    return out


# ============================================================================
# Encoder-aware pipeline resolver
# ============================================================================


def for_encoder(preset: str, encoder: str) -> PreprocessSpec:
    """Resolve `(preset, encoder)` into the concrete spec.

    ``preset="v2"`` chooses ``v2_reve`` for REVE, ``v2_tfm`` for TFM, and
    ``v2_reve`` (closest match) for DIVER-1 by default.
    """
    if preset != "v2":
        if preset not in PIPELINES:
            raise ValueError(f"unknown preprocess preset: {preset!r}")
        return PIPELINES[preset]
    return {
        "reve": V2_REVE,
        "tfm": V2_TFM,
        "diver1": V2_REVE,
    }.get(encoder, V2_REVE)
