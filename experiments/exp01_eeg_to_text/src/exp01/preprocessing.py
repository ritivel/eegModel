"""Per-row EEG preprocessing pipelines.

The Apr-30 pilot showed that the §4.3 gap goes the wrong way (noise > eeg)
because we feed REVE/TFM raw, unfiltered, unnormalised EEG — both encoders
were pretrained on data that had been bandpass-filtered, line-noise-notched,
and per-recording z-scored + clipped. With that mismatch, the encoder
outputs essentially-uninformative features and the bridge cannot align them
to text in any number of contrastive steps.

Recipe sources (each verified against the corresponding paper / repo):
  - REVE (NeurIPS 2025, arXiv 2510.21585 §3.1.1):
      bandpass 0.5–99.5 Hz, resample 200 Hz, per-recording z-score, ±15-σ clip
  - TFM-Tokenizer (ICLR 2026, arXiv 2502.16060 §B.2 / B.6):
      bandpass 0.1–75 Hz, notch 50 Hz, resample 200 Hz, STFT(n_fft=200,hop=100)
  - Défossez & King 2025 (Nat. Comm., 10521):
      bandpass 0.1–40 Hz, RobustScaler, clip ±5, resample 50 Hz (smaller for
      MEG/EEG word-decoding; included as a future option for the word-CL track)

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
    happen for low-SR sources where 0.4*sr < 75 Hz). Falls back to a
    low-pass if ``lo <= 0`` (degenerate config).
    """
    from scipy import signal

    nyq = 0.5 * sr
    lo_w = lo / nyq if lo > 0 else None
    hi_w = hi / nyq if hi < nyq else None

    # Tighten cutoffs that violate Nyquist (be conservative — leave a 5%
    # gap below Nyquist so the filter has finite stop-band).
    if hi_w is not None and hi_w >= 0.99:
        hi_w = 0.95

    if lo_w is None and hi_w is None:
        return None  # no-op
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
    # ``sosfiltfilt`` needs at least ``padlen`` samples; ZuCo sentences are
    # 1k–10k samples so we're safe, but DERCo word fragments can be ~50 samples.
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
    """Anti-aliased polyphase resample on the time axis. Replaces the previous
    ``F.interpolate(mode="linear")`` which aliased high-frequency content into
    the band (the worst case is `eeg_sem_relev` at 2858 Hz → 200 Hz).
    """
    from scipy import signal
    from math import gcd

    if abs(src_sr - dst_sr) < 0.5:
        return eeg
    # Round to integers for resample_poly — its polyphase factors must be int.
    a, b = int(round(dst_sr)), int(round(src_sr))
    g = gcd(a, b)
    up, down = a // g, b // g
    return signal.resample_poly(eeg, up=up, down=down, axis=-1).astype("float32", copy=False)


def zscore_per_recording(eeg: np.ndarray, *, eps: float = 1e-6,
                         clip_sigma: float | None = 15.0) -> np.ndarray:
    """Z-score with statistics computed across *all* (channel, time) samples
    of the row. After normalisation, optionally clip values exceeding
    ``±clip_sigma`` standard deviations.

    REVE pretraining used statistics across the recording session and a
    15-σ clip; we use per-row stats as a proxy because the parquet schema
    doesn't carry session ids. This matches Défossez & King 2025's
    RobustScaler + clamp[-5,5] pattern for unaligned recording sources.
    """
    mu = float(eeg.mean())
    sd = float(eeg.std()) + eps
    out = (eeg - mu) / sd
    if clip_sigma is not None:
        out = np.clip(out, -clip_sigma, clip_sigma)
    return out.astype("float32", copy=False)


def zscore_per_channel(eeg: np.ndarray, *, eps: float = 1e-6,
                       clip_sigma: float | None = 5.0) -> np.ndarray:
    """Per-channel z-score (Brain4FMs / Graph-Enhanced ICLR 2026 recipe).

    Uses each channel's own mean / std, which removes inter-channel scale
    differences as well as the global scale. Default clip is ±5σ to match
    Défossez & King 2025's RobustScaler behaviour.
    """
    mu = eeg.mean(axis=-1, keepdims=True)
    sd = eeg.std(axis=-1, keepdims=True) + eps
    out = (eeg - mu) / sd
    if clip_sigma is not None:
        out = np.clip(out, -clip_sigma, clip_sigma)
    return out.astype("float32", copy=False)


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels (CAR). Standard EEG referencing
    when the original recording reference is unknown / inconsistent.
    Acts as a coarse spatial high-pass.
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
    notch_hz: float | None  # None means skip notch
    target_sr: int | None   # None means leave at native sr
    z_score: str            # "row" (per-recording-equivalent) or "channel"
    clip_sigma: float | None
    car: bool

    def apply(self, eeg: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
        """Apply the pipeline to a (C, T) row at ``sr``. Returns (eeg, new_sr)."""
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


# v1 — what we used for the Apr-30 pilot (no preprocessing in this module;
# the collator does linear-interp resample). Kept for reproducibility.
V1_NOOP = PreprocessSpec(
    name="v1",
    bandpass_lo=None, bandpass_hi=None,
    notch_hz=None,
    target_sr=None,
    z_score="none",
    clip_sigma=None,
    car=False,
)

# v2_reve — REVE pretraining recipe (arXiv 2510.21585 §3.1.1).
V2_REVE = PreprocessSpec(
    name="v2_reve",
    bandpass_lo=0.5, bandpass_hi=99.5,
    notch_hz=50.0,           # ZuCo Switzerland and DERCo are 50 Hz mains
    target_sr=200,
    z_score="row",
    clip_sigma=15.0,         # REVE: "values exceeding 15 standard deviations were clipped"
    car=False,
)

# v2_tfm — TFM-Tokenizer / LaBraM / BIOT pretraining recipe.
V2_TFM = PreprocessSpec(
    name="v2_tfm",
    bandpass_lo=0.1, bandpass_hi=75.0,
    notch_hz=50.0,
    target_sr=200,
    z_score="row",           # TFM ingests STFT magnitude; per-row z-score on
                             # the time-domain input keeps the magnitude scale
                             # consistent across recordings.
    clip_sigma=15.0,
    car=False,
)

# v2_dk25 — Défossez & King 2025 recipe (smaller for word-CL).
V2_DK25 = PreprocessSpec(
    name="v2_dk25",
    bandpass_lo=0.1, bandpass_hi=40.0,
    notch_hz=50.0,
    target_sr=50,
    z_score="row",
    clip_sigma=5.0,          # paper uses RobustScaler + clamp[-5,5]; close enough
    car=True,
)


PIPELINES: dict[str, PreprocessSpec] = {
    "v1": V1_NOOP,
    "v2_reve": V2_REVE,
    "v2_tfm": V2_TFM,
    "v2_dk25": V2_DK25,
}


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
        "diver1": V2_REVE,  # DIVER-1 was MAE-pretrained on similar clinical data
    }.get(encoder, V2_REVE)
