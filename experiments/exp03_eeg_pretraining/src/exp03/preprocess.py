"""Per-recording preprocessing pipelines for exp03.

Two pipelines are exposed, both run offline once and produce parquet shards
that every later experiment reads. The split between them is the explicit
"minimum offline, maximum in-model" decision documented in
``mini_experiments.md`` §4.1:

  ``preprocess_minimal``
      The PRIMARY pipeline. Six lines of numpy: NaN sanitation, per-channel
      z-score, ±5σ clip, 4-second non-overlapping windowing, iid-channel
      expansion, float16 cast. Native 500 Hz preserved. Notch filtering,
      bandpass filtering, and resampling are *deliberately not done* — they
      are hypotheses tested by exp02 (frontend), exp05 (multi-rate), and
      exp14 (context length).

  ``preprocess_v2_clean``
      The literature-comparability cell ONLY (used by exp02 F0-prep, never
      by the §4.4 winner-picker). Adds 60 Hz notch + 0.5–100 Hz Butterworth
      bandpass + 500 → 250 Hz polyphase resample on top of the minimal
      pipeline. Reproduces the input the BENDR / LaBraM / CBraMod / REVE
      papers measured their numbers on.

Filters use ``scipy.signal``; resampling uses polyphase ``resample_poly``
for proper anti-aliasing. Everything runs on numpy arrays so it can execute
inside ``ProcessPoolExecutor`` workers without holding the GIL.

Output schema (one row per (channel, 4-sec window) after iid expansion):

    subject_id      string         e.g. "NDARABCD1234"
    site            string         RU / CBIC / CUNY / SI (HBN site code)
    recording_id    string         e.g. "task-RestingState_run-1"
    task_label      int8           0..5  (RestingState=0 / SequenceLearning=1 /
                                          SymbolSearch=2 / SurroundSuppression=3 /
                                          ContrastChangeDetection=4 / Video=5)
    channel_idx     int16          0..127 (HydroCel position; see hbn.HYDROCEL_128_NAMES)
    channel_name    string         e.g. "E62"
    window_idx      int32          0, 1, 2, ... per recording
    window_start_s  float32        offset within recording, in seconds
    sample_rate_hz  int16          500 for minimal, 250 for v2_clean
    n_samples       int16          2000 for minimal, 1000 for v2_clean
    signal          list<float16>  the actual EEG window (length = n_samples)
    age             float32        subject age at recording, NaN if missing
    sex             string         "M" / "F" / "" (missing)
    adhd            int8           1 if any DSM-V Dx column matches /ADHD|attention.deficit/i
                                   0 if none, -1 if dx columns are all empty (label missing)
    pipeline        string         "minimal" or "v2_clean"
    src_sha256_8    string         first 8 hex chars of sha256(raw .set+.fdt) — provenance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Spec constants (one place to change so they stay in sync with §4.1)
# ---------------------------------------------------------------------------

WINDOW_SECONDS: float = 4.0
CLIP_SIGMA: float = 5.0
ZSCORE_EPS: float = 1e-6

V2_CLEAN_NOTCH_HZ: float = 60.0          # US power line; HBN sites are all US
V2_CLEAN_NOTCH_Q: float = 30.0           # tight notch ~1.7 Hz wide
V2_CLEAN_BAND_LO: float = 0.5            # Hz, high-pass cutoff
V2_CLEAN_BAND_HI: float = 100.0          # Hz, low-pass cutoff (anti-alias before 250 Hz resample)
V2_CLEAN_BAND_ORDER: int = 4             # Butterworth order (zero-phase filtfilt)
V2_CLEAN_TARGET_SR: int = 250            # Hz


# ---------------------------------------------------------------------------
# Spec dataclass (printable provenance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessSpec:
    """Identifies which pipeline a parquet shard was produced by."""

    name: str                       # "minimal" or "v2_clean"
    target_sr: int                  # output sample rate after pipeline
    notch_hz: float | None
    band_lo: float | None
    band_hi: float | None
    z_score: str = "channel"
    clip_sigma: float = CLIP_SIGMA
    window_seconds: float = WINDOW_SECONDS

    @property
    def n_samples(self) -> int:
        return int(round(self.window_seconds * self.target_sr))


SPEC_MINIMAL = PreprocessSpec(
    name="minimal",
    target_sr=500,
    notch_hz=None,
    band_lo=None,
    band_hi=None,
)

SPEC_V2_CLEAN = PreprocessSpec(
    name="v2_clean",
    target_sr=V2_CLEAN_TARGET_SR,
    notch_hz=V2_CLEAN_NOTCH_HZ,
    band_lo=V2_CLEAN_BAND_LO,
    band_hi=V2_CLEAN_BAND_HI,
)

SPECS: dict[str, PreprocessSpec] = {
    SPEC_MINIMAL.name: SPEC_MINIMAL,
    SPEC_V2_CLEAN.name: SPEC_V2_CLEAN,
}


# ---------------------------------------------------------------------------
# Building blocks (each is independently testable, all pure numpy)
# ---------------------------------------------------------------------------


def sanitise_nan(eeg: np.ndarray) -> np.ndarray:
    """Replace NaN/±inf with 0 in-place-but-safe.

    Required for training stability — HBN raw has occasional electrode-contact
    gaps that produce NaN samples, which would propagate through any filter
    and contaminate every downstream window.
    """
    return np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def zscore_per_channel(eeg: np.ndarray, *, eps: float = ZSCORE_EPS,
                       clip_sigma: float | None = CLIP_SIGMA) -> np.ndarray:
    """Per-channel z-score with optional symmetric clip.

    Per-channel because electrode impedances differ across sensors of the
    same recording — frontal electrodes ride larger eye-movement signals,
    high-impedance contacts have larger scale. Per-channel normalisation
    makes the optimiser's life possible without throwing away inter-channel
    relative-amplitude information *within* each window.

    The ±5σ clip catches single-sample electrode-pop artifacts that survive
    any filter and would otherwise dominate L2 loss.
    """
    eeg = eeg.astype(np.float32, copy=False)
    mu = eeg.mean(axis=-1, keepdims=True)
    sd = eeg.std(axis=-1, keepdims=True) + eps
    out = (eeg - mu) / sd
    if clip_sigma is not None:
        np.clip(out, -clip_sigma, clip_sigma, out=out)
    return out


def notch_filter(eeg: np.ndarray, sr: float, hz: float, q: float) -> np.ndarray:
    """Zero-phase IIR notch (60 Hz line for HBN's US sites).

    Skipped if 2*hz >= sr (Nyquist violation). The Q factor is the standard
    EEG default of 30 → ~1.7 Hz wide notch around the line frequency,
    leaving most of the 40–80 Hz γ band intact.
    """
    if 2.0 * hz >= sr:
        return eeg
    from scipy import signal as scipy_signal

    b, a = scipy_signal.iirnotch(w0=hz, Q=q, fs=sr)
    padlen = min(3 * max(len(a), len(b)), max(0, eeg.shape[-1] - 1))
    if padlen <= 0:
        return eeg
    return scipy_signal.filtfilt(b, a, eeg, axis=-1, padlen=padlen).astype(np.float32, copy=False)


def bandpass_filter(eeg: np.ndarray, sr: float, lo: float, hi: float,
                    order: int = V2_CLEAN_BAND_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass via SOS sections + sosfiltfilt.

    Zero-phase matters because exp07 will test phase-aware losses — a
    phase-shifting filter (filt only, no filtfilt) would bias them.
    Falls back gracefully to high-pass / low-pass / no-op if the cutoffs
    sit outside [eps, Nyquist].
    """
    from scipy import signal as scipy_signal

    nyq = 0.5 * sr
    lo_w = lo / nyq if lo > 0 else None
    hi_w = hi / nyq if hi < nyq else None
    if hi_w is not None and hi_w >= 0.99:
        hi_w = 0.95
    if lo_w is None and hi_w is None:
        return eeg

    if lo_w is None:
        sos = scipy_signal.butter(order, hi_w, btype="lowpass", output="sos")
    elif hi_w is None:
        sos = scipy_signal.butter(order, lo_w, btype="highpass", output="sos")
    else:
        sos = scipy_signal.butter(order, [lo_w, hi_w], btype="bandpass", output="sos")

    padlen = min(3 * (sos.shape[0] * 2), max(0, eeg.shape[-1] - 1))
    if padlen <= 0:
        return eeg
    return scipy_signal.sosfiltfilt(sos, eeg, axis=-1, padlen=padlen).astype(np.float32, copy=False)


def resample_polyphase(eeg: np.ndarray, src_sr: float, dst_sr: int) -> np.ndarray:
    """Anti-aliased polyphase resample on the time axis.

    Uses scipy's ``resample_poly`` (Kaiser-windowed sinc) — strictly better
    than naive subsampling, which would alias high-frequency content into
    the passband.
    """
    if abs(src_sr - dst_sr) < 0.5:
        return eeg
    from math import gcd
    from scipy import signal as scipy_signal

    a, b = int(round(dst_sr)), int(round(src_sr))
    g = gcd(a, b)
    up, down = a // g, b // g
    return scipy_signal.resample_poly(eeg, up=up, down=down, axis=-1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Pipelines (compose the blocks above)
# ---------------------------------------------------------------------------


def preprocess_minimal(eeg: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
    """Apply the SPEC_MINIMAL pipeline.

    Input:  eeg (C, T) at ``sr`` Hz, raw mV.
    Output: (eeg', sr')  with sr' == sr (no resampling), eeg' float32 (C, T).

    Steps: NaN sanitation → per-channel z-score (with ±5σ clip).
    No notch, no bandpass, no resample — those are model-side hypotheses.
    """
    eeg = sanitise_nan(np.asarray(eeg, dtype=np.float32))
    eeg = zscore_per_channel(eeg)
    return eeg, float(sr)


def preprocess_v2_clean(eeg: np.ndarray, sr: float) -> tuple[np.ndarray, float]:
    """Apply the SPEC_V2_CLEAN pipeline (F0-prep literature cell only).

    Input:  eeg (C, T) at ``sr`` Hz, raw mV.
    Output: (eeg', sr')  with sr' == 250 Hz (or original if already lower),
            eeg' float32 (C, T').

    Steps: NaN sanitation → 60 Hz notch (Q=30) → 0.5–100 Hz Butterworth
    bandpass (order 4, zero-phase) → polyphase resample to 250 Hz →
    per-channel z-score (with ±5σ clip).
    """
    eeg = sanitise_nan(np.asarray(eeg, dtype=np.float32))
    eeg = notch_filter(eeg, sr, V2_CLEAN_NOTCH_HZ, V2_CLEAN_NOTCH_Q)
    eeg = bandpass_filter(eeg, sr, V2_CLEAN_BAND_LO, V2_CLEAN_BAND_HI)
    eeg = resample_polyphase(eeg, sr, V2_CLEAN_TARGET_SR)
    eeg = zscore_per_channel(eeg)
    return eeg, float(V2_CLEAN_TARGET_SR)


def apply_pipeline(eeg: np.ndarray, sr: float, pipeline: str) -> tuple[np.ndarray, float]:
    """Dispatch to the named pipeline."""
    if pipeline == "minimal":
        return preprocess_minimal(eeg, sr)
    if pipeline == "v2_clean":
        return preprocess_v2_clean(eeg, sr)
    raise ValueError(f"unknown pipeline: {pipeline!r} (valid: {list(SPECS)})")


# ---------------------------------------------------------------------------
# Windowing + iid-channel expansion
# ---------------------------------------------------------------------------


def window_4s(eeg: np.ndarray, sr: float,
              window_seconds: float = WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Slice (C, T) → (n_windows, C, T_win) non-overlapping.

    Drops the trailing partial window if it doesn't fit. Returns the
    windows tensor and a (n_windows,) array of window-start times in seconds.
    """
    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise ValueError(f"expected (C, T), got shape {eeg.shape}")
    C, T = eeg.shape
    win_samples = int(round(window_seconds * sr))
    if win_samples <= 0 or T < win_samples:
        return np.empty((0, C, win_samples), dtype=eeg.dtype), np.empty((0,), dtype=np.float32)

    n_windows = T // win_samples
    truncated = eeg[:, : n_windows * win_samples]
    windows = truncated.reshape(C, n_windows, win_samples).transpose(1, 0, 2)  # (n_windows, C, T_win)
    starts = (np.arange(n_windows) * window_seconds).astype(np.float32)
    return windows, starts


def iid_expand_rows(windows: np.ndarray, starts: np.ndarray, channel_names: list[str],
                    *, base_metadata: dict) -> list[dict]:
    """Expand (n_windows, C, T_win) into ``n_windows × C`` row dicts.

    Each row is one (channel, 4-sec window) iid training example, suitable
    for direct conversion to a parquet shard via ``rows_to_parquet_table``.
    The signal is cast to float16 here (the parquet write format).
    """
    if windows.size == 0:
        return []
    n_windows, C, T_win = windows.shape
    if C != len(channel_names):
        raise ValueError(f"channel count mismatch: {C} vs {len(channel_names)}")

    # Pre-cast once to avoid 11.5M individual float16 conversions.
    sig16 = windows.astype(np.float16, copy=False)

    rows: list[dict] = []
    for w in range(n_windows):
        win_start = float(starts[w])
        for c in range(C):
            rows.append({
                **base_metadata,
                "channel_idx": np.int16(c),
                "channel_name": channel_names[c],
                "window_idx": np.int32(w),
                "window_start_s": np.float32(win_start),
                "n_samples": np.int16(T_win),
                "signal": sig16[w, c, :].tolist(),  # parquet list<float16> column
            })
    return rows


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


PARQUET_SCHEMA_FIELDS = (
    "subject_id", "site", "recording_id", "task_label",
    "channel_idx", "channel_name", "window_idx", "window_start_s",
    "sample_rate_hz", "n_samples", "signal",
    "age", "sex", "adhd",
    "pipeline", "src_sha256_8",
)


def rows_to_parquet_table(rows: list[dict]):
    """Build a pyarrow Table with an explicit schema (avoids type inference)."""
    import pyarrow as pa

    if not rows:
        raise ValueError("no rows to write")

    schema = pa.schema([
        pa.field("subject_id", pa.string()),
        pa.field("site", pa.string()),
        pa.field("recording_id", pa.string()),
        pa.field("task_label", pa.int8()),
        pa.field("channel_idx", pa.int16()),
        pa.field("channel_name", pa.string()),
        pa.field("window_idx", pa.int32()),
        pa.field("window_start_s", pa.float32()),
        pa.field("sample_rate_hz", pa.int16()),
        pa.field("n_samples", pa.int16()),
        pa.field("signal", pa.list_(pa.float16())),
        pa.field("age", pa.float32()),
        pa.field("sex", pa.string()),
        pa.field("adhd", pa.int8()),
        pa.field("pipeline", pa.string()),
        pa.field("src_sha256_8", pa.string()),
    ])

    columns = {name: [] for name in PARQUET_SCHEMA_FIELDS}
    for r in rows:
        for name in PARQUET_SCHEMA_FIELDS:
            columns[name].append(r.get(name))
    return pa.table(columns, schema=schema)


def write_parquet_shard(rows: list[dict], path) -> int:
    """Write rows to a parquet file and return the number of rows written."""
    import pyarrow.parquet as pq

    table = rows_to_parquet_table(rows)
    pq.write_table(
        table, str(path),
        compression="zstd", compression_level=3,
        # Write stats per column so a downstream filter on subject/task is fast.
        write_statistics=True,
    )
    return len(rows)
