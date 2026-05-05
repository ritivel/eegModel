"""Dataset / DataLoader scaffolding for eegfm.

Three sources of EEG batches, all returning the same `(B, T_samples)` shape
of float32 normalised single-channel windows:

    1. `synthetic_batch(B, T, kind=...)` — generate dummy data on the fly.
       Used by Check E (shape audit) and Check A (loss-at-init reference).
       Three flavours:
           "gauss"      iid N(0, 1) per sample (matched-noise twin baseline)
           "ar1"        AR(1) process at ρ=0.95 — has 1/f-like spectrum
           "constant"   per-channel constant equal to global mean (used by
                        Check B's input-independent baseline)

    2. `ParquetWindowDataset` — torch IterableDataset reading our offline
       parquet shards (the canonical preprocessed corpus). Yields rows
       lazily so we don't load 783 GB into RAM. Used by Checks C and D
       (one-batch overfit; random-init feature extraction for linear probe).

    3. `single_recording_overfit_batch(...)` — extract exactly 4 windows
       from a single subject+task parquet shard. The exact "fixed batch"
       Check C needs; the same batch must be returned reproducibly across
       optimization steps for the overfit to be meaningful.

All loaders apply nothing on top of what's in the parquet — windows are
already z-scored and ±5σ-clipped per `preprocess.py` SPEC_MINIMAL. We do
the float16→float32 cast at load time because torch ops on float16 are
much slower on most ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset


# =============================================================================
# Synthetic batches (Checks E, A)
# =============================================================================


def synthetic_batch(
    B: int = 8,
    T: int = 2000,
    kind: Literal["gauss", "ar1", "constant"] = "gauss",
    *,
    rho: float = 0.95,
    constant_value: float = 0.0,
    device: str | torch.device = "cpu",
    seed: int | None = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a (B, T) batch of synthetic single-channel EEG-shaped windows.

    Args:
        B: batch size
        T: number of samples per window. Default 2000 = 4 s @ 500 Hz.
        kind: which synthetic process. See module doc.
        rho: AR(1) coefficient. Default 0.95 gives a 1/f-like spectrum
             vaguely similar to real EEG.
        constant_value: only used when kind="constant". 0.0 matches our
             z-scored target's per-channel mean.
        device: torch device.
        seed: deterministic seed (None to draw from global rng).
        dtype: output dtype.

    Returns:
        torch.Tensor of shape (B, T), unit-variance approximately, dtype.
    """
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    if kind == "gauss":
        x = torch.randn(B, T, generator=g, dtype=torch.float32)
    elif kind == "ar1":
        # Vectorised AR(1) via truncated impulse-response convolution.
        # Recursion x[t] = rho*x[t-1] + sigma*eps[t] has closed form
        #     x[t] = sum_{k=0..t} rho^k * sigma * eps[t-k]
        # which is a 1-D convolution with kernel [sigma, sigma*rho, sigma*rho^2, ...].
        # Truncate when rho^k drops below 1e-5 (k≈220 at rho=0.95). Earlier
        # versions had a Python `for t in range(T)` loop that bottlenecked
        # Check B at ~10x slower-than-GPU (2000 iters × 16 batch / step).
        import torch.nn.functional as F
        sigma = math.sqrt(1.0 - rho * rho)
        max_lag = max(2, int(math.ceil(math.log(1e-5) / math.log(rho))))
        max_lag = min(max_lag, T)
        kernel = (rho ** torch.arange(max_lag, dtype=torch.float32)) * sigma
        eps = torch.randn(B, T + max_lag - 1, generator=g, dtype=torch.float32)
        # F.conv1d is cross-correlation; flip the kernel for convolution.
        kernel_flipped = kernel.flip(0).view(1, 1, -1)
        x = F.conv1d(eps.unsqueeze(1), kernel_flipped).squeeze(1)
        # Output length T + max_lag - 1 - max_lag + 1 = T  ✓
    elif kind == "constant":
        x = torch.full((B, T), float(constant_value), dtype=torch.float32)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return x.to(device=device, dtype=dtype)


# =============================================================================
# Parquet shard reader (Checks C, D)
# =============================================================================


@dataclass
class ParquetIndexEntry:
    """One parquet shard's location + a couple of size hints."""
    path: Path
    subject_id: str
    recording_id: str
    n_rows: int


class ParquetWindowDataset(IterableDataset):
    """IterableDataset over parquet shards from `derived/<pipeline>/`.

    Yields dicts of:
        signal:        (T_samples,) float32 tensor
        subject_id:    str
        site:          str
        task_label:    int
        channel_idx:   int
        attention:     float (CBCL Pearson-z; NaN if missing)
        externalizing: float
        p_factor:      float
        internalizing: float

    Reads parquet shards lazily via pyarrow. Optionally filters to a
    subset of subjects (for splits) and/or a limited number of windows
    per shard (for fast iteration on dev boxes).

    The dataset is *iid* over (channel, window) — every row in the parquet
    is one independent training example per `mini_experiments.md` §4.1.
    Multi-worker DataLoader users: shards are partitioned across workers
    by `worker_id` so two workers never read the same shard.
    """

    def __init__(
        self,
        derived_root: Path,
        *,
        subject_filter: set[str] | None = None,
        max_windows_per_shard: int | None = None,
        shuffle_within_shard: bool = True,
        rng_seed: int = 0,
    ):
        super().__init__()
        self.derived_root = Path(derived_root)
        if not self.derived_root.exists():
            raise FileNotFoundError(
                f"derived_root {self.derived_root} does not exist; "
                f"did you `eegfm sync-derived-down --pipeline minimal`?"
            )
        self.subject_filter = subject_filter
        self.max_windows_per_shard = max_windows_per_shard
        self.shuffle_within_shard = shuffle_within_shard
        self.rng_seed = rng_seed

        # Build a flat index of all (subject, recording) shards available.
        self._index = self._build_index()
        if not self._index:
            raise RuntimeError(
                f"no parquet shards under {self.derived_root}; "
                f"sub-dir layout should be derived/<pipeline>/sub-<id>/<recording>.parquet"
            )

    def _build_index(self) -> list[ParquetIndexEntry]:
        idx: list[ParquetIndexEntry] = []
        for sub_dir in sorted(self.derived_root.glob("sub-*")):
            sub_id = sub_dir.name[len("sub-"):]
            if self.subject_filter is not None and sub_id not in self.subject_filter:
                continue
            for shard in sorted(sub_dir.glob("*.parquet")):
                idx.append(ParquetIndexEntry(
                    path=shard, subject_id=sub_id, recording_id=shard.stem, n_rows=0,
                ))
        return idx

    def _iter_shard(self, entry: ParquetIndexEntry) -> Iterator[dict]:
        # Lazy import — avoid pulling pyarrow at module-import time on
        # CPU-only dev boxes that may have a partial install.
        import pyarrow.parquet as pq

        table = pq.read_table(entry.path, columns=[
            "subject_id", "site", "recording_id", "task_label",
            "channel_idx", "channel_name", "window_idx", "window_start_s",
            "n_samples", "signal",
            "p_factor", "attention", "internalizing", "externalizing",
            "age", "sex",
        ])
        n = table.num_rows
        order = np.arange(n)
        if self.shuffle_within_shard:
            rng = np.random.default_rng(self.rng_seed + hash(entry.recording_id) % (2**31))
            rng.shuffle(order)
        if self.max_windows_per_shard is not None:
            order = order[: self.max_windows_per_shard]

        # Pull the columns we need into numpy once; row access is fast then.
        sig_col = table.column("signal").to_pylist()
        # Other columns: pull once.
        cols = {name: table.column(name).to_pylist() for name in [
            "subject_id", "site", "recording_id", "task_label",
            "channel_idx", "channel_name", "window_idx", "window_start_s", "n_samples",
            "p_factor", "attention", "internalizing", "externalizing", "age", "sex",
        ]}

        for i in order.tolist():
            sig = np.asarray(sig_col[i], dtype=np.float32)        # cast f16 -> f32
            yield {
                "signal": torch.from_numpy(sig),
                "subject_id": cols["subject_id"][i],
                "site": cols["site"][i],
                "recording_id": cols["recording_id"][i],
                "task_label": int(cols["task_label"][i]),
                "channel_idx": int(cols["channel_idx"][i]),
                "channel_name": cols["channel_name"][i],
                "window_idx": int(cols["window_idx"][i]),
                "window_start_s": float(cols["window_start_s"][i]),
                "n_samples": int(cols["n_samples"][i]),
                "p_factor": float(cols["p_factor"][i]),
                "attention": float(cols["attention"][i]),
                "internalizing": float(cols["internalizing"][i]),
                "externalizing": float(cols["externalizing"][i]),
                "age": float(cols["age"][i]),
                "sex": cols["sex"][i] or "",
            }

    def __iter__(self) -> Iterator[dict]:
        # Two-level sharding: first by DDP rank (so different processes read
        # disjoint shards), then by DataLoader worker (so different worker
        # threads within a process read disjoint shards too).
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank, world_size = 0, 1
        except ImportError:
            rank, world_size = 0, 1
        rank_shards = self._index[rank::world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = rank_shards
        else:
            shards = rank_shards[worker_info.id :: worker_info.num_workers]

        for entry in shards:
            yield from self._iter_shard(entry)


def collate_signal_batch(rows: list[dict]) -> dict:
    """Default collate fn for `ParquetWindowDataset` (CPU/single-process eval).

    Stacks signals into (B, T_samples), keeps metadata as Python lists for
    quick filtering / split-by-subject in eval code. NOT compatible with
    `accelerate` DDP `dispatch_batches=True` because the str list values
    can't be torch.cat'd across ranks; use :func:`collate_signal_batch_train`
    in that case.
    """
    signals = torch.stack([r["signal"] for r in rows], dim=0)
    keys_int = ["task_label", "channel_idx", "window_idx", "n_samples"]
    keys_float = ["window_start_s", "p_factor", "attention",
                  "internalizing", "externalizing", "age"]
    keys_str = ["subject_id", "site", "recording_id", "channel_name", "sex"]
    out = {"signal": signals}
    for k in keys_int:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    for k in keys_float:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.float32)
    for k in keys_str:
        out[k] = [r[k] for r in rows]
    return out


def collate_signal_batch_train(rows: list[dict]) -> dict:
    """Tensor-only collate for the training loop (DDP-safe).

    Same as :func:`collate_signal_batch` minus the ``str`` metadata fields
    (subject_id, site, recording_id, channel_name, sex). Required for
    ``accelerate.Accelerator(dispatch_batches=True)`` because accelerate
    recursively ``torch.cat``s every dict value across ranks; lists of
    strings raise. The training loop only consumes ``batch["signal"]``
    anyway — see :mod:`eegfm.train`.
    """
    signals = torch.stack([r["signal"] for r in rows], dim=0)
    keys_int = ["task_label", "channel_idx", "window_idx", "n_samples"]
    keys_float = ["window_start_s", "p_factor", "attention",
                  "internalizing", "externalizing", "age"]
    out = {"signal": signals}
    for k in keys_int:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    for k in keys_float:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.float32)
    return out


# =============================================================================
# G4 LeJEPA — multi-view augmentation collate
# =============================================================================
#
# LeJEPA (Balestriero & LeCun, arXiv:2511.08544) requires V augmented views
# per sample to compute its invariance loss. The original recipe is image
# multi-crop (2 global + 6 local). For EEG, image augmentations don't apply
# — instead we use a small, EEG-validated set of stochastic augmentations
# whose composition produces V different views of the same 4-second window:
#
#   1. Random temporal masking — zero out a random contiguous span of length
#      `mask_span_frac * T` samples. EEG2Rep (KDD 2024 best paper) showed
#      that "preserving 50% of EEG recordings" is the sweet spot, so we
#      default to mask_frac=0.5 with random span placement per view.
#
#   2. Gaussian noise injection — add iid N(0, sigma_noise) noise. Defaults
#      to sigma=0.05 in the z-scored space; just enough to produce view
#      variability without dominating the signal.
#
#   3. Random sign flip — multiply by ±1 with probability 0.5. EEG polarity
#      is reference-dependent; many augmentation libraries (BENDR, EEG2Rep)
#      include this as a free invariance-inducing transform.
#
# All views have the SAME length T (no global/local crop-size split) so they
# tensor-stack cleanly into (B, V, T) without requiring multi-resolution
# encoder forward passes. The "global vs local" notion from LeJEPA's image
# recipe is approximated here by varying the mask fraction across views: the
# first n_views_global views use mask_frac_global (lighter), the remaining
# n_views_local views use mask_frac_local (heavier).
# =============================================================================


@dataclass
class MultiViewConfig:
    """Knobs for :class:`MultiViewCollate`. Defaults follow EEG2Rep + LeJEPA."""

    n_views_global: int = 2
    n_views_local: int = 4

    mask_frac_global: float = 0.30        # fraction of timesteps to zero in global views
    mask_frac_local: float = 0.60         # fraction in local views (heavier mask)
    mask_n_spans: int = 2                 # number of contiguous spans per view (rest of mask is split)

    sigma_noise: float = 0.05             # iid Gaussian noise std (z-scored space)
    sign_flip_prob: float = 0.5           # per-view probability of multiplying by -1

    seed: int = 0                         # for reproducibility of view sampling


class MultiViewCollate:
    """Collate fn that emits (B, V, T) multi-view batches for G4 LeJEPA.

    Drop-in replacement for :func:`collate_signal_batch_train` when training
    paradigm == "lejepa". Each input row's signal of shape (T,) is expanded
    into V augmented views of shape (T,), stacked to (V, T), and the per-row
    (V, T) tensors are stacked across the batch into (B, V, T). The trainer
    then passes ``batch["signal"]`` of shape (B, V, T) to the model, which
    reshapes to (B*V, T) for the encoder and (V, B, D) for the LeJEPA head.

    Augmentations are stochastic per-view per-sample; the same source signal
    produces V different views. We use a per-batch torch.Generator seeded
    deterministically by ``cfg.seed + global_step`` so that runs with the
    same seed are reproducible (matters for the noise-twin diagnostic).
    """

    def __init__(self, cfg: MultiViewConfig | None = None):
        self.cfg = cfg if cfg is not None else MultiViewConfig()
        self.n_views = self.cfg.n_views_global + self.cfg.n_views_local
        if self.n_views < 2:
            raise ValueError(
                f"LeJEPA needs at least 2 views per sample; "
                f"got n_views_global+n_views_local = {self.n_views}."
            )
        self._step = 0  # for view-augmentation seeding; advances per call

    def _sample_one_view(
        self,
        signal: torch.Tensor,                       # (T,)
        mask_frac: float,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Apply one stochastic augmentation: time-mask + noise + sign flip."""
        T = signal.size(0)
        view = signal.clone()

        # 1) Random temporal mask: split mask_frac*T samples into n_spans
        #    contiguous spans at random positions.
        n_mask_total = int(round(mask_frac * T))
        if n_mask_total > 0:
            n_spans = max(1, self.cfg.mask_n_spans)
            span_len = max(1, n_mask_total // n_spans)
            for _ in range(n_spans):
                if span_len >= T:
                    view.zero_()
                    break
                start = int(torch.randint(0, T - span_len + 1, (1,), generator=gen).item())
                view[start : start + span_len] = 0.0

        # 2) Gaussian noise injection (iid in z-scored space)
        if self.cfg.sigma_noise > 0:
            noise = torch.randn(T, generator=gen) * self.cfg.sigma_noise
            view = view + noise

        # 3) Random sign flip (EEG polarity invariance)
        if self.cfg.sign_flip_prob > 0:
            if float(torch.rand(1, generator=gen).item()) < self.cfg.sign_flip_prob:
                view = -view

        return view

    def __call__(self, rows: list[dict]) -> dict:
        # Per-call generator so view sampling is deterministic given (cfg.seed, _step).
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.cfg.seed + self._step)
        self._step += 1

        per_sample_views: list[torch.Tensor] = []
        for row in rows:
            signal = row["signal"]                             # (T,)
            views = []
            for _ in range(self.cfg.n_views_global):
                views.append(self._sample_one_view(signal, self.cfg.mask_frac_global, gen))
            for _ in range(self.cfg.n_views_local):
                views.append(self._sample_one_view(signal, self.cfg.mask_frac_local, gen))
            per_sample_views.append(torch.stack(views, dim=0))  # (V, T)

        signals = torch.stack(per_sample_views, dim=0)          # (B, V, T)

        # Match collate_signal_batch_train's tensor-only contract: keep
        # the per-sample int/float metadata so accelerate's DDP-cat works.
        keys_int = ["task_label", "channel_idx", "window_idx", "n_samples"]
        keys_float = ["window_start_s", "p_factor", "attention",
                      "internalizing", "externalizing", "age"]
        out: dict[str, torch.Tensor] = {"signal": signals}
        for k in keys_int:
            out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
        for k in keys_float:
            out[k] = torch.tensor([r[k] for r in rows], dtype=torch.float32)
        return out


# =============================================================================
# Single-recording overfit batch (Check C)
# =============================================================================


def single_recording_overfit_batch(
    derived_root: Path,
    *,
    n_windows: int = 4,
    subject_id: str | None = None,
    recording_id: str | None = None,
    channel_idx: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Return a fixed (n_windows, T_samples) batch from one parquet shard.

    Per `01_sanity_baselines/README.md` Check C: take 4 EEG epochs from one
    recording (one channel; one task) and train the model on JUST those 4
    examples for ≤ 1000 steps. The SSL loss must drive to <1% of init.

    If `subject_id`/`recording_id` are None, picks the first shard found.
    """
    derived_root = Path(derived_root)
    shards = sorted(derived_root.glob("sub-*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {derived_root}")

    if subject_id is not None:
        shards = [s for s in shards if f"sub-{subject_id}" in str(s)]
    if recording_id is not None:
        shards = [s for s in shards if s.stem == recording_id]
    if not shards:
        raise FileNotFoundError(
            f"no shard matching subject_id={subject_id!r} recording_id={recording_id!r}"
        )

    shard = shards[0]
    import pyarrow.parquet as pq
    table = pq.read_table(shard, columns=[
        "subject_id", "recording_id", "task_label", "channel_idx",
        "window_idx", "n_samples", "signal",
    ])
    df_chan_idx = np.asarray(table.column("channel_idx").to_pylist())
    df_window_idx = np.asarray(table.column("window_idx").to_pylist())
    sig_col = table.column("signal").to_pylist()

    sel = np.where(df_chan_idx == channel_idx)[0]
    if len(sel) < n_windows:
        raise ValueError(
            f"shard {shard.name} has only {len(sel)} windows on channel {channel_idx}; "
            f"need {n_windows}. Try a different channel_idx or recording."
        )
    # Take the first n_windows in window-order so the batch is reproducible.
    order = np.argsort(df_window_idx[sel])[:n_windows]
    chosen = sel[order]

    sigs = np.stack([np.asarray(sig_col[i], dtype=np.float32) for i in chosen], axis=0)
    return {
        "signal": torch.from_numpy(sigs).to(device=device, dtype=dtype),
        "subject_id": table.column("subject_id").to_pylist()[chosen[0]],
        "recording_id": table.column("recording_id").to_pylist()[chosen[0]],
        "task_label": int(table.column("task_label").to_pylist()[chosen[0]]),
        "channel_idx": int(channel_idx),
        "window_indices": [int(df_window_idx[i]) for i in chosen],
        "n_samples": int(table.column("n_samples").to_pylist()[chosen[0]]),
    }


# =============================================================================
# Convenience: list all available shards
# =============================================================================


def list_shards(derived_root: Path) -> list[ParquetIndexEntry]:
    """List `(subject_id, recording_id, path)` for every parquet shard under root."""
    derived_root = Path(derived_root)
    out = []
    for sub_dir in sorted(derived_root.glob("sub-*")):
        sub_id = sub_dir.name[len("sub-"):]
        for shard in sorted(sub_dir.glob("*.parquet")):
            out.append(ParquetIndexEntry(
                path=shard, subject_id=sub_id, recording_id=shard.stem, n_rows=0,
            ))
    return out
