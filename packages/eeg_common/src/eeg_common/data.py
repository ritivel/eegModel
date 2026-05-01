"""HuggingFace dataset download + per-row parquet streamer.

Eight unified-schema sources are supported (5 ZuCo + 3 word-level-only). Test
splits are always restricted to ZUCO_SOURCES; training spans all 8.

Storage-aware: every function and class that touches the filesystem takes a
:class:`eeg_common.storage.Storage` argument so two experiments can coexist
in the same Python process.

This module is import-safe: no torch / datasets imports at module level so a
local introspection of splits doesn't spin up the heavy stack.
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from . import preprocessing
from .splits import sent_hash
from .storage import Storage

DATASET_REPO = "tankalapavankalyan/exp01-eeg-to-text-sentences"

ZUCO_SOURCES = ("zuco_v1_sr", "zuco_v1_nr", "zuco_v1_tsr", "zuco_v2_nr", "zuco_v2_tsr")

ALL_SOURCES = ZUCO_SOURCES + ("derco_preprocessed", "emmt", "eeg_sem_relev")


# ============================================================================
# Path resolution
# ============================================================================


def _hf_dataset_snapshots_dir(storage: Storage) -> Path:
    return (storage.hf_cache
            / "datasets--tankalapavankalyan--exp01-eeg-to-text-sentences"
            / "snapshots")


def shard_paths(storage: Storage, source: str) -> list[Path]:
    """All parquet shards for a dataset source, from the HF cache."""
    base = _hf_dataset_snapshots_dir(storage)
    paths = sorted(Path(p) for p in glob.glob(f"{base}/*/data/{source}__*.parquet"))
    if not paths:
        # ``emmt`` is a single file, not __sub-... pattern.
        paths = sorted(Path(p) for p in glob.glob(f"{base}/*/data/{source}*.parquet"))
    return paths


# ============================================================================
# Download
# ============================================================================


def download_dataset(storage: Storage) -> None:
    """One-shot download of the gated HF dataset into the HF cache.

    Idempotent: ``snapshot_download`` skips files already present.
    """
    from huggingface_hub import snapshot_download

    storage.ensure_dirs()
    print(f"[download] {DATASET_REPO} -> {storage.hf_cache}", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        cache_dir=str(storage.hf_cache),
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["data/*.parquet", "*.md"],
    )
    paths = sorted(_hf_dataset_snapshots_dir(storage).glob("*/data/*.parquet"))
    total_gb = sum(p.stat().st_size for p in paths) / 1024 ** 3
    print(f"[download] {len(paths)} parquet shards ({total_gb:.1f} GB) cached",
          flush=True)


# ============================================================================
# Torch dataset + noise twin
# ============================================================================


class EEGSentenceDataset:
    """Streaming parquet reader. Returns dict rows; the encoder-specific
    collator (in the experiment's trainer) handles tensorisation, resampling,
    and channel layout.

    Args:
        storage: Storage instance (locates the parquet shards in the HF cache).
        sources: which ``dataset`` values to include.
        subject_filter: only keep rows whose ``participant_id`` is in this set.
        sentence_filter: only keep rows whose normalised text hashes into this
            set (used to apply the Yin unique-sentence partition).
        noise: ``None`` for raw EEG, ``"gauss"`` for the Jo et al. noise twin
            (per-channel zero-mean Gaussian with the EEG's per-channel std).
        eval_only: skips SpecAugment when True.
        preprocess: a :class:`PreprocessSpec` applied per row before the noise
            twin step.
        specaugment: kwargs dict forwarded to :func:`preprocessing.specaugment`
            (or ``None`` to disable). Applied only when ``not eval_only``.
    """

    _WORD_LEVEL_ONLY_SOURCES = frozenset({"derco_preprocessed", "emmt", "eeg_sem_relev"})

    def __init__(
        self,
        storage: Storage,
        sources: Iterable[str] = ZUCO_SOURCES,
        subject_filter: Iterable[str] | None = None,
        sentence_filter: Iterable[str] | None = None,
        noise: str | None = None,
        eval_only: bool = False,
        preprocess: "preprocessing.PreprocessSpec | None" = None,
        specaugment: dict | None = None,
        # ----- Quality filters added after the May-1 audit (see findings.md §2.3) -----
        drop_sources: Iterable[str] = (),
        min_text_chars: int = 0,
        max_text_chars: int | None = None,
        max_seconds: float | None = None,
        drop_nan_rows: bool = False,
        drop_zero_rows: bool = False,
    ):
        import pyarrow.parquet as pq

        self.storage = storage
        self.specaugment_kwargs = specaugment
        self.subject_filter = set(subject_filter) if subject_filter else None
        self.sentence_filter = set(sentence_filter) if sentence_filter else None
        self.noise = noise
        self.eval_only = eval_only
        self.preprocess = preprocess
        self.drop_sources = set(drop_sources)
        self.min_text_chars = max(0, int(min_text_chars))
        self.max_text_chars = int(max_text_chars) if max_text_chars else None
        self.max_seconds = float(max_seconds) if max_seconds else None
        self.drop_nan_rows = bool(drop_nan_rows)
        self.drop_zero_rows = bool(drop_zero_rows)

        files: list[tuple[Path, str]] = []
        for src in sources:
            if src in self.drop_sources:
                continue
            for p in shard_paths(storage, src):
                files.append((p, src))
        self.files = files

        # Index = (file_idx, row_group_idx, row_in_group_idx) for every row
        # that survives the filters. We read ONLY cheap metadata columns at
        # filter time — see the long comment in exp01.data for why.
        self._index: list[tuple[int, int, int]] = []
        per_source_kept: dict[str, int] = {}
        per_source_total: dict[str, int] = {}
        n_drop_text = 0
        n_drop_seconds = 0
        n_drop_degenerate = 0
        for fi, (f, src) in enumerate(files):
            try:
                pf = pq.ParquetFile(f)
            except Exception as e:
                print(f"[data] WARN: failed to open {f.name}: {e}", flush=True)
                continue
            word_level_source = src in self._WORD_LEVEL_ONLY_SOURCES
            apply_sentence_filter_here = src in ZUCO_SOURCES
            cols_meta = ["sentence_text", "participant_id", "num_channels",
                         "num_words", "sampling_rate_hz"]
            if self.max_seconds is not None:
                # Need a per-row T estimate. ``num_samples`` lives in some shards;
                # otherwise fall back to peeking at the EEG arr length below.
                try:
                    if "num_samples" in pf.schema_arrow.names:
                        cols_meta.append("num_samples")
                except Exception:
                    pass
            kept = 0
            total = 0
            for rg_idx in range(pf.num_row_groups):
                try:
                    t = pf.read_row_group(rg_idx, columns=cols_meta)
                except Exception as e:
                    print(f"[data] WARN: failed to read rg{rg_idx} of {f.name}: {e}",
                          flush=True)
                    continue
                texts = t["sentence_text"].to_pylist()
                pids = t["participant_id"].to_pylist()
                ncs = t["num_channels"].to_pylist()
                nws = t["num_words"].to_pylist()
                srs = t["sampling_rate_hz"].to_pylist()
                nsamples = (t["num_samples"].to_pylist()
                            if "num_samples" in t.column_names else [None] * len(texts))
                total += len(texts)
                for ri_in_rg, (text, pid, nc, nw, sr, nsamp) in enumerate(
                        zip(texts, pids, ncs, nws, srs, nsamples)):
                    if self.subject_filter and str(pid) not in self.subject_filter:
                        continue
                    if (apply_sentence_filter_here and self.sentence_filter
                            and sent_hash(text or "") not in self.sentence_filter):
                        continue
                    if (nc or 0) < 2:
                        n_drop_degenerate += 1
                        continue
                    if not word_level_source and (nw or 0) < 1:
                        n_drop_degenerate += 1
                        continue
                    txt = text or ""
                    if self.min_text_chars and len(txt.strip()) < self.min_text_chars:
                        n_drop_text += 1
                        continue
                    if self.max_text_chars is not None and len(txt) > self.max_text_chars:
                        n_drop_text += 1
                        continue
                    if (self.max_seconds is not None and nsamp is not None and sr
                            and (nsamp / float(sr)) > self.max_seconds):
                        n_drop_seconds += 1
                        continue
                    self._index.append((fi, rg_idx, ri_in_rg))
                    kept += 1
            per_source_kept[src] = per_source_kept.get(src, 0) + kept
            per_source_total[src] = per_source_total.get(src, 0) + total

        per_source_str = ", ".join(
            f"{s}={per_source_kept.get(s, 0)}/{per_source_total.get(s, 0)}"
            for s in sources if s not in self.drop_sources
        )
        filt_str = (f"min_text={self.min_text_chars} max_text={self.max_text_chars} "
                    f"max_sec={self.max_seconds} drop_sources={sorted(self.drop_sources) or '-'} "
                    f"drop_nan={self.drop_nan_rows} drop_zero={self.drop_zero_rows}")
        print(
            f"[data] EEGSentenceDataset: {len(self._index)} rows kept "
            f"(degenerate dropped={n_drop_degenerate}, text dropped={n_drop_text}, "
            f"too-long dropped={n_drop_seconds}); "
            f"per-source kept/total: {per_source_str}; "
            f"subject_filter={'set('+str(len(self.subject_filter))+')' if self.subject_filter else 'None'}, "
            f"sentence_filter={'set('+str(len(self.sentence_filter))+')' if self.sentence_filter else 'None'}, "
            f"noise={self.noise}, preprocess={self.preprocess.name if self.preprocess else 'None'}, "
            f"filters=[{filt_str}]",
            flush=True,
        )
        if not self._index:
            for fi, (f, src) in enumerate(files[:3]):
                try:
                    pf = pq.ParquetFile(f)
                    t = pf.read_row_group(0, columns=["sentence_text", "participant_id"])
                    sample_pids = sorted({str(p) for p in t["participant_id"].to_pylist()})[:5]
                    sample_texts = [s for s in t["sentence_text"].to_pylist() if s][:2]
                    print(f"[data]   {f.name}: pids={sample_pids} texts={sample_texts}",
                          flush=True)
                except Exception as e:
                    print(f"[data]   {f.name}: <{e}>", flush=True)

        self._cache: dict[int, "pq.ParquetFile"] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        import pyarrow.parquet as pq

        fi, rg_idx, ri_in_rg = self._index[i]
        if fi not in self._cache:
            self._cache[fi] = pq.ParquetFile(self.files[fi][0])
        cols = [
            "sentence_text",
            "participant_id",
            "dataset",
            "sampling_rate_hz",
            "channel_names",
            "num_channels",
            "sentence_eeg",
            "word_eeg_segments",
        ]
        row = (self._cache[fi]
               .read_row_group(rg_idx, columns=cols)
               .slice(ri_in_rg, 1)
               .to_pylist()[0])

        eeg = _row_to_array(row)
        sr = float(row["sampling_rate_hz"])

        # Runtime-only quality fallback: if a row is all NaN or all zeros,
        # quietly substitute the next surviving row in the index. The startup
        # filter step doesn't peek at the actual EEG arrays for speed, so a
        # small fraction of degenerate rows can slip through.
        if self.drop_nan_rows or self.drop_zero_rows:
            bad = False
            if self.drop_nan_rows and np.isnan(eeg).any():
                bad = True
            if self.drop_zero_rows and (eeg.size <= 1 or float(np.abs(eeg).max()) < 1e-12):
                bad = True
            if bad:
                # Walk forward; bounded by len(self._index) to avoid infinite recursion.
                return self.__getitem__((i + 1) % len(self._index))

        if self.preprocess is not None:
            eeg, sr = self.preprocess.apply(eeg, sr)

        if self.noise:
            mu = eeg.mean(axis=1, keepdims=True)
            sd = eeg.std(axis=1, keepdims=True) + 1e-6
            seed = hash((row["participant_id"], row["sentence_text"])) & 0xFFFFFFFF
            rng = np.random.default_rng(seed=seed)
            eeg = (rng.standard_normal(size=eeg.shape).astype("float32") * sd + mu).astype("float32")
            # If the source row was NaN-rich, the noise is too. Replace with N(0,1).
            if not np.isfinite(eeg).all():
                eeg = rng.standard_normal(size=eeg.shape).astype("float32")

        if self.specaugment_kwargs is not None and not self.eval_only:
            sa_seed = hash(("specaug", row["participant_id"],
                            row["sentence_text"])) & 0xFFFFFFFF
            sa_rng = np.random.default_rng(seed=sa_seed)
            eeg = preprocessing.specaugment(eeg, sr, rng=sa_rng,
                                             **self.specaugment_kwargs)

        return {
            "eeg": eeg,
            "sr": sr,
            "channels": list(row["channel_names"] or []),
            "text": str(row["sentence_text"]),
            "participant_id": str(row["participant_id"]),
            "dataset": str(row["dataset"]),
        }


def _row_to_array(row: dict) -> np.ndarray:
    """Extract a (channels, time) float32 array from a unified-schema row.

    Prefers ``sentence_eeg`` (channels-first, full sentence) when available
    (ZuCo); falls back to concatenating ``word_eeg_segments`` along time
    (DERCo, EMMT, eeg_sem_relev). A row whose neither column has usable EEG
    returns a (n_channels, 1) zero placeholder so the collator's pad step
    doesn't crash.
    """
    if row.get("sentence_eeg"):
        return np.asarray(row["sentence_eeg"], dtype="float32")

    segs = row.get("word_eeg_segments") or []
    arrs = []
    n_channels_seen = 0
    for w in segs:
        if not w:
            continue
        a = np.asarray(w, dtype="float32")
        if a.ndim == 2:
            arrs.append(a)
            n_channels_seen = max(n_channels_seen, a.shape[0])
    if not arrs:
        c = max(int(row.get("num_channels") or 1), 1)
        return np.zeros((c, 1), dtype="float32")
    if len({a.shape[0] for a in arrs}) > 1:
        arrs = [a for a in arrs if a.shape[0] == n_channels_seen]
        if not arrs:
            return np.zeros((n_channels_seen, 1), dtype="float32")
    return np.concatenate(arrs, axis=1)
