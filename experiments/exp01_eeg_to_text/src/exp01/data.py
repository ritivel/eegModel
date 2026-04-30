"""Data: HF download, Yin et al. (2024) splits, Jo et al. (2024) noise twin.

This module is import-safe on the local machine (no torch / datasets imports
at module level) so the local Modal entrypoints can introspect splits without
spinning up the heavy stack.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import storage

DATASET_REPO = "tankalapavankalyan/exp01-eeg-to-text-sentences"

# ZuCo-only sources we evaluate on (§4 of report). Training sees all 8 sources;
# evaluation rows are restricted to these.
ZUCO_SOURCES = ("zuco_v1_sr", "zuco_v1_nr", "zuco_v1_tsr", "zuco_v2_nr", "zuco_v2_tsr")

ALL_SOURCES = ZUCO_SOURCES + ("derco_preprocessed", "emmt", "eeg_sem_relev")


# ============================================================================
# Path resolution: parquet shards live in the HF cache under
# ``$EXP01_DATA_ROOT/hf/datasets--.../snapshots/<commit>/data/``.
# ============================================================================


def _hf_dataset_snapshots_dir() -> Path:
    return (storage.HF_CACHE / "datasets--tankalapavankalyan--exp01-eeg-to-text-sentences"
            / "snapshots")


def shard_paths(source: str) -> list[Path]:
    """All parquet shards for a dataset source, from the HF cache."""
    base = _hf_dataset_snapshots_dir()
    paths = sorted(Path(p) for p in glob.glob(f"{base}/*/data/{source}__*.parquet"))
    if not paths:
        # ``emmt`` is a single file, not __sub-... pattern.
        paths = sorted(Path(p) for p in glob.glob(f"{base}/*/data/{source}*.parquet"))
    return paths


# ============================================================================
# 1. Download
# ============================================================================


def download_dataset() -> None:
    """One-shot download of the gated HF dataset into the HF cache.

    Idempotent: ``snapshot_download`` skips files already present.
    """
    from huggingface_hub import snapshot_download

    storage.ensure_dirs()
    print(f"[download] {DATASET_REPO} -> {storage.HF_CACHE}", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        cache_dir=str(storage.HF_CACHE),
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["data/*.parquet", "*.md"],
    )
    paths = sorted(_hf_dataset_snapshots_dir().glob("*/data/*.parquet"))
    total_gb = sum(p.stat().st_size for p in paths) / 1024**3
    print(f"[download] {len(paths)} parquet shards ({total_gb:.1f} GB) cached", flush=True)


# ============================================================================
# 2. Splits — Yin et al. (2024) unique-sentence + leave-N-subjects-out
# ============================================================================


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_subjects: tuple[str, ...]
    dev_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_sent_hashes: frozenset
    dev_sent_hashes: frozenset
    test_sent_hashes: frozenset


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _hash(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:16]


def make_folds(*, n_folds: int = 5, n_test: int = 3, n_dev: int = 3, seed: int = 20260430) -> list[FoldSplit]:
    """Compute LNSO subject folds + Yin unique-sentence assignment.

    The unique-sentence pool is split 80/10/10 once; the same text-level
    partition is reused across folds so cross-fold variance is purely from
    *which subjects* are held out, not *which sentences*. This matches §4.1+§4.2.
    """
    import random
    import pyarrow.parquet as pq

    storage.ensure_dirs()

    # Step 1: collect unique normalised sentences and their subject sets.
    text_to_subjects: dict[str, set[str]] = {}
    n_total_files = sum(len(shard_paths(src)) for src in ZUCO_SOURCES)
    print(f"[splits] scanning {n_total_files} ZuCo shards (sentence_text + participant_id)", flush=True)
    seen = 0
    for src in ZUCO_SOURCES:
        paths = shard_paths(src)
        for path in paths:
            t0 = time.time()
            t = pq.read_table(path, columns=["sentence_text", "participant_id"])
            for s, p in zip(t["sentence_text"].to_pylist(), t["participant_id"].to_pylist()):
                if not s:
                    continue
                text_to_subjects.setdefault(_norm(s), set()).add(str(p))
            seen += 1
            print(f"[splits]   ({seen}/{n_total_files}) {path.name}: "
                  f"{len(t)} rows in {(time.time()-t0)*1000:.0f}ms "
                  f"(unique-so-far={len(text_to_subjects)})", flush=True)

    unique = sorted(text_to_subjects.keys())
    print(f"[splits] unique ZuCo sentences: {len(unique)}", flush=True)

    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    train_cut, dev_cut = int(0.8 * n), int(0.9 * n)
    train_t = frozenset(_hash(s) for s in unique[:train_cut])
    dev_t = frozenset(_hash(s) for s in unique[train_cut:dev_cut])
    test_t = frozenset(_hash(s) for s in unique[dev_cut:])

    # Step 2: fold subjects. Pool is derived from the actual ``participant_id``
    # values observed in the parquets (matches what ``EEGSentenceDataset`` will
    # filter against at training time).
    all_subjects: set[str] = set()
    for subjects in text_to_subjects.values():
        all_subjects.update(subjects)
    pool = sorted(all_subjects)
    print(f"[splits] subject pool: {len(pool)} subjects: {pool}", flush=True)
    folds = []
    for i in range(n_folds):
        rng_fold = random.Random(seed + i)
        shuffled = pool[:]
        rng_fold.shuffle(shuffled)
        test_s = tuple(shuffled[:n_test])
        dev_s = tuple(shuffled[n_test : n_test + n_dev])
        train_s = tuple(s for s in pool if s not in test_s and s not in dev_s)
        folds.append(
            FoldSplit(
                fold=i,
                train_subjects=train_s,
                dev_subjects=dev_s,
                test_subjects=test_s,
                train_sent_hashes=train_t,
                dev_sent_hashes=dev_t,
                test_sent_hashes=test_t,
            )
        )
    return folds


def write_splits() -> None:
    """Persist folds to ``$EXP01_DATA_ROOT/splits/fold_*.json``."""
    storage.SPLITS.mkdir(parents=True, exist_ok=True)
    for fold in make_folds():
        path = storage.SPLITS / f"fold_{fold.fold}.json"
        path.write_text(
            json.dumps(
                {
                    "fold": fold.fold,
                    "train_subjects": list(fold.train_subjects),
                    "dev_subjects": list(fold.dev_subjects),
                    "test_subjects": list(fold.test_subjects),
                    "train_sent_hashes": sorted(fold.train_sent_hashes),
                    "dev_sent_hashes": sorted(fold.dev_sent_hashes),
                    "test_sent_hashes": sorted(fold.test_sent_hashes),
                },
                separators=(",", ":"),
            )
        )
        print(f"[splits] wrote {path}", flush=True)


def load_fold(fold: int) -> FoldSplit:
    raw = json.loads((storage.SPLITS / f"fold_{fold}.json").read_text())
    return FoldSplit(
        fold=raw["fold"],
        train_subjects=tuple(raw["train_subjects"]),
        dev_subjects=tuple(raw["dev_subjects"]),
        test_subjects=tuple(raw["test_subjects"]),
        train_sent_hashes=frozenset(raw["train_sent_hashes"]),
        dev_sent_hashes=frozenset(raw["dev_sent_hashes"]),
        test_sent_hashes=frozenset(raw["test_sent_hashes"]),
    )


# ============================================================================
# 3. Torch dataset + noise twin
# ============================================================================


class EEGSentenceDataset:
    """Streaming parquet reader. Returns dict rows; the encoder-specific
    collator (in ``encoders.py`` / ``model.py``) handles tensorisation,
    resampling, and channel layout.

    Args:
        sources: which ``dataset`` values to include (training spans all 8;
            eval is restricted to ZUCO_SOURCES upstream).
        subject_filter: only keep rows whose ``participant_id`` is in this set.
        sentence_filter: only keep rows whose normalised text hashes into this
            set (used to apply the Yin unique-sentence partition).
        noise: ``None`` for raw EEG, ``"gauss"`` for the Jo et al. noise twin
            (per-channel zero-mean Gaussian with the EEG's per-channel std).
    """

    def __init__(
        self,
        sources: Iterable[str] = ZUCO_SOURCES,
        subject_filter: Iterable[str] | None = None,
        sentence_filter: Iterable[str] | None = None,
        noise: str | None = None,
        eval_only: bool = False,
    ):
        import pyarrow.parquet as pq

        self.subject_filter = set(subject_filter) if subject_filter else None
        self.sentence_filter = set(sentence_filter) if sentence_filter else None
        self.noise = noise
        self.eval_only = eval_only

        files: list[Path] = []
        for src in sources:
            files.extend(shard_paths(src))
        self.files = files

        # Index = (file_idx, row_group_idx, row_in_group_idx) for every row
        # that survives the filters. Row-group bookkeeping matters because
        # ZuCo parquets sometimes have multiple row groups; doing
        # ``read_row_group(0)`` for every row was the source of the original
        # IndexError.
        self._index: list[tuple[int, int, int]] = []
        per_file_kept = []
        per_file_total = []
        for fi, f in enumerate(files):
            try:
                pf = pq.ParquetFile(f)
            except Exception as e:
                print(f"[data] WARN: failed to open {f.name}: {e}", flush=True)
                per_file_kept.append(0); per_file_total.append(0)
                continue
            kept = 0; total = 0
            for rg_idx in range(pf.num_row_groups):
                try:
                    t = pf.read_row_group(rg_idx, columns=["sentence_text", "participant_id"])
                except Exception as e:
                    print(f"[data] WARN: failed to read rg{rg_idx} of {f.name}: {e}", flush=True)
                    continue
                texts = t["sentence_text"].to_pylist()
                pids = t["participant_id"].to_pylist()
                total += len(texts)
                for ri_in_rg, (text, pid) in enumerate(zip(texts, pids)):
                    if self.subject_filter and str(pid) not in self.subject_filter:
                        continue
                    if self.sentence_filter and _hash(text or "") not in self.sentence_filter:
                        continue
                    self._index.append((fi, rg_idx, ri_in_rg))
                    kept += 1
            per_file_kept.append(kept); per_file_total.append(total)

        # Loud diagnostic: helps catch silent split / filter issues fast.
        n_files_with_data = sum(1 for k in per_file_kept if k > 0)
        print(
            f"[data] EEGSentenceDataset: {len(self._index)} rows kept from "
            f"{n_files_with_data}/{len(files)} files (total scanned={sum(per_file_total)}); "
            f"subject_filter={'set('+str(len(self.subject_filter))+')' if self.subject_filter else 'None'}, "
            f"sentence_filter={'set('+str(len(self.sentence_filter))+')' if self.sentence_filter else 'None'}, "
            f"noise={self.noise}",
            flush=True,
        )
        if not self._index:
            # Surface the first 3 source files' subject/sentence samples so
            # the user can see what the filter saw.
            for fi, f in enumerate(files[:3]):
                try:
                    pf = pq.ParquetFile(f)
                    t = pf.read_row_group(0, columns=["sentence_text", "participant_id"])
                    sample_pids = sorted({str(p) for p in t["participant_id"].to_pylist()})[:5]
                    sample_texts = [s for s in t["sentence_text"].to_pylist() if s][:2]
                    print(f"[data]   {f.name}: pids={sample_pids} texts={sample_texts}", flush=True)
                except Exception as e:
                    print(f"[data]   {f.name}: <{e}>", flush=True)

        self._cache: dict[int, "pq.ParquetFile"] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        import pyarrow.parquet as pq

        fi, rg_idx, ri_in_rg = self._index[i]
        if fi not in self._cache:
            self._cache[fi] = pq.ParquetFile(self.files[fi])
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

        eeg = _row_to_array(row)  # (channels, time) float32
        if self.noise:
            mu = eeg.mean(axis=1, keepdims=True)
            sd = eeg.std(axis=1, keepdims=True) + 1e-6
            rng = np.random.default_rng(seed=hash((row["participant_id"], row["sentence_text"])) & 0xFFFFFFFF)
            eeg = (rng.standard_normal(size=eeg.shape).astype("float32") * sd + mu).astype("float32")

        return {
            "eeg": eeg,
            "sr": float(row["sampling_rate_hz"]),
            "channels": list(row["channel_names"] or []),
            "text": str(row["sentence_text"]),
            "participant_id": str(row["participant_id"]),
            "dataset": str(row["dataset"]),
        }


def _row_to_array(row: dict):
    """Extract a (channels, time) float32 array from a unified-schema row.

    Prefers ``sentence_eeg`` (channels-first, full sentence) when available
    (ZuCo); falls back to concatenating ``word_eeg_segments`` along time.
    """
    import numpy as np

    if row.get("sentence_eeg"):
        # list[list[float]] of shape [channels][time]
        return np.asarray(row["sentence_eeg"], dtype="float32")

    segs = row.get("word_eeg_segments") or []
    arrs = []
    for w in segs:
        if not w:
            continue
        a = np.asarray(w, dtype="float32")  # (channels, time_w)
        if a.ndim == 2:
            arrs.append(a)
    if not arrs:
        # 1-channel zero placeholder; collator will skip.
        return np.zeros((1, 1), dtype="float32")
    # All segments share channels by construction; widths vary. Concatenate.
    return np.concatenate(arrs, axis=1)
