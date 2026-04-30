"""Yin et al. (2024) unique-sentence + leave-N-subjects-out fold builder.

Persisted to ``<storage.splits>/fold_<n>.json`` and re-loaded by every
training and eval entry-point. Same fold definitions are reused across
exp01 and exp02 so cross-experiment cell pairs are directly comparable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .storage import Storage


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_subjects: tuple[str, ...]
    dev_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_sent_hashes: frozenset
    dev_sent_hashes: frozenset
    test_sent_hashes: frozenset


def normalise(text: str) -> str:
    return " ".join(text.lower().strip().split())


def sent_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()[:16]


def make_folds(
    storage: Storage,
    *,
    n_folds: int = 5,
    n_test: int = 3,
    n_dev: int = 3,
    seed: int = 20260430,
    include_non_zuco_in_train_pool: bool = True,
) -> list[FoldSplit]:
    """Compute LNSO subject folds + Yin unique-sentence assignment.

    The unique-sentence pool is split 80 / 10 / 10 once over ``ZUCO_SOURCES``;
    the same text-level partition is reused across folds so cross-fold
    variance is purely from *which subjects* are held out, not *which
    sentences*. Test subjects are restricted to ZuCo (per Yin et al. 2024 /
    our §4 protocol).

    Bug-fix: previously the *training-subject pool* was also restricted to
    ZuCo subjects, which silently dropped DERCo / EMMT / eeg_sem_relev
    participants. With ``include_non_zuco_in_train_pool=True`` (the default)
    we add those subjects to ``train_subjects`` as well so the row-filter
    step in ``EEGSentenceDataset`` actually keeps them.
    """
    import random
    import pyarrow.parquet as pq

    from .data import ALL_SOURCES, ZUCO_SOURCES, shard_paths

    storage.ensure_dirs()

    text_to_subjects: dict[str, set[str]] = {}
    n_total_files = sum(len(shard_paths(storage, src)) for src in ZUCO_SOURCES)
    print(f"[splits] scanning {n_total_files} ZuCo shards (sentence_text + participant_id)",
          flush=True)
    seen = 0
    for src in ZUCO_SOURCES:
        for path in shard_paths(storage, src):
            t0 = time.time()
            t = pq.read_table(path, columns=["sentence_text", "participant_id"])
            for s, p in zip(t["sentence_text"].to_pylist(), t["participant_id"].to_pylist()):
                if not s:
                    continue
                text_to_subjects.setdefault(normalise(s), set()).add(str(p))
            seen += 1
            print(
                f"[splits]   ({seen}/{n_total_files}) {path.name}: "
                f"{len(t)} rows in {(time.time() - t0) * 1000:.0f}ms "
                f"(unique-so-far={len(text_to_subjects)})",
                flush=True,
            )

    unique = sorted(text_to_subjects.keys())
    print(f"[splits] unique ZuCo sentences: {len(unique)}", flush=True)

    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    train_cut, dev_cut = int(0.8 * n), int(0.9 * n)
    train_t = frozenset(sent_hash(s) for s in unique[:train_cut])
    dev_t = frozenset(sent_hash(s) for s in unique[train_cut:dev_cut])
    test_t = frozenset(sent_hash(s) for s in unique[dev_cut:])

    zuco_pool: set[str] = set()
    for subjects in text_to_subjects.values():
        zuco_pool.update(subjects)
    zuco_pool_sorted = sorted(zuco_pool)
    print(f"[splits] ZuCo subject pool: {len(zuco_pool_sorted)} subjects: {zuco_pool_sorted}",
          flush=True)

    non_zuco_subjects: set[str] = set()
    if include_non_zuco_in_train_pool:
        for src in ALL_SOURCES:
            if src in ZUCO_SOURCES:
                continue
            for path in shard_paths(storage, src):
                try:
                    t = pq.read_table(path, columns=["participant_id"])
                except Exception as e:
                    print(f"[splits] WARN: could not read {path.name}: {e}", flush=True)
                    continue
                for p in t["participant_id"].to_pylist():
                    if p:
                        non_zuco_subjects.add(str(p))
        print(f"[splits] non-ZuCo (train-only) subjects: {len(non_zuco_subjects)}",
              flush=True)

    folds: list[FoldSplit] = []
    for i in range(n_folds):
        rng_fold = random.Random(seed + i)
        shuffled = zuco_pool_sorted[:]
        rng_fold.shuffle(shuffled)
        test_s = tuple(shuffled[:n_test])
        dev_s = tuple(shuffled[n_test: n_test + n_dev])
        train_zuco = [s for s in zuco_pool_sorted if s not in test_s and s not in dev_s]
        train_s = tuple(train_zuco) + tuple(sorted(non_zuco_subjects))
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


def write_splits(storage: Storage) -> None:
    """Persist folds to ``<storage.splits>/fold_*.json``."""
    storage.splits.mkdir(parents=True, exist_ok=True)
    for fold in make_folds(storage):
        path = storage.splits / f"fold_{fold.fold}.json"
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


def load_fold(storage: Storage, fold: int) -> FoldSplit:
    raw = json.loads((storage.splits / f"fold_{fold}.json").read_text())
    return FoldSplit(
        fold=raw["fold"],
        train_subjects=tuple(raw["train_subjects"]),
        dev_subjects=tuple(raw["dev_subjects"]),
        test_subjects=tuple(raw["test_subjects"]),
        train_sent_hashes=frozenset(raw["train_sent_hashes"]),
        dev_sent_hashes=frozenset(raw["dev_sent_hashes"]),
        test_sent_hashes=frozenset(raw["test_sent_hashes"]),
    )
