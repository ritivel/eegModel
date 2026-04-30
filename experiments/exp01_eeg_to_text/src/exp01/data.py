"""exp01 data — thin wrappers around :mod:`eeg_common.data` and
:mod:`eeg_common.splits` that bind the exp01 storage so callers don't have
to pass it explicitly.

Re-exports module-level constants (``ZUCO_SOURCES``, ``ALL_SOURCES``,
``DATASET_REPO``) and a ``FoldSplit`` alias so existing call-sites keep
working unchanged.
"""

from __future__ import annotations

from typing import Iterable

from eeg_common import data as _data
from eeg_common import splits as _splits
from eeg_common.data import (
    ALL_SOURCES,
    DATASET_REPO,
    ZUCO_SOURCES,
)
from eeg_common.splits import FoldSplit

from . import preprocessing  # noqa: F401  — re-exported so exp01.data.preprocessing still resolves
from . import storage


__all__ = [
    "ALL_SOURCES",
    "DATASET_REPO",
    "ZUCO_SOURCES",
    "FoldSplit",
    "EEGSentenceDataset",
    "download_dataset",
    "load_fold",
    "make_folds",
    "shard_paths",
    "write_splits",
    "_hf_dataset_snapshots_dir",  # used by exp01.cli inspect-channels
]


# -----------------------------------------------------------------------------
# Storage-bound wrappers
# -----------------------------------------------------------------------------


def shard_paths(source: str):
    return _data.shard_paths(storage.STORAGE, source)


def _hf_dataset_snapshots_dir():
    return _data._hf_dataset_snapshots_dir(storage.STORAGE)


def download_dataset() -> None:
    _data.download_dataset(storage.STORAGE)


def make_folds(**kwargs) -> list[FoldSplit]:
    return _splits.make_folds(storage.STORAGE, **kwargs)


def write_splits() -> None:
    _splits.write_splits(storage.STORAGE)


def load_fold(fold: int) -> FoldSplit:
    return _splits.load_fold(storage.STORAGE, fold)


class EEGSentenceDataset(_data.EEGSentenceDataset):
    """Storage-bound subclass: binds ``storage.STORAGE`` so exp01 callers
    don't have to pass it explicitly.
    """

    def __init__(
        self,
        sources: Iterable[str] = ZUCO_SOURCES,
        subject_filter: Iterable[str] | None = None,
        sentence_filter: Iterable[str] | None = None,
        noise: str | None = None,
        eval_only: bool = False,
        preprocess=None,
        specaugment=None,
    ):
        super().__init__(
            storage.STORAGE,
            sources=sources,
            subject_filter=subject_filter,
            sentence_filter=sentence_filter,
            noise=noise,
            eval_only=eval_only,
            preprocess=preprocess,
            specaugment=specaugment,
        )
