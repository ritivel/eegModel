"""Shared building blocks for the eegModel experiments.

The light-weight modules (storage, preprocessing, splits, data) are re-exported
at the package root for convenience; the torch-dependent ``encoders`` module is
NOT eagerly imported so light-touch introspection (e.g. fold builders, CLI
metadata) doesn't pay the torch import cost.

Heavy imports::

    from eeg_common.encoders import REVEEncoder, TFMEncoder, load_encoder
"""

from .storage import Storage, from_env
from .preprocessing import (
    PIPELINES,
    PreprocessSpec,
    V1_NOOP,
    V2_DK25,
    V2_REVE,
    V2_TFM,
    for_encoder,
    specaugment,
)
from .splits import FoldSplit, load_fold, make_folds, write_splits
from .data import (
    ALL_SOURCES,
    DATASET_REPO,
    EEGSentenceDataset,
    ZUCO_SOURCES,
    download_dataset,
    shard_paths,
)

__all__ = [
    "Storage",
    "from_env",
    "PreprocessSpec",
    "PIPELINES",
    "V1_NOOP",
    "V2_REVE",
    "V2_TFM",
    "V2_DK25",
    "for_encoder",
    "specaugment",
    "FoldSplit",
    "make_folds",
    "write_splits",
    "load_fold",
    "DATASET_REPO",
    "ZUCO_SOURCES",
    "ALL_SOURCES",
    "EEGSentenceDataset",
    "download_dataset",
    "shard_paths",
]
