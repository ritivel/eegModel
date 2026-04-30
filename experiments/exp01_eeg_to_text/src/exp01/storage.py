"""Local single-box storage layout (exp01).

Thin wrapper around :mod:`eeg_common.storage` that binds an :class:`Storage`
instance to ``$EXP01_DATA_ROOT`` (default ``./data``) and re-exports the
old module-level constants and helpers so existing call-sites keep working.

Layout::

    $EXP01_DATA_ROOT/
      hf/                          HuggingFace cache (models + dataset)
      splits/                      per-fold split JSONs
      runs/<cell_id>/              live training checkpoints + jsonl logs
      eval/<cell_id>/              metrics.json + predictions.parquet
      wandb/                       wandb local run dirs (synced to cloud)
"""

from __future__ import annotations

from pathlib import Path

from eeg_common import storage as _common
from eeg_common.storage import Storage

STORAGE: Storage = _common.from_env("EXP01_DATA_ROOT")

DATA_ROOT: Path = STORAGE.data_root
HF_CACHE: Path = STORAGE.hf_cache
WANDB_DIR: Path = STORAGE.wandb_dir
SPLITS: Path = STORAGE.splits
RUNS: Path = STORAGE.runs
EVAL: Path = STORAGE.eval


def ensure_dirs() -> None:
    STORAGE.ensure_dirs()


def cell_run_dir(cell_id: str) -> Path:
    return STORAGE.cell_run_dir(cell_id)


def cell_eval_dir(cell_id: str) -> Path:
    return STORAGE.cell_eval_dir(cell_id)
