"""Local single-box storage layout.

Everything lives under ``$EXP01_DATA_ROOT`` (default ``./data``):

  $EXP01_DATA_ROOT/
    hf/                          HuggingFace cache (models + dataset)
    splits/                      per-fold split JSONs
    runs/<cell_id>/              live training checkpoints + jsonl logs +
                                 sample_gens.jsonl, stats.jsonl
    eval/<cell_id>/              metrics.json + predictions.parquet
    wandb/                       wandb local run dirs (synced to cloud)
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("EXP01_DATA_ROOT", "./data")).expanduser().resolve()

HF_CACHE = DATA_ROOT / "hf"
WANDB_DIR = DATA_ROOT / "wandb"
SPLITS = DATA_ROOT / "splits"
RUNS = DATA_ROOT / "runs"
EVAL = DATA_ROOT / "eval"


def ensure_dirs() -> None:
    for p in (HF_CACHE, WANDB_DIR, SPLITS, RUNS, EVAL):
        p.mkdir(parents=True, exist_ok=True)


def cell_run_dir(cell_id: str) -> Path:
    p = RUNS / cell_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def cell_eval_dir(cell_id: str) -> Path:
    p = EVAL / cell_id
    p.mkdir(parents=True, exist_ok=True)
    return p
