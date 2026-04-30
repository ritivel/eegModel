"""Filesystem paths for one experiment's artifacts.

Each experiment binds its own ``Storage`` instance from its own env var (e.g.
``EXP01_DATA_ROOT`` for exp01, ``EXP02_DATA_ROOT`` for exp02). The shared
modules in ``eeg_common`` accept a ``Storage`` argument anywhere they need to
hit disk so two experiments can coexist in the same Python process.

Layout under ``<data_root>``::

    hf/        HuggingFace cache (model weights + dataset parquets)
    splits/    fold JSONs (Yin et al. unique-sentence + LNSO)
    runs/      training checkpoints + jsonl logs
    eval/      eval metrics + predictions parquet
    wandb/     local wandb run dirs (synced to cloud)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Storage:
    """All on-disk paths for one experiment.

    Construct with :func:`from_env` or pass an explicit ``data_root`` if you
    want a one-off Storage for a unit test.
    """

    data_root: Path

    @property
    def hf_cache(self) -> Path:
        return self.data_root / "hf"

    @property
    def splits(self) -> Path:
        return self.data_root / "splits"

    @property
    def runs(self) -> Path:
        return self.data_root / "runs"

    @property
    def eval(self) -> Path:
        return self.data_root / "eval"

    @property
    def wandb_dir(self) -> Path:
        return self.data_root / "wandb"

    def ensure_dirs(self) -> None:
        for p in (self.hf_cache, self.splits, self.runs, self.eval, self.wandb_dir):
            p.mkdir(parents=True, exist_ok=True)

    def cell_run_dir(self, cell_id: str) -> Path:
        p = self.runs / cell_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cell_eval_dir(self, cell_id: str) -> Path:
        p = self.eval / cell_id
        p.mkdir(parents=True, exist_ok=True)
        return p


def from_env(env_var: str, default: str = "./data") -> Storage:
    """Build a ``Storage`` rooted at ``$<env_var>`` (or ``default`` if unset).

    The env-var lookup is at *call* time, not import time, so a CLI can
    ``os.environ["EXP02_DATA_ROOT"] = ...`` before constructing its storage.
    """
    raw = os.environ.get(env_var, default)
    return Storage(Path(raw).expanduser().resolve())
