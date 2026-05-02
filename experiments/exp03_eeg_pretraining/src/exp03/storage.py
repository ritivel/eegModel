"""Filesystem + S3 paths for exp03 artifacts.

Layout (canonical):

    Local NVMe (ephemeral, fast working set on the GPU box):
        $EXP03_DATA_ROOT/                 — root, default /opt/dlami/nvme/eeg
        $EXP03_DATA_ROOT/raw/hbn/         — raw HBN .set/.fdt (downloaded once)
        $EXP03_DATA_ROOT/derived/
            hbn_minimal_500hz/            — minimum-offline parquet shards (PRIMARY)
            hbn_v2_clean_250hz/           — F0-prep cell only (notch + bandpass + 250 Hz)
        $EXP03_DATA_ROOT/runs/<exp>/<id>/ — checkpoints + metrics during training
        $EXP03_DATA_ROOT/models/          — HF cache (REVE / TFM / etc., later)

    S3 warehouse (persistent, cross-cloud):
        s3://eegmodel-warehouse/derived/hbn_minimal_500hz/        (mirror of above)
        s3://eegmodel-warehouse/derived/hbn_v2_clean_250hz/       (mirror of above)
        s3://eegmodel-warehouse/runs/exp03/<NN_name>/<run_id>/    (synced at job end)
        s3://eegmodel-warehouse/models/hf_cache/                  (mirror of above)

Raw HBN is *not* mirrored to our bucket — FCP-INDI's `s3://fcp-indi/...` is
the canonical NIH-funded source and will outlive our project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_WAREHOUSE_BUCKET = "eegmodel-warehouse"
S3_WAREHOUSE_REGION = "us-west-2"

# FCP-INDI HBN-EEG public bucket (NIH-funded, no credentials needed for read).
HBN_S3_BUCKET = "fcp-indi"
HBN_S3_PREFIX = "data/Projects/HBN/EEG"

# Derived-shard pipeline names (one folder per pipeline variant, so the same
# subject+task can have multiple parquet shards under different pipelines).
PIPELINE_MINIMAL = "hbn_minimal_500hz"        # primary, per mini_experiments.md §4.1
PIPELINE_V2_CLEAN = "hbn_v2_clean_250hz"      # F0-prep literature-comparability cell only

DEFAULT_DATA_ROOT = "/opt/dlami/nvme/eeg"     # GPU-box NVMe scratch; override via env on other machines


# ---------------------------------------------------------------------------
# Storage class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Storage:
    """All on-disk paths for one machine + S3 prefixes for the warehouse.

    Construct with :func:`from_env` so the data root comes from
    ``$EXP03_DATA_ROOT`` and falls back to ``DEFAULT_DATA_ROOT``. Pass an
    explicit ``data_root`` only for unit tests.
    """

    data_root: Path
    s3_bucket: str = S3_WAREHOUSE_BUCKET
    s3_region: str = S3_WAREHOUSE_REGION

    # --- local paths -------------------------------------------------------

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def raw_hbn(self) -> Path:
        return self.raw_root / "hbn"

    @property
    def derived_root(self) -> Path:
        return self.data_root / "derived"

    def derived_pipeline(self, pipeline: str) -> Path:
        """e.g. derived/hbn_minimal_500hz/"""
        return self.derived_root / pipeline

    @property
    def runs_root(self) -> Path:
        return self.data_root / "runs"

    def run_dir(self, experiment: str, run_id: str) -> Path:
        """e.g. runs/01_sanity_baselines/2026-05-02T12-00-00_xyz/"""
        p = self.runs_root / experiment / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def hf_cache(self) -> Path:
        return self.models_dir / "hf_cache"

    @property
    def scratch(self) -> Path:
        return self.data_root / "scratch"

    # --- S3 prefixes (no client; just URI builders) ------------------------

    def s3_uri(self, *parts: str) -> str:
        suffix = "/".join(p.strip("/") for p in parts if p)
        return f"s3://{self.s3_bucket}/{suffix}"

    def s3_derived(self, pipeline: str) -> str:
        return self.s3_uri("derived", pipeline)

    def s3_run(self, experiment: str, run_id: str) -> str:
        return self.s3_uri("runs", "exp03", experiment, run_id)

    @property
    def s3_models(self) -> str:
        return self.s3_uri("models", "hf_cache")

    # --- mkdir all the things -----------------------------------------------

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_hbn,
            self.derived_pipeline(PIPELINE_MINIMAL),
            self.derived_pipeline(PIPELINE_V2_CLEAN),
            self.runs_root,
            self.hf_cache,
            self.scratch,
        ):
            p.mkdir(parents=True, exist_ok=True)


def from_env(env_var: str = "EXP03_DATA_ROOT", default: str = DEFAULT_DATA_ROOT) -> Storage:
    """Construct a Storage rooted at $EXP03_DATA_ROOT (or default).

    Lookup happens at call time, not import time, so tests / scripts can set
    the env var before constructing.
    """
    raw = os.environ.get(env_var, default)
    return Storage(data_root=Path(raw).expanduser().resolve())
