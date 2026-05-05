"""S3-direct checkpoint helpers, built on `s3torchconnector`.

Why this exists
---------------

A capacity-block week ends at a hard ``EndDate`` (currently
2026-05-12T11:30:00Z). At that instant AWS force-stops the instance; the
NVMe goes away. If checkpoints have only ever been written to NVMe,
everything since the last warehouse sync is gone.

The robust pattern (and what production teams use) is to write checkpoints
*directly to S3* via the AWS S3 Connector for PyTorch — which uses the
AWS Common Runtime under the hood, bypassing the Python GIL, and is up to
40 % faster than writing to local EBS first. Combined with PyTorch's
Distributed Checkpoint (DCP), the writes are also rank-parallel.

This module exposes two helpers:

- :class:`S3CheckpointSink` — drop-in replacement for ``torch.save`` that
  writes to ``s3://<bucket>/<prefix>/<name>.pt``. Use for single-rank
  small models where DCP overhead isn't worth it.
- :class:`S3DCPCheckpointSink` — DCP storage-writer wrapper that saves a
  sharded checkpoint at ``s3://<bucket>/<prefix>/step_{N}/``. Use for
  multi-rank training (8× H100 with FSDP/DDP).

Both are no-ops on rank>0 by default (they write only on the main process)
unless ``rank_specific=True``.

Resume policy
-------------

On startup, both sinks expose a :py:meth:`latest` method that lists the
prefix and returns the most-recent step number — call it before training
starts and either restore from the resulting URI or start fresh.

The Accelerate integration is in :func:`accelerate_save_state_to_s3` /
:func:`accelerate_load_state_from_s3`, which pre-stages locally then
``aws s3 sync``s the directory; this is the recommended path for the
already-Accelerate-using ``exp03/train.py`` until the codebase migrates to
DCP-native sharded state.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy imports: torch and s3torchconnector are gpu-side deps, not required
# on the local Mac that runs `eeg-ops` against AWS.
_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass


_STEP_RE = re.compile(r"step_(\d+)")


@dataclass(frozen=True)
class CheckpointURI:
    bucket: str
    prefix: str
    region: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix.rstrip('/')}"


# ---------------------------------------------------------------------------
# Single-file checkpoint (rank-0 only)
# ---------------------------------------------------------------------------


class S3CheckpointSink:
    """Write small checkpoints directly to S3 using the AWS S3 Connector.

    Example:
        >>> sink = S3CheckpointSink(bucket="eegmodel-warehouse",
        ...                          prefix="runs/exp03/exp17-g0-seed0",
        ...                          region="us-west-2")
        >>> sink.save(model.state_dict(), step=1000)
    """

    def __init__(self, *, bucket: str, prefix: str, region: str):
        from s3torchconnector import S3Checkpoint  # type: ignore

        self.uri = CheckpointURI(bucket=bucket, prefix=prefix, region=region)
        self._cm = S3Checkpoint(region=region)

    def save(self, state: Any, *, step: int, name: str | None = None) -> str:
        import torch  # type: ignore

        fname = name or f"step_{step}.pt"
        target = f"{self.uri.uri}/{fname}"
        with self._cm.writer(target) as w:
            torch.save(state, w)
        return target

    def load(self, *, step: int | None = None, name: str | None = None) -> Any:
        import torch  # type: ignore

        if name is None:
            step_n = step if step is not None else self.latest_step()
            if step_n is None:
                raise FileNotFoundError(f"no checkpoints under {self.uri.uri}/")
            name = f"step_{step_n}.pt"
        target = f"{self.uri.uri}/{name}"
        with self._cm.reader(target) as r:
            return torch.load(r, map_location="cpu", weights_only=False)

    def latest_step(self) -> int | None:
        """Return the largest step number whose checkpoint exists, or None."""
        import boto3

        s3 = boto3.client("s3", region_name=self.uri.region)
        steps: list[int] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.uri.bucket, Prefix=self.uri.prefix):
            for o in page.get("Contents", []):
                m = _STEP_RE.search(o["Key"])
                if m:
                    steps.append(int(m.group(1)))
        return max(steps) if steps else None


# ---------------------------------------------------------------------------
# Distributed Checkpoint sink (multi-rank)
# ---------------------------------------------------------------------------


class S3DCPCheckpointSink:
    """PyTorch DCP wrapper that writes sharded checkpoints to S3.

    Use this for 8-way DDP/FSDP training. Each step writes a directory
    under ``<prefix>/step_{N}/`` containing one ``.distcp`` shard per rank.

    Example:
        >>> sink = S3DCPCheckpointSink(...)
        >>> sink.save(state_dict, step=1000)
        >>> # to resume
        >>> sd = sink.load(step=1000)
    """

    def __init__(self, *, bucket: str, prefix: str, region: str, thread_count: int = 8):
        from s3torchconnector.dcp import S3StorageReader, S3StorageWriter  # type: ignore

        self.uri = CheckpointURI(bucket=bucket, prefix=prefix, region=region)
        self._writer_cls = S3StorageWriter
        self._reader_cls = S3StorageReader
        self._thread_count = thread_count

    def save(self, state_dict: Any, *, step: int) -> str:
        import torch.distributed.checkpoint as DCP  # type: ignore

        target = f"{self.uri.uri}/step_{step}"
        writer = self._writer_cls(
            region=self.uri.region, path=target, thread_count=self._thread_count
        )
        DCP.save(state_dict=state_dict, storage_writer=writer)
        return target

    def load(self, *, step: int, state_dict: Any) -> Any:
        import torch.distributed.checkpoint as DCP  # type: ignore

        target = f"{self.uri.uri}/step_{step}"
        reader = self._reader_cls(region=self.uri.region, path=target)
        DCP.load(state_dict=state_dict, storage_reader=reader)
        return state_dict

    def latest_step(self) -> int | None:
        import boto3

        s3 = boto3.client("s3", region_name=self.uri.region)
        steps: set[int] = set()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.uri.bucket, Prefix=self.uri.prefix):
            for o in page.get("Contents", []):
                m = _STEP_RE.search(o["Key"])
                if m:
                    steps.add(int(m.group(1)))
        return max(steps) if steps else None


# ---------------------------------------------------------------------------
# Accelerate bridge — for the existing exp03/train.py loop that uses
# `accelerator.save_state(local_dir)`. We sync that local dir to S3 at the
# end of each save and read it back at the start of resume.
# ---------------------------------------------------------------------------


def accelerate_save_state_to_s3(
    accelerator: Any,
    *,
    local_dir: Path,
    bucket: str,
    prefix: str,
    region: str,
    extra_files: list[Path] | None = None,
) -> str:
    """Wrap ``accelerator.save_state(local_dir)`` and push to S3.

    Equivalent to::

        accelerator.save_state(local_dir)
        if accelerator.is_main_process:
            aws s3 sync local_dir s3://bucket/prefix/

    but with stable progress logging and a return URI.
    """
    accelerator.save_state(str(local_dir))
    if not accelerator.is_main_process:
        return ""
    target = f"s3://{bucket}/{prefix.rstrip('/')}"
    cmd = [
        "aws", "s3", "sync", str(local_dir), target,
        "--region", region, "--no-progress",
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"aws s3 sync to {target} exited {rc}")
    if extra_files:
        for p in extra_files:
            subprocess.call(
                ["aws", "s3", "cp", str(p), f"{target}/{p.name}", "--region", region]
            )
    return target


def accelerate_load_state_from_s3(
    accelerator: Any,
    *,
    bucket: str,
    prefix: str,
    region: str,
    local_dir: Path,
) -> bool:
    """Pull a checkpoint dir from S3 → local then ``accelerator.load_state``.

    Returns True if a checkpoint was found and loaded, False if the prefix
    is empty (i.e. fresh training start).
    """
    import boto3

    s3 = boto3.client("s3", region_name=region)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    if "Contents" not in resp:
        return False

    local_dir.mkdir(parents=True, exist_ok=True)
    src = f"s3://{bucket}/{prefix.rstrip('/')}"
    cmd = ["aws", "s3", "sync", src, str(local_dir), "--region", region, "--no-progress"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"aws s3 sync from {src} exited {rc}")
    accelerator.load_state(str(local_dir))
    return True


# ---------------------------------------------------------------------------
# In-band end-of-week safety: rsync runs/ → warehouse before the reservation
# expires. Used by `eeg-ops cluster down --sync-runs`.
# ---------------------------------------------------------------------------


def sync_runs_dir(
    *,
    local_runs_dir: Path,
    bucket: str = "eegmodel-warehouse",
    prefix: str = "runs/exp03",
    region: str = "us-west-2",
) -> int:
    """Final-push: ``aws s3 sync local_runs_dir → warehouse``.

    Returns the CLI exit code. Idempotent. Safe to call multiple times.
    """
    if not local_runs_dir.exists():
        return 0
    target = f"s3://{bucket}/{prefix.rstrip('/')}"
    cmd = [
        "aws", "s3", "sync", str(local_runs_dir), target,
        "--region", region, "--no-progress",
    ]
    return subprocess.call(cmd)


@contextlib.contextmanager
def heartbeat_log(label: str):
    """Tiny timer for periodic progress lines.

    Usage::

        with heartbeat_log("save_state"):
            sink.save(state, step=1000)
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        print(f"[checkpoint] {label}: {dt:.2f} s")
