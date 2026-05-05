"""Cross-region S3 mirror replication.

The warehouse lives in ``us-west-2``; the GPU box lives in whichever region
we're renting. We mirror ``s3://<warehouse>/derived/`` into
``s3://<region-cache>/derived/`` once at the start of the rental so the
GPU box pulls in-region (~5 min instead of ~2-4 h cross-region).

Implementation: ``aws s3 sync`` server-side, idempotent (skips matching
ETags), one prefix at a time so log lines stay attributable.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import boto3
from botocore.exceptions import ClientError

from .config import RegionConfig, WAREHOUSE_BUCKET, WAREHOUSE_REGION

DEFAULT_PIPELINES: tuple[str, ...] = (
    "hbn_minimal_500hz",
    "hbn_v2_clean_250hz",
    "tuab_v2_clean_250hz",
    "tuev_v2_clean_250hz",
)


def ensure_cache_bucket(*, region_cfg: RegionConfig) -> str:
    """Create the regional cache bucket if it doesn't exist. Idempotent."""
    s3 = boto3.client("s3", region_name=region_cfg.region)
    bucket = region_cfg.cache_bucket
    try:
        s3.head_bucket(Bucket=bucket)
        return bucket
    except ClientError as e:
        if e.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
            raise
    create_kwargs: dict = {"Bucket": bucket}
    if region_cfg.region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {
            "LocationConstraint": region_cfg.region
        }
    s3.create_bucket(**create_kwargs)
    s3.put_bucket_versioning(
        Bucket=bucket, VersioningConfiguration={"Status": "Enabled"}
    )
    return bucket


def sync_pipelines(
    *,
    region_cfg: RegionConfig,
    pipelines: Sequence[str] = DEFAULT_PIPELINES,
    dry_run: bool = False,
) -> int:
    """``aws s3 sync`` each pipeline prefix from warehouse → cache.

    Returns the largest non-zero exit code across pipelines (0 = all OK).
    """
    worst = 0
    for pl in pipelines:
        src = f"s3://{WAREHOUSE_BUCKET}/derived/{pl}/"
        dst = f"s3://{region_cfg.cache_bucket}/derived/{pl}/"
        cmd = [
            "aws", "s3", "sync", src, dst,
            "--source-region", WAREHOUSE_REGION,
            "--region", region_cfg.region,
            "--no-progress",
        ]
        if dry_run:
            cmd.append("--dryrun")
        rc = subprocess.call(cmd)
        worst = max(worst, rc)
    return worst


def measure_cache_size(*, region_cfg: RegionConfig) -> tuple[int, int]:
    """Return ``(n_objects, total_bytes)`` for the regional cache bucket."""
    s3 = boto3.client("s3", region_name=region_cfg.region)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    size = 0
    for page in paginator.paginate(Bucket=region_cfg.cache_bucket):
        for o in page.get("Contents", []):
            n += 1
            size += o["Size"]
    return n, size
