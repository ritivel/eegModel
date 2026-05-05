"""Thin wrappers around the SkyPilot Python API and CLI.

We use the CLI for ``sky launch`` / ``sky exec`` / ``sky down`` (subprocess) so
we get the standard SkyPilot streaming output verbatim — no need to
reimplement progress bars. We use the Python API only for declarative things
like patching ``~/.sky/config.yaml`` to register a capacity reservation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ruamel.yaml import YAML


SKY_CONFIG_PATH = Path.home() / ".sky" / "config.yaml"


def register_capacity_reservation(reservation_id: str) -> None:
    """Add ``reservation_id`` to ``aws.specific_reservations`` in
    ``~/.sky/config.yaml``. Idempotent.

    SkyPilot reads this file at launch time; once the reservation ID is in
    the list, ``sky launch`` automatically routes the request into the
    reserved capacity (zero-cost path in the optimizer).
    See https://docs.skypilot.co/en/stable/reservations/reservations.html.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    if SKY_CONFIG_PATH.exists():
        cfg = yaml.load(SKY_CONFIG_PATH) or {}
    else:
        cfg = {}
    aws_cfg = cfg.setdefault("aws", {})
    reservations = aws_cfg.setdefault("specific_reservations", [])
    if reservation_id not in reservations:
        reservations.append(reservation_id)
    # NOTE: deliberately do NOT set ``aws.prioritize_reservations = True``.
    # That flag tells the optimizer to scan every AWS region globally
    # looking for "open" reservations to consume. Our capacity blocks are
    # always ``targeted``, and ``specific_reservations`` plus a region-pinned
    # ``infra:`` in the task YAML is sufficient for SkyPilot to route the
    # launch into the reserved capacity at zero on-demand cost. Setting
    # ``prioritize_reservations`` triggers a 5+ min ``ec2.<region>`` probe
    # that fails on regions whose endpoint is unreachable from this network
    # (e.g. me-south-1).
    aws_cfg.pop("prioritize_reservations", None)
    SKY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(cfg, SKY_CONFIG_PATH)


def set_remote_identity(role_arn_or_name: str) -> None:
    """Set ``aws.remote_identity`` in ``~/.sky/config.yaml`` so launched
    instances assume the IAM role for S3/CloudWatch access.

    Either an ARN (``arn:aws:iam::123:role/Foo``) or a bare role name works;
    SkyPilot resolves both. Idempotent.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    if SKY_CONFIG_PATH.exists():
        cfg = yaml.load(SKY_CONFIG_PATH) or {}
    else:
        cfg = {}
    cfg.setdefault("aws", {})["remote_identity"] = role_arn_or_name
    SKY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(cfg, SKY_CONFIG_PATH)


def unregister_capacity_reservation(reservation_id: str) -> None:
    """Remove ``reservation_id`` from the SkyPilot config (run after teardown
    so an expired ID doesn't slow down future ``sky launch`` calls)."""
    if not SKY_CONFIG_PATH.exists():
        return
    yaml = YAML()
    cfg = yaml.load(SKY_CONFIG_PATH) or {}
    res = cfg.get("aws", {}).get("specific_reservations", [])
    if reservation_id in res:
        res.remove(reservation_id)
        yaml.dump(cfg, SKY_CONFIG_PATH)


def launch(
    *,
    cluster_name: str,
    yaml_path: Path,
    retry_until_up: bool = True,
    extra_envs: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run ``sky launch`` for the given task YAML; returns the CLI exit code."""
    args = ["sky", "launch", "-c", cluster_name, "-y"]
    if retry_until_up:
        args.append("--retry-until-up")
    for k, v in (extra_envs or {}).items():
        args.extend(["--env", f"{k}={v}"])
    if extra_args:
        args.extend(extra_args)
    args.append(str(yaml_path))
    return subprocess.call(args, env={**os.environ})


def exec_remote(*, cluster_name: str, command: str) -> int:
    return subprocess.call(["sky", "exec", cluster_name, command], env={**os.environ})


def status(*, cluster_name: str | None = None) -> int:
    args = ["sky", "status"]
    if cluster_name:
        args.append(cluster_name)
    return subprocess.call(args, env={**os.environ})


def down(*, cluster_name: str, purge: bool = False) -> int:
    args = ["sky", "down", cluster_name, "-y"]
    if purge:
        args.append("--purge")
    return subprocess.call(args, env={**os.environ})
