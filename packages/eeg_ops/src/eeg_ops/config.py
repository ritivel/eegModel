"""Resolved per-region config for the cluster-lifecycle CLI.

Two layers:

* :class:`RegionConfig` — static defaults baked in by the package authors
  (AMI IDs, default subnets, bucket names per region, expected GPU type).
  These are knowable once per region and rarely change. Looked up by
  :func:`region_config`.
* :class:`State` — mutable state across CLI invocations, persisted at
  ``~/.config/eeg-ops/state.toml``. This is where `eeg-ops capacity buy`
  stashes the resulting CapacityReservationId so `eeg-ops cluster up` can
  read it without re-typing.

The state file is intentionally a TOML doc (not JSON) so the user can
hand-edit and read it. We never put secrets in it; secrets stay in
``$WANDB_API_KEY`` / ``$HF_TOKEN`` / the AWS credential chain.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# Default state-file location. `XDG_CONFIG_HOME` is honored.
_XDG = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
DEFAULT_STATE_PATH = _XDG / "eeg-ops" / "state.toml"


# ---------------------------------------------------------------------------
# Static per-region defaults
# ---------------------------------------------------------------------------

# Canonical warehouse — single bucket, single region; we mirror from here
# into the regional cache for whichever region we're renting GPUs in.
WAREHOUSE_BUCKET = "eegmodel-warehouse"
WAREHOUSE_REGION = "us-west-2"


@dataclass(frozen=True)
class RegionConfig:
    """All knobs that depend only on which AWS region we're operating in."""

    region: str
    az: str
    """Default availability zone for the capacity block. Within ``ap-south-1``,
    AWS only sells P5 capacity blocks in ``ap-south-1a`` today (May 2026)."""

    subnet_id: str
    """Default-for-AZ subnet ID. We hard-code this rather than discover at
    runtime so two engineers running the CLI side-by-side don't end up with
    different subnets and confused logs. Override via :func:`region_config`
    or by editing this map when you bring up a new account."""

    vpc_id: str

    cache_bucket: str
    """In-region S3 bucket name for the preprocessed-shard mirror. Created
    lazily by ``eeg-ops data prewarm``."""

    dlami_pytorch_ami: str
    """Most-recent AWS Deep Learning AMI (PyTorch 2.7, Ubuntu 22.04) ID for
    the region. Re-resolve periodically via:
        aws ec2 describe-images --owners amazon \\
          --filters Name=name,Values='Deep Learning OSS Nvidia Driver AMI GPU PyTorch* (Ubuntu 22.04)*'
    """

    instance_profile_name: str = "eeg-ops-instance-profile"
    """Created by ``eeg-ops iam create`` — gives the box S3 read-write to both
    cache and warehouse plus CloudWatch logs/metrics."""


# Authoritative table. Add a row when you start using a new region.
_REGION_TABLE: dict[str, RegionConfig] = {
    "ap-south-1": RegionConfig(
        region="ap-south-1",
        az="ap-south-1a",
        subnet_id="subnet-062d8f2ac72af66c1",
        vpc_id="vpc-03b70f34f216ded41",
        cache_bucket="eeg-mumbai-139156132535",
        dlami_pytorch_ami="ami-0eb5a4f0d81f671e3",
    ),
}


def region_config(region: str) -> RegionConfig:
    """Look up the static config for an AWS region we know about."""
    if region not in _REGION_TABLE:
        raise KeyError(
            f"unknown region {region!r}; add a RegionConfig entry to "
            f"eeg_ops.config._REGION_TABLE first"
        )
    return _REGION_TABLE[region]


# ---------------------------------------------------------------------------
# Mutable state across CLI calls
# ---------------------------------------------------------------------------


@dataclass
class CapacityState:
    """One-row record for a purchased reservation. Multiple may be present
    in the state file; the active one is keyed by ``reservation_id``."""

    reservation_id: str
    offering_id: str
    region: str
    az: str
    instance_type: str
    duration_hours: int
    upfront_fee_usd: float
    start_date: str  # ISO 8601 UTC; AWS returns "+00:00" suffix, we keep that
    end_date: str
    state: str = "payment-pending"


@dataclass
class State:
    """Top-level state container persisted at ``~/.config/eeg-ops/state.toml``.

    Empty defaults make the file optional — first call to a CLI command can
    create it on demand.
    """

    aws_account_id: str | None = None
    active_reservation_id: str | None = None
    """Which reservation `eeg-ops cluster up` will target."""

    cluster_name: str | None = None
    """The current SkyPilot cluster name (if any)."""

    notion_chat_session_id: str | None = None
    """Page ID of the currently-open Notion chat Session row, if the AI
    assistant has opened one. Set by ``eeg-ops session open --type chat``."""

    notion_rental_session_id: str | None = None
    """Page ID of the currently-open Notion gpu_rental Session row, if any.
    Auto-set by ``eeg-ops capacity buy`` and cleared by ``cluster down``."""

    capacities: dict[str, CapacityState] = field(default_factory=dict)
    """All known reservations by ID. Lifecycle is append-only; we never delete
    rows, just mark expired ones via state='expired'."""

    @classmethod
    def load(cls, path: Path | None = None) -> "State":
        """Load state from disk, or return empty defaults if the file is missing."""
        path = path or DEFAULT_STATE_PATH
        if not path.exists():
            return cls()
        raw = tomllib.loads(path.read_text())
        caps_raw = raw.pop("capacities", {})
        caps = {k: CapacityState(**v) for k, v in caps_raw.items()}
        return cls(**raw, capacities=caps)

    def save(self, path: Path | None = None) -> None:
        """Atomic write of the state file (creates parent dirs as needed)."""
        path = path or DEFAULT_STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # We avoid the `tomli_w`/`tomllib` dependency split by hand-writing.
        # The schema is small and stable; tests pin the format.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(_render_toml(self))
        tmp.replace(path)

    # ---- ergonomic helpers -------------------------------------------------

    def get_active(self) -> CapacityState | None:
        if self.active_reservation_id is None:
            return None
        return self.capacities.get(self.active_reservation_id)

    def upsert_capacity(self, c: CapacityState, *, set_active: bool = True) -> None:
        self.capacities[c.reservation_id] = c
        if set_active:
            self.active_reservation_id = c.reservation_id


def _render_toml(s: State) -> str:
    """Tiny hand-rolled TOML writer — covers exactly our schema.

    We don't use ``tomli_w`` to avoid a runtime dep for one writer call.
    """
    lines: list[str] = ["# Managed by eeg-ops; safe to hand-edit.", ""]
    if s.aws_account_id:
        lines.append(f'aws_account_id = "{s.aws_account_id}"')
    if s.active_reservation_id:
        lines.append(f'active_reservation_id = "{s.active_reservation_id}"')
    if s.cluster_name:
        lines.append(f'cluster_name = "{s.cluster_name}"')
    if s.notion_chat_session_id:
        lines.append(f'notion_chat_session_id = "{s.notion_chat_session_id}"')
    if s.notion_rental_session_id:
        lines.append(f'notion_rental_session_id = "{s.notion_rental_session_id}"')
    lines.append("")
    for cid, c in sorted(s.capacities.items()):
        lines.append(f"[capacities.{cid}]")
        for k, v in asdict(c).items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif v is None:
                continue
            else:
                raise TypeError(f"can't serialize {k}={v!r} in state.toml")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# Convenience: replace bits of a State (functional update for tests).
def _with(s: State, **kwargs: Any) -> State:
    return replace(s, **kwargs)
