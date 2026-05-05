"""State-file round-tripping and region-config lookup smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from eeg_ops.config import (
    CapacityState,
    State,
    region_config,
    WAREHOUSE_BUCKET,
)


def test_region_config_known_region_returns_dlami_and_subnet():
    rc = region_config("ap-south-1")
    assert rc.region == "ap-south-1"
    assert rc.az == "ap-south-1a"
    assert rc.dlami_pytorch_ami.startswith("ami-")
    assert rc.subnet_id.startswith("subnet-")
    assert rc.cache_bucket
    assert rc.cache_bucket != WAREHOUSE_BUCKET


def test_region_config_unknown_region_raises():
    with pytest.raises(KeyError):
        region_config("antarctica-1")


def test_state_round_trip_empty(tmp_path: Path):
    p = tmp_path / "state.toml"
    s = State()
    s.save(p)
    s2 = State.load(p)
    assert s2.active_reservation_id is None
    assert s2.capacities == {}


def test_state_round_trip_with_capacity(tmp_path: Path):
    p = tmp_path / "state.toml"
    s = State()
    cap = CapacityState(
        reservation_id="cr-aaaa",
        offering_id="cb-bbbb",
        region="ap-south-1",
        az="ap-south-1a",
        instance_type="p5.48xlarge",
        duration_hours=168,
        upfront_fee_usd=5285.95,
        start_date="2026-05-05T11:30:00+00:00",
        end_date="2026-05-12T11:30:00+00:00",
        state="active",
    )
    s.upsert_capacity(cap)
    s.save(p)

    s2 = State.load(p)
    assert s2.active_reservation_id == "cr-aaaa"
    assert "cr-aaaa" in s2.capacities
    loaded = s2.capacities["cr-aaaa"]
    assert loaded.upfront_fee_usd == pytest.approx(5285.95)
    assert loaded.region == "ap-south-1"
    assert loaded.duration_hours == 168


def test_state_get_active_returns_record_or_none(tmp_path: Path):
    s = State()
    assert s.get_active() is None

    cap = CapacityState(
        reservation_id="cr-x", offering_id="cb-x", region="ap-south-1",
        az="ap-south-1a", instance_type="p5.48xlarge",
        duration_hours=168, upfront_fee_usd=1.0,
        start_date="x", end_date="y",
    )
    s.upsert_capacity(cap)
    got = s.get_active()
    assert got is not None and got.reservation_id == "cr-x"
