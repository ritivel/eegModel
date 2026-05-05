"""SkyPilot config patcher unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from eeg_ops import sky as sky_mod


@pytest.fixture
def isolated_sky_config(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(sky_mod, "SKY_CONFIG_PATH", p)
    return p


def test_register_capacity_reservation_creates_file(isolated_sky_config: Path):
    sky_mod.register_capacity_reservation("cr-123")
    cfg = YAML().load(isolated_sky_config.read_text())
    assert cfg["aws"]["specific_reservations"] == ["cr-123"]
    # Deliberately NOT set: targeted CRs don't need (and break on) the
    # global region scan that prioritize_reservations triggers.
    assert "prioritize_reservations" not in cfg["aws"]


def test_register_capacity_reservation_strips_prioritize_if_present(
    isolated_sky_config: Path,
):
    isolated_sky_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_sky_config.write_text(
        "aws:\n  prioritize_reservations: true\n  specific_reservations: []\n"
    )
    sky_mod.register_capacity_reservation("cr-1")
    cfg = YAML().load(isolated_sky_config.read_text())
    assert "prioritize_reservations" not in cfg["aws"]
    assert cfg["aws"]["specific_reservations"] == ["cr-1"]


def test_register_capacity_reservation_idempotent(isolated_sky_config: Path):
    sky_mod.register_capacity_reservation("cr-123")
    sky_mod.register_capacity_reservation("cr-123")  # again
    cfg = YAML().load(isolated_sky_config.read_text())
    assert cfg["aws"]["specific_reservations"] == ["cr-123"]


def test_register_capacity_reservation_appends_distinct(isolated_sky_config: Path):
    sky_mod.register_capacity_reservation("cr-1")
    sky_mod.register_capacity_reservation("cr-2")
    cfg = YAML().load(isolated_sky_config.read_text())
    assert cfg["aws"]["specific_reservations"] == ["cr-1", "cr-2"]


def test_unregister_capacity_reservation(isolated_sky_config: Path):
    sky_mod.register_capacity_reservation("cr-1")
    sky_mod.register_capacity_reservation("cr-2")
    sky_mod.unregister_capacity_reservation("cr-1")
    cfg = YAML().load(isolated_sky_config.read_text())
    assert cfg["aws"]["specific_reservations"] == ["cr-2"]


def test_unregister_capacity_reservation_no_file_is_noop(isolated_sky_config: Path):
    # File doesn't exist yet — function must not raise.
    sky_mod.unregister_capacity_reservation("cr-x")
    assert not isolated_sky_config.exists()
