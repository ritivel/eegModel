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
    assert cfg["aws"]["prioritize_reservations"] is True


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
