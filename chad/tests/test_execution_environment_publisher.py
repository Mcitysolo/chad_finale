"""Tests for chad.ops.execution_environment_publisher.

Covers:
- exec_mode reflects active CHAD_EXECUTION_MODE for paper / dry_run / live
- the publisher overwrites a malformed prior file without crashing
- a missing prior file does not crash the publisher
- atomic write helper (tmp + rename) is used
- fail-soft wrapper swallows exceptions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from chad.ops import execution_environment_publisher as eep


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_paper_mode_publishes_exec_mode_paper(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    out = eep.publish_once(tmp_path)
    file_obj = _read(tmp_path / "execution_environment.json")
    assert out == file_obj
    assert file_obj["exec_mode"] == "paper"
    assert file_obj["live_enabled"] is False
    assert file_obj["ibkr_dry_run"] is True
    assert file_obj["source"] == "systemd_env"
    assert file_obj["schema_version"] == "execution_environment.v1"
    assert isinstance(file_obj["ts_utc"], str) and file_obj["ts_utc"].endswith("Z")
    assert file_obj["ttl_seconds"] == 60
    assert "bootstrap" not in file_obj["notes"].lower()


def test_dry_run_mode_publishes_exec_mode_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "dry_run")
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["exec_mode"] == "dry_run"
    assert file_obj["live_enabled"] is False
    assert file_obj["ibkr_dry_run"] is True


def test_live_mode_publishes_exec_mode_live(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["exec_mode"] == "live"
    assert file_obj["live_enabled"] is True
    assert file_obj["ibkr_dry_run"] is False


def test_missing_env_falls_back_to_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAD_EXECUTION_MODE", raising=False)
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["exec_mode"] == "dry_run"


def test_invalid_env_value_falls_back_to_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "WAT")
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["exec_mode"] == "dry_run"


def test_publisher_overwrites_malformed_prior_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    target = tmp_path / "execution_environment.json"
    target.write_text("{ this is :: not valid json", encoding="utf-8")
    file_obj = eep.publish_once(tmp_path)
    on_disk = _read(target)
    assert file_obj == on_disk
    assert on_disk["exec_mode"] == "paper"


def test_publisher_handles_missing_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    sub = tmp_path / "fresh_runtime"
    file_obj = eep.publish_once(sub)
    assert (sub / "execution_environment.json").exists()
    assert file_obj["exec_mode"] == "paper"


def test_publisher_uses_atomic_helper(tmp_path, monkeypatch):
    """Ensure the publisher routes through the canonical atomic writer."""
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    real_writer = eep.write_runtime_state_json
    calls: Dict[str, Any] = {}

    def _spy(path, state, *, ttl_seconds, inject_ts=True):
        calls["path"] = path
        calls["ttl_seconds"] = ttl_seconds
        calls["inject_ts"] = inject_ts
        return real_writer(path, state, ttl_seconds=ttl_seconds, inject_ts=inject_ts)

    with mock.patch.object(eep, "write_runtime_state_json", side_effect=_spy):
        eep.publish_once(tmp_path)

    assert calls["path"] == tmp_path / "execution_environment.json"
    assert calls["ttl_seconds"] == eep.DEFAULT_TTL_SECONDS
    assert calls["inject_ts"] is True


def test_publish_once_safe_returns_false_on_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    with mock.patch.object(eep, "write_runtime_state_json", side_effect=OSError("boom")):
        ok = eep.publish_once_safe(tmp_path)
    assert ok is False


def test_publish_once_safe_returns_true_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    ok = eep.publish_once_safe(tmp_path)
    assert ok is True
    assert (tmp_path / "execution_environment.json").exists()


def test_kraken_enabled_truthy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    monkeypatch.setenv("CHAD_KRAKEN_ENABLED", "0")
    monkeypatch.delenv("KRAKEN_ENABLED", raising=False)
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["kraken_enabled"] is False

    monkeypatch.setenv("CHAD_KRAKEN_ENABLED", "1")
    file_obj = eep.publish_once(tmp_path)
    assert file_obj["kraken_enabled"] is True
