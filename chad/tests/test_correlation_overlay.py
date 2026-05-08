"""Tests for the correlation overlay producer/consumer wiring.

Covers four cases:
  1. Valid dynamic_caps_correlation.json applies overlay.
  2. Missing source falls back with explicit reason in health file.
  3. Invalid correlation_governed_weights falls back with explicit reason.
  4. Producer output from correlation_layer is accepted by the consumer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from chad.risk import correlation_layer
from chad.risk.correlation_strategy import CorrelationOverlayStrategy


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _no_op_refresher(out_path: Path):
    return False, "test_disabled_refresher", {}


def _ok_refresher(out_path: Path):
    return True, None, {}


def test_valid_correlation_file_applies_overlay(tmp_path: Path):
    c_path = tmp_path / "dynamic_caps_correlation.json"
    health = tmp_path / "correlation_overlay_health.json"
    _write(
        c_path,
        {
            "correlation_governed_weights": {
                "alpha": 0.4,
                "beta": 0.6,
            }
        },
    )

    s = CorrelationOverlayStrategy(c_path=c_path, health_path=health, refresher=_ok_refresher)
    out = s.apply(
        repo_root=tmp_path,
        base_weights={"alpha": 0.5, "beta": 0.5, "gamma": 0.1},
        log=logging.getLogger("t"),
    )
    assert out["alpha"] == 0.4
    assert out["beta"] == 0.6
    # passthrough for keys absent from overlay
    assert out["gamma"] == 0.1

    h = json.loads(health.read_text())
    assert h["ok"] is True
    assert h["status"] == "ok"
    assert h["weights_count"] == 2


def test_missing_file_falls_back_with_reason(tmp_path: Path):
    c_path = tmp_path / "dynamic_caps_correlation.json"  # does not exist
    health = tmp_path / "correlation_overlay_health.json"

    s = CorrelationOverlayStrategy(c_path=c_path, health_path=health, refresher=_no_op_refresher)
    base = {"alpha": 0.5, "beta": 0.5}
    out = s.apply(repo_root=tmp_path, base_weights=base, log=logging.getLogger("t"))
    assert out == base

    h = json.loads(health.read_text())
    assert h["ok"] is False
    assert h["status"] == "missing"
    assert h["reason"] == "test_disabled_refresher"
    assert h["weights_count"] == 0


def test_invalid_correlation_weights_falls_back(tmp_path: Path):
    c_path = tmp_path / "dynamic_caps_correlation.json"
    health = tmp_path / "correlation_overlay_health.json"
    _write(c_path, {"correlation_governed_weights": "not-a-dict"})

    s = CorrelationOverlayStrategy(c_path=c_path, health_path=health, refresher=_no_op_refresher)
    base = {"alpha": 0.5}
    out = s.apply(repo_root=tmp_path, base_weights=base, log=logging.getLogger("t"))
    assert out == base

    h = json.loads(health.read_text())
    assert h["ok"] is False
    assert h["status"] == "invalid"
    assert h["reason"] == "invalid correlation_governed_weights"


def test_empty_correlation_weights_falls_back(tmp_path: Path):
    c_path = tmp_path / "dynamic_caps_correlation.json"
    health = tmp_path / "correlation_overlay_health.json"
    _write(c_path, {"correlation_governed_weights": {}})

    s = CorrelationOverlayStrategy(c_path=c_path, health_path=health, refresher=_no_op_refresher)
    out = s.apply(
        repo_root=tmp_path,
        base_weights={"alpha": 0.5},
        log=logging.getLogger("t"),
    )
    assert out == {"alpha": 0.5}
    h = json.loads(health.read_text())
    assert h["status"] == "invalid"


def test_producer_output_accepted_by_consumer(tmp_path: Path):
    """End-to-end: correlation_layer.refresh writes a file the consumer accepts."""
    quarantine = tmp_path / "dynamic_caps_quarantine.json"
    cycle = tmp_path / "full_execution_cycle_last.json"
    out_path = tmp_path / "dynamic_caps_correlation.json"
    health = tmp_path / "correlation_overlay_health.json"

    _write(
        quarantine,
        {
            "quarantine_weights": {
                "alpha": 0.3,
                "beta_trend": 0.2,
                "gamma": 0.1,
                "delta": 0.2,
                "omega": 0.05,
                "crypto": 0.05,
                "forex": 0.05,
                "alpha_crypto": 0.025,
                "alpha_forex": 0.025,
            }
        },
    )
    _write(cycle, {})

    ok, reason, payload = correlation_layer.refresh(
        quarantine_path=quarantine,
        cycle_path=cycle,
        out_path=out_path,
    )
    assert ok, reason
    assert "correlation_governed_weights" in payload
    assert out_path.exists()

    # Consumer (with passing refresher) should accept the producer's output.
    s = CorrelationOverlayStrategy(c_path=out_path, health_path=health, refresher=_ok_refresher)
    base = {k: 1.0 / 9 for k in payload["correlation_governed_weights"].keys()}
    out = s.apply(repo_root=tmp_path, base_weights=base, log=logging.getLogger("t"))

    assert set(out.keys()) == set(base.keys())
    h = json.loads(health.read_text())
    assert h["ok"] is True
    assert h["status"] == "ok"
    assert h["weights_count"] == len(payload["correlation_governed_weights"])


def test_producer_fails_when_quarantine_missing(tmp_path: Path):
    quarantine = tmp_path / "missing.json"
    cycle = tmp_path / "cycle.json"
    out_path = tmp_path / "out.json"

    ok, reason, payload = correlation_layer.refresh(
        quarantine_path=quarantine,
        cycle_path=cycle,
        out_path=out_path,
    )
    assert ok is False
    assert reason == "quarantine_weights missing"
    assert payload == {}
    assert not out_path.exists()
