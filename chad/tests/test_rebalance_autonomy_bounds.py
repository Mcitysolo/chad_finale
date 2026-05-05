from __future__ import annotations

import json
import subprocess
from pathlib import Path

PYTHON = "/home/ubuntu/chad_finale/venv/bin/python3"
SCRIPT = "/home/ubuntu/chad_finale/ops/rebalance_auto_executor_paper.py"


def _neutral_event_risk_path(tmp_path: Path) -> str:
    p = tmp_path / "event_risk.json"
    p.write_text(json.dumps({
        "schema_version": "event_risk.v1",
        "severity": "low",
        "elevated_risk": False,
        "risk_score": 0.0,
        "ts_utc": "2026-01-01T00:00:00Z",
        "ttl_seconds": 1800,
        "windows": [],
    }))
    return str(p)


def _run(env: dict, args: list[str]) -> dict:
    p = subprocess.run(
        [PYTHON, SCRIPT, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0
    return json.loads(p.stdout)


def test_execute_blocks_caps_not_met(tmp_path: Path) -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/chad_finale/runtime",
        "CHAD_CONFIG_DIR": "/home/ubuntu/chad_finale/config",
        "CHAD_AUTO_EXECUTE_REBALANCE": "1",
        "CHAD_PORTFOLIO_PROFILE": "BALANCED",
        # isolate from live runtime event_risk.json
        "CHAD_EVENT_RISK_PATH": _neutral_event_risk_path(tmp_path),
    }
    obj = _run(env, ["--execute"])
    assert obj.get("blocked") is True
    assert obj.get("reason") in ("CAPS_NOT_MET", "DRIFT_NOT_MET", "COOLDOWN_NOT_MET", "EVENT_RISK_BLOCKED")
    # In current system state, caps are expected to be the blocker.
    if obj.get("reason") == "CAPS_NOT_MET":
        assert obj.get("checks", {}).get("caps_ok") is False


def test_preview_includes_would_execute_flag(tmp_path: Path) -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/chad_finale/runtime",
        "CHAD_CONFIG_DIR": "/home/ubuntu/chad_finale/config",
        "CHAD_PORTFOLIO_PROFILE": "BALANCED",
        # isolate from live runtime event_risk.json
        "CHAD_EVENT_RISK_PATH": _neutral_event_risk_path(tmp_path),
    }
    obj = _run(env, [])
    assert obj.get("mode") == "PREVIEW"
    assert "would_execute" in obj
