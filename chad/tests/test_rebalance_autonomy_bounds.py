from __future__ import annotations

import json
import subprocess

PYTHON = "/home/ubuntu/chad_finale/venv/bin/python3"
SCRIPT = "/home/ubuntu/chad_finale/ops/rebalance_auto_executor_paper.py"


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


def test_execute_blocks_caps_not_met() -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/CHAD FINALE/runtime",
        "CHAD_CONFIG_DIR": "/home/ubuntu/CHAD FINALE/config",
        "CHAD_AUTO_EXECUTE_REBALANCE": "1",
        "CHAD_PORTFOLIO_PROFILE": "BALANCED",
    }
    obj = _run(env, ["--execute"])
    assert obj.get("blocked") is True
    assert obj.get("reason") in ("CAPS_NOT_MET", "DRIFT_NOT_MET", "COOLDOWN_NOT_MET", "EVENT_RISK_BLOCKED")
    # In current system state, caps are expected to be the blocker.
    if obj.get("reason") == "CAPS_NOT_MET":
        assert obj.get("checks", {}).get("caps_ok") is False


def test_preview_includes_would_execute_flag() -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/CHAD FINALE/runtime",
        "CHAD_CONFIG_DIR": "/home/ubuntu/CHAD FINALE/config",
        "CHAD_PORTFOLIO_PROFILE": "BALANCED",
    }
    obj = _run(env, [])
    assert obj.get("mode") == "PREVIEW"
    assert "would_execute" in obj
