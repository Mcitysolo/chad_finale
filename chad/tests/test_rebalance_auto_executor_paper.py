from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = "/home/ubuntu/chad_finale/ops/rebalance_auto_executor_paper.py"
PYTHON = "/home/ubuntu/chad_finale/venv/bin/python3"


def test_execute_blocks_when_disabled() -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/CHAD FINALE/runtime",
        # critical: AUTO_EXECUTE disabled
        "CHAD_AUTO_EXECUTE_REBALANCE": "0",
    }
    p = subprocess.run(
        [PYTHON, SCRIPT, "--execute"],
        check=False,
        capture_output=True,
        text=True,
        env={**env, **dict(**env)},
    )
    assert p.returncode == 0
    obj = json.loads(p.stdout.strip())
    assert obj.get("blocked") is True
    assert obj.get("reason") == "AUTO_EXECUTE_DISABLED"


def test_preview_smoke() -> None:
    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": "/home/ubuntu/CHAD FINALE/runtime",
        "CHAD_PORTFOLIO_PROFILE": "BALANCED",
    }
    p = subprocess.run(
        [PYTHON, SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        env={**env, **dict(**env)},
    )
    assert p.returncode == 0
    obj = json.loads(p.stdout)
    assert obj.get("ok") is True
    assert obj.get("mode") == "PREVIEW"
    assert int(obj.get("receipts_count", 0)) >= 1
