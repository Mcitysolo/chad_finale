from __future__ import annotations

import json
import subprocess
from pathlib import Path

PYTHON = "/home/ubuntu/chad_finale/venv/bin/python3"
SCRIPT = "/home/ubuntu/chad_finale/ops/rotation_publish.py"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _run(env: dict[str, str]) -> dict:
    p = subprocess.run(
        [PYTHON, SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_rotation_publish_uses_config_rules(tmp_path: Path) -> None:
    root = tmp_path / "chad"
    runtime = root / "runtime"
    reports = root / "reports" / "rotation"
    config_dir = root / "config"

    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        runtime / "macro_state.json",
        {
            "ts_utc": "2026-03-05T22:00:00Z",
            "risk_label": "risk_on",
            "ttl_seconds": 3600,
        },
    )
    _write_json(
        runtime / "event_risk.json",
        {
            "ts_utc": "2026-03-05T22:00:00Z",
            "severity": "medium",
            "ttl_seconds": 3600,
        },
    )
    _write_json(
        runtime / "sector_rotation.json",
        {
            "ts_utc": "2026-03-05T22:00:00Z",
            "provider_status": "bootstrap_no_provider",
            "ttl_seconds": 3600,
        },
    )

    _write_json(
        config_dir / "rotation_rules.json",
        {
            "schema_version": "rotation_rules.v1",
            "enabled": True,
            "advisory_only": True,
            "write_pointer": True,
            "ttl_seconds": 3600,
            "fail_closed_on_missing_inputs": True,
            "fail_closed_event_risk_severities": ["high", "unknown"],
            "max_abs_tilt": 0.03,
            "inputs": {
                "macro_state_path": str(runtime / "macro_state.json"),
                "event_risk_path": str(runtime / "event_risk.json"),
                "sector_rotation_path": str(runtime / "sector_rotation.json"),
            },
            "output": {
                "reports_dir": str(reports),
                "pointer_path": str(runtime / "rotation_state.json"),
            },
            "regime_rules": {
                "event_risk_override": {
                    "growth_momo": -0.03,
                    "hedge": 0.03,
                    "reason": "event_risk_high_or_unknown",
                },
                "risk_off": {
                    "growth_momo": -0.03,
                    "hedge": 0.03,
                    "reason": "macro_risk_off",
                },
                "risk_on": {
                    "growth_momo": 0.03,
                    "hedge": -0.03,
                    "reason": "macro_risk_on_custom",
                },
                "neutral": {
                    "growth_momo": 0.0,
                    "hedge": 0.0,
                    "reason": "macro_neutral",
                },
            },
            "symbol_blocks": [{"symbol": "TSLA", "reason": "rotation_test_block"}],
            "notes": [
                "rotation test contract",
                "advisory only",
            ],
        },
    )

    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": str(runtime),
        "CHAD_ROTATION_REPORTS_DIR": str(reports),
        "CHAD_ROTATION_RULES_PATH": str(config_dir / "rotation_rules.json"),
    }

    obj = _run(env)
    assert obj.get("ok") is True
    assert obj.get("config_loaded") is True
    assert obj.get("config_path") == str(config_dir / "rotation_rules.json")

    pointer = json.loads((runtime / "rotation_state.json").read_text(encoding="utf-8"))
    assert pointer.get("schema_version") == "rotation_state.v1"

    latest = Path(pointer["latest_rotation_path"])
    assert latest.is_file()

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == "rotation.v1"
    assert payload.get("regime") == "risk_on"
    assert payload.get("config", {}).get("config_loaded") is True
    assert payload.get("config", {}).get("schema_version") == "rotation_rules.v1"
    assert payload.get("config", {}).get("max_abs_tilt") == 0.03
    assert payload.get("symbol_blocks") == [{"symbol": "TSLA", "reason": "rotation_test_block"}]

    tilts = payload.get("tilts", [])
    assert len(tilts) == 2
    assert tilts[0]["sleeve"] == "growth_momo"
    assert tilts[0]["delta_weight"] == 0.03
    assert tilts[0]["reason"] == "macro_risk_on_custom"
    assert tilts[1]["sleeve"] == "hedge"
    assert tilts[1]["delta_weight"] == -0.03
    assert tilts[1]["reason"] == "macro_risk_on_custom"


def test_rotation_publish_fail_closed_on_missing_inputs(tmp_path: Path) -> None:
    root = tmp_path / "chad"
    runtime = root / "runtime"
    reports = root / "reports" / "rotation"
    config_dir = root / "config"

    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        runtime / "macro_state.json",
        {
            "ts_utc": "2026-03-05T22:00:00Z",
            "risk_label": "risk_off",
            "ttl_seconds": 3600,
        },
    )
    _write_json(
        config_dir / "rotation_rules.json",
        {
            "schema_version": "rotation_rules.v1",
            "enabled": True,
            "advisory_only": True,
            "write_pointer": True,
            "ttl_seconds": 3600,
            "fail_closed_on_missing_inputs": True,
            "fail_closed_event_risk_severities": ["high", "unknown"],
            "max_abs_tilt": 0.05,
            "inputs": {
                "macro_state_path": str(runtime / "macro_state.json"),
                "event_risk_path": str(runtime / "event_risk.json"),
                "sector_rotation_path": str(runtime / "sector_rotation.json"),
            },
            "output": {
                "reports_dir": str(reports),
                "pointer_path": str(runtime / "rotation_state.json"),
            },
            "regime_rules": {
                "event_risk_override": {
                    "growth_momo": -0.05,
                    "hedge": 0.05,
                    "reason": "event_risk_high_or_unknown",
                },
                "risk_off": {
                    "growth_momo": -0.05,
                    "hedge": 0.05,
                    "reason": "macro_risk_off",
                },
                "risk_on": {
                    "growth_momo": 0.05,
                    "hedge": -0.05,
                    "reason": "macro_risk_on",
                },
                "neutral": {
                    "growth_momo": 0.0,
                    "hedge": 0.0,
                    "reason": "macro_neutral",
                },
            },
            "symbol_blocks": [],
            "notes": ["fail closed test"],
        },
    )

    env = {
        "PYTHONPATH": "/home/ubuntu/chad_finale",
        "CHAD_RUNTIME_DIR": str(runtime),
        "CHAD_ROTATION_REPORTS_DIR": str(reports),
        "CHAD_ROTATION_RULES_PATH": str(config_dir / "rotation_rules.json"),
    }

    obj = _run(env)
    assert obj.get("ok") is True
    assert obj.get("config_loaded") is True

    pointer = json.loads((runtime / "rotation_state.json").read_text(encoding="utf-8"))
    latest = Path(pointer["latest_rotation_path"])
    payload = json.loads(latest.read_text(encoding="utf-8"))

    assert payload.get("regime") == "unknown"
    assert payload.get("tilts") == []
    assert payload.get("symbol_blocks") == []
    assert str(payload.get("notes", "")).startswith("fail_closed_missing_inputs:")
    assert payload.get("config", {}).get("config_loaded") is True
