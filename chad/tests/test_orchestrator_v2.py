from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from chad.core.orchestrator import (
    Orchestrator,
    OrchestratorCycleResult,
    OrchestratorSettings,
)


def _write_snapshot(path: Path, ibkr: float, coinbase: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "ibkr_equity": ibkr,
        "coinbase_equity": coinbase,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_settings_from_env_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clear env to hit defaults
    for key in (
        "CHAD_DAILY_RISK_PCT",
        "CHAD_ORCH_INTERVAL_SECONDS",
        "CHAD_ORCH_RUN_FOREVER",
        "CHAD_ORCH_LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)

    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )

    assert settings.daily_risk_pct == pytest.approx(5.0)
    assert settings.loop_interval_seconds == pytest.approx(60.0)
    assert settings.run_forever is False
    assert settings.log_level == "INFO"
    assert settings.portfolio_snapshot_path == snapshot_path
    assert settings.dynamic_caps_path == caps_path


def test_refresh_dynamic_caps_uses_snapshot_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 10% risk for easier math
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "10.0")

    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    # 100k + 50k total equity
    _write_snapshot(snapshot_path, ibkr=100_000.0, coinbase=50_000.0)

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)

    result: OrchestratorCycleResult = orch.refresh_dynamic_caps()

    assert result.used_fallback_snapshot is False
    assert result.total_equity == pytest.approx(150_000.0)
    assert result.daily_risk_fraction == pytest.approx(0.10)
    assert result.portfolio_risk_cap == pytest.approx(15_000.0)
    assert result.dynamic_caps_path == caps_path
    assert caps_path.is_file()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity"] == pytest.approx(150_000.0)
    assert data["portfolio_risk_cap"] == pytest.approx(15_000.0)
    assert abs(float(data["sum_normalized_weights"]) - 1.0) < 1e-6


def test_refresh_dynamic_caps_falls_back_when_snapshot_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 5% risk on 200k fallback = 10k
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "5.0")
    monkeypatch.setenv("CHAD_IBKR_EQUITY_FALLBACK", "200000")
    monkeypatch.setenv("CHAD_COINBASE_EQUITY_FALLBACK", "0")

    snapshot_path = tmp_path / "missing_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)

    result = orch.refresh_dynamic_caps()

    assert result.used_fallback_snapshot is True
    assert result.total_equity == pytest.approx(200_000.0)
    assert result.daily_risk_fraction == pytest.approx(0.05)
    assert result.portfolio_risk_cap == pytest.approx(10_000.0)
    assert caps_path.is_file()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity"] == pytest.approx(200_000.0)
    assert data["portfolio_risk_cap"] == pytest.approx(10_000.0)
