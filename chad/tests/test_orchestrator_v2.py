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


# ---------------------------------------------------------------------------
# BOX-034A Inc 3 Step 1a — total_equity_currency / total_equity_currency_ok
# ---------------------------------------------------------------------------

from chad.core.orchestrator import _derive_total_equity_currency_ok  # noqa: E402


def _write_snapshot_with_currency(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_derive_currency_ok_both_active_and_ok() -> None:
    # ibkr_ok=True, kraken_ok=True, both > 0 -> True
    assert _derive_total_equity_currency_ok(
        ibkr_equity=100_000.0,
        kraken_equity=250.0,
        ibkr_currency_ok=True,
        kraken_currency_ok=True,
    ) is True


def test_derive_currency_ok_active_leg_not_ok() -> None:
    # ibkr_ok=True, kraken_ok=False (active) -> False
    assert _derive_total_equity_currency_ok(
        ibkr_equity=100_000.0,
        kraken_equity=250.0,
        ibkr_currency_ok=True,
        kraken_currency_ok=False,
    ) is False


def test_derive_currency_ok_inactive_leg_excluded() -> None:
    # ibkr_ok=True, kraken=0, coinbase excluded -> True (only ibkr active)
    assert _derive_total_equity_currency_ok(
        ibkr_equity=100_000.0,
        kraken_equity=0.0,
        ibkr_currency_ok=True,
        kraken_currency_ok=None,
    ) is True


def test_derive_currency_ok_active_leg_flag_absent_fail_closed() -> None:
    # ibkr active but its _ok flag is absent (None) -> False (fail-closed)
    assert _derive_total_equity_currency_ok(
        ibkr_equity=100_000.0,
        kraken_equity=0.0,
        ibkr_currency_ok=None,
        kraken_currency_ok=None,
    ) is False


def test_derive_currency_ok_all_zero_fail_closed() -> None:
    # No active legs -> False (fail-closed)
    assert _derive_total_equity_currency_ok(
        ibkr_equity=0.0,
        kraken_equity=0.0,
        ibkr_currency_ok=True,
        kraken_currency_ok=True,
    ) is False


def test_refresh_dynamic_caps_tags_currency_ok_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "10.0")
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)

    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    _write_snapshot_with_currency(
        snapshot_path,
        {
            "ibkr_equity": 100_000.0,
            "ibkr_equity_currency": "CAD",
            "ibkr_equity_currency_ok": True,
            "coinbase_equity": 0.0,
            "kraken_equity": 250.0,
            "kraken_equity_currency": "CAD",
            "kraken_equity_currency_ok": True,
        },
    )

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)
    orch.refresh_dynamic_caps()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity_currency"] == "CAD"
    assert data["total_equity_currency_ok"] is True


def test_refresh_dynamic_caps_tags_currency_ok_false_when_active_leg_untrusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "10.0")
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)

    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    _write_snapshot_with_currency(
        snapshot_path,
        {
            "ibkr_equity": 100_000.0,
            "ibkr_equity_currency_ok": True,
            "coinbase_equity": 0.0,
            "kraken_equity": 250.0,
            "kraken_equity_currency_ok": False,
        },
    )

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)
    orch.refresh_dynamic_caps()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity_currency"] == "CAD"
    assert data["total_equity_currency_ok"] is False


def test_refresh_dynamic_caps_fallback_currency_ok_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Env-fallback path carries no currency tags -> fail-closed False.
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "5.0")
    monkeypatch.setenv("CHAD_IBKR_EQUITY_FALLBACK", "200000")
    monkeypatch.setenv("CHAD_COINBASE_EQUITY_FALLBACK", "0")
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)

    snapshot_path = tmp_path / "missing_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)
    orch.refresh_dynamic_caps()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity_currency"] == "CAD"
    assert data["total_equity_currency_ok"] is False


def test_refresh_dynamic_caps_respects_base_currency_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAD_DAILY_RISK_PCT", "10.0")
    monkeypatch.setenv("CHAD_BASE_CURRENCY", "usd")  # lower-case to verify normalization

    snapshot_path = tmp_path / "portfolio_snapshot.json"
    caps_path = tmp_path / "dynamic_caps.json"

    _write_snapshot_with_currency(
        snapshot_path,
        {
            "ibkr_equity": 100_000.0,
            "ibkr_equity_currency_ok": True,
            "coinbase_equity": 0.0,
            "kraken_equity": 0.0,
        },
    )

    settings = OrchestratorSettings.from_env(
        portfolio_snapshot_path=snapshot_path,
        dynamic_caps_path=caps_path,
    )
    orch = Orchestrator(settings=settings)
    orch.refresh_dynamic_caps()

    data = json.loads(caps_path.read_text(encoding="utf-8"))
    assert data["total_equity_currency"] == "USD"
    # only ibkr active and ok -> True
    assert data["total_equity_currency_ok"] is True
