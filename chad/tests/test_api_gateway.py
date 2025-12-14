"""
Tests for backend.api_gateway (CHAD API Gateway – Phase 7).

These tests assert:

* The gateway imports and the FastAPI app object exists.
* /health, /risk-state, /live-gate, /shadow, and / all respond 200 (or 403 for
  the intentionally disabled /orders endpoint).
* No endpoint can ever report allow_ibkr_live=True in this Phase-7 build.
* IBKR is hard-locked to DRY_RUN at the execution config level.
* Shadow statistics and mode snapshots are well-formed.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi.testclient import TestClient

from backend.api_gateway import app


def _get(client: TestClient, path: str) -> Dict[str, Any]:
    response = client.get(path)
    assert response.status_code == 200, f"GET {path} failed: {response.text}"
    data = response.json()
    assert isinstance(data, dict), f"Expected dict JSON for {path}"
    return data


def test_app_import_and_root_ok() -> None:
    """
    Basic sanity: app must be importable and root must respond with a pointer
    to the main informational endpoints.
    """
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "CHAD API Gateway"
    msg = data.get("message", "")
    assert "/health" in msg
    assert "/risk-state" in msg
    assert "/live-gate" in msg
    assert "/shadow" in msg


def test_health_endpoint_reports_gateway_healthy() -> None:
    """
    /health should report the gateway as healthy and DRY_RUN-only.
    """
    client = TestClient(app)
    data = _get(client, "/health")

    assert data["service"] == "CHAD API Gateway"
    assert data["healthy"] is True

    details = data.get("details", {})
    assert isinstance(details, dict)

    # Execution snapshot should be present and DRY_RUN-locked for IBKR.
    execution = details.get("execution", {})
    assert isinstance(execution, dict)
    assert execution.get("ibkr_enabled") is True
    assert execution.get("ibkr_dry_run") is True
    # Mode string should be something like "dry_run".
    assert isinstance(execution.get("exec_mode"), str)

    mode = details.get("mode", {})
    assert isinstance(mode, dict)
    assert "chad_mode" in mode
    assert isinstance(mode.get("live_enabled"), bool)

    shadow = details.get("shadow", {})
    assert isinstance(shadow, dict)
    assert "state" in shadow
    assert "paper_only" in shadow
    assert isinstance(shadow.get("paper_only"), bool)


def test_live_gate_enforces_no_live_in_phase7() -> None:
    """
    /live-gate is the single source of truth for whether IBKR live trading
    could be allowed. In Phase 7 it must always deny live.
    """
    client = TestClient(app)
    data = _get(client, "/live-gate")

    # Execution snapshot assertions
    execution = data.get("execution", {})
    assert isinstance(execution, dict)
    assert execution.get("ibkr_enabled") is True
    # Hard lock: must be DRY_RUN.
    assert execution.get("ibkr_dry_run") is True
    assert isinstance(execution.get("exec_mode"), str)

    # Mode snapshot assertions
    mode = data.get("mode", {})
    assert isinstance(mode, dict)
    assert "chad_mode" in mode
    assert isinstance(mode.get("live_enabled"), bool)

    # Shadow snapshot assertions
    shadow = data.get("shadow", {})
    assert isinstance(shadow, dict)
    assert "state" in shadow
    assert isinstance(shadow.get("paper_only"), bool)
    assert isinstance(shadow.get("sizing_factor"), (int, float))
    reasons = data.get("reasons", [])
    assert isinstance(reasons, list)
    assert len(reasons) >= 1

    # The core guarantees we care about:
    assert data.get("allow_ibkr_live") is False
    # Paper / what-if execution is allowed.
    assert data.get("allow_ibkr_paper") is True


def test_risk_state_shape_is_consistent() -> None:
    """
    /risk-state should mirror the structure of the CLI risk snapshot:
    - mode snapshot,
    - optional dynamic caps,
    - shadow snapshot.
    """
    client = TestClient(app)
    data = _get(client, "/risk-state")

    mode = data.get("mode", {})
    assert isinstance(mode, dict)
    assert "chad_mode" in mode
    assert isinstance(mode.get("live_enabled"), bool)

    # dynamic_caps may legitimately be None if the orchestrator has not
    # written runtime/dynamic_caps.json yet. We only assert shape if present.
    dynamic_caps = data.get("dynamic_caps")
    if dynamic_caps is not None:
        assert isinstance(dynamic_caps, dict)
        assert isinstance(dynamic_caps.get("total_equity"), (int, float))
        assert isinstance(dynamic_caps.get("daily_risk_fraction"), (int, float))
        assert isinstance(dynamic_caps.get("portfolio_risk_cap"), (int, float))
        strategy_caps = dynamic_caps.get("strategy_caps", {})
        assert isinstance(strategy_caps, dict)
        for key in ("alpha", "beta", "gamma", "omega", "delta", "crypto", "forex"):
            assert isinstance(strategy_caps.get(key), (int, float)), key

    shadow = data.get("shadow", {})
    assert isinstance(shadow, dict)
    assert "state" in shadow
    assert isinstance(shadow.get("paper_only"), bool)
    stats = shadow.get("stats", {})
    assert isinstance(stats, dict)
    # Only check keys & types, not particular values.
    for key in (
        "total_trades",
        "win_rate",
        "total_pnl",
        "max_drawdown",
        "sharpe_like",
        "live_trades",
        "paper_trades",
    ):
        assert key in stats, f"missing stats key: {key}"


def test_shadow_endpoint_returns_shadow_only() -> None:
    """
    /shadow should expose only the shadow snapshot wrapper, with the same
    structure as used in risk-state and live-gate.
    """
    client = TestClient(app)
    data = _get(client, "/shadow")

    assert "shadow" in data
    shadow = data["shadow"]
    assert isinstance(shadow, dict)
    assert "state" in shadow
    assert isinstance(shadow.get("paper_only"), bool)
    assert "stats" in shadow
    stats = shadow["stats"]
    assert isinstance(stats, dict)
    assert "total_trades" in stats
    assert "total_pnl" in stats


def test_orders_endpoint_is_disabled() -> None:
    """
    /orders must be explicitly disabled in Phase 7.

    Any attempt to POST here should return 403, proving that there is no
    HTTP-based trading path.
    """
    client = TestClient(app)
    resp = client.post("/orders", json={"dummy": "payload"})
    assert resp.status_code == 403
    data = resp.json()
    assert "disabled in Phase 7" in data.get("detail", "")
