from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api_gateway import app


def test_status_endpoint_shape() -> None:
    c = TestClient(app)
    r = c.get("/status")
    assert r.status_code == 200
    data = r.json()

    # Top-level
    assert data["service"] == "CHAD API Gateway"
    assert "ts_utc" in data

    # Core snapshots
    assert "execution" in data and isinstance(data["execution"], dict)
    assert "mode" in data and isinstance(data["mode"], dict)
    assert "live_gate" in data and isinstance(data["live_gate"], dict)
    assert "shadow" in data and isinstance(data["shadow"], dict)

    # Runtime files map
    rf = data.get("runtime_files")
    assert isinstance(rf, dict)

    # Required keys we expect to always report (exists can be true/false depending on runtime)
    for key in ("feed_state", "positions_snapshot", "reconciliation_state", "dynamic_caps", "operator_intent", "portfolio_snapshot", "scr_state", "tier_state"):
        assert key in rf, key
        assert isinstance(rf[key], dict)
        assert "exists" in rf[key]
        assert "path" in rf[key]
