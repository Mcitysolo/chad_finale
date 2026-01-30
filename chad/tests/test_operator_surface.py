from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.operator_surface import router


def test_operator_surface_endpoints_shape() -> None:
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)

    r = c.get("/version")
    assert r.status_code == 200
    v = r.json()
    assert "ts_utc" in v
    assert "git" in v and isinstance(v["git"], dict)

    r2 = c.get("/status")
    assert r2.status_code == 200
    s = r2.json()
    assert s["service"] == "CHAD Operator Surface"
    assert "runtime_files" in s and isinstance(s["runtime_files"], dict)
    assert "failed_units" in s and isinstance(s["failed_units"], dict)

    r3 = c.get("/why_blocked")
    assert r3.status_code == 200
    w = r3.json()
    assert "summary" in w
    assert "operator_intent" in w
    assert "shadow" in w
    assert "live_gate" in w
