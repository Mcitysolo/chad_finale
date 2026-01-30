from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.operator_surface_v2 import build_operator_router


def test_operator_surface_v2_shapes() -> None:
    app = FastAPI()
    app.include_router(build_operator_router())
    c = TestClient(app)

    assert c.get("/op/version").status_code == 200
    assert c.get("/op/status").status_code == 200
    assert c.get("/op/why_blocked").status_code == 200
    assert c.get("/op/risk_explain").status_code == 200
    assert c.get("/op/perf_snapshot").status_code == 200
    assert c.get("/op/what_if_caps", params={"equity": 20000, "daily_risk_fraction": 0.05}).status_code == 200
    assert c.get("/op/brief").status_code == 200
