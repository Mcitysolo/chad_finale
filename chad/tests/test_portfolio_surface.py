from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import os

from backend.portfolio_surface import router as portfolio_router


def test_portfolio_routes_smoke(tmp_path: Path) -> None:
    # Build fake runtime/config in tmp, point engine there
    repo = tmp_path
    runtime = repo / "runtime"
    config = repo / "config"
    runtime.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)

    # minimal profiles
    (config / "portfolio_profiles.json").write_text(
        json.dumps(
            {
                "schema_version": "portfolio_profiles.v1",
                "profiles": {"BALANCED": {"max_symbols": 3, "targets": []}},
            }
        ),
        encoding="utf-8",
    )

    # fallback universe
    (config / "universe.json").write_text(json.dumps({"tickers": ["AAPL", "MSFT", "SPY"]}), encoding="utf-8")

    # positions snapshot (simple)
    (runtime / "positions_snapshot.json").write_text(
        json.dumps(
            {
                "positions": [
                    {"symbol": "AAPL", "qty": 1, "marketPrice": 100, "marketValue": 100},
                    {"symbol": "MSFT", "qty": 1, "marketPrice": 200, "marketValue": 200},
                ]
            }
        ),
        encoding="utf-8",
    )

    os.environ["CHAD_REPO_DIR"] = str(repo)
    os.environ["CHAD_RUNTIME_DIR"] = str(runtime)
    os.environ["CHAD_CONFIG_DIR"] = str(config)

    app = FastAPI()
    app.include_router(portfolio_router)
    c = TestClient(app)

    r = c.get("/portfolio/active")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "positions" in data

    r2 = c.get("/portfolio/targets/BALANCED")
    assert r2.status_code == 200
    t = r2.json()
    assert t["ok"] is True
    assert t["profile"] == "BALANCED"
    assert len(t["targets"]) == 3

    r3 = c.get("/portfolio/rebalance/latest", params={"profile": "BALANCED"})
    assert r3.status_code == 200
    rb = r3.json()
    assert rb["ok"] is True
    assert "diffs" in rb
