"""Verify dashboard portfolio block exposes Today Realized + Total Paper PnL.

The dashboard renames the legacy "Realized P&L" card to "Today Realized P&L"
and adds a "Total Paper Performance" card sourced from scr_state.stats.total_pnl.
This test pins the field names and fallback behaviour the frontend relies on.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test_value_12345")
    sys.modules.pop("chad.dashboard.api", None)
    mod = importlib.import_module("chad.dashboard.api")
    return mod


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_portfolio_exposes_new_fields(api, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNTIME", tmp_path)
    _write(tmp_path / "pnl_state.json", {
        "account_equity": 100000.0,
        "realized_pnl": 12.34,
    })
    _write(tmp_path / "scr_state.json", {
        "sizing_factor": 0.1,
        "stats": {
            "total_pnl": 9880.5,
            "effective_trades": 190,
            "win_rate": 0.7736842105263158,
        },
    })

    builder = api.StateBuilder()
    p = builder._portfolio()

    assert p["today_realized_pnl"] == 12.34
    assert p["total_paper_pnl"] == 9880.5
    assert p["effective_trades"] == 190
    assert abs(p["win_rate"] - 0.7737) < 1e-3
    assert p["sizing_factor"] == 0.1
    assert p["sizing_factor_pct_label"] == "10%"
    assert p["win_rate_pct_label"] == "77.4%"
    # Backwards compat — old field still present for any consumer.
    assert p["realized_pnl"] == 12.34


def test_portfolio_missing_state_returns_none_not_zero(api, tmp_path, monkeypatch):
    """When source files are missing, fields are None so the UI can show '—'."""
    monkeypatch.setattr(api, "RUNTIME", tmp_path)
    # Both files absent — pure fallback path.
    builder = api.StateBuilder()
    p = builder._portfolio()

    assert p["today_realized_pnl"] is None
    assert p["total_paper_pnl"] is None
    assert p["effective_trades"] is None
    assert p["win_rate"] is None
    assert p["sizing_factor"] is None
    # Labels fall back to em-dash, never "$0.00".
    assert p["today_realized_pnl_label"] == "—"
    assert p["total_paper_pnl_label"] == "—"
    assert p["sizing_factor_pct_label"] == "—"
    assert p["win_rate_pct_label"] == "—"
