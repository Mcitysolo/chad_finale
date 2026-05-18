"""GAP-002 regression: Prometheus exposes canonical SCR PnL truth,
existing chad_paper_* / chad_scr_* metrics are NOT renamed, the false
"SCR-aligned" claim is removed from chad/ops/metrics_server.py, and the
dashboard portfolio payload carries non-breaking definition fields.

The fix is additive + text-only — no metric rename, no value change.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest


# -----------------------------
# Exporter: canonical SCR PnL metric + rename-safety + label-cleanup
# -----------------------------


def _load_scr_lines(tmp_path: Path, scr_payload: dict) -> list:
    """Invoke metrics_server._scr_lines() against a monkeypatched
    RUNTIME_SCR_PATH pointing at a tmp scr_state.json."""
    from chad.ops import metrics_server as mod

    scr_path = tmp_path / "scr_state.json"
    scr_path.write_text(json.dumps(scr_payload), encoding="utf-8")

    orig = mod.RUNTIME_SCR_PATH
    mod.RUNTIME_SCR_PATH = scr_path
    try:
        return mod._scr_lines()
    finally:
        mod.RUNTIME_SCR_PATH = orig


def test_canonical_scr_pnl_metric_emitted(tmp_path: Path) -> None:
    """chad_scr_total_pnl (and siblings) mirror runtime/scr_state.json::stats."""
    scr_payload = {
        "state": "CONFIDENT",
        "paper_only": False,
        "sizing_factor": 1.0,
        "stats": {
            "effective_trades": 196,
            "excluded_manual": 6,
            "excluded_untrusted": 321,
            "excluded_nonfinite": 0,
            "total_pnl": 9905.125,
            "win_rate": 0.7551020408163265,
            "sharpe_like": 5.727005179212231,
            "max_drawdown": -290.0,
        },
    }
    lines = _load_scr_lines(tmp_path, scr_payload)
    by_name = {ln.name: ln.value for ln in lines if not ln.labels}

    assert "chad_scr_total_pnl" in by_name, (
        "chad_scr_total_pnl must be emitted as the canonical SCR PnL truth"
    )
    assert abs(by_name["chad_scr_total_pnl"] - 9905.125) < 1e-6
    assert abs(by_name["chad_scr_win_rate"] - 0.7551020408163265) < 1e-6
    assert abs(by_name["chad_scr_sharpe_like"] - 5.727005179212231) < 1e-6
    assert abs(by_name["chad_scr_max_drawdown"] - (-290.0)) < 1e-6


def test_canonical_scr_pnl_metric_defensive_against_missing_stats(tmp_path: Path) -> None:
    """Missing stats keys must coerce to 0.0 without raising (consistent
    with sibling chad_scr_* fields)."""
    scr_payload = {"state": "WARMUP", "paper_only": True, "sizing_factor": 0.0, "stats": {}}
    lines = _load_scr_lines(tmp_path, scr_payload)
    by_name = {ln.name: ln.value for ln in lines if not ln.labels}
    for k in (
        "chad_scr_total_pnl",
        "chad_scr_win_rate",
        "chad_scr_sharpe_like",
        "chad_scr_max_drawdown",
    ):
        assert k in by_name, f"{k} must always be emitted"
        assert by_name[k] == 0.0


def test_no_existing_metric_renamed(tmp_path: Path) -> None:
    """Rename-safety regression: every metric name in the public consumer
    surface must still appear in rendered Prometheus output."""
    from chad.ops import metrics_server as mod

    scr_path = tmp_path / "scr_state.json"
    scr_path.write_text(
        json.dumps({"state": "WARMUP", "paper_only": True, "sizing_factor": 0.0, "stats": {}}),
        encoding="utf-8",
    )
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    # one trivial trade so paper rollup emits values, not just zeros
    env = {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": {
            "strategy": "manual",
            "symbol": "EUR",
            "side": "BUY",
            "notional": 1.0,
            "pnl": 0.25,
            "is_live": False,
            "tags": [],
            "extra": {},
        },
        "prev_hash": "GENESIS",
        "record_hash": "x",
    }
    (trades_dir / "trade_history_20990101.ndjson").write_text(json.dumps(env) + "\n")

    paper_lines = mod._paper_rollup_metrics(trades_dir)

    orig = mod.RUNTIME_SCR_PATH
    mod.RUNTIME_SCR_PATH = scr_path
    try:
        scr_lines = mod._scr_lines()
    finally:
        mod.RUNTIME_SCR_PATH = orig

    text = mod._render_prometheus(paper_lines + scr_lines)
    for required in (
        "chad_paper_total_pnl",
        "chad_paper_avg_pnl",
        "chad_paper_win_rate",
        "chad_paper_trades_total",
        "chad_paper_sharpe_like",
        "chad_paper_pnl_untrusted_count",
        "chad_scr_state",
        "chad_scr_effective_trades",
        "chad_scr_excluded_manual",
        "chad_scr_excluded_untrusted",
        "chad_scr_sizing_factor",
        "chad_scr_paper_only",
    ):
        assert required in text, f"{required} must still be emitted (rename-safety)"


def test_false_scr_aligned_label_removed() -> None:
    """metrics_server source must no longer assert the chad_paper_* set
    is 'SCR-aligned'; corrected wording must be present."""
    from chad.ops import metrics_server as mod

    src = inspect.getsource(mod)
    # Original false claim — must be gone.
    assert "LEAN set (SCR-aligned)" not in src, (
        "False label 'LEAN set (SCR-aligned)' must be removed from chad_paper_* docs"
    )
    # Corrected wording must be present.
    assert "does NOT consult" in src
    assert "DIVERGES from SCR canonical" in src or "DIVERGES from SCR" in src


# -----------------------------
# Dashboard: additive definition fields + non-breaking guard
# -----------------------------


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test_value_12345")
    sys.modules.pop("chad.dashboard.api", None)
    return importlib.import_module("chad.dashboard.api")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dashboard_pnl_definition_fields_present(api, tmp_path: Path, monkeypatch) -> None:
    """Additive fields surface SCR-effective definition without
    changing existing field names or values."""
    monkeypatch.setattr(api, "RUNTIME", tmp_path)
    _write(tmp_path / "pnl_state.json", {"account_equity": 100000.0, "realized_pnl": 12.34})
    _write(
        tmp_path / "scr_state.json",
        {
            "sizing_factor": 0.1,
            "stats": {
                "total_pnl": 9880.5,
                "effective_trades": 190,
                "win_rate": 0.7736842105263158,
            },
        },
    )
    _write(
        tmp_path / "epoch_state.json",
        {"active_epoch": "CHAD_v8.9_Paper_Epoch_2", "epoch_started_at_utc": "2026-05-04T00:54:30Z"},
    )

    builder = api.StateBuilder()
    p = builder._portfolio()

    # Non-breaking guard: existing values are byte-identical to pre-change.
    assert p["total_paper_pnl"] == 9880.5
    assert p["total_paper_pnl_label"] == "+$9,880"
    assert p["today_realized_pnl"] == 12.34
    assert p["effective_trades"] == 190
    assert p["sizing_factor"] == 0.1

    # Additive definition fields.
    assert "total_paper_pnl_definition" in p
    assert isinstance(p["total_paper_pnl_definition"], str)
    assert "SCR-effective" in p["total_paper_pnl_definition"]
    assert "Epoch-2" in p["total_paper_pnl_definition"]

    assert "total_paper_pnl_source" in p
    assert isinstance(p["total_paper_pnl_source"], str)
    assert "scr_state.json" in p["total_paper_pnl_source"]

    assert p["total_paper_pnl_window_days"] == 60
    assert "total_paper_pnl_epoch" in p
    assert p["total_paper_pnl_epoch"] == "CHAD_v8.9_Paper_Epoch_2"


def test_dashboard_pnl_definition_epoch_defensive_when_missing(api, tmp_path: Path, monkeypatch) -> None:
    """If runtime/epoch_state.json is absent/unparseable, total_paper_pnl_epoch
    is None — never raises."""
    monkeypatch.setattr(api, "RUNTIME", tmp_path)
    # No epoch_state.json file written.
    _write(tmp_path / "pnl_state.json", {"account_equity": 100000.0, "realized_pnl": None})
    _write(tmp_path / "scr_state.json", {"stats": {"total_pnl": 0.0}})

    builder = api.StateBuilder()
    p = builder._portfolio()

    assert "total_paper_pnl_epoch" in p
    assert p["total_paper_pnl_epoch"] is None
    # And the definition/source/window fields are still present.
    assert isinstance(p["total_paper_pnl_definition"], str)
    assert p["total_paper_pnl_window_days"] == 60
