"""W4A-6 — LC5 short-horizon drawdown budgets (5d/20d, epoch-scoped, v2).

Trust guards (PLAN_W4A §5.1 / P7): sample-depth counts, epoch-eligibility
(rows before the epoch start are excluded from the windows), null when a
horizon has no rows. The 60d fields are unchanged (report-only).
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.risk.drawdown_guard import compute_drawdown, report_to_state_dict


def _hist(tmp_path, rows):
    p = tmp_path / "equity_history.ndjson"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _row(date, eq):
    return {"date_utc": date, "total_equity_cad": eq,
            "schema_version": "equity_history.v2"}


def _epoch(tmp_path, start_date):
    p = tmp_path / "epoch_state.json"
    p.write_text(json.dumps({
        "schema_version": "epoch_state.v1",
        "epoch_started_at_utc": f"{start_date}T00:00:00Z",
    }))
    return p


def _compute(tmp_path, rows, *, epoch=None, current=None):
    hp = _hist(tmp_path, rows)
    snap = None
    if current is not None:
        snap = tmp_path / "portfolio_snapshot.json"
        snap.write_text(json.dumps({"ibkr_equity": current}))
    return compute_drawdown(
        equity_history_path=hp,
        portfolio_snapshot_path=snap or (tmp_path / "nope_snap.json"),
        pnl_state_path=tmp_path / "nope_pnl.json",
        epoch_state_path=_epoch(tmp_path, epoch) if epoch else None,
    )


# --------------------------------------------------------------------------- #
# Window math
# --------------------------------------------------------------------------- #

def test_5d_drawdown_from_window_peak(tmp_path):
    rows = [
        _row("2026-07-15", 1000.0),
        _row("2026-07-16", 1050.0),  # peak within trailing 5
        _row("2026-07-17", 1030.0),
        _row("2026-07-18", 1020.0),
        _row("2026-07-19", 1010.0),
    ]
    rep = _compute(tmp_path, rows, current=1000.0)
    # trailing-5 peak (incl current appended? no — current measured vs rows) = 1050
    assert rep.sample_count_5d == 5
    assert abs(rep.dd_5d_pct - ((1000.0 - 1050.0) / 1050.0 * 100.0)) < 1e-6


def test_20d_wider_window(tmp_path):
    rows = [_row(f"2026-06-{d:02d}", 1000.0 + d) for d in range(1, 26)]
    rep = _compute(tmp_path, rows, current=1005.0)
    # 20d uses the trailing 20 rows (days 06..25), peak = 1025
    assert rep.sample_count_20d == 20
    assert abs(rep.dd_20d_pct - ((1005.0 - 1025.0) / 1025.0 * 100.0)) < 1e-6
    # 5d uses trailing 5 (days 21..25), peak = 1025 too
    assert rep.sample_count_5d == 5


def test_insufficient_5d_samples_reports_count(tmp_path):
    rows = [_row("2026-07-18", 1000.0), _row("2026-07-19", 990.0)]
    rep = _compute(tmp_path, rows, current=980.0)
    assert rep.sample_count_5d == 2  # < 5 → LC5 will not enforce (W4A-7)
    assert rep.dd_5d_pct is not None  # still reports the number


def test_zero_rows_null_dd(tmp_path):
    rep = _compute(tmp_path, [], current=1000.0)
    assert rep.dd_5d_pct is None and rep.sample_count_5d == 0
    assert rep.dd_20d_pct is None and rep.sample_count_20d == 0


# --------------------------------------------------------------------------- #
# Epoch scoping (H3 phantom-HWM guard, P7)
# --------------------------------------------------------------------------- #

def test_pre_epoch_rows_excluded_from_windows(tmp_path):
    rows = [
        _row("2026-06-01", 5000.0),   # PHANTOM pre-epoch peak (H3)
        _row("2026-06-29", 4800.0),   # still pre-epoch
        _row("2026-07-01", 1000.0),   # epoch begins
        _row("2026-07-02", 1010.0),
        _row("2026-07-03", 990.0),
    ]
    rep = _compute(tmp_path, rows, epoch="2026-07-01", current=1000.0)
    # Only the 3 epoch-eligible rows count; the 5000 phantom peak is excluded.
    assert rep.sample_count_5d == 3
    assert rep.dd_5d_pct is not None
    # peak among epoch rows = 1010, NOT 5000
    assert abs(rep.dd_5d_pct - ((1000.0 - 1010.0) / 1010.0 * 100.0)) < 1e-6


def test_no_epoch_uses_all_dated_rows(tmp_path):
    rows = [_row("2026-06-01", 5000.0), _row("2026-07-01", 1000.0)]
    rep = _compute(tmp_path, rows, current=1000.0)  # no epoch path
    assert rep.sample_count_5d == 2
    # peak 5000 now counts (no epoch guard)
    assert abs(rep.dd_5d_pct - ((1000.0 - 5000.0) / 5000.0 * 100.0)) < 1e-6


# --------------------------------------------------------------------------- #
# Schema v2 + backward compat
# --------------------------------------------------------------------------- #

def test_state_dict_is_v2_with_budget_fields(tmp_path):
    rows = [_row("2026-07-18", 1000.0), _row("2026-07-19", 990.0)]
    rep = _compute(tmp_path, rows, current=980.0)
    state = report_to_state_dict(rep, ts_utc="2026-07-23T14:00:00Z")
    assert state["schema_version"] == "drawdown_state.v2"
    for k in ("dd_5d_pct", "dd_20d_pct", "sample_count_5d", "sample_count_20d"):
        assert k in state
    # legacy fields preserved unchanged
    for k in ("current_equity_cad", "hwm_cad", "drawdown_pct", "halt",
              "enforcement_active", "sample_count", "lookback_days"):
        assert k in state
    assert state["enforcement_active"] is False


def test_null_dd_serializes_as_null(tmp_path):
    rep = _compute(tmp_path, [], current=1000.0)
    state = report_to_state_dict(rep, ts_utc="2026-07-23T14:00:00Z")
    assert state["dd_5d_pct"] is None and state["dd_20d_pct"] is None


def test_60d_hwm_unchanged_by_windows(tmp_path):
    """The 60d field still measures vs the full-window HWM (report-only)."""
    rows = [_row("2026-07-15", 1200.0), _row("2026-07-19", 1000.0)]
    rep = _compute(tmp_path, rows, current=1000.0)
    # 60d HWM = max(1200, 1000, current 1000) = 1200
    assert abs(rep.drawdown_pct - ((1000.0 - 1200.0) / 1200.0 * 100.0)) < 1e-6
    assert rep.enforcement_active is False
