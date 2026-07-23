"""W4A-7 — LC5 enforcement: ladder factor, staleness guard, emergency block.

Includes the compute engine (ladder/staleness/emergency) and the sizing helper.
The D5-rider "exits always free" named regression lives in
test_w4a_lc5_emergency_exits_free.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from chad.risk.fuse_box import (
    FuseBoxConfig,
    apply_lc5_factor,
    compute_lc5_state,
)

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)


def _cfg(tmp_path):
    """Config with the real ladder."""
    p = tmp_path / "fuse_box.json"
    p.write_text(json.dumps({
        "lc5_ladder": {
            "dd_5d": [{"at_pct": -3.0, "factor": 0.75},
                      {"at_pct": -5.0, "factor": 0.50},
                      {"at_pct": -8.0, "factor": 0.25}],
            "dd_20d": [{"at_pct": -8.0, "factor": 0.50},
                       {"at_pct": -12.0, "factor": 0.25}],
            "emergency_halt_pct": -15.0,
            "stale_max_seconds": 3600,
        },
    }))
    return FuseBoxConfig.load(p)


def _dd_state(tmp_path, *, dd_5d=None, dd_20d=None, n5=5, n20=20, halt=False,
              age_seconds=10, ttl=300, corrupt=False, missing=False):
    p = tmp_path / "drawdown_state.json"
    if missing:
        return p
    if corrupt:
        p.write_text("{broken")
        return p
    # Freshness (read_runtime_state_json) compares ts_utc to REAL wall-clock,
    # so the drawdown_state ts must be stamped relative to now, not the fixed
    # NOW used for the session-window logic.
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    p.write_text(json.dumps({
        "schema_version": "drawdown_state.v2",
        "ts_utc": ts.isoformat().replace("+00:00", "Z"),
        "ttl_seconds": ttl,
        "status": "ok",
        "drawdown_pct": -1.0,
        "halt": halt,
        "dd_5d_pct": dd_5d,
        "dd_20d_pct": dd_20d,
        "sample_count_5d": n5,
        "sample_count_20d": n20,
    }))
    return p


# --------------------------------------------------------------------------- #
# Ladder
# --------------------------------------------------------------------------- #

def test_no_drawdown_factor_1(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-1.0, dd_20d=-2.0)
    lc5, ev = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["factor"] == 1.0
    assert lc5["staleness"] == "fresh"


def test_5d_first_rung(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-3.5, dd_20d=-1.0)
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["factor"] == 0.75


def test_min_across_windows(tmp_path):
    """5d at -6 → 0.50; 20d at -9 → 0.50; min = 0.50."""
    dd = _dd_state(tmp_path, dd_5d=-6.0, dd_20d=-9.0)
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["factor"] == 0.50


def test_deepest_rung_wins(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-9.0, dd_20d=-1.0)  # 5d past -8 → 0.25
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["factor"] == 0.25


def test_depth_guard_shallow_window_ignored(tmp_path):
    """dd_5d at -9 but only 3 samples (<5) → that window does not enforce."""
    dd = _dd_state(tmp_path, dd_5d=-9.0, dd_20d=-1.0, n5=3)
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["factor"] == 1.0  # 5d ignored, 20d shallow drawdown


# --------------------------------------------------------------------------- #
# Emergency (D5)
# --------------------------------------------------------------------------- #

def test_emergency_on_existing_halt_boolean(tmp_path):
    """D5: the existing −15% halt boolean wired as emergency."""
    dd = _dd_state(tmp_path, dd_5d=-1.0, dd_20d=-1.0, halt=True)
    lc5, ev = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["emergency"] is True
    assert any(e["event"] == "lc5_emergency" for e in ev)


def test_emergency_on_window_below_threshold(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-16.0, dd_20d=-1.0)
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), drawdown_state_path=dd)
    assert lc5["emergency"] is True


def test_emergency_cleared_event(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-1.0, dd_20d=-1.0, halt=False)
    prior = {"emergency": True, "staleness": "fresh"}
    lc5, ev = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                                drawdown_state_path=dd)
    assert lc5["emergency"] is False
    assert any(e["event"] == "lc5_emergency_cleared" for e in ev)


# --------------------------------------------------------------------------- #
# Staleness (§5.4) — the mandatory A4-grade guard
# --------------------------------------------------------------------------- #

def test_stale_holds_last_factor(tmp_path):
    """Stale within stale_max: hold the prior factor, never tighten/loosen."""
    dd = _dd_state(tmp_path, dd_5d=-9.0, age_seconds=400, ttl=300)  # stale
    prior = {"factor": 0.5, "worst_factor_session": 0.5, "staleness": "fresh"}
    lc5, ev = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                                drawdown_state_path=dd)
    assert lc5["factor"] == 0.5  # held, NOT the -9 → 0.25 the fresh value implies
    assert lc5["staleness"] == "stale"
    assert any(e["marker"] == "FUSE_LC5_DRAWDOWN_STALE" for e in ev)


def test_missing_holds_last(tmp_path):
    dd = _dd_state(tmp_path, missing=True)
    prior = {"factor": 0.75, "worst_factor_session": 0.75, "staleness": "fresh"}
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                               drawdown_state_path=dd)
    assert lc5["factor"] == 0.75


def test_corrupt_holds_last(tmp_path):
    dd = _dd_state(tmp_path, corrupt=True)
    prior = {"factor": 0.5, "worst_factor_session": 0.5, "staleness": "fresh"}
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                               drawdown_state_path=dd)
    assert lc5["factor"] == 0.5


def test_stale_beyond_max_degrades_to_worst(tmp_path):
    """Stale > stale_max_seconds: degrade to the session worst rung (never up)."""
    dd = _dd_state(tmp_path, age_seconds=7200, ttl=300)  # 2h stale, max 3600
    prior = {"factor": 0.75, "worst_factor_session": 0.25, "staleness": "fresh"}
    lc5, _ = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                               drawdown_state_path=dd)
    assert lc5["factor"] == 0.25  # worst, not the gentler 0.75 last-applied


def test_restored_event_on_recovery(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-1.0)  # fresh
    prior = {"factor": 0.5, "staleness": "stale"}
    lc5, ev = compute_lc5_state(NOW, config=_cfg(tmp_path), prior_lc5=prior,
                                drawdown_state_path=dd)
    assert lc5["staleness"] == "fresh"
    assert any(e["marker"] == "FUSE_LC5_DRAWDOWN_RESTORED" for e in ev)


def test_worst_factor_resets_on_session_roll(tmp_path):
    dd = _dd_state(tmp_path, dd_5d=-1.0)  # fresh, factor 1.0
    prior = {"factor": 0.25, "worst_factor_session": 0.25, "staleness": "fresh",
             "session_window_start": "2026-07-22T00:00:00Z"}
    lc5, _ = compute_lc5_state(
        NOW, config=_cfg(tmp_path), prior_lc5=prior,
        prior_session_window_start="2026-07-22T00:00:00Z",
        session_window_start_utc=datetime(2026, 7, 23, tzinfo=timezone.utc),
        drawdown_state_path=dd,
    )
    # new session → worst resets, fresh factor 1.0
    assert lc5["worst_factor_session"] == 1.0
    assert lc5["factor"] == 1.0


# --------------------------------------------------------------------------- #
# Sizing helper (D4 mechanics — SCR-CAUTIOUS rounding)
# --------------------------------------------------------------------------- #

def test_apply_factor_equity_floor_min_1():
    assert apply_lc5_factor(10.0, 0.5, "STK") == 5.0
    assert apply_lc5_factor(3.0, 0.5, "STK") == 1.0   # floor(1.5)=1
    assert apply_lc5_factor(1.0, 0.25, "STK") == 1.0  # min 1 share


def test_apply_factor_fut_round_min_1():
    assert apply_lc5_factor(4.0, 0.5, "FUT") == 2.0
    assert apply_lc5_factor(3.0, 0.5, "FUT") == 2.0   # round(1.5)=2
    assert apply_lc5_factor(1.0, 0.25, "FUT") == 1.0  # min 1 contract


def test_apply_factor_1_is_noop():
    assert apply_lc5_factor(7.0, 1.0, "STK") == 7.0
