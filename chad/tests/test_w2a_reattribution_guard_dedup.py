"""W2A-3 (D7 proof) — UNH re-attribution leaves NO broker_sync|UNH over-count.

Executable half of docs/W2A_item2_guard_dedup_proof.md. Proves, against the REAL reader
functions, that writing an OPEN `gamma|UNH`=228 alongside the ever-present `broker_sync|UNH`=228
mirror never sums to 456/451 in any consumer — the guard dual-books the same shares and every
reader compares legs like-with-like, never sums them. No re-attribution script exists until
this passes (D7).
"""
from __future__ import annotations

from datetime import datetime, timezone

from chad.core.position_guard import (
    _agg_guard_broker_mirror,
    _agg_guard_strategy,
    detect_guard_vs_broker_drift_v2,
    detect_guard_vs_independent_snapshot_drift,
)
from chad.risk.position_exit_overlay import (
    _broker_signed_by_symbol,
    evaluate_positions,
    load_overlay_config,
)


# --------------------------------------------------------------------------- #
# The two guard states: today's live state, and the post-re-attribution state.
# --------------------------------------------------------------------------- #
def _current_guard() -> dict:
    """Live 2026-07-20: gamma|UNH CLOSED (223), broker_sync|UNH open 228."""
    return {
        "_version": 41,
        "gamma|UNH": {
            "open": False, "strategy": "gamma", "symbol": "UNH", "side": "SELL",
            "quantity": 223.0, "last_state": "CLOSED", "source": "paper_ledger_rebuild",
        },
        "broker_sync|UNH": {
            "open": True, "strategy": "broker_sync", "symbol": "UNH", "side": "BUY",
            "quantity": 228.0, "source": "broker_truth_rebuild",
        },
    }


def _post_reattribution_guard() -> dict:
    """After item-2 + one cycle: gamma|UNH open 228 AND broker_sync|UNH re-opened 228."""
    return {
        "_version": 42,
        "gamma|UNH": {
            "open": True, "strategy": "gamma", "symbol": "UNH", "side": "BUY",
            "quantity": 228.0, "last_state": "OPEN", "source": "paper_ledger_rebuild",
        },
        "broker_sync|UNH": {
            "open": True, "strategy": "broker_sync", "symbol": "UNH", "side": "BUY",
            "quantity": 228.0, "source": "broker_truth_rebuild",
        },
    }


# --------------------------------------------------------------------------- #
# Reader 1 — detect_guard_vs_broker_drift_v2 (the position_guard_drift emitter).
# --------------------------------------------------------------------------- #
def test_drift_v2_today_flags_broker_untracked():
    """Baseline: today UNH is an actionable broker_untracked_position (guard 0, broker 228)."""
    d = detect_guard_vs_broker_drift_v2(_current_guard())
    unh = [r for r in d["drifts"] if r["symbol"] == "UNH"]
    assert len(unh) == 1
    assert unh[0]["drift_kind"] == "broker_untracked_position"
    assert unh[0]["guard_qty"] == 0.0
    assert unh[0]["broker_qty"] == 228.0
    assert d["drift_count"] >= 1


def test_drift_v2_after_reattribution_is_matched_no_overcount():
    """After re-attribution guard_qty == broker_qty == 228 (compared, not summed) → NO drift.

    The 456/451 phantom would only appear if the reader summed the two legs; it does not."""
    d = detect_guard_vs_broker_drift_v2(_post_reattribution_guard())
    assert [r for r in d["drifts"] if r["symbol"] == "UNH"] == []
    assert d["counts_by_kind"]["qty_mismatch"] == 0
    assert d["counts_by_kind"]["broker_untracked_position"] == 0
    assert d["drift_count"] == 0  # the actionable drift that existed today is GONE


# --------------------------------------------------------------------------- #
# Reader 2 — detect_guard_vs_independent_snapshot_drift (v4, three independent legs).
# --------------------------------------------------------------------------- #
def _fresh_snapshot(ts: str = "2026-07-20T20:00:00Z") -> dict:
    return {"ts_utc": ts, "ttl_seconds": 300, "positions": [{"symbol": "UNH", "position": 228.0}]}


def test_drift_v4_after_reattribution_all_legs_agree():
    snap = _fresh_snapshot()
    # 'now' just after the snapshot ts so the independent leg is fresh (not blind).
    now = datetime(2026, 7, 20, 20, 0, 10, tzinfo=timezone.utc).timestamp()
    d = detect_guard_vs_independent_snapshot_drift(_post_reattribution_guard(), snap, now=now)
    assert d["independent_leg"] == "fresh"
    # broker=228, mirror=228, strategy=228 — each compared to the independent leg, never summed.
    assert [r for r in d["drifts"] if r["symbol"] == "UNH"] == []
    assert d["drift_count"] == 0


def test_aggregation_legs_are_separate_never_summed():
    """The two guard legs are each 228 and DISTINCT; summing them (=456) is what the codebase
    forbids. Assert both aggregators independently, proving neither is 456."""
    post = _post_reattribution_guard()
    assert _agg_guard_strategy(post) == {"UNH": 228.0}
    assert _agg_guard_broker_mirror(post) == {"UNH": 228.0}


# --------------------------------------------------------------------------- #
# Reader 3 — the exit overlay (item-2's consumer): manages gamma|UNH, skips the mirror.
# --------------------------------------------------------------------------- #
def _overlay_config():
    return load_overlay_config({
        "mode": "shadow",
        "atr_period": 14,
        "atr_trail_mult": 3.0,
        "hard_stop_loss_pct": 0.05,
        "min_bars_for_atr": 20,
        "max_hold_days": {"default": 5, "equity": 5},
    })


def test_exit_overlay_manages_gamma_not_broker_sync():
    post = _post_reattribution_guard()
    open_positions = {k: v for k, v in post.items()
                      if isinstance(v, dict) and v.get("open")}
    result = evaluate_positions(
        open_positions=open_positions,
        guard_state=post,
        bars_by_symbol={},
        price_by_symbol={"UNH": 424.97},
        anchors={},
        config=_overlay_config(),
        now_utc=datetime(2026, 7, 20, 20, 0, 0, tzinfo=timezone.utc),
    )
    keys = {v.position_key for v in result.verdicts}
    assert "gamma|UNH" in keys            # the strategy leg IS managed (item-2's goal)
    assert "broker_sync|UNH" not in keys  # the mirror is skipped (position_exit_overlay.py:633)


def test_exit_overlay_reduce_only_cap_is_broker_truth_not_double():
    """The reduce-only cross-check equals broker truth (228), NOT the summed 456 — so a close
    can never oversell even with both keys open."""
    post = _post_reattribution_guard()
    assert _broker_signed_by_symbol(post) == {"UNH": 228.0}
