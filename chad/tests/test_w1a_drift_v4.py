"""
W1A-4 — pure independent-snapshot drift detector.

detect_guard_vs_independent_snapshot_drift compares the guard's two same-source
legs (broker_sync mirror + strategy) SEPARATELY against the ONE independent leg
(positions_snapshot.json). It is the pure lift of sentinel EXS4. These tests
lock the correctness rules that matter most: never sum the dual-booked legs
(no 2x phantom), freshness-gate the independent leg to blind (never false OK,
never false indict), and treat excluded symbols as info-only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from chad.core.position_guard import detect_guard_vs_independent_snapshot_drift as detect

TS = "2026-07-15T12:00:00Z"
EPOCH = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
NOW = EPOCH + 30  # 30s after the snapshot -> fresh


def _guard_dualbook(symbol: str = "UNH", qty: float = 273.0) -> dict:
    """Guard as it really is: gamma|SYM AND broker_sync|SYM for the SAME shares."""
    return {
        "_version": 7,
        f"gamma|{symbol}": {"symbol": symbol, "quantity": qty, "side": "BUY", "open": True, "strategy": "gamma"},
        f"broker_sync|{symbol}": {"symbol": symbol, "quantity": qty, "side": "BUY", "open": True, "strategy": "broker_sync"},
    }


def _snap(positions: list, ttl: float = 300, ts: str = TS) -> dict:
    return {"ts_utc": ts, "ttl_seconds": ttl, "positions": positions}


# ---------------------------------------------------------------------------
# Agreement + the dual-book invariant (highest-priority correctness rule)
# ---------------------------------------------------------------------------


def test_dualbook_agreement_no_spurious_2x_drift() -> None:
    guard = _guard_dualbook("UNH", 273.0)
    snap = _snap([{"symbol": "UNH", "position": 273.0}])
    r = detect(guard, snap, now=NOW)
    assert r["independent_leg"] == "fresh"
    assert r["drift_count"] == 0
    assert r["info_count"] == 0
    assert r["drifts"] == []  # NOT a 2x phantom from summing mirror+strategy


# ---------------------------------------------------------------------------
# Actionable drift kinds
# ---------------------------------------------------------------------------


def test_mirror_side_flip_is_actionable() -> None:
    # Guard mirror says +273, independent collector says -273 (a flip).
    guard = _guard_dualbook("UNH", 273.0)
    snap = _snap([{"symbol": "UNH", "position": -273.0}])
    r = detect(guard, snap, now=NOW)
    assert r["drift_count"] == 1
    row = r["drifts"][0]
    assert row["drift_kind"] == "mirror_vs_independent_broker"
    assert row["is_excluded"] is False
    assert row["independent_broker_qty"] == -273.0


def test_phantom_guard_entry_when_snapshot_empty() -> None:
    guard = {"_version": 7, "gamma|XYZ": {"symbol": "XYZ", "quantity": 100, "side": "BUY", "open": True, "strategy": "gamma"}}
    snap = _snap([])
    r = detect(guard, snap, now=NOW)
    assert r["drift_count"] == 1
    assert r["drifts"][0]["drift_kind"] == "phantom_guard_entry"


# ---------------------------------------------------------------------------
# Excluded (mixed-ownership) symbols are informational, never actionable
# ---------------------------------------------------------------------------


def test_excluded_symbol_is_info_only() -> None:
    guard = _guard_dualbook("UNH", 273.0)
    snap = _snap([{"symbol": "UNH", "position": -273.0}])  # would be actionable...
    r = detect(guard, snap, now=NOW, excluded_symbols={"UNH"})
    assert r["drift_count"] == 0  # ...but excluded -> not actionable
    assert r["info_count"] == 1
    assert r["drifts"][0]["drift_kind"] == "mixed_ownership_info"
    assert r["drifts"][0]["is_excluded"] is True


# ---------------------------------------------------------------------------
# Freshness gate: blind, never a false OK / false indict
# ---------------------------------------------------------------------------


def test_stale_snapshot_is_blind_not_false_indict() -> None:
    guard = _guard_dualbook("UNH", 273.0)
    snap = _snap([{"symbol": "UNH", "position": -273.0}], ttl=300)  # drift present...
    # ...but the snapshot is age > ttl*3 -> it cannot prove anything.
    r = detect(guard, snap, now=EPOCH + 300 * 3 + 60)
    assert r["independent_leg"] == "blind"
    assert r["blind_reason"] == "snapshot_stale"
    assert r["drift_count"] == 0  # never indicts on a stale leg
    assert r["snapshot_age_seconds"] is not None


def test_missing_snapshot_is_blind_no_crash() -> None:
    r = detect(_guard_dualbook(), None, now=NOW)
    assert r["independent_leg"] == "blind"
    assert r["blind_reason"] == "snapshot_unreadable"
    assert r["drift_count"] == 0


def test_no_timestamp_snapshot_is_blind() -> None:
    guard = _guard_dualbook()
    snap = {"ttl_seconds": 300, "positions": [{"symbol": "UNH", "position": 273.0}]}  # no ts_utc
    r = detect(guard, snap, now=NOW)
    assert r["independent_leg"] == "blind"
    assert r["blind_reason"] == "snapshot_no_timestamp"


def test_guard_unreadable_is_blind() -> None:
    r = detect(None, _snap([]), now=NOW)  # type: ignore[arg-type]
    assert r["independent_leg"] == "blind"
    assert r["blind_reason"] == "guard_unreadable"


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_schema_shape_and_generation_echo() -> None:
    guard = _guard_dualbook("UNH", 273.0)
    snap = _snap([{"symbol": "UNH", "position": 273.0}])
    r = detect(guard, snap, now=NOW)
    for key in ("schema_version", "independent_leg", "blind_reason", "snapshot_age_seconds",
                "snapshot_generation", "drift_count", "info_count", "counts_by_kind", "drifts"):
        assert key in r
    assert r["schema_version"] == "position_guard_drift.v4"
    assert r["snapshot_generation"] == 7  # echoed from guard _version
