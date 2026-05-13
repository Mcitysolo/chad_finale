"""Gap-4 (v9.1 audit) — meta forwarding into closed_trade.v1.

Verifies that the strategy meta block (setup_family, stop_width_usd,
stop_width_points, session_window, r_target_1, r_target_2, ...) survives
the FIFO matcher and lands on closed_trade.v1 records so the analytics
layer (setup_family_expectancy_updater) can bucket alpha_intraday_micro
trades correctly.
"""

from __future__ import annotations

import json
import logging
import pathlib

import pytest

from chad.execution.trade_closer import (
    ClosedTrade,
    TradeCloser,
    _sanitize_meta,
)


# ---------------------------------------------------------------------------
# Helpers (mirror chad/tests/test_trade_closer.py style)
# ---------------------------------------------------------------------------


_ALPHA_MICRO = "alpha_intraday_micro"


def _fill(
    fid: str,
    side: str,
    qty: float,
    px: float,
    *,
    strategy: str = _ALPHA_MICRO,
    symbol: str = "MES",
    ts: str = "2026-05-12T13:35:00+00:00",
    seq: int = 1,
    meta=None,
) -> dict:
    payload = {
        "schema_version": "paper_exec_fill.v4",
        "fill_id": fid,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "fill_price": px,
        "fill_time_utc": ts,
        "entry_time_utc": ts,
        "is_live": False,
        "reject": False,
        "status": "paper_fill",
    }
    # Only attach a top-level `meta` field when explicitly asked: the
    # production fill writer does not yet emit one, so omitting it
    # exercises the position_guard fallback path (and Test 5/6).
    if meta is not None:
        payload["meta"] = meta
    return {
        "payload": payload,
        "sequence_id": seq,
        "timestamp_utc": ts,
        "prev_hash": "GENESIS",
        "record_hash": fid,
    }


def _write_fills(path: pathlib.Path, fills: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for f in fills:
            fh.write(json.dumps(f) + "\n")


def _full_micro_meta() -> dict:
    return {
        "setup_family": "ORB",
        "stop_width_usd": 12.50,
        "stop_width_points": 2.5,
        "session_window": "RTH_OPEN_30",
        "risk_budget_usd": 500.0,
        "r_target_1": 1.0,
        "r_target_2": 2.0,
    }


def _make_closer(tmp_path: pathlib.Path, *, guard_path=None) -> TradeCloser:
    return TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=tmp_path / "state.json",
        routing_path=tmp_path / "profit_routing.json",
        position_guard_path=guard_path,
    )


def _round_trip(
    tmp_path: pathlib.Path,
    *,
    open_meta=None,
    close_meta=None,
    guard_meta=None,
) -> dict:
    """Open a long, close it, and return the closed_trade.v1 payload dict.

    When *guard_meta* is provided, a position_guard.json is written next to
    the temp state so trade_closer's lot-creation lookup can find it.
    """
    guard_path = None
    if guard_meta is not None:
        guard_path = tmp_path / "position_guard.json"
        guard_path.write_text(
            json.dumps(
                {
                    f"{_ALPHA_MICRO}|MES": {
                        "open": True,
                        "strategy": _ALPHA_MICRO,
                        "symbol": "MES",
                        "side": "BUY",
                        "quantity": 1.0,
                        "last_state": "OPEN",
                        "meta": guard_meta,
                    }
                }
            ),
            encoding="utf-8",
        )
    closer = _make_closer(tmp_path, guard_path=guard_path)
    fills = [
        _fill("open-1", "BUY", 1, 5000.0, seq=1, meta=open_meta),
        _fill("close-1", "SELL", 1, 5010.0, seq=2, meta=close_meta),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260512.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260512")
    assert len(closed) == 1, "round trip should produce exactly one closed_trade"
    return closed[0].to_payload()


# ---------------------------------------------------------------------------
# Tests 1-4: full alpha_intraday_micro meta survives the round trip
# ---------------------------------------------------------------------------


def test_meta_forwarding_setup_family_present(tmp_path):
    """Test 1: setup_family="ORB" appears on closed_trade.v1.meta."""
    payload = _round_trip(tmp_path, open_meta=_full_micro_meta())
    assert isinstance(payload.get("meta"), dict)
    assert payload["meta"].get("setup_family") == "ORB"


def test_meta_forwarding_stop_width_usd_present(tmp_path):
    """Test 2: stop_width_usd survives onto the closed_trade.v1 payload."""
    payload = _round_trip(tmp_path, open_meta=_full_micro_meta())
    assert payload["meta"].get("stop_width_usd") == pytest.approx(12.50)


def test_meta_forwarding_stop_width_points_present(tmp_path):
    """Test 3: stop_width_points survives onto the closed_trade.v1 payload."""
    payload = _round_trip(tmp_path, open_meta=_full_micro_meta())
    assert payload["meta"].get("stop_width_points") == pytest.approx(2.5)


def test_meta_forwarding_session_window_present(tmp_path):
    """Test 4: session_window survives onto the closed_trade.v1 payload."""
    payload = _round_trip(tmp_path, open_meta=_full_micro_meta())
    assert payload["meta"].get("session_window") == "RTH_OPEN_30"


# ---------------------------------------------------------------------------
# Tests 5-6: missing / None meta defaults safely to {}
# ---------------------------------------------------------------------------


def test_meta_forwarding_missing_defaults_to_empty(tmp_path):
    """Test 5: alpha_intraday_micro round trip with no meta on either fill
    (and no position_guard entry) emits payload["meta"] == {} without
    crashing."""
    payload = _round_trip(tmp_path)
    assert payload.get("meta") == {}


def test_meta_forwarding_none_defaults_to_empty(tmp_path):
    """Test 6: a fill payload that carries an explicit ``meta=None`` value
    must NOT crash _extract_fill or _sanitize_meta — the closer treats it
    as absent and emits meta={}."""
    closer = _make_closer(tmp_path)
    fills = [
        _fill("open-1", "BUY", 1, 5000.0, seq=1),
        _fill("close-1", "SELL", 1, 5010.0, seq=2),
    ]
    # Force payload.meta = None (cannot use _fill(meta=None) helper because
    # that branch omits the key — here we want the key present and == None).
    fills[0]["payload"]["meta"] = None
    fills[1]["payload"]["meta"] = None
    _write_fills(tmp_path / "fills" / "FILLS_20260512.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260512")
    assert len(closed) == 1
    assert closed[0].to_payload().get("meta") == {}


# ---------------------------------------------------------------------------
# Test 7: schema fields preserved
# ---------------------------------------------------------------------------


def test_meta_forwarding_existing_schema_fields_preserved(tmp_path):
    """Test 7: every existing closed_trade.v1 field must still be present
    after the additive `meta` key is introduced."""
    payload = _round_trip(tmp_path, open_meta=_full_micro_meta())
    required = (
        "schema_version",
        "strategy",
        "symbol",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "pnl",
        "gross_pnl",
        "net_pnl",
        "fees",
        "exit_time_utc",
    )
    missing = [k for k in required if k not in payload]
    assert not missing, f"closed_trade.v1 missing required fields: {missing}"
    assert payload["schema_version"] == "closed_trade.v1"


# ---------------------------------------------------------------------------
# Test 8: non-JSON-serializable meta values are sanitized
# ---------------------------------------------------------------------------


def test_meta_forwarding_non_json_value_sanitized(caplog):
    """Test 8: a non-JSON-serializable value (a set, bytes) inside meta
    must not crash ClosedTrade.to_payload(). The emitted payload must
    remain JSON-serializable (so the ledger writer can persist it) and
    a WARNING is logged.

    The sanitizer is the defense-in-depth surface for in-memory meta
    paths — programmatic lot construction in tests, legacy state files,
    and any future writer that forwards strategy objects without first
    serializing them — even though paper_exec_evidence_writer's own JSON
    write would catch wholly-unserializable values upstream."""
    bad_meta = {
        "setup_family": "ORB",
        "stop_width_usd": 12.5,
        # canonical "JSON cannot encode this" cases
        "tags_set": {"a", "b"},
        "raw_bytes": b"\xde\xad",
    }
    ct = ClosedTrade(
        strategy=_ALPHA_MICRO,
        symbol="MES",
        side="BUY",
        entry_price=5000.0,
        exit_price=5010.0,
        quantity=1.0,
        entry_time_utc="2026-05-12T13:35:00+00:00",
        exit_time_utc="2026-05-12T14:00:00+00:00",
        pnl=50.0,
        contract_multiplier=5.0,
        fill_ids=["a", "b"],
        meta=bad_meta,
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.trade_closer"):
        payload = ct.to_payload()

    # Sanitized meta still carries the known-safe scalars.
    assert payload["meta"].get("setup_family") == "ORB"
    assert payload["meta"].get("stop_width_usd") == pytest.approx(12.5)
    # Unsafe values were either dropped or stringified; payload must remain
    # JSON-serializable end-to-end.
    json.dumps(payload)
    # At least one WARNING about sanitization must have been emitted.
    sanitize_warnings = [
        r for r in caplog.records
        if "trade_closer_meta_sanitize" in r.getMessage()
        or "trade_closer_meta_drop" in r.getMessage()
    ]
    assert sanitize_warnings, "expected at least one sanitize/drop warning"


# ---------------------------------------------------------------------------
# Test 9: end-to-end — updater consumes synthetic closed_trade.v1 record
# ---------------------------------------------------------------------------


def test_meta_forwarding_updater_can_consume_setup_family(tmp_path):
    """Test 9: write a synthetic closed_trade.v1 record with
    meta.setup_family="ORB" and meta.stop_width_usd, then run
    SetupFamilyExpectancyUpdater against it — the ORB family must
    increment trades and report a non-null avg_r."""
    from chad.analytics.setup_family_expectancy_updater import (
        SetupFamilyExpectancyUpdater,
    )

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "setup_family_expectancy.json"

    # Build a closed_trade.v1 record matching the production schema.
    ct = ClosedTrade(
        strategy=_ALPHA_MICRO,
        symbol="MES",
        side="BUY",
        entry_price=5000.0,
        exit_price=5010.0,
        quantity=1.0,
        entry_time_utc="2026-05-12T13:35:00+00:00",
        exit_time_utc="2026-05-12T14:00:00+00:00",
        pnl=50.0,
        contract_multiplier=5.0,
        fill_ids=["open-1", "close-1"],
        meta=_full_micro_meta(),
    )
    record = {
        "payload": ct.to_payload(),
        "sequence_id": 1,
        "timestamp_utc": "2026-05-12T14:00:00+00:00",
        "prev_hash": "GENESIS",
        "record_hash": "abc",
    }
    (trades_dir / "trade_history_20260512.ndjson").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    # Lookback must cover the synthetic 2026-05-12 record.
    import datetime as _dt
    updater = SetupFamilyExpectancyUpdater(
        trades_dir=trades_dir,
        output_path=output_path,
        lookback_days=3650,
        now=_dt.datetime(2026, 5, 13, tzinfo=_dt.timezone.utc),
    )
    result = updater.run()

    orb = result["families"]["ORB"]
    assert orb["trades"] == 1, "ORB family should have one trade after consuming the synthetic record"
    assert orb["avg_r"] is not None, "ORB avg_r must be non-null when stop_width_usd is present"
    assert orb["avg_r"] == pytest.approx(50.0 / 12.5)


# ---------------------------------------------------------------------------
# Defensive: _sanitize_meta unit checks (cheap, isolates sanitizer logic)
# ---------------------------------------------------------------------------


def test_sanitize_meta_non_dict_returns_empty():
    """Sanitizer must coerce non-dict inputs (None, list, str, int) to {}."""
    assert _sanitize_meta(None) == {}
    assert _sanitize_meta("ORB") == {}
    assert _sanitize_meta([1, 2, 3]) == {}
    assert _sanitize_meta(42) == {}


def test_sanitize_meta_preserves_json_safe_scalars():
    """JSON-safe scalars (str, int, float, bool, None, nested list) survive
    unchanged through the sanitizer."""
    src = {
        "setup_family": "ORB",
        "stop_width_usd": 12.5,
        "active": True,
        "missing": None,
        "tags": ["a", "b"],
    }
    out = _sanitize_meta(src)
    assert out == src
    # Round-trip JSON-safe.
    json.dumps(out)
