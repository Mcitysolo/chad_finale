"""W4A-2 (D2a) — forward-only live-regime stamp in closed_trade.v1.

The D2 rider (audits/W4A_GO_RECORD.md §2): the stamp is fail-unknown —
missing/stale/corrupt/unrecognised classifier state stamps None, never a
guess. Absent key (pre-W4A-2 rows) and None must read identically as
"unknown" in the fuse counter.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from chad.execution.trade_closer import ClosedTrade, current_regime_stamp
from chad.risk.fuse_box import _norm_regime


def _write_regime(path, regime, *, age_seconds=10, ttl=360):
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    path.write_text(json.dumps({
        "schema_version": "regime_state.v1",
        "ts_utc": ts.isoformat().replace("+00:00", "Z"),
        "ttl_seconds": ttl,
        "regime": regime,
    }))


def _trade(**kw):
    base = dict(
        strategy="alpha", symbol="SPY", side="BUY",
        entry_price=500.0, exit_price=501.0, quantity=1.0,
        entry_time_utc="2026-07-23T13:00:00+00:00",
        exit_time_utc="2026-07-23T13:30:00+00:00",
        pnl=1.0, contract_multiplier=1.0, fill_ids=["a", "b"],
    )
    base.update(kw)
    return ClosedTrade(**base)


def test_stamp_fresh_regime(tmp_path):
    p = tmp_path / "regime_state.json"
    _write_regime(p, "ranging")
    assert current_regime_stamp(p) == "ranging"


def test_stamp_stale_regime_is_none(tmp_path):
    p = tmp_path / "regime_state.json"
    _write_regime(p, "ranging", age_seconds=3600, ttl=360)
    assert current_regime_stamp(p) is None


def test_stamp_missing_file_is_none(tmp_path):
    assert current_regime_stamp(tmp_path / "nope.json") is None


def test_stamp_corrupt_file_is_none(tmp_path):
    p = tmp_path / "regime_state.json"
    p.write_text("{broken")
    assert current_regime_stamp(p) is None


def test_stamp_unrecognised_vocabulary_is_none(tmp_path):
    p = tmp_path / "regime_state.json"
    _write_regime(p, "sideways_chop")
    assert current_regime_stamp(p) is None


def test_stamp_fresh_unknown_is_stamped(tmp_path):
    """A FRESH classifier honestly reporting 'unknown' is a valid stamp —
    it still lands in the unknown bucket, but the provenance differs from
    'classifier was dead' only in evidence, not in counting."""
    p = tmp_path / "regime_state.json"
    _write_regime(p, "unknown")
    assert current_regime_stamp(p) == "unknown"


def test_payload_carries_regime_key(monkeypatch):
    import chad.execution.trade_closer as tc

    monkeypatch.setattr(tc, "_regime_stamp_cache", (True, "volatile"))
    payload = _trade().to_payload()
    assert payload["regime"] == "volatile"


def test_payload_regime_none_when_unavailable(monkeypatch):
    import chad.execution.trade_closer as tc

    monkeypatch.setattr(tc, "_regime_stamp_cache", (True, None))
    payload = _trade().to_payload()
    assert "regime" in payload and payload["regime"] is None


def test_absent_and_none_read_identically_unknown():
    """Fuse-side contract: pre-W4A-2 rows (no key) and None-stamped rows
    both normalize to the unknown bucket (D2 rider)."""
    assert _norm_regime(None) == "unknown"
    assert _norm_regime("") == "unknown"
    assert _norm_regime("unknown") == "unknown"
    assert _norm_regime("ranging") == "ranging"


def test_stamp_additive_only_shape(monkeypatch):
    """The stamp is ONE additive key — no other payload key appears or
    disappears vs the pre-W4A-2 shape (hash-chain rows are forward-only)."""
    import chad.execution.trade_closer as tc

    monkeypatch.setattr(tc, "_regime_stamp_cache", (True, None))
    payload = _trade().to_payload()
    expected = {
        "schema_version", "strategy", "symbol", "side", "pnl", "gross_pnl",
        "commission", "slippage", "fees", "net_pnl", "entry_time_utc",
        "exit_time_utc", "fill_price", "entry_price", "exit_price",
        "quantity", "contract_multiplier", "notional", "fill_ids", "broker",
        "account_id", "is_live", "tags", "pnl_breakdown", "meta", "regime",
    }
    assert set(payload.keys()) == expected
