"""W4A-4 — LC3 buckets (symbol + sector, anti-revenge).

Prime pin (PLAN_W4A §4): the unmapped sector bucket is COUNT-ONLY and can
never trip — a missing symbol_sectors row must not create a blockable bucket.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from chad.risk.fuse_box import (
    UNMAPPED_SECTOR,
    FuseBoxConfig,
    TrustedClose,
    build_lc3_buckets,
    compute_bucket_stats,
    load_sector_map,
    make_sector_lookup,
)

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)


def _close(symbol="TLT", pnl=-1.0, strategy="gamma", regime="unknown", ts=NOW):
    return TrustedClose(
        strategy=strategy, symbol=symbol, side="BUY", pnl=pnl, exit_ts=ts,
        regime=regime, setup_family=None, fill_ids=("a", "b"),
    )


def _cfg(tmp_path, obj=None):
    p = tmp_path / "fuse_box.json"
    p.write_text(json.dumps(obj or {"defaults": {"consecutive_losers": 3}}))
    return FuseBoxConfig.load(p)


SECTOR_MAP = {"TLT": "rates", "ZN": "rates", "SPY": "index_etf"}


# --------------------------------------------------------------------------- #
# Sector map loading
# --------------------------------------------------------------------------- #

def test_load_real_sector_map():
    m = load_sector_map()
    assert m.get("TLT") == "rates"
    assert m.get("SPY") == "index_etf"
    assert m.get("BTC-USD") == "crypto"


def test_load_missing_map_empty(tmp_path):
    assert load_sector_map(tmp_path / "nope.json") == {}


def test_load_corrupt_map_empty(tmp_path):
    p = tmp_path / "symbol_sectors.json"
    p.write_text("{broken")
    assert load_sector_map(p) == {}


def test_reserved_unmapped_name_ignored(tmp_path, caplog):
    p = tmp_path / "symbol_sectors.json"
    p.write_text(json.dumps({"sectors": {"unmapped": ["FOO"], "rates": ["TLT"]}}))
    with caplog.at_level(logging.WARNING, logger="chad.risk.fuse_box"):
        m = load_sector_map(p)
    assert "FOO" not in m and m.get("TLT") == "rates"
    assert any("RESERVED_NAME" in r.getMessage() for r in caplog.records)


def test_sector_lookup_unmapped_fallback():
    lookup = make_sector_lookup(SECTOR_MAP)
    assert lookup("TLT") == "rates"
    assert lookup("DOGE") == UNMAPPED_SECTOR


# --------------------------------------------------------------------------- #
# Symbol buckets
# --------------------------------------------------------------------------- #

def test_symbol_buckets_from_observed(tmp_path):
    cfg = _cfg(tmp_path)
    closes = [_close("TLT"), _close("TLT"), _close("SPY")]
    buckets = build_lc3_buckets(cfg, closes, SECTOR_MAP)
    syms = {b.fuse_id for b in buckets if b.kind == "symbol"}
    assert syms == {"symbol:TLT", "symbol:SPY"}


def test_symbol_bucket_trips_on_streak(tmp_path):
    cfg = _cfg(tmp_path)
    closes = [_close("TLT", -1), _close("TLT", -2), _close("TLT", -3)]
    bucket = next(
        b for b in build_lc3_buckets(cfg, closes, SECTOR_MAP)
        if b.fuse_id == "symbol:TLT"
    )
    stats = compute_bucket_stats(bucket, closes)
    assert stats.tripped and stats.consecutive_losers == 3


# --------------------------------------------------------------------------- #
# Sector buckets
# --------------------------------------------------------------------------- #

def test_sector_bucket_sums_symbols(tmp_path):
    cfg = _cfg(tmp_path)
    closes = [_close("TLT", -1), _close("ZN", -2), _close("TLT", -3)]
    lookup = make_sector_lookup(SECTOR_MAP)
    bucket = next(
        b for b in build_lc3_buckets(cfg, closes, SECTOR_MAP)
        if b.fuse_id == "sector:rates"
    )
    stats = compute_bucket_stats(bucket, closes, lookup)
    assert stats.matched == 3  # TLT + ZN + TLT all map to rates
    assert stats.consecutive_losers == 3
    assert stats.tripped


def test_unmapped_sector_bucket_never_trips(tmp_path, caplog):
    """The core LC3 safety pin: 3 losers on an unmapped symbol produce a
    count-only bucket that CANNOT trip."""
    cfg = _cfg(tmp_path)
    closes = [_close("DOGE", -1), _close("DOGE", -2), _close("DOGE", -3)]
    lookup = make_sector_lookup(SECTOR_MAP)
    with caplog.at_level(logging.WARNING, logger="chad.risk.fuse_box"):
        buckets = build_lc3_buckets(cfg, closes, SECTOR_MAP)
    sec = next(b for b in buckets if b.fuse_id == f"sector:{UNMAPPED_SECTOR}")
    assert sec.consecutive_losers is None
    assert sec.session_net_pnl_usd is None
    stats = compute_bucket_stats(sec, closes, lookup)
    assert stats.matched == 3  # counted (visible)
    assert not stats.tripped   # but never trippable
    assert any("FUSE_SECTOR_UNMAPPED" in r.getMessage() for r in caplog.records)


def test_unmapped_symbol_still_gets_symbol_bucket(tmp_path):
    """An unmapped symbol still gets its own symbol: bucket (that CAN trip) —
    only the SECTOR grain is disarmed for it."""
    cfg = _cfg(tmp_path)
    closes = [_close("DOGE", -1), _close("DOGE", -2), _close("DOGE", -3)]
    bucket = next(
        b for b in build_lc3_buckets(cfg, closes, SECTOR_MAP)
        if b.fuse_id == "symbol:DOGE"
    )
    assert bucket.consecutive_losers == 3
    stats = compute_bucket_stats(bucket, closes)
    assert stats.tripped


def test_sector_threshold_override(tmp_path):
    cfg = _cfg(tmp_path, {
        "defaults": {"consecutive_losers": 3},
        "sector_fuse": {"consecutive_losers": 2},
    })
    closes = [_close("TLT", -1), _close("ZN", -2)]
    lookup = make_sector_lookup(SECTOR_MAP)
    sec = next(
        b for b in build_lc3_buckets(cfg, closes, SECTOR_MAP)
        if b.fuse_id == "sector:rates"
    )
    assert sec.consecutive_losers == 2
    assert compute_bucket_stats(sec, closes, lookup).tripped


# --------------------------------------------------------------------------- #
# Full LC3 cycle wiring
# --------------------------------------------------------------------------- #

def test_lc3_cycle_end_to_end(tmp_path):
    import pathlib

    from chad.risk.fuse_box import run_evaluator_cycle
    from chad.tests.test_w4a_fuse_box import (
        _fill_row, _trade_row, _write_ndjson,
    )

    DATE = "20260723"
    fills = []
    trades = []
    for i in range(3):
        fills += [
            _fill_row(f"o{i}", "gamma", "TLT", "BUY", 1.0, 90.0, "paper_fill",
                      "2026-07-23T10:00:00+00:00", i),
            _fill_row(f"c{i}", "gamma", "TLT", "SELL", 1.0, 89.0, "paper_fill",
                      "2026-07-23T13:00:00+00:00", 10 + i),
        ]
        trades.append(
            _trade_row("gamma", "TLT", -3.0, [f"o{i}", f"c{i}"],
                       exit_ts=f"2026-07-23T13:0{i}:00+00:00")
        )
    kw = dict(
        env={"CHAD_FUSE_LC3": "shadow"},
        trades_dir=tmp_path / "trades",
        fills_dir=tmp_path / "fills",
        runtime_dir=tmp_path / "runtime",
        epoch_state_path=tmp_path / "runtime" / "epoch_state.json",
        state_path=tmp_path / "runtime" / "fuse_box_state.json",
        evidence_dir=tmp_path / "ev",
        notify_fn=lambda *a, **k: True,
    )
    _write_ndjson(pathlib.Path(kw["fills_dir"]) / f"FILLS_{DATE}.ndjson", fills)
    _write_ndjson(
        pathlib.Path(kw["trades_dir"]) / f"trade_history_{DATE}.ndjson", trades
    )
    state = run_evaluator_cycle(NOW, **kw)
    fuses = {r["fuse_id"]: r for r in state["fuses"]}
    assert fuses["symbol:TLT"]["tripped"] is True
    assert fuses["sector:rates"]["tripped"] is True
