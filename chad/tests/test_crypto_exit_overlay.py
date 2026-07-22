"""UC1/U2-A — chad/risk/crypto_exit_overlay.py.

Covers the pure core (lot aggregation, the three conditions, fail-closed marks), the
reduce-only invariant that the Kraken book does NOT enforce for us, the anchor-merge
property (the equity lane's live-proven wipe defect, deliberately not inherited), and the
headline end-to-end proof: a crypto lot closed by the overlay in ACTIVE routes through the
REAL trusted-fill engine and produces a Stage-2-ADMISSIBLE row.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from chad.risk import crypto_exit_overlay as cxo
from chad.risk.position_exit_overlay import load_overlay_config

UTC = timezone.utc
NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=UTC)


def _cfg(mode: str = "shadow", **kw):
    payload = {
        "mode": mode,
        "atr_period": 14,
        "atr_trail_mult": 3.0,
        "hard_stop_loss_pct": 0.12,
        "min_bars_for_atr": 16,
        "max_hold_days": {"crypto": 4.0, "default": 4.0},
    }
    payload.update(kw)
    return load_overlay_config(payload)


def _lot(strategy="alpha_crypto", symbol="SOL-USD", direction="long", qty=2.5,
         entry=76.6, fee_pu=0.2, opened="2026-07-12T22:04:18Z"):
    return (strategy, symbol, direction, qty, entry, fee_pu, opened)


def _bars(n=30, close=76.0, rng=1.0):
    return [{"high": close + rng, "low": close - rng, "close": close} for _ in range(n)]


# --------------------------------------------------------------------------- #
# Lot aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_sums_qty_and_uses_fee_inclusive_vwap_basis():
    # Two lots, different prices -> qty-weighted, fee-inclusive basis.
    rows = [
        _lot(qty=2.0, entry=100.0, fee_pu=1.0),
        _lot(qty=8.0, entry=50.0, fee_pu=0.5),
    ]
    snaps = cxo.aggregate_lots(rows)
    assert len(snaps) == 1
    s = snaps[0]
    assert s.qty == 10.0
    # (2*(100+1) + 8*(50+0.5)) / 10 = (202 + 404) / 10 = 60.6
    assert s.entry_price == pytest.approx(60.6)
    assert s.side == "BUY"
    assert s.lots == 2


def test_aggregate_uses_earliest_open_so_max_hold_measures_the_real_holding_period():
    rows = [
        _lot(qty=1.0, opened="2026-07-15T00:00:00Z"),
        _lot(qty=1.0, opened="2026-07-12T00:00:00Z"),  # oldest surviving lot
    ]
    s = cxo.aggregate_lots(rows)[0]
    assert s.opened_at_utc == "2026-07-12T00:00:00Z"


def test_aggregate_ignores_zero_and_negative_remaining_lots():
    rows = [_lot(qty=0.0), _lot(qty=-1.0), _lot(qty=3.0)]
    snaps = cxo.aggregate_lots(rows)
    assert len(snaps) == 1 and snaps[0].qty == 3.0


def test_aggregate_separates_strategies_and_symbols():
    rows = [
        _lot(strategy="alpha_crypto", symbol="SOL-USD", qty=1.0),
        _lot(strategy="alpha_crypto", symbol="BTC-USD", qty=2.0),
        _lot(strategy="beta", symbol="SOL-USD", qty=3.0),
    ]
    snaps = {s.key: s.qty for s in cxo.aggregate_lots(rows)}
    assert snaps == {"alpha_crypto|SOL-USD": 1.0, "alpha_crypto|BTC-USD": 2.0, "beta|SOL-USD": 3.0}


def test_short_book_maps_to_sell_side():
    s = cxo.aggregate_lots([_lot(direction="short")])[0]
    assert s.side == "SELL"


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
def test_hold_when_no_condition_met():
    snaps = cxo.aggregate_lots([_lot(qty=2.5, entry=76.6, fee_pu=0.0)])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 76.5},
        bars_by_symbol={"SOL-USD": _bars(close=76.5, rng=0.5)},
        anchors={"alpha_crypto|SOL-USD": {"peak": 76.6, "trough": 76.4,
                                          "first_seen_utc": "2026-07-17T00:00:00Z"}},
        config=_cfg(), now_utc=datetime(2026, 7, 13, tzinfo=UTC),
    )
    assert [v.verdict for v in r.verdicts] == ["HOLD"]
    assert r.close_intents == []


def test_hard_stop_fires_against_the_real_cost_basis():
    # basis 100+0 = 100; 12% stop = 88.0; mark 87 -> fire.
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=100.0, fee_pu=0.0)])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 87.0},
        bars_by_symbol={}, anchors={}, config=_cfg(),
        now_utc=datetime(2026, 7, 13, tzinfo=UTC),
    )
    v = r.verdicts[0]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "hard_stop_loss"
    assert v.entry_price == pytest.approx(100.0)
    assert v.hard_stop_price == pytest.approx(88.0)


def test_hard_stop_basis_includes_entry_fee():
    # basis = 100 + 5 = 105; stop = 92.4. A mark of 93 must NOT fire (it would if fees
    # were ignored, since 100*0.88 = 88).
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=100.0, fee_pu=5.0)])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 93.0}, bars_by_symbol={},
        anchors={}, config=_cfg(), now_utc=datetime(2026, 7, 13, tzinfo=UTC),
    )
    assert r.verdicts[0].hard_stop_price == pytest.approx(92.4)
    assert r.verdicts[0].verdict == "HOLD"


def test_max_hold_fires_from_the_real_lot_open_time():
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=76.6, opened="2026-07-12T00:00:00Z")])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 76.6}, bars_by_symbol={},
        anchors={}, config=_cfg(),
        now_utc=datetime(2026, 7, 17, tzinfo=UTC),  # 5 days > 4.0 max_hold
    )
    v = r.verdicts[0]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "max_hold"
    assert v.age_days == pytest.approx(5.0, abs=0.01)


def test_atr_trailing_stop_fires_from_peak():
    # peak 100, atr 1.0, mult 3.0 -> stop 97.0; mark 96 -> fire.
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=90.0, fee_pu=0.0)])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 96.0},
        bars_by_symbol={"SOL-USD": _bars(n=30, close=96.0, rng=0.5)},
        anchors={"alpha_crypto|SOL-USD": {"peak": 100.0, "trough": 90.0,
                                          "first_seen_utc": "2026-07-16T00:00:00Z"}},
        config=_cfg(), now_utc=datetime(2026, 7, 13, tzinfo=UTC),
    )
    v = r.verdicts[0]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "atr_trailing_stop"
    assert v.peak == pytest.approx(100.0)


def test_condition_precedence_hard_stop_wins_over_max_hold():
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=100.0, fee_pu=0.0,
                                     opened="2026-07-01T00:00:00Z")])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 50.0}, bars_by_symbol={},
        anchors={}, config=_cfg(), now_utc=NOW,
    )
    assert r.verdicts[0].reason == "hard_stop_loss"


# --------------------------------------------------------------------------- #
# Fail-closed
# --------------------------------------------------------------------------- #
def test_missing_mark_is_fail_closed_skip_not_a_close():
    snaps = cxo.aggregate_lots([_lot(qty=1.0, entry=100.0, opened="2026-07-01T00:00:00Z")])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={}, bars_by_symbol={}, anchors={},
        config=_cfg(), now_utc=NOW,
    )
    # Max-hold is long past, but with no mark we must NOT propose a close.
    assert r.verdicts[0].verdict == "SKIP_NO_DATA"
    assert r.close_intents == []


def test_stale_tick_yields_no_mark_and_therefore_no_close(tmp_path):
    prices = tmp_path / "kraken_prices.json"
    prices.write_text(json.dumps({
        "ts_utc": "2026-07-17T00:00:00Z",
        "ticks": {"SOL-USD": {"bid": 73.6, "ask": 73.61, "last": 73.6,
                              "ts_utc": "2026-07-17T00:00:00Z"}},
    }))
    loader = cxo._default_marks_loader(prices, max_age_seconds=60.0)
    # now_epoch is real time, far past the embedded ts -> stale -> fail closed.
    # (W3B-6: loader now returns (marks, meta); the stale-tick contract is
    # unchanged — no mark, no meta.)
    assert loader(["SOL-USD"]) == ({}, {})


def test_empty_book_yields_no_verdicts_and_no_closes():
    r = cxo.evaluate_crypto_positions(
        snapshots=[], marks_by_symbol={}, bars_by_symbol={}, anchors={},
        config=_cfg(), now_utc=NOW,
    )
    assert r.verdicts == [] and r.close_intents == []


# --------------------------------------------------------------------------- #
# Reduce-only — the book will NOT protect us
# --------------------------------------------------------------------------- #
def test_close_qty_never_exceeds_open_lot_qty():
    snaps = cxo.aggregate_lots([_lot(qty=2.5), _lot(qty=2.5), _lot(qty=2.5)])
    r = cxo.evaluate_crypto_positions(
        snapshots=snaps, marks_by_symbol={"SOL-USD": 1.0}, bars_by_symbol={},
        anchors={}, config=_cfg(), now_utc=NOW,
    )
    v = r.verdicts[0]
    assert v.verdict == "WOULD_CLOSE"
    assert v.close_qty == 7.5 == v.open_qty
    assert r.close_intents[0]["quantity"] == 7.5


def test_reclamp_clamps_down_to_a_shrunken_book():
    intents = [{"position_key": "alpha_crypto|SOL-USD", "quantity": 12.5,
                "symbol": "SOL-USD", "strategy": "alpha_crypto"}]
    out = cxo.reduce_only_reclamp_crypto(intents, {"alpha_crypto|SOL-USD": 4.0})
    assert out[0]["quantity"] == 4.0


def test_reclamp_drops_intent_when_book_went_flat():
    intents = [{"position_key": "alpha_crypto|SOL-USD", "quantity": 12.5,
                "symbol": "SOL-USD", "strategy": "alpha_crypto"}]
    assert cxo.reduce_only_reclamp_crypto(intents, {"alpha_crypto|SOL-USD": 0.0}) == []
    assert cxo.reduce_only_reclamp_crypto(intents, {}) == []


def test_close_intent_side_is_opposite_of_the_book():
    long_snap = cxo.aggregate_lots([_lot(direction="long", qty=1.0, entry=100.0)])
    r = cxo.evaluate_crypto_positions(
        snapshots=long_snap, marks_by_symbol={"SOL-USD": 50.0}, bars_by_symbol={},
        anchors={}, config=_cfg(), now_utc=NOW)
    assert r.close_intents[0]["close_side"] == "SELL"

    short_snap = cxo.aggregate_lots([_lot(direction="short", qty=1.0, entry=100.0)])
    r2 = cxo.evaluate_crypto_positions(
        snapshots=short_snap, marks_by_symbol={"SOL-USD": 200.0}, bars_by_symbol={},
        anchors={}, config=_cfg(), now_utc=NOW)
    assert r2.close_intents[0]["close_side"] == "BUY"


def test_dispatch_pair_round_trips_back_to_the_canonical_symbol():
    """The close must be booked under the SAME symbol the lots are held under.

    A slash form (SOL/USD) passes through pair_to_canonical unchanged, so the fill would be
    booked as symbol 'SOL/USD' while the book holds 'SOL-USD' — the "close" would open a
    second, parallel position instead of reducing the first. Pin the round-trip.
    """
    from chad.core.kraken_trusted_fill_engine import pair_to_canonical

    for canonical in ("SOL-USD", "BTC-USD", "ETH-USD"):
        pair = cxo._canonical_to_pair(canonical)
        assert pair_to_canonical(pair) == canonical, (
            f"{canonical} dispatched as {pair!r} books under "
            f"{pair_to_canonical(pair)!r} — reduce-only would be violated"
        )
    # BTC specifically must go out as Kraken's XBT naming.
    assert cxo._canonical_to_pair("BTC-USD") == "XBTUSD"


def test_idempotency_key_is_deterministic_and_carries_no_timestamp():
    a = cxo._close_idempotency_key("alpha_crypto", "SOL-USD", "sell", 12.5)
    b = cxo._close_idempotency_key("alpha_crypto", "SOL-USD", "sell", 12.5)
    assert a == b, "collision across cycles IS the storm defense — must not embed a clock"
    assert cxo._close_idempotency_key("alpha_crypto", "SOL-USD", "sell", 12.4) != a


# --------------------------------------------------------------------------- #
# Anchor merge — the equity lane's wipe defect, deliberately NOT inherited
# --------------------------------------------------------------------------- #
def _overlay(tmp_path, *, mode="shadow", lots, marks=None, engine=None, open_qty=None):
    return cxo.CryptoExitOverlay(
        _cfg(mode),
        evidence_path=tmp_path / "ev.ndjson",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=lambda: lots,
        marks_loader=lambda syms: (marks if marks is not None else {}),
        bars_loader=lambda syms: {},
        open_qty_loader=open_qty,
        engine_factory=(lambda: engine) if engine is not None else None,
        env={},
    )


def test_blind_cycle_does_not_wipe_a_live_anchor(tmp_path):
    """The equity lane replaces the state file with only what it saw this cycle, so one
    empty read destroys peak/entry (live-proven twice on gamma|UNH). This lane merges."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema_version": "crypto_exit_overlay_state.v1",
                                 "anchors": {"alpha_crypto|SOL-USD": {
                                     "peak": 120.0, "trough": 70.0,
                                     "first_seen_utc": "2026-07-12T00:00:00Z"}}}))
    ov = _overlay(tmp_path, lots=[])  # book read returns nothing this cycle
    ov._state_path = state
    ov.run_cycle(now_utc=NOW)
    after = json.loads(state.read_text())["anchors"]
    assert "alpha_crypto|SOL-USD" in after, "a blind cycle must not erase the trail"
    assert after["alpha_crypto|SOL-USD"]["peak"] == 120.0


def test_peak_survives_across_cycles_and_ratchets_up(tmp_path):
    lots = [_lot(qty=1.0, entry=76.6)]
    ov = _overlay(tmp_path, lots=lots, marks={"SOL-USD": 80.0})
    ov.run_cycle(now_utc=datetime(2026, 7, 13, tzinfo=UTC))
    ov2 = _overlay(tmp_path, lots=lots, marks={"SOL-USD": 78.0})
    ov2.run_cycle(now_utc=datetime(2026, 7, 13, 1, tzinfo=UTC))
    anchors = json.loads((tmp_path / "state.json").read_text())["anchors"]
    assert anchors["alpha_crypto|SOL-USD"]["peak"] == 80.0, "peak must ratchet, not follow spot"


def test_anchor_pruned_only_when_book_confirms_flat(tmp_path):
    lots = [_lot(qty=1.0, entry=76.6)]
    ov = _overlay(tmp_path, lots=lots, marks={"SOL-USD": 80.0})
    ov.run_cycle(now_utc=NOW)
    assert "alpha_crypto|SOL-USD" in json.loads((tmp_path / "state.json").read_text())["anchors"]
    # Book successfully read and genuinely flat -> prune is correct here.
    ov2 = _overlay(tmp_path, lots=[_lot(strategy="other", qty=1.0)], marks={"SOL-USD": 80.0})
    ov2.run_cycle(now_utc=NOW)
    assert "alpha_crypto|SOL-USD" not in json.loads(
        (tmp_path / "state.json").read_text())["anchors"]


# --------------------------------------------------------------------------- #
# Mode gating
# --------------------------------------------------------------------------- #
def test_shadow_mode_submits_nothing(tmp_path):
    class _Boom:
        def process_intent(self, intent):  # pragma: no cover - must never be called
            raise AssertionError("shadow mode must never dispatch")

    ov = _overlay(tmp_path, mode="shadow", lots=[_lot(qty=1.0, entry=100.0)],
                  marks={"SOL-USD": 50.0}, engine=_Boom())
    r = ov.run_cycle(now_utc=NOW)
    assert len(r.close_intents) == 1  # it PROPOSED
    # ...and did not dispatch (no AssertionError raised).


def test_off_mode_evaluates_nothing_but_still_heartbeats(tmp_path):
    ov = _overlay(tmp_path, mode="off", lots=[_lot(qty=1.0, entry=100.0)],
                  marks={"SOL-USD": 50.0})
    r = ov.run_cycle(now_utc=NOW)
    assert r.evaluated is False
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["mode"] == "off" and hb["evaluated"] == 0


def test_crypto_kill_switch_is_independent_of_the_equity_lane(tmp_path):
    ov = cxo.CryptoExitOverlay(
        _cfg("shadow"), evidence_path=None, state_path=None, heartbeat_path=None,
        lots_loader=lambda: [], marks_loader=lambda s: {}, bars_loader=lambda s: {},
        env={"CHAD_POSITION_EXIT_OVERLAY": "off"},  # equity switch must NOT bind crypto
    )
    assert ov.mode == "shadow"
    ov2 = cxo.CryptoExitOverlay(
        _cfg("shadow"), evidence_path=None, state_path=None, heartbeat_path=None,
        lots_loader=lambda: [], marks_loader=lambda s: {}, bars_loader=lambda s: {},
        env={"CHAD_CRYPTO_EXIT_OVERLAY": "off"},
    )
    assert ov2.mode == "off"


def test_loader_failure_is_non_fatal_and_submits_nothing(tmp_path):
    def _boom():
        raise RuntimeError("book unavailable")

    ov = cxo.CryptoExitOverlay(
        _cfg("active"), evidence_path=None, state_path=None,
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=_boom, marks_loader=lambda s: {}, bars_loader=lambda s: {}, env={},
    )
    r = ov.run_cycle(now_utc=NOW)  # must not raise into the trading loop
    assert r.evaluated is False
    assert json.loads((tmp_path / "hb.json").read_text())["healthy"] is False


def test_evidence_rows_are_lane_tagged_crypto(tmp_path):
    ov = _overlay(tmp_path, lots=[_lot(qty=1.0, entry=100.0)], marks={"SOL-USD": 50.0})
    ov.run_cycle(now_utc=NOW)
    rows = [json.loads(l) for l in (tmp_path / "ev.ndjson").read_text().splitlines()]
    assert rows and all(r["lane"] == "crypto" for r in rows)
    assert rows[0]["asset_class"] == "crypto"
    assert rows[0]["schema_version"] == "exit_overlay.v2"


def test_build_default_crypto_overlay_refuses_real_paths_under_pytest(tmp_path):
    from pathlib import Path

    with pytest.raises(RuntimeError, match="test-write leak guard"):
        cxo.build_default_crypto_overlay(
            repo_root=Path("/home/ubuntu/chad_finale"),
            config_path=Path("/home/ubuntu/chad_finale/config/crypto_exit_overlay.json"),
        )


def test_shipped_crypto_config_is_valid_and_ships_shadow():
    from pathlib import Path

    payload = json.loads(Path("/home/ubuntu/chad_finale/config/crypto_exit_overlay.json")
                         .read_text())
    cfg = load_overlay_config(payload)
    assert cfg.mode == "shadow", "the crypto lane must ship SHADOW, never active"
    assert cfg.max_hold_for("crypto") == 4.0


# --------------------------------------------------------------------------- #
# END-TO-END: crypto lot -> overlay ACTIVE close -> trusted engine -> Stage-2
# --------------------------------------------------------------------------- #
def test_active_close_routes_through_trusted_engine_and_is_stage2_admissible(tmp_path,
                                                                             monkeypatch):
    """The headline claim: an overlay crypto exit produces a SCOREABLE round-trip.

    Drives the REAL RoundTripBook + REAL TrustedFillEngine + REAL Stage-2 trust predicate.
    """
    from chad.core import kraken_trusted_fill_engine as tfe
    from chad.validation import trade_log_adapter as tla

    book_db = tmp_path / "book.sqlite3"
    book = tfe.RoundTripBook(db_path=book_db, now_iso=lambda: "2026-07-12T22:04:18Z")

    # Open a real lot through the real engine: BUY 5 SOL.
    class _Intent:
        def __init__(self, side, volume):
            self.pair, self.side, self.volume = "SOLUSD", side, volume
            self.strategy, self.ordertype = "alpha_crypto", "market"
            self.markers, self.idempotency_key, self.trace_id = (), None, None

    class _Tick:
        def get_touch(self, symbol, *, now_epoch):
            return tfe.Touch(bid=100.0, ask=100.2, last=100.1, ts_utc="2026-07-12T22:04:18Z")

    captured = []
    engine = tfe.TrustedFillEngine(
        book=book, tick_source=_Tick(), now_fn=lambda: 1_800_000_000.0,
        evidence_writer=lambda kw: captured.append(kw),
        trade_history_writer=lambda kw: None,
    )
    open_res = engine.process_intent(_Intent("buy", 5.0))
    assert open_res["trusted"] is True
    assert book.open_qty("alpha_crypto", "SOL-USD") == 5.0

    # Now the overlay evaluates that real book and closes it in ACTIVE.
    rows = [r for r in sqlite3.connect(str(book_db)).execute(
        "SELECT strategy, symbol, direction, qty_remaining, entry_price, "
        "entry_fee_per_unit, opened_at_utc FROM kraken_trusted_lots ORDER BY rowid ASC")]
    ov = cxo.CryptoExitOverlay(
        _cfg("active"),
        evidence_path=tmp_path / "ev.ndjson", state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=lambda: rows,
        marks_loader=lambda syms: {"SOL-USD": 50.0},   # deep loss -> hard stop fires
        bars_loader=lambda syms: {},
        open_qty_loader=book.open_qty,
        engine_factory=lambda: engine,
        env={},
    )
    result = ov.run_cycle(now_utc=NOW)
    assert [v.reason for v in result.verdicts] == ["hard_stop_loss"]

    # The book is FLAT -> the round-trip realized, and did NOT flip into a short.
    assert book.open_qty("alpha_crypto", "SOL-USD") == 0.0

    # The close produced a Stage-2 ADMISSIBLE evidence row.
    assert len(captured) == 2, "one open + one close evidence row"
    close_kw = captured[-1]
    assert close_kw["status"] == "paper_fill"
    assert close_kw["asset_class"] == "crypto"
    assert close_kw["extra"]["provenance"] == tfe.PROVENANCE_SIMULATED_LIVE_TICKS
    assert close_kw["extra"]["trust_state"] == "TRUSTED"

    record = {
        "status": close_kw["status"],
        "asset_class": close_kw["asset_class"],
        "symbol": close_kw["symbol"],
        "price": close_kw["fill_price"],
        "quantity": close_kw["quantity"],
        "tags": list(close_kw.get("tags") or []),
        "extra": dict(close_kw["extra"]),
    }
    assert tla.trust_exclusion(record) is None, (
        "an overlay crypto exit must bank SCOREABLE evidence — "
        f"got exclusion={tla.trust_exclusion(record)!r}"
    )


def test_active_close_cannot_flip_the_book_short(tmp_path):
    """The Kraken book flips residuals by design; the overlay must never hand it an
    oversized close. This is the INCIDENT-0713 (TLT) mechanism in the crypto lane."""
    from chad.core import kraken_trusted_fill_engine as tfe

    book = tfe.RoundTripBook(db_path=tmp_path / "b.sqlite3",
                             now_iso=lambda: "2026-07-12T00:00:00Z")

    class _Tick:
        def get_touch(self, symbol, *, now_epoch):
            return tfe.Touch(bid=100.0, ask=100.2, last=100.1, ts_utc="2026-07-12T00:00:00Z")

    class _Intent:
        def __init__(self, side, volume):
            self.pair, self.side, self.volume = "SOLUSD", side, volume
            self.strategy, self.ordertype = "alpha_crypto", "market"
            self.markers, self.idempotency_key, self.trace_id = (), None, None

    engine = tfe.TrustedFillEngine(
        book=book, tick_source=_Tick(), now_fn=lambda: 1_800_000_000.0,
        evidence_writer=lambda kw: None, trade_history_writer=lambda kw: None)
    engine.process_intent(_Intent("buy", 3.0))
    assert book.open_qty("alpha_crypto", "SOL-USD") == 3.0

    # A stale/oversized proposal (10) against a book holding 3 must be clamped to 3.
    stale = [{"position_key": "alpha_crypto|SOL-USD", "symbol": "SOL-USD",
              "strategy": "alpha_crypto", "close_side": "SELL", "quantity": 10.0,
              "reason": "crypto_exit_overlay_hard_stop_loss"}]
    clamped = cxo.reduce_only_reclamp_crypto(
        stale, {"alpha_crypto|SOL-USD": book.open_qty("alpha_crypto", "SOL-USD")})
    assert clamped[0]["quantity"] == 3.0

    engine.process_intent(_Intent("sell", clamped[0]["quantity"]))
    assert book.open_qty("alpha_crypto", "SOL-USD") == 0.0, "flat, not short"
