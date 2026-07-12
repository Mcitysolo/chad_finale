"""U1 CRYPTO-TRUST — trusted Kraken paper-fill engine (pure core + orchestrator).

All I/O to tmp_path; no live feed / broker. Proves: live-tick marking with
freshness fail-closed, the documented slippage model, real taker fee, FIFO
round-trip realized PnL, and Stage-2-admissible trusted labels (NO validate_only
/ pnl_untrusted).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from chad.execution.kraken_trading_config import load_kraken_trading_config
from chad.core import kraken_trusted_fill_engine as tfe

_CFG = load_kraken_trading_config()
_NOW = 1_800_000_000.0  # fixed epoch for deterministic freshness


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prices(symbol="SOL-USD", bid=170.0, ask=170.2, last=170.1, age_s=0.0):
    return {
        "ts_utc": _iso(_NOW - age_s),
        "ticks": {symbol: {"bid": bid, "ask": ask, "last": last, "ts_utc": _iso(_NOW - age_s)}},
    }


class _FakeTickSource:
    def __init__(self, touch):
        self._touch = touch

    def get_touch(self, symbol, *, now_epoch):
        return self._touch


class _Intent:
    def __init__(self, pair="SOLUSD", side="buy", volume=1.0, markers=()):
        self.strategy = "alpha_crypto"
        self.pair = pair
        self.side = side
        self.volume = volume
        self.ordertype = "market"
        self.markers = markers
        self.idempotency_key = ""
        self.trace_id = ""


# --------------------------------------------------------------------------- #
# Touch parsing + freshness
# --------------------------------------------------------------------------- #

def test_read_touch_fresh_ok():
    t = tfe.read_touch_from_prices(_prices(), "SOL-USD", now_epoch=_NOW, max_age_seconds=30.0)
    assert t is not None
    assert t.bid == 170.0 and t.ask == 170.2
    assert t.mid == pytest.approx(170.1)


def test_read_touch_stale_fails_closed():
    t = tfe.read_touch_from_prices(_prices(age_s=120.0), "SOL-USD", now_epoch=_NOW, max_age_seconds=30.0)
    assert t is None


def test_read_touch_missing_symbol_or_crossed():
    assert tfe.read_touch_from_prices(_prices(), "BTC-USD", now_epoch=_NOW, max_age_seconds=30.0) is None
    crossed = _prices(bid=171.0, ask=170.0)  # ask < bid
    assert tfe.read_touch_from_prices(crossed, "SOL-USD", now_epoch=_NOW, max_age_seconds=30.0) is None


# --------------------------------------------------------------------------- #
# Slippage + fee
# --------------------------------------------------------------------------- #

def test_slippage_model_buy_above_sell_below_mid():
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    buy_px, buy_bps = tfe.model_fill_price("buy", touch, _CFG.slippage_impact_floor_bps)
    sell_px, sell_bps = tfe.model_fill_price("sell", touch, _CFG.slippage_impact_floor_bps)
    assert buy_px > touch.mid > sell_px
    # half-spread frac = (0.2/2)/170.1 ~ 5.88 bps + 5 bps floor
    assert buy_bps == pytest.approx(sell_bps)
    assert buy_bps == pytest.approx(((0.1 / 170.1) + 5e-4) * 1e4, rel=1e-6)


def test_simulate_fill_fee_is_real_taker():
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    fill = tfe.simulate_fill(strategy="alpha_crypto", pair="SOLUSD", symbol="SOL-USD",
                             side="buy", qty=1.0, touch=touch, config=_CFG)
    assert fill.fee == pytest.approx(fill.notional * 26.0 / 1e4)
    assert fill.fee_model == "kraken_paper_v1"
    assert fill.provenance == "SIMULATED_AGAINST_LIVE_TICKS"


# --------------------------------------------------------------------------- #
# FIFO round-trip book
# --------------------------------------------------------------------------- #

def _book(tmp_path):
    return tfe.RoundTripBook(db_path=tmp_path / "book.sqlite3",
                             now_iso=lambda: "2026-07-12T00:00:00Z")


def _mk_fill(side, qty, price, fee=0.0):
    return tfe.TrustedFill(strategy="alpha_crypto", pair="SOLUSD", symbol="SOL-USD",
                           side=side, qty=qty, fill_price=price, notional=price * qty,
                           fee=fee, slippage_bps=0.0, mid=price, touch_ts_utc="")


def test_book_long_roundtrip_realizes_pnl_net_of_fees(tmp_path):
    book = _book(tmp_path)
    assert book.record(_mk_fill("buy", 1.0, 100.0, fee=0.26)) == []   # open long
    rts = book.record(_mk_fill("sell", 1.0, 110.0, fee=0.286))        # close
    assert len(rts) == 1
    rt = rts[0]
    assert rt.direction == "long" and rt.qty == pytest.approx(1.0)
    # gross 10.0 - entry_fee 0.26 - exit_fee 0.286
    assert rt.realized_pnl == pytest.approx(10.0 - 0.26 - 0.286)
    assert book.open_qty("alpha_crypto", "SOL-USD") == pytest.approx(0.0)


def test_book_short_roundtrip(tmp_path):
    book = _book(tmp_path)
    assert book.record(_mk_fill("sell", 2.0, 100.0, fee=0.52)) == []  # open short
    rts = book.record(_mk_fill("buy", 2.0, 90.0, fee=0.468))          # close (profit down)
    assert len(rts) == 1
    assert rts[0].direction == "short"
    assert rts[0].realized_pnl == pytest.approx((100.0 - 90.0) * 2.0 - 0.52 - 0.468)


def test_book_fifo_partial_and_multi_lot(tmp_path):
    book = _book(tmp_path)
    book.record(_mk_fill("buy", 1.0, 100.0, fee=0.26))   # lot A
    book.record(_mk_fill("buy", 1.0, 120.0, fee=0.312))  # lot B
    rts = book.record(_mk_fill("sell", 1.5, 130.0, fee=0.507))  # close 1.5 FIFO
    # matches lot A fully (1.0) then lot B partially (0.5)
    assert len(rts) == 2
    assert rts[0].entry_price == pytest.approx(100.0) and rts[0].qty == pytest.approx(1.0)
    assert rts[1].entry_price == pytest.approx(120.0) and rts[1].qty == pytest.approx(0.5)
    assert book.open_qty("alpha_crypto", "SOL-USD") == pytest.approx(0.5)  # 0.5 of lot B remains


def test_book_flip_residual_opens_opposite(tmp_path):
    book = _book(tmp_path)
    book.record(_mk_fill("buy", 1.0, 100.0, fee=0.26))       # long 1.0
    rts = book.record(_mk_fill("sell", 1.5, 110.0, fee=0.0)) # close 1.0, flip 0.5 short
    assert len(rts) == 1 and rts[0].qty == pytest.approx(1.0)
    assert book.open_qty("alpha_crypto", "SOL-USD") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def _engine(tmp_path, touch, dedup=None):
    ev_rows, th_rows = [], []
    eng = tfe.TrustedFillEngine(
        config=_CFG,
        tick_source=_FakeTickSource(touch),
        book=_book(tmp_path),
        now_fn=lambda: _NOW,
        dedup=dedup,
        evidence_writer=lambda kw: (ev_rows.append(kw) or "FILLS.ndjson"),
        trade_history_writer=lambda kw: (th_rows.append(kw) or "trade_history.ndjson"),
    )
    return eng, ev_rows, th_rows


def test_engine_open_then_close_emits_trusted_admissible_rows(tmp_path):
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    eng, ev_rows, th_rows = _engine(tmp_path, touch)

    r_open = eng.process_intent(_Intent(side="buy", volume=1.0))
    assert r_open["trusted"] is True and r_open["leg"] == "open"
    assert len(ev_rows) == 1 and len(th_rows) == 0  # open writes FILLS only

    r_close = eng.process_intent(_Intent(side="sell", volume=1.0))
    assert r_close["trusted"] is True and r_close["leg"] == "close"
    assert len(th_rows) == 1  # close writes a realized trade_history row

    tr = th_rows[0]
    assert tr["broker"] == "kraken_paper"
    assert tr["quantity"] > 0 and tr["fill_price"] > 0 and tr["notional"] > 0
    assert isinstance(tr["pnl"], float)  # finite realized pnl
    # Trust contract: NO untrust flags anywhere.
    assert "validate_only" not in tr["tags"] and "pnl_untrusted" not in tr["tags"]
    assert "validate_only" not in tr["extra"] and "pnl_untrusted" not in tr["extra"]
    assert tr["extra"]["fee_model"] == "kraken_paper_v1"
    assert tr["extra"]["provenance"] == "SIMULATED_AGAINST_LIVE_TICKS"
    # FILLS evidence also trusted + fee pre-set (suppresses ibkr_fixed_v1 restamp)
    assert ev_rows[0]["fee_amount"] > 0 and ev_rows[0]["extra"]["fee_model"] == "kraken_paper_v1"
    assert "validate_only" not in ev_rows[0]["tags"] and "pnl_untrusted" not in ev_rows[0]["tags"]


def test_engine_stale_touch_fails_closed(tmp_path):
    eng, ev_rows, th_rows = _engine(tmp_path, None)  # FakeTickSource returns None
    r = eng.process_intent(_Intent())
    assert r == {"trusted": False, "reason": "no_fresh_touch"}
    assert ev_rows == [] and th_rows == []


def test_engine_dedup_does_not_double_record(tmp_path):
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    seen = set()
    def dedup(key):
        if key in seen:
            return False
        seen.add(key)
        return True
    eng, ev_rows, th_rows = _engine(tmp_path, touch, dedup=dedup)
    r1 = eng.process_intent(_Intent(side="buy", volume=1.0))
    r2 = eng.process_intent(_Intent(side="buy", volume=1.0))  # same minute+params
    assert r1["reason"] == "filled"
    assert r2["reason"] == "dedup"
    assert len(ev_rows) == 1  # second call did not write / double-record


def test_engine_carries_min_size_markers_into_tags(tmp_path):
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    eng, ev_rows, th_rows = _engine(tmp_path, touch)
    eng.process_intent(_Intent(side="buy", volume=1.0, markers=("CRYPTO_MIN_SIZE_BUMP",)))
    assert "CRYPTO_MIN_SIZE_BUMP" in ev_rows[0]["tags"]
