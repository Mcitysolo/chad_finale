"""W4B-3 (J16): advice aggregation + the overlay's advice rule.

Pins:
  AGGREGATOR (exit_advice.load_advice_by_symbol):
    - fresh SELL rows aggregate per symbol (strategies, max_confidence, reasons);
    - stale rows (> advice_ttl_seconds) expire; future-dated rows never count;
    - excluded symbols never aggregate (set wall + row flag wall);
    - BUY rows never aggregate; missing-confidence rows never aggregate;
    - unreadable evidence -> {} (never raises).
  OVERLAY RULE (evaluate_positions):
    - kernel precedence absolute: a firing hard-stop wins over advice;
    - record mode: ADVICE_WOULD_CLOSE verdict, NO close intent (nothing to
      submit even in ACTIVE);
    - consume mode: WOULD_CLOSE + reduce-only intent reason
      exit_overlay_strategy_advice, qty clamped to broker truth;
    - owning-strategy wall: advice from a non-owning strategy never fires;
    - long-only wall: SELL-side (short) legs never take advice;
    - advice_mode=off byte-identical: result equals a no-advice evaluation;
    - excluded symbols SKIP_EXCLUDED before the advice rule can ever see them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.core import exit_advice as ea
from chad.risk.position_exit_overlay import (
    ExitOverlayConfig,
    evaluate_positions,
)

_NOW = datetime(2026, 7, 23, 15, 0, 0, tzinfo=timezone.utc)


def _write_rows(directory: Path, rows, day=None):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (day or _NOW).strftime("%Y%m%d")
    with (directory / f"exit_advice_{stamp}.ndjson").open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(symbol="PSQ", strategy="gamma", ts=None, side="SELL",
         confidence=0.7, excluded=False, reason="trend_break_exit"):
    return {
        "schema_version": "exit_advice.v1",
        "ts_utc": (ts or _NOW).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site": "router", "mode": "record",
        "strategy": strategy, "source_strategies": [strategy],
        "symbol": symbol, "side": side, "size": 5.0,
        "confidence": confidence, "asset_class": "etf",
        "reason": reason, "excluded": excluded, "position_open": True,
    }


# --------------------------------------------------------------------------- #
# aggregator
# --------------------------------------------------------------------------- #

def test_aggregates_fresh_rows(tmp_path):
    _write_rows(tmp_path, [
        _row(ts=_NOW - timedelta(seconds=60)),
        _row(ts=_NOW - timedelta(seconds=30), strategy="alpha", confidence=0.9,
             reason="vol_spike_exit"),
    ])
    advice = ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=tmp_path / "missing.json",
    )
    assert set(advice) == {"PSQ"}
    a = advice["PSQ"]
    assert a["strategies"] == ["alpha", "gamma"]
    assert a["max_confidence"] == 0.9
    assert a["reasons"] == ["trend_break_exit", "vol_spike_exit"]
    assert a["count_fresh"] == 2


def test_stale_and_future_rows_never_count(tmp_path):
    _write_rows(tmp_path, [
        _row(ts=_NOW - timedelta(seconds=600)),           # stale (> 180s)
        _row(ts=_NOW + timedelta(seconds=60)),            # future-dated
    ])
    advice = ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=tmp_path / "missing.json",
    )
    assert advice == {}


def test_ttl_config_is_honored(tmp_path):
    _write_rows(tmp_path, [_row(ts=_NOW - timedelta(seconds=600))])
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"advice_ttl_seconds": 900}))
    advice = ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=cfg,
    )
    assert set(advice) == {"PSQ"}                          # 600s < 900s ttl


def test_excluded_walls(tmp_path):
    _write_rows(tmp_path, [
        _row(symbol="MSFT"),                               # wall 1: the set
        _row(symbol="TLT", excluded=True),                 # wall 2: the row flag
    ])
    advice = ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path,
        excluded_symbols=frozenset({"MSFT"}),
        config_path=tmp_path / "missing.json",
    )
    assert advice == {}


def test_buy_and_no_confidence_rows_never_aggregate(tmp_path):
    _write_rows(tmp_path, [
        _row(side="BUY"),
        _row(confidence=None),
    ])
    advice = ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=tmp_path / "missing.json",
    )
    assert advice == {}


def test_min_confidence_filter(tmp_path):
    _write_rows(tmp_path, [_row(confidence=0.3)])
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"min_confidence": 0.5}))
    assert ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=cfg) == {}


def test_yesterday_file_read_near_midnight(tmp_path):
    midnight = datetime(2026, 7, 24, 0, 1, 0, tzinfo=timezone.utc)
    _write_rows(tmp_path, [_row(ts=midnight - timedelta(seconds=90))],
                day=midnight - timedelta(days=1))
    advice = ea.load_advice_by_symbol(
        now=midnight, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=tmp_path / "missing.json",
    )
    assert set(advice) == {"PSQ"}


def test_unreadable_evidence_returns_empty(tmp_path):
    (tmp_path / f"exit_advice_{_NOW.strftime('%Y%m%d')}.ndjson").write_text(
        "not-json\n{\"半\": tru\n")
    assert ea.load_advice_by_symbol(
        now=_NOW, evidence_dir=tmp_path, excluded_symbols=frozenset(),
        config_path=tmp_path / "missing.json") == {}


# --------------------------------------------------------------------------- #
# overlay rule
# --------------------------------------------------------------------------- #

_CFG = ExitOverlayConfig(
    mode="shadow", atr_period=14, atr_trail_mult=2.5,
    hard_stop_loss_pct=0.08, min_bars_for_atr=16,
    max_hold_days={"equity": 20.0, "etf": 30.0, "default": 20.0},
)


def _book(symbol="PSQ", strategy="gamma", side="BUY", qty=5.0, broker_qty=5.0):
    open_positions = {
        f"{strategy}|{symbol}": {
            "symbol": symbol, "side": side, "quantity": qty, "open": True,
            "strategy": strategy,
        },
    }
    signed = broker_qty if side == "BUY" else -broker_qty
    guard_state = {
        f"broker_sync|{symbol}": {
            "symbol": symbol, "side": "BUY" if signed >= 0 else "SELL",
            "quantity": abs(signed), "open": True,
        },
    }
    return open_positions, guard_state


def _advice(symbol="PSQ", strategies=("gamma",)):
    return {symbol: {
        "symbol": symbol, "strategies": list(strategies),
        "latest_ts_utc": "2026-07-23T14:59:30Z", "max_confidence": 0.7,
        "reasons": ["trend_break_exit"], "count_fresh": 2,
    }}


def _eval(open_positions, guard_state, *, advice=None, mode="off",
          price=33.0, excluded=frozenset()):
    return evaluate_positions(
        open_positions=open_positions,
        guard_state=guard_state,
        bars_by_symbol={},
        price_by_symbol={s: price for s in
                         {v["symbol"] for v in open_positions.values()}},
        anchors={},
        config=_CFG,
        now_utc=_NOW,
        excluded_symbols=excluded,
        advice_by_symbol=advice or {},
        advice_mode=mode,
    )


def test_record_mode_emits_advice_would_close_no_intent():
    op, gs = _book()
    res = _eval(op, gs, advice=_advice(), mode="record")
    kinds = {v.verdict for v in res.verdicts}
    assert "ADVICE_WOULD_CLOSE" in kinds
    v = next(v for v in res.verdicts if v.verdict == "ADVICE_WOULD_CLOSE")
    assert v.reason == "strategy_advice"
    assert v.close_qty == 5.0
    assert res.close_intents == []                        # evidence only


def test_consume_mode_emits_reduce_only_intent():
    op, gs = _book(qty=8.0, broker_qty=5.0)               # guard bloated vs broker
    res = _eval(op, gs, advice=_advice(), mode="consume")
    v = next(v for v in res.verdicts if v.verdict == "WOULD_CLOSE")
    assert v.reason == "strategy_advice"
    assert v.close_qty == 5.0                             # clamped to broker truth
    assert len(res.close_intents) == 1
    ci = res.close_intents[0]
    assert ci["reason"] == "exit_overlay_strategy_advice"
    assert ci["quantity"] == 5.0
    assert ci["close_side"] == "SELL"


def test_kernel_precedence_hard_stop_beats_advice():
    op, gs = _book()
    # anchor forces a hard stop: entry 100, price 33 -> deep below stop
    res = evaluate_positions(
        open_positions=op, guard_state=gs, bars_by_symbol={},
        price_by_symbol={"PSQ": 33.0},
        anchors={"gamma|PSQ": {"entry_price": 100.0, "peak": 100.0,
                               "trough": 100.0, "first_seen_utc": "2026-07-20T00:00:00Z"}},
        config=_CFG, now_utc=_NOW, excluded_symbols=frozenset(),
        advice_by_symbol=_advice(), advice_mode="consume",
    )
    v = next(v for v in res.verdicts if v.verdict == "WOULD_CLOSE")
    assert v.reason == "hard_stop_loss"                   # kernel wins
    assert res.close_intents[0]["reason"] == "exit_overlay_hard_stop_loss"


def test_owning_strategy_wall():
    op, gs = _book(strategy="gamma")
    res = _eval(op, gs, advice=_advice(strategies=("alpha",)), mode="consume")
    kinds = {v.verdict for v in res.verdicts}
    assert kinds == {"HOLD"}                              # alpha can't close gamma's leg
    assert res.close_intents == []


def test_long_only_wall():
    op, gs = _book(side="SELL", qty=5.0, broker_qty=5.0)  # short leg
    res = _eval(op, gs, advice=_advice(), mode="consume")
    assert all(v.verdict != "WOULD_CLOSE" or v.reason != "strategy_advice"
               for v in res.verdicts)
    assert not any(ci["reason"] == "exit_overlay_strategy_advice"
                   for ci in res.close_intents)


def test_off_mode_byte_identical():
    op, gs = _book()
    with_advice = _eval(op, gs, advice=_advice(), mode="off")
    without = _eval(op, gs, advice=None, mode="off")
    assert [v.to_dict() for v in with_advice.verdicts] == \
           [v.to_dict() for v in without.verdicts]
    assert with_advice.close_intents == without.close_intents


def test_excluded_symbol_skips_before_advice():
    op, gs = _book(symbol="MSFT")
    res = _eval(op, gs, advice=_advice(symbol="MSFT"), mode="consume",
                excluded={"MSFT"})
    kinds = [v.verdict for v in res.verdicts]
    assert kinds == ["SKIP_EXCLUDED"]
    assert res.close_intents == []


def test_advice_would_close_not_counted_as_would_close():
    """record-mode verdicts must not inflate the would_close set (heartbeat)."""
    op, gs = _book()
    res = _eval(op, gs, advice=_advice(), mode="record")
    assert res.would_close == []
