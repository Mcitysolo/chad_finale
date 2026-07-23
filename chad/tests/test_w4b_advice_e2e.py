"""W4B-4 (J16): end-to-end advice chain + receipt regressions + review tool.

The full chain, one test each:
  recorder rows -> aggregator -> overlay(ACTIVE, CHAD_EXIT_ADVICE=record)
    -> ADVICE_WOULD_CLOSE evidence, adapter NEVER called (the D6-rider ladder:
       consume is a separate flip);
  same chain with consume -> exactly one reduce-only close reaches the fake
    adapter, carrying the W4B-2 forensic stamps (reason/position_key/action);
  the gamma|MSFT receipt (operator-excluded): recorded flagged, never
    aggregated, SKIP_EXCLUDED at the overlay, apply_close_intents backstop —
    all three walls proven behaviorally;
  the review tool's GO criteria detect violations (excluded close, clamp
    breach, pre-flip consumption).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.core import exit_advice as ea
import chad.risk.position_exit_overlay as pxo

_LOG = logging.getLogger("test_w4b_advice_e2e")
NOW = datetime(2026, 7, 23, 15, 30, 0, tzinfo=timezone.utc)


def _cfg(mode):
    return pxo.ExitOverlayConfig(
        mode=mode, atr_period=14, atr_trail_mult=2.5, hard_stop_loss_pct=0.08,
        min_bars_for_atr=16,
        max_hold_days={"equity": 20.0, "etf": 30.0, "default": 20.0},
    )


def _guard(strat, broker):
    state = {"_version": 1}
    for key, (sym, side, qty) in strat.items():
        state[key] = {"open": True, "symbol": sym, "side": side, "quantity": qty,
                      "strategy": key.split("|")[0],
                      "opened_at": (NOW - timedelta(days=1)).isoformat()}
    for sym, (side, qty) in broker.items():
        state[f"broker_sync|{sym}"] = {"open": False, "symbol": sym, "side": side,
                                       "quantity": qty, "strategy": "broker_sync"}
    return state


def _open(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


class _FakeAdapter:
    def __init__(self):
        self.calls = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        return []


def _record_and_load(tmp_path, symbol="PSQ", strategy="gamma"):
    """Run the real recorder, then the real aggregator, on tmp paths."""
    evidence = tmp_path / "advice"
    from chad.types import AssetClass, SignalSide, TradeSignal
    sig = TradeSignal(strategy=strategy, symbol=symbol, side=SignalSide.SELL,
                      size=5.0, confidence=0.7, asset_class=AssetClass.ETF,
                      meta={"reason": "trend_break_exit"})
    n = ea.record_dropped_urges(
        [sig], site="router", mode="record",
        excluded_symbols=frozenset({"MSFT"}),
        evidence_dir=evidence, state_path=tmp_path / "advice_state.json",
        logger=_LOG, now=NOW,
    )
    assert n == 1
    advice = ea.load_advice_by_symbol(
        now=NOW + timedelta(seconds=30), evidence_dir=evidence,
        excluded_symbols=frozenset({"MSFT"}),
        config_path=tmp_path / "missing.json",
    )
    return advice


def _runner(tmp_path, state, env, advice):
    return pxo.PositionExitOverlay(
        _cfg("active"),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        guard_loader=lambda: state,
        open_positions_loader=lambda: _open(state),
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {"PSQ": 33.0, "MSFT": 500.0},
        env=env,
        excluded_symbols=frozenset({"MSFT"}),
        advice_loader=lambda now: advice,
    )


def test_e2e_record_mode_evidence_only_no_submit(tmp_path):
    advice = _record_and_load(tmp_path)
    assert set(advice) == {"PSQ"}
    state = _guard({"gamma|PSQ": ("PSQ", "BUY", 5.0)}, {"PSQ": ("BUY", 5.0)})
    runner = _runner(tmp_path, state, {"CHAD_EXIT_ADVICE": "record"}, advice)
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW + timedelta(seconds=30))
    assert adapter.calls == []                            # record NEVER submits
    kinds = {v.verdict for v in res.verdicts}
    assert "ADVICE_WOULD_CLOSE" in kinds
    ev = list((tmp_path / "evi").glob("exit_overlay_*.ndjson"))
    rows = [json.loads(l) for l in ev[0].read_text().splitlines()]
    adv = [r for r in rows if r["verdict"] == "ADVICE_WOULD_CLOSE"]
    assert adv and adv[0]["reason"] == "strategy_advice"
    assert adv[0]["close_qty"] == 5.0


def test_e2e_consume_mode_submits_with_forensic_stamps(tmp_path):
    advice = _record_and_load(tmp_path)
    # guard bloated (8) vs broker truth (5) -> the clamp must win
    state = _guard({"gamma|PSQ": ("PSQ", "BUY", 8.0)}, {"PSQ": ("BUY", 5.0)})
    runner = _runner(tmp_path, state, {"CHAD_EXIT_ADVICE": "consume"}, advice)
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW + timedelta(seconds=30))
    assert [v.reason for v in res.would_close] == ["strategy_advice"]
    assert len(adapter.calls) == 1
    intent = adapter.calls[0][0]
    assert intent.symbol == "PSQ"
    assert str(getattr(intent.side, "value", intent.side)).upper() == "SELL"
    assert intent.quantity == 5.0                         # broker-clamped, never guard
    # W4B-2 forensic stamps travel to the adapter boundary
    assert intent.meta["action"] == "CLOSE"
    assert intent.meta["reason"] == "exit_overlay_strategy_advice"
    assert intent.meta["position_key"] == "gamma|PSQ"
    assert intent.meta["close_origin"] == "apply_close_intents"


def test_receipt_gamma_msft_three_walls(tmp_path):
    """The standing gamma|MSFT urge (operator-excluded) can never become a close."""
    from chad.types import AssetClass, SignalSide, TradeSignal
    evidence = tmp_path / "advice"
    sig = TradeSignal(strategy="gamma", symbol="MSFT", side=SignalSide.SELL,
                      size=16.0, confidence=0.7, asset_class=AssetClass.EQUITY,
                      meta={"reason": "trend_break_exit"})
    # Wall 0 (recorder): recorded, flagged — never silently dropped
    ea.record_dropped_urges(
        [sig], site="router", mode="record", excluded_symbols=frozenset({"MSFT"}),
        evidence_dir=evidence, state_path=tmp_path / "s.json", logger=_LOG, now=NOW)
    row = json.loads((evidence / f"exit_advice_{NOW.strftime('%Y%m%d')}.ndjson").read_text())
    assert row["excluded"] is True
    # Wall 1 (aggregator): never consumable
    advice = ea.load_advice_by_symbol(
        now=NOW + timedelta(seconds=10), evidence_dir=evidence,
        excluded_symbols=frozenset({"MSFT"}), config_path=tmp_path / "m.json")
    assert advice == {}
    # Wall 2 (overlay): even if advice were forged, SKIP_EXCLUDED fires first
    state = _guard({"beta|MSFT": ("MSFT", "BUY", 16.0)}, {"MSFT": ("BUY", 34.0)})
    forged = {"MSFT": {"symbol": "MSFT", "strategies": ["beta"],
                       "latest_ts_utc": "x", "max_confidence": 1.0,
                       "reasons": ["forged"], "count_fresh": 9}}
    runner = _runner(tmp_path, state, {"CHAD_EXIT_ADVICE": "consume"}, forged)
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW + timedelta(seconds=10))
    assert [v.verdict for v in res.verdicts] == ["SKIP_EXCLUDED"]
    assert adapter.calls == []
    # Wall 3 (apply_close_intents chokepoint): a hand-built close dict is refused
    import chad.core.position_reconciler as pr
    if "MSFT" in pr._EFFECTIVE_NON_CHAD_SYMBOLS:  # production floor includes MSFT
        pr.apply_close_intents([{
            "symbol": "MSFT", "close_side": "SELL", "quantity": 16.0,
            "strategy": "beta", "reason": "exit_overlay_strategy_advice",
            "position_key": "beta|MSFT",
        }], adapter)
        assert adapter.calls == []


def test_review_tool_detects_violations_and_go(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exit_advice_report",
        Path(__file__).resolve().parents[2] / "scripts" / "exit_advice_report.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    advice_dir = tmp_path / "advice"
    overlay_dir = tmp_path / "overlay"
    advice_dir.mkdir()
    overlay_dir.mkdir()
    day = NOW.strftime("%Y%m%d")
    (advice_dir / f"exit_advice_{day}.ndjson").write_text("\n".join([
        json.dumps({"ts_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "site": "router",
                    "strategy": "gamma", "symbol": "PSQ", "side": "SELL",
                    "excluded": False}),
        json.dumps({"ts_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "site": "router",
                    "strategy": "gamma", "symbol": "MSFT", "side": "SELL",
                    "excluded": True}),
        # violation: excluded per SSOT but the recorder failed to flag it
        json.dumps({"ts_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "site": "router",
                    "strategy": "beta", "symbol": "AAPL", "side": "SELL",
                    "excluded": False}),
    ]) + "\n")
    (overlay_dir / f"exit_overlay_{day}.ndjson").write_text("\n".join([
        # clean would-fire
        json.dumps({"verdict": "ADVICE_WOULD_CLOSE", "reason": "strategy_advice",
                    "symbol": "PSQ", "strategy": "gamma", "close_qty": 5.0,
                    "broker_confirmed_qty": 5.0, "ts_utc": "t"}),
        # violation: clamp breach
        json.dumps({"verdict": "ADVICE_WOULD_CLOSE", "reason": "strategy_advice",
                    "symbol": "TLT", "strategy": "gamma", "close_qty": 9.0,
                    "broker_confirmed_qty": 5.0, "ts_utc": "t"}),
        # violation: consumed close on an excluded symbol
        json.dumps({"verdict": "WOULD_CLOSE", "reason": "strategy_advice",
                    "symbol": "MSFT", "strategy": "beta", "close_qty": 1.0,
                    "broker_confirmed_qty": 1.0, "ts_utc": "t"}),
        # non-advice rows are ignored
        json.dumps({"verdict": "WOULD_CLOSE", "reason": "hard_stop_loss",
                    "symbol": "UNH", "strategy": "gamma", "close_qty": 1.0,
                    "broker_confirmed_qty": 1.0, "ts_utc": "t"}),
    ]) + "\n")

    report = tool.build_report(
        advice_dir=advice_dir, overlay_dir=overlay_dir,
        excluded=["MSFT", "AAPL"], days=2, now=NOW,
    )
    assert report["schema_version"] == "exit_advice_review.v1"
    assert report["advice_rows"]["total"] == 3
    assert report["advice_rows"]["excluded_flagged"] == 1
    assert report["advice_rows"]["by_tuple_truncated"] is False
    assert len(report["advice_would_close"]) == 2
    assert len(report["consumed_advice_closes"]) == 1
    v = report["violations"]
    assert [r["symbol"] for r in v["clamp_exceeds_broker"]] == ["TLT"]
    assert [r["symbol"] for r in v["advice_close_on_excluded"]] == ["MSFT"]
    assert [r["symbol"] for r in v["excluded_unflagged"]] == ["AAPL"]
    g = report["go_criteria"]
    assert g["recorder_flag_wall_intact"] is False
    assert g["zero_advice_closes_on_excluded"] is False
    assert g["all_clamps_within_broker"] is False
    assert g["would_fire_corpus_nonempty"] is True
    assert g["nothing_consumed_pre_flip"] is False
