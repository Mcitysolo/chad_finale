"""U2 integration tests — position_exit_overlay runner end-to-end + live_loop wiring.

Covers: SHADOW runs, logs verdicts, writes evidence, submits NOTHING; flip to ACTIVE routes a
reduce-only close through the REAL reconciler submit path (apply_close_intents ->
paper_adapter.submit_strategy_trade_intents) and the close is provably RTH-gate-subject; the
submit-time reduce-only reclamp drops a phantom; and the live_loop hot path is wired at the
correct cycle point (after the reconciler, before market_metrics/intent planning).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.risk import position_exit_overlay as pxo

UTC = timezone.utc
NOW = datetime(2026, 7, 13, 15, 0, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(mode):
    return pxo.load_overlay_config({
        "mode": mode, "atr_period": 14, "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08, "min_bars_for_atr": 16,
        "max_hold_days": {"equity": 20.0, "etf": 30.0, "default": 20.0},
    })


def _guard(strat, broker):
    state = {"_version": 1}
    for key, (sym, side, qty, days) in strat.items():
        state[key] = {"open": True, "symbol": sym, "side": side, "quantity": qty,
                      "strategy": key.split("|")[0],
                      "opened_at": (NOW - timedelta(days=days)).isoformat()}
    for sym, (side, qty) in broker.items():
        state[f"broker_sync|{sym}"] = {"open": False, "symbol": sym, "side": side,
                                       "quantity": qty, "strategy": "broker_sync"}
    return state


def _open(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


class _FakeAdapter:
    """Records submit calls; returns [] so apply_close_intents does no evidence/guard I/O."""

    def __init__(self):
        self.calls = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        return []


def _runner(mode, tmp_path, state, prices, anchors):
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({"anchors": anchors}))
    return pxo.PositionExitOverlay(
        _cfg(mode),
        evidence_path=tmp_path / "evi",
        state_path=sp,
        guard_loader=lambda: state,
        open_positions_loader=lambda: _open(state),
        bars_loader=lambda syms: {},
        price_loader=lambda syms: prices,
        env={},
    )


def test_shadow_logs_verdict_and_submits_nothing(tmp_path, caplog):
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    anchors = {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                             "first_seen_utc": NOW.isoformat()}}
    runner = _runner("shadow", tmp_path, state, {"BAC": 90.0}, anchors)  # 90 < hard_stop 92
    adapter = _FakeAdapter()
    with caplog.at_level(logging.INFO):
        res = runner.run_cycle(adapter, now_utc=NOW)
    assert [v.reason for v in res.would_close] == ["hard_stop_loss"]
    assert adapter.calls == []  # SHADOW submits nothing
    assert any(pxo.MARKER_SHADOW in r.getMessage() and "WOULD_CLOSE" in r.getMessage()
               for r in caplog.records)
    ev = list((tmp_path / "evi").glob("exit_overlay_*.ndjson"))
    assert ev and json.loads(ev[0].read_text().splitlines()[0])["verdict"] == "WOULD_CLOSE"


def test_active_reduce_only_close_reaches_submit_path(tmp_path):
    # TSLA is NOT operator-excluded (unlike BAC), so the GAP-001 exclusion chokepoint in
    # apply_close_intents is not what governs here — the reduce-only clamp is.
    # guard says 100, broker confirms 60 -> reduce-only close of 60 reaches the adapter.
    state = _guard({"gamma|TSLA": ("TSLA", "BUY", 100, 1)}, {"TSLA": ("BUY", 60)})
    anchors = {"gamma|TSLA": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    runner = _runner("active", tmp_path, state, {"TSLA": 90.0}, anchors)
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW)
    assert res.would_close  # a close was proposed
    # It reached the REAL submit path (apply_close_intents -> adapter.submit_strategy_trade_intents)
    assert len(adapter.calls) == 1
    intent = adapter.calls[0][0]
    assert intent.symbol == "TSLA"
    assert str(getattr(intent.side, "value", intent.side)).upper() == "SELL"  # close of a long
    assert intent.quantity == 60.0            # reduce-only: clamped to broker-confirmed 60
    assert intent.sec_type == "STK"

    # ...and the close is provably subject to the RTH gate (respects it): equity is RTH-gated,
    # and outside the session the same gate the adapter applies would block it.
    from chad.execution.ibkr_adapter import resolve_asset_class
    from chad.execution.rth_gate import is_rth_gated_asset, rth_block_reason
    ac = resolve_asset_class("TSLA")
    assert is_rth_gated_asset(ac)
    closed = datetime(2026, 7, 13, 6, 0, tzinfo=UTC)  # 02:00 ET -> session closed
    assert rth_block_reason(ac, closed, {}) is not None  # would be RTH-blocked outside hours


def test_active_phantom_dropped_at_submit_time(tmp_path):
    # A WOULD_CLOSE is proposed against a confirmed position, but at submit the fresh guard shows
    # NO broker truth (phantom) -> reduce-only reclamp drops it -> nothing submitted.
    state = _guard({"gamma|TSLA": ("TSLA", "BUY", 100, 1)}, {"TSLA": ("BUY", 100)})
    anchors = {"gamma|TSLA": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({"anchors": anchors}))
    # guard_loader returns confirmed state on first read (eval), phantom on the submit re-read.
    reads = {"n": 0}

    def _guard_loader():
        reads["n"] += 1
        return state if reads["n"] == 1 else _guard({"gamma|TSLA": ("TSLA", "BUY", 100, 1)}, {})

    runner = pxo.PositionExitOverlay(
        _cfg("active"),
        evidence_path=tmp_path / "evi", state_path=sp,
        guard_loader=_guard_loader,
        open_positions_loader=lambda: _open(state),
        bars_loader=lambda syms: {}, price_loader=lambda syms: {"TSLA": 90.0},
        env={},
    )
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW)
    assert res.would_close             # overlay proposed the close at eval time
    assert adapter.calls == []          # ...but submit-time reclamp dropped the phantom


def test_live_loop_wired_after_reconciler_before_market_metrics():
    """Structural wiring guard: the overlay runs in run_once, after the reconciler and before
    the market_metrics publisher (i.e. before intent planning)."""
    src = (REPO_ROOT / "chad" / "core" / "live_loop.py").read_text()
    assert "from chad.risk.position_exit_overlay import build_default_overlay" in src
    assert "_exit_overlay.run_cycle(_paper_adapter)" in src
    i_recon = src.index("reconcile_positions_with_signals")
    i_overlay = src.index("build_default_overlay")
    i_metrics = src.index("MarketMetricsPublisher")
    assert i_recon < i_overlay < i_metrics, "overlay must sit after reconciler, before metrics"
