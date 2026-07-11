"""WKF U2 — market-hours (RTH) gate on the equity/ETF submit path.

Two layers:
  * the pure gate module (chad.execution.rth_gate + market_hours.equity_rth_is_open):
    window edges (DST-aware), asset-class scope, env toggle;
  * the adapter integration (IbkrAdapter.submit_strategy_trade_intents, dry-run,
    injected clock): a Saturday equity intent is blocked with status=market_closed
    and writes NO idempotency row; a Monday-14:00 equity intent submits; a
    futures intent is exempt; CHAD_RTH_GATE=0 disables the gate.

No live broker, no network, no runtime writes.
"""
from __future__ import annotations

from datetime import datetime, timezone

from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig
from chad.execution.ibkr_executor import StrategyTradeIntent
from chad.execution.rth_gate import (
    rth_block_reason,
    rth_gate_enabled,
    is_rth_gated_asset,
)
from chad.utils.market_hours import equity_rth_is_open


def _u(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# 2026-07-11 is a Saturday; 2026-07-13 is a Monday (verified).
_SAT_0240 = _u("2026-07-11T02:40")          # F2 case
_SAT_2034 = _u("2026-07-11T20:34")          # F2 case
_MON_1400 = _u("2026-07-13T14:00")          # in-session (EDT)


# ---------------------------------------------------------------------------
# Pure window / gate unit tests
# ---------------------------------------------------------------------------

def test_rth_window_edges_edt_dst_aware():
    # EDT regime: session is 13:30-20:00 UTC.
    assert equity_rth_is_open(_u("2026-07-13T13:29")) is False
    assert equity_rth_is_open(_u("2026-07-13T13:30")) is True
    assert equity_rth_is_open(_u("2026-07-13T19:59")) is True
    assert equity_rth_is_open(_u("2026-07-13T20:00")) is False
    # EST regime: session shifts to 14:30-21:00 UTC (DST-correct).
    assert equity_rth_is_open(_u("2026-01-15T14:29")) is False
    assert equity_rth_is_open(_u("2026-01-15T14:30")) is True
    assert equity_rth_is_open(_u("2026-01-15T20:59")) is True
    assert equity_rth_is_open(_u("2026-01-15T21:00")) is False
    # Weekends always closed.
    assert equity_rth_is_open(_SAT_0240) is False
    assert equity_rth_is_open(_SAT_2034) is False


def test_gate_scope_and_toggle():
    assert is_rth_gated_asset("equity") is True
    assert is_rth_gated_asset("etf") is True
    assert is_rth_gated_asset("futures") is False
    assert is_rth_gated_asset("crypto") is False
    assert is_rth_gated_asset("options") is False

    # Default ON; only explicit falsy disables.
    assert rth_gate_enabled({}) is True
    assert rth_gate_enabled({"CHAD_RTH_GATE": "1"}) is True
    assert rth_gate_enabled({"CHAD_RTH_GATE": "garbage"}) is True
    assert rth_gate_enabled({"CHAD_RTH_GATE": "0"}) is False
    assert rth_gate_enabled({"CHAD_RTH_GATE": "false"}) is False


def test_block_reason_matrix():
    # equity/ETF off-hours → block; futures/crypto exempt; in-session → pass.
    assert rth_block_reason("equity", _SAT_0240, {}) == "market_closed_outside_rth"
    assert rth_block_reason("etf", _SAT_2034, {}) == "market_closed_outside_rth"
    assert rth_block_reason("equity", _MON_1400, {}) is None
    assert rth_block_reason("futures", _SAT_0240, {}) is None
    assert rth_block_reason("crypto", _SAT_0240, {}) is None
    # Gate disabled → never blocks even off-hours.
    assert rth_block_reason("equity", _SAT_0240, {"CHAD_RTH_GATE": "0"}) is None


# ---------------------------------------------------------------------------
# Adapter integration through the full submit path
# ---------------------------------------------------------------------------

def _adapter(tmp_path, now):
    return IbkrAdapter(
        config=IbkrConfig(dry_run=True, state_db_path=tmp_path / "s.db",
                          enable_idempotency=True),
        now_fn=lambda: now,
    )


def _equity_intent(symbol="AAPL"):
    return StrategyTradeIntent(
        strategy="alpha", symbol=symbol, sec_type="STK", exchange="SMART",
        currency="USD", side="BUY", order_type="LMT", quantity=10.0,
        notional_estimate=1000.0, limit_price=100.0,
    )


def _futures_intent(symbol="MES"):
    return StrategyTradeIntent(
        strategy="alpha_futures", symbol=symbol, sec_type="FUT", exchange="CME",
        currency="USD", side="BUY", order_type="LMT", quantity=1.0,
        notional_estimate=5000.0, limit_price=5000.0,
    )


def _idempo_row(adapter, raw):
    key = adapter._compute_idempotency_key(adapter._intent_from_trade_intent(raw))
    return adapter._idempotency.get(key)


def test_saturday_equity_intent_blocked_no_idempotency_row(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("CHAD_RTH_GATE", raising=False)  # default ON
    adapter = _adapter(tmp_path, _SAT_0240)
    raw = _equity_intent("AAPL")
    with caplog.at_level("WARNING"):
        out = adapter.submit_strategy_trade_intents([raw])

    assert len(out) == 1
    assert out[0].status == "market_closed"
    # blocked BEFORE the claim → no idempotency row (resubmittable later)
    assert _idempo_row(adapter, raw) is None
    # marker emitted
    assert any(m.startswith("RTH_GATE_BLOCK symbol=AAPL") for m in caplog.messages)


def test_monday_in_session_equity_intent_submits(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAD_RTH_GATE", raising=False)
    adapter = _adapter(tmp_path, _MON_1400)
    raw = _equity_intent("AAPL")
    out = adapter.submit_strategy_trade_intents([raw])

    assert out[0].status == "dry_run", "in-session equity must pass the RTH gate"
    assert _idempo_row(adapter, raw) is not None, "passed order claims idempotency"


def test_futures_intent_exempt_offhours(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAD_RTH_GATE", raising=False)
    monkeypatch.delenv("CHAD_DISABLE_FUTURES_EXECUTION", raising=False)
    adapter = _adapter(tmp_path, _SAT_0240)
    out = adapter.submit_strategy_trade_intents([_futures_intent("MES")])
    # RTH gate must NOT block futures (their session differs). Whatever the
    # terminal status is, it is never the RTH block status.
    assert out[0].status != "market_closed"


def test_gate_disabled_allows_offhours_equity(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_RTH_GATE", "0")
    adapter = _adapter(tmp_path, _SAT_0240)
    raw = _equity_intent("AAPL")
    out = adapter.submit_strategy_trade_intents([raw])
    assert out[0].status == "dry_run", "gate disabled → off-hours equity submits"
    assert _idempo_row(adapter, raw) is not None
