"""W2B-4 — the UNH churn regression fixture (defect (b), D7 characterized diff).

The PFF1 receipt (docs/PFF1_churn_case_for_contextbuilder.md): the exit overlay
reduce-only-closed the 273-share UNH gamma long, and within minutes gamma
**re-bought** because it was position-blind — the empty book made gamma's
50-unit cap (gamma.py:304) unreachable, so the BUY branch (gamma.py:386) fired
every cycle. Injecting the real book (UNH=228) makes the cap gate fire and gamma
returns ``[]`` before it can re-buy.

Hermetic: this test inlines the churn shape (UNH 228 long, uptrend that gamma
would enter) and drives the REAL ``gamma_handler`` — it does NOT read the live
``exit_overlay`` ndjson (the non-hermetic coupling D8 warns about). The proof is
a set-diff over signal keys ``(strategy, symbol, side)``: OFF (empty book) emits
the churn re-buy, ON (injected book) does not, and the diff is EXACTLY that one
key — nothing added (no new SELL surface from this injection). ``max_position_units``
is pinned to 50 so the proof is about the injection, not the threshold.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from chad.strategies import gamma as gamma_mod
from chad.strategies.gamma import GammaParams, gamma_handler
from chad.types import (
    AssetClass,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyName,
)

# The churn symbol + the book gamma re-bought to (PFF1 receipt).
_SYMBOL = "UNH"
_INJECTED_QTY = 228.0
_ENTRY_KEY = (StrategyName.GAMMA, _SYMBOL, SignalSide.BUY)
_PARAMS = GammaParams(max_position_units=50.0)  # pinned: proof is about the injection


def _make_bars(n: int = 200, start: float = 300.0, step: float = 1.0) -> List[Dict[str, float]]:
    """Deterministic uptrend OHLC. A steady rise puts fast EMA above slow EMA
    with a trend-regime slope and an ATR% inside gamma's [0.0015, 0.04] band, so
    the trend-momentum BUY branch is reachable on an empty book."""
    out: List[Dict[str, float]] = []
    px = start
    for _ in range(n):
        o = px
        c = px + step
        h = max(o, c) + 0.10
        l = min(o, c) - 0.10
        out.append({"open": o, "high": h, "low": l, "close": c, "volume": 5_000_000})
        px = c
    return out


_BARS = _make_bars()
_PRICE = _BARS[-1]["close"] + 5.0  # above fast EMA -> momentum entry


def _make_ctx(*, positions: Dict[str, Position] | None = None) -> MarketContext:
    now = datetime.now(timezone.utc)
    tick = MarketTick(symbol=_SYMBOL, price=_PRICE, size=100.0,
                      exchange=None, timestamp=now, source="test")
    return MarketContext(
        now=now,
        ticks={_SYMBOL: tick},
        legend=None,
        portfolio=PortfolioSnapshot(timestamp=now, cash=1_000_000.0, positions=positions or {}),
        bars={_SYMBOL: _BARS},
    )


def _keyset(signals) -> set:
    return {(s.strategy, s.symbol, s.side) for s in signals}


def _gamma_keys(monkeypatch, *, positions) -> set:
    """Run the real gamma_handler over a single-symbol universe with a clean
    in-memory state (so the OFF and ON runs are independent)."""
    monkeypatch.setattr(gamma_mod, "get_trade_universe", lambda: [_SYMBOL])
    gamma_mod._STATE.pos.clear()
    return _keyset(gamma_handler(_make_ctx(positions=positions), _PARAMS))


def test_off_book_emits_the_churn_rebuy(monkeypatch):
    """Sanity: on the empty book gamma emits the UNH re-buy (today's behaviour /
    the churn). If this ever stops firing the regression below is vacuous."""
    off = _gamma_keys(monkeypatch, positions={})
    assert _ENTRY_KEY in off


def test_injected_book_suppresses_the_rebuy(monkeypatch):
    """The fix: inject the real 228-share long -> gamma's 50-unit cap gate
    (gamma.py:304) fires and gamma returns [] before the BUY branch."""
    on = _gamma_keys(
        monkeypatch,
        positions={_SYMBOL: Position(_SYMBOL, AssetClass.EQUITY, _INJECTED_QTY, 424.97)},
    )
    assert _ENTRY_KEY not in on


def test_characterized_diff_is_exactly_the_rebuy(monkeypatch):
    """D7 characterized diff: injecting the book removes EXACTLY the churn re-buy
    and adds nothing (no new SELL surface enabled by this injection alone — a
    large leg is frozen by the cap before any native exit can fire)."""
    off = _gamma_keys(monkeypatch, positions={})
    on = _gamma_keys(
        monkeypatch,
        positions={_SYMBOL: Position(_SYMBOL, AssetClass.EQUITY, _INJECTED_QTY, 424.97)},
    )
    assert off - on == {_ENTRY_KEY}
    assert on - off == set()
