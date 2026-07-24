#!/usr/bin/env python3
"""
chad/tests/test_w6a_futures_signal_path.py

W6A-6 — WAKE VERIFICATION: signal-path proof for gamma_futures and
alpha_futures. Fixture only. No execution, ever.

Why this exists
---------------
The lane brief assumed both strategies were benched for want of bar data. The
audit found otherwise: both need ``min_bars=40`` and their symbols carry
209-251 bars, so they were never data-starved. A read-only probe against the
real 2026-07-23 bars showed both strategies executing their full evaluation
path and emitting zero signals — because no setup had triggered, not because
anything was broken.

That leaves a real question this file answers: with data flowing, does the
signal path actually work end to end? These tests construct bar series that
*do* trigger each strategy's documented setup and assert a well-formed
TradeSignal comes out.

What this does NOT do
---------------------
This proves the signal path, nothing more. In production both strategies stay
gated by:
  * the regime roster — regime was "ranging" on 2026-07-23, which excludes
    alpha_futures and gamma_futures (they are on-roster in trending_bull /
    trending_bear / volatile / unknown), and
  * operator intent — live_gate reported EXIT_ONLY per INCIDENT-0723.

Neither gate is touched here, and neither should be: waking these strategies
in production is a regime and operator-intent decision, not a data one.
Futures EXECUTION remains permanently disabled by
CHAD_DISABLE_FUTURES_EXECUTION=1, which this lane never reads or modifies.
No order can result from anything in this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import pytest

from chad.strategies.alpha_futures import build_alpha_futures_signals
from chad.strategies.gamma_futures import (
    _resolve_gamma_universe,
    build_gamma_futures_signals,
)
from chad.types import SignalSide

EQUITY = 219_865.0


def _bar(close: float, *, high: float = None, low: float = None, volume: float = 500_000.0) -> Dict:
    return {
        "open": close,
        "high": high if high is not None else close * 1.002,
        "low": low if low is not None else close * 0.998,
        "close": close,
        "volume": volume,
        "ts_utc": "2026-07-22",
    }


def _truly_flat(base: float, n: int = 60) -> List[Dict]:
    """Constant price with high == low == close.

    A series that merely oscillates is NOT flat for a breakout strategy: the
    synthetic ``high = close * 1.002`` alone clears the prior 20-bar high and
    fires alpha_futures. The negative control has to be genuinely featureless.
    """
    return [_bar(base, high=base, low=base) for _ in range(n)]


def _flat_then_breakout(base: float, n: int = 60) -> List[Dict]:
    """Quiet range, then a decisive push through the prior high."""
    bars = [_bar(base + (i % 3) * 0.05) for i in range(n - 1)]
    bars.append(_bar(base * 1.06, high=base * 1.065))
    return bars


def _flat_then_capitulation(base: float, n: int = 60) -> List[Dict]:
    """Quiet range, then a sharp selloff: drives RSI under 25 and price
    through the lower Bollinger band (gamma_futures Case 2 -> BUY)."""
    bars = [_bar(base + (i % 3) * 0.05) for i in range(n - 12)]
    price = base
    for _ in range(12):
        price *= 0.985
        bars.append(_bar(price, high=price * 1.001, low=price * 0.99))
    return bars


def _ctx(bars: Dict[str, List[Dict]]) -> SimpleNamespace:
    prices = {sym: rows[-1]["close"] for sym, rows in bars.items()}
    return SimpleNamespace(
        bars=bars,
        ohlcv=bars,
        prices=prices,
        equity=EQUITY,
        account_equity=EQUITY,
        portfolio=SimpleNamespace(positions={}, equity=EQUITY),
        regime="trending_bull",
    )


# ---------------------------------------------------------------------------
# gamma_futures — mean reversion. No session/overnight gate on this path.
# ---------------------------------------------------------------------------


def test_gamma_futures_emits_signal_on_capitulation() -> None:
    """RSI oversold + close through the lower band is gamma_futures' Case 2."""
    ctx = _ctx({
        "MCL": _flat_then_capitulation(70.0),
        "MYM": _flat_then_capitulation(44_000.0),
        "M2K": _flat_then_capitulation(2_300.0),
    })

    signals = build_gamma_futures_signals(ctx)

    assert signals, "gamma_futures produced no signal on a textbook reversion setup"
    for sig in signals:
        assert sig.symbol in ("MCL", "MYM", "M2K")
        assert sig.side == SignalSide.BUY
        assert 0.0 < float(sig.confidence) <= 1.0
        assert sig.symbol not in ("MES", "MNQ", "MGC"), "must stay disjoint from alpha"


def test_gamma_futures_silent_without_a_setup() -> None:
    """The counterpart assertion: a quiet series must produce nothing. Without
    this, 'it emits' proves only that it emits indiscriminately."""
    signals = build_gamma_futures_signals(_ctx({
        "MCL": _truly_flat(70.0),
        "MYM": _truly_flat(44_000.0),
        "M2K": _truly_flat(2_300.0),
    }))
    assert signals == []


def test_gamma_universe_resolves_from_real_bar_availability() -> None:
    ctx = _ctx({
        "MCL": _flat_then_capitulation(70.0),
        "MYM": _flat_then_capitulation(44_000.0),
        "M2K": _flat_then_capitulation(2_300.0),
    })
    assert _resolve_gamma_universe(ctx) == ("MCL", "MYM", "M2K")


# ---------------------------------------------------------------------------
# alpha_futures — momentum. MES/MNQ are exempt from the MCL/MGC overnight
# gate, so this path needs no clock pinning.
# ---------------------------------------------------------------------------


def test_alpha_futures_emits_signal_on_breakout() -> None:
    ctx = _ctx({
        "MES": _flat_then_breakout(6_400.0),
        "MNQ": _flat_then_breakout(23_000.0),
    })

    signals = build_alpha_futures_signals(ctx)

    assert signals, "alpha_futures produced no signal on a textbook breakout"
    for sig in signals:
        assert sig.symbol in ("MES", "MNQ")
        assert sig.side == SignalSide.BUY
        assert 0.0 < float(sig.confidence) <= 1.0


def test_alpha_futures_silent_without_a_setup() -> None:
    assert build_alpha_futures_signals(_ctx({
        "MES": _truly_flat(6_400.0),
        "MNQ": _truly_flat(23_000.0),
    })) == []


# ---------------------------------------------------------------------------
# The starvation premise, pinned
# ---------------------------------------------------------------------------


def test_min_bars_threshold_is_far_below_real_history() -> None:
    """The brief's premise was starvation. Both strategies need 40 bars; the
    real files carried 209-251. Pin the threshold so a future raise that WOULD
    starve them is a deliberate, visible act."""
    from chad.strategies.alpha_futures import StrategyTuning
    from chad.strategies.gamma_futures import GammaFuturesTuning

    assert StrategyTuning().min_bars == 40
    assert GammaFuturesTuning().min_bars == 40


def test_starves_below_min_bars() -> None:
    """Genuine starvation must still produce nothing — the honest negative."""
    short = [_bar(70.0) for _ in range(20)]
    assert build_gamma_futures_signals(_ctx({"MCL": short})) == []
