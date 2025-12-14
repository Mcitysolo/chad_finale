#!/usr/bin/env python3
"""
chad/tests/test_daily_throttle.py

Unit tests for CHAD's risk.daily_throttle module.

This suite validates:
    - Daily notional parsing from NDJSON
    - Symbol caps
    - Strategy caps
    - Global caps
    - Acceptance / rejection ordering
    - Deterministic behavior
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, date
from pathlib import Path

import pytest

from chad.risk.daily_throttle import (
    DailyThrottleConfig,
    throttle_signals,
    evaluate_throttle,
    _load_today_notional,
)
from chad.types import (
    RoutedSignal,
    AssetClass,
    SignalSide,
    StrategyName,
)


def _make_signal(symbol: str, qty: float, price: float, strat: StrategyName) -> RoutedSignal:
    """Helper: build RoutedSignal with minimal attributes."""
    now = datetime.now(timezone.utc)
    return RoutedSignal(
        symbol=symbol,
        side=SignalSide.BUY,
        net_size=qty,
        source_strategies=(strat,),
        confidence=0.5,
        asset_class=AssetClass.EQUITY,
        created_at=now,
        meta={},
    )


def _write_ndjson(tmp_path: Path, entries: list[dict]) -> Path:
    """Write NDJSON entries to a file for testing."""
    p = tmp_path / "execution_log.ndjson"
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p


def test_load_today_notional(tmp_path: Path) -> None:
    """Verify correct notional parsing from NDJSON for today's date."""
    today = datetime.now(timezone.utc).date()

    entries = [
        {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "context_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "SPY",
            "side": "buy",
            "net_size": 5.0,
            "strategies": ["beta"],
            "price": 100.0,
            "legend_weight": 0.1,
        }
    ]

    log_path = _write_ndjson(tmp_path, entries)
    usage = _load_today_notional(log_path, today)

    assert usage.total_notional == 500.0
    assert usage.symbol_notional["SPY"] == 500.0
    assert usage.strategy_notional["beta"] == 500.0


def test_symbol_cap_rejection(tmp_path: Path) -> None:
    """Signal should be rejected if it exceeds per-symbol cap."""
    today = datetime.now(timezone.utc).date()
    price_map = {"SPY": 100.0}

    # Existing usage = 400
    existing = [
        {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "context_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "SPY",
            "side": "buy",
            "net_size": 4.0,
            "strategies": ["beta"],
            "price": 100.0,
        }
    ]
    log_path = _write_ndjson(tmp_path, existing)

    # Throttle config: symbol cap = 500
    cfg = DailyThrottleConfig(
        per_symbol={"SPY": 500.0},
        per_strategy={"beta": 999_999.0},
        global_cap=999_999.0,
    )

    # New signal = 200 notional -> 400 + 200 = 600 > 500 → reject
    s = _make_signal("SPY", qty=2.0, price=100.0, strat=StrategyName.BETA)

    decision = evaluate_throttle([s], cfg, price_map, today=today, log_path=log_path)

    assert decision.accepted == []
    assert len(decision.rejected) == 1
    assert "symbol cap exceeded" in decision.rejected[0]


def test_strategy_cap_rejection(tmp_path: Path) -> None:
    """Signal rejected if it exceeds strategy cap."""
    today = datetime.now(timezone.utc).date()
    price_map = {"AAPL": 50.0}

    # Existing usage for beta = 400
    existing = [
        {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "context_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "AAPL",
            "side": "buy",
            "net_size": 8.0,
            "strategies": ["beta"],
            "price": 50.0,
        }
    ]
    log_path = _write_ndjson(tmp_path, existing)

    # Strategy cap = 500
    cfg = DailyThrottleConfig(
        per_symbol={"AAPL": 999_999.0},
        per_strategy={"beta": 500.0},
        global_cap=999_999.0,
    )

    # New signal = 200 notional → 400 + 200 = 600 > 500 → reject
    s = _make_signal("AAPL", qty=4.0, price=50.0, strat=StrategyName.BETA)

    decision = evaluate_throttle([s], cfg, price_map, today=today, log_path=log_path)
    assert decision.accepted == []
    assert len(decision.rejected) == 1
    assert "strategy cap exceeded" in decision.rejected[0]


def test_global_cap_rejection(tmp_path: Path) -> None:
    """Global cap is enforced last."""
    today = datetime.now(timezone.utc).date()
    price_map = {"MSFT": 200.0}

    # Existing usage = 900
    existing = [
        {
            "cycle_timestamp": datetime.now(timezone.utc).isoformat(),
            "context_timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "MSFT",
            "side": "buy",
            "net_size": 4.5,
            "strategies": ["beta"],
            "price": 200.0,
        }
    ]
    log_path = _write_ndjson(tmp_path, existing)

    cfg = DailyThrottleConfig(
        per_symbol={"MSFT": 999_999.0},
        per_strategy={"beta": 999_999.0},
        global_cap=1000.0,
    )

    # New signal = 400 → 900 + 400 = 1300 > 1000 → reject
    s = _make_signal("MSFT", qty=2.0, price=200.0, strat=StrategyName.BETA)

    decision = evaluate_throttle([s], cfg, price_map, today=today, log_path=log_path)
    assert decision.accepted == []
    assert len(decision.rejected) == 1
    assert "global cap exceeded" in decision.rejected[0]


def test_accept_when_under_all_caps(tmp_path: Path) -> None:
    """Signal accepted when symbol, strategy, and global caps allow."""
    today = datetime.now(timezone.utc).date()
    price_map = {"QQQ": 400.0}

    # Existing usage = 0
    log_path = _write_ndjson(tmp_path, [])

    cfg = DailyThrottleConfig(
        per_symbol={"QQQ": 10000.0},
        per_strategy={"beta": 5000.0},
        global_cap=15000.0,
    )

    # New signal = 400*2 = 800
    s = _make_signal("QQQ", qty=2.0, price=400.0, strat=StrategyName.BETA)

    decision = evaluate_throttle([s], cfg, price_map, today=today, log_path=log_path)
    assert len(decision.accepted) == 1
    assert decision.rejected == []


def test_multiple_signals_apply_incrementally(tmp_path: Path) -> None:
    """
    Signals are processed sequentially and usage accumulates for later ones.
    """
    today = datetime.now(timezone.utc).date()
    price_map = {"SPY": 100.0}

    log_path = _write_ndjson(tmp_path, [])

    cfg = DailyThrottleConfig(
        per_symbol={"SPY": 600.0},       # symbol cap
        per_strategy={"beta": 600.0},    # strategy cap
        global_cap=600.0,                # global cap
    )

    s1 = _make_signal("SPY", 3.0, 100.0, StrategyName.BETA)  # 300 notional → OK
    s2 = _make_signal("SPY", 2.0, 100.0, StrategyName.BETA)  # 200 notional → OK (total 500)
    s3 = _make_signal("SPY", 2.0, 100.0, StrategyName.BETA)  # 200 notional → would push to 700 → reject

    decision = evaluate_throttle([s1, s2, s3], cfg, price_map, today=today, log_path=log_path)

    # s1 and s2 accepted, s3 rejected
    assert len(decision.accepted) == 2
    assert len(decision.rejected) == 1
    assert "global cap exceeded" in decision.rejected[0]
