#!/usr/bin/env python3
"""
chad/risk/daily_throttle.py

Production-grade daily notional throttle for CHAD.

This module enforces per-day notional caps along three dimensions:

    * Per symbol  (e.g. max notional in SPY per day)
    * Per strategy (e.g. max notional for BetaBrain per day)
    * Global total (max aggregate notional per day across all symbols/strategies)

Design goals
------------
- Deterministic: signals are processed in order, with caps applied consistently.
- Testable: pure logic, no hidden I/O; log reading is isolated and optional.
- Safe by default: if inputs are missing or malformed, the throttle rejects
  those signals rather than silently allowing risk drift.

This module is intentionally synchronous and minimal in surface area while
remaining fully typed and robust.  It is designed to be “boring reliable” in
production and easy to reason about under audit.

The unit tests in ``chad/tests/test_daily_throttle.py`` are treated as a
contract and kept fully compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from chad.types import RoutedSignal, StrategyName


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DailyThrottleConfig:
    """
    Immutable configuration for daily notional caps.

    Attributes
    ----------
    per_symbol:
        Mapping from symbol (e.g. "SPY") to maximum per-day notional.
        Symbols not present are treated as having no per-symbol cap.
    per_strategy:
        Mapping from strategy name (e.g. "beta") to maximum per-day notional.
        Strategies not present are treated as having no per-strategy cap.
    global_cap:
        Maximum total notional across all symbols and strategies per day.
        A value of 0.0 means "no trading allowed".
    """

    per_symbol: Mapping[str, float]
    per_strategy: Mapping[str, float]
    global_cap: float

    def __post_init__(self) -> None:
        # Defensive validation: caps must be finite and non-negative.
        for k, v in self.per_symbol.items():
            if v < 0:
                raise ValueError(f"per_symbol cap for {k!r} must be non-negative")
        for k, v in self.per_strategy.items():
            if v < 0:
                raise ValueError(f"per_strategy cap for {k!r} must be non-negative")
        if self.global_cap < 0:
            raise ValueError("global_cap must be non-negative")


@dataclass(frozen=True)
class DailyUsage:
    """
    Snapshot of notional usage for a given trading day.

    The test suite expects this shape from ``_load_today_notional``:
        usage.symbol_notional["SPY"] -> float
        usage.strategy_notional["beta"] -> float
        usage.total_notional -> float
    """

    symbol_notional: Mapping[str, float]
    strategy_notional: Mapping[str, float]
    total_notional: float

    @classmethod
    def empty(cls) -> "DailyUsage":
        """Return an empty usage snapshot with zero notional everywhere."""
        return cls(symbol_notional={}, strategy_notional={}, total_notional=0.0)


@dataclass(frozen=True)
class ThrottleDecision:
    """
    Result of applying the daily throttle to a batch of signals.

    Attributes
    ----------
    accepted:
        Signals that passed all caps and are allowed to proceed.
    rejected:
        Human-readable messages describing why particular signals were rejected.
        Messages are ordered corresponding to the rejected signals in input order.
    symbol_notional:
        Final per-symbol notional after applying accepted signals.
    strategy_notional:
        Final per-strategy notional after applying accepted signals.
    total_notional:
        Final global notional after applying accepted signals.
    """

    accepted: List[RoutedSignal]
    rejected: List[str]
    symbol_notional: Mapping[str, float]
    strategy_notional: Mapping[str, float]
    total_notional: float


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _parse_iso_date(ts: str) -> Optional[date]:
    """
    Parse an ISO-8601 timestamp string to a date.

    This is intentionally tolerant: any parsing failure returns None rather
    than raising, so a single bad record cannot break the throttle.
    """
    try:
        # Python's fromisoformat handles both naive and offset-aware stamps.
        return datetime.fromisoformat(ts).date()
    except Exception:
        return None


def _accumulate_notional_from_log_line(
    record: Dict[str, object],
    target_day: date,
    symbol_notional: MutableMapping[str, float],
    strategy_notional: MutableMapping[str, float],
) -> float:
    """
    Given a JSON record, update usage maps if it belongs to ``target_day``.

    Returns the notional added (0.0 if the record is ignored).
    """
    ts_raw = record.get("cycle_timestamp") or record.get("timestamp")
    if not isinstance(ts_raw, str):
        return 0.0

    rec_day = _parse_iso_date(ts_raw)
    if rec_day != target_day:
        return 0.0

    symbol = record.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        return 0.0

    try:
        net_size = float(record.get("net_size", 0.0))
        price = float(record.get("price", 0.0))
    except (TypeError, ValueError):
        return 0.0

    notional = abs(net_size) * price
    if notional <= 0.0:
        return 0.0

    symbol_notional[symbol] = symbol_notional.get(symbol, 0.0) + notional

    strategies = record.get("strategies") or []
    if isinstance(strategies, str):
        strategies = [strategies]
    if isinstance(strategies, Iterable):
        for s in strategies:
            s_key = str(s)
            strategy_notional[s_key] = strategy_notional.get(s_key, 0.0) + notional

    return notional


# --------------------------------------------------------------------------- #
# Log-based usage loading
# --------------------------------------------------------------------------- #


def _load_today_notional(log_path: Path, day: date) -> DailyUsage:
    """
    Load per-day notional usage from an NDJSON execution log.

    Parameters
    ----------
    log_path:
        Path to an NDJSON file where each line is a JSON object representing a
        past execution / "would-trade" record.
    day:
        The calendar date (UTC) for which notional should be accumulated.

    Returns
    -------
    DailyUsage
        Per-symbol, per-strategy, and global notional for ``day``.

    Behaviour
    ---------
    * If the log file does not exist, an empty usage snapshot is returned.
    * Malformed lines or records are ignored rather than causing failures.
    """
    if not log_path.exists():
        return DailyUsage.empty()

    symbol_usage: Dict[str, float] = {}
    strategy_usage: Dict[str, float] = {}
    total_notional = 0.0

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip broken lines; logging layer should surface them separately.
                continue
            if not isinstance(record, dict):
                continue
            total_notional += _accumulate_notional_from_log_line(
                record, day, symbol_usage, strategy_usage
            )

    return DailyUsage(
        symbol_notional=symbol_usage,
        strategy_notional=strategy_usage,
        total_notional=total_notional,
    )


# --------------------------------------------------------------------------- #
# Core throttling logic
# --------------------------------------------------------------------------- #


def _strategy_key(strat: object) -> str:
    """
    Normalise a strategy identifier (enum or string) into a key.

    This supports both StrategyName enums and plain strings so that the
    throttle continues to work even if the StrategyName type evolves.
    """
    if isinstance(strat, StrategyName):
        return strat.value  # type: ignore[no-any-return]
    return str(strat)


def evaluate_throttle(
    signals: Sequence[RoutedSignal],
    config: DailyThrottleConfig,
    price_map: Mapping[str, float],
    *,
    today: Optional[date] = None,
    log_path: Optional[Path] = None,
) -> ThrottleDecision:
    """
    Apply daily notional caps to a batch of routed signals.

    Parameters
    ----------
    signals:
        Ordered sequence of RoutedSignal objects produced by CHAD’s pipeline.
    config:
        DailyThrottleConfig specifying per-symbol, per-strategy, and global caps.
    price_map:
        Mapping from symbol -> price to use for notional calculations.
        Tests provide this explicitly; in production this would typically be
        built from the latest market data snapshot.
    today:
        Date to treat as "today" for log usage.  Defaults to UTC today.
    log_path:
        Optional NDJSON log file path containing existing executions for the day.
        If provided and the file exists, usage is bootstrapped from that file.
        If None, usage starts at zero.

    Returns
    -------
    ThrottleDecision
        Accepted signals, rejection reasons, and the final usage snapshot.
    """
    day = today or datetime.utcnow().date()

    if log_path is not None:
        base_usage = _load_today_notional(log_path, day)
        symbol_usage: Dict[str, float] = dict(base_usage.symbol_notional)
        strategy_usage: Dict[str, float] = dict(base_usage.strategy_notional)
        total_usage: float = base_usage.total_notional
    else:
        symbol_usage = {}
        strategy_usage = {}
        total_usage = 0.0

    accepted: List[RoutedSignal] = []
    rejected: List[str] = []

    for sig in signals:
        symbol = sig.symbol
        price = price_map.get(symbol)
        if price is None:
            rejected.append(f"{symbol}: no price available; rejecting signal")
            continue

        try:
            notional = abs(float(sig.net_size)) * float(price)
        except (TypeError, ValueError):
            rejected.append(f"{symbol}: invalid size/price; rejecting signal")
            continue

        # 1) Global cap check – applied first so "global cap exceeded" appears
        #    in scenarios where both global and symbol caps would otherwise
        #    trip, matching the test expectations.
        new_total_usage = total_usage + notional
        if new_total_usage > config.global_cap:
            rejected.append(
                f"{symbol}: global cap exceeded "
                f"(used {total_usage:.2f} + {notional:.2f} > {config.global_cap:.2f})"
            )
            continue

        # 2) Per-symbol cap
        used_symbol = symbol_usage.get(symbol, 0.0)
        new_symbol_total = used_symbol + notional
        symbol_cap = config.per_symbol.get(symbol)
        if symbol_cap is not None and new_symbol_total > symbol_cap:
            rejected.append(
                f"{symbol}: symbol cap exceeded "
                f"(used {used_symbol:.2f} + {notional:.2f} > {symbol_cap:.2f})"
            )
            continue

        # 3) Per-strategy cap
        violation: Optional[str] = None
        for strat in sig.source_strategies:
            s_key = _strategy_key(strat)
            used_strat = strategy_usage.get(s_key, 0.0)
            new_strat_total = used_strat + notional
            strat_cap = config.per_strategy.get(s_key)
            if strat_cap is not None and new_strat_total > strat_cap:
                violation = (
                    f"{symbol}: strategy cap exceeded for {s_key} "
                    f"(used {used_strat:.2f} + {notional:.2f} > {strat_cap:.2f})"
                )
                break

        if violation is not None:
            rejected.append(violation)
            continue

        # If we reach here the signal is accepted; update usage.
        accepted.append(sig)
        total_usage = new_total_usage
        symbol_usage[symbol] = new_symbol_total
        for strat in sig.source_strategies:
            s_key = _strategy_key(strat)
            strategy_usage[s_key] = strategy_usage.get(s_key, 0.0) + notional

    return ThrottleDecision(
        accepted=accepted,
        rejected=rejected,
        symbol_notional=symbol_usage,
        strategy_notional=strategy_usage,
        total_notional=total_usage,
    )


def throttle_signals(
    signals: Sequence[RoutedSignal],
    config: DailyThrottleConfig,
    price_map: Mapping[str, float],
    *,
    today: Optional[date] = None,
    log_path: Optional[Path] = None,
) -> ThrottleDecision:
    """
    Convenience wrapper used by CHAD callers.

    This function simply forwards to :func:`evaluate_throttle` while keeping
    a backwards-compatible signature.
    """
    return evaluate_throttle(
        signals=signals,
        config=config,
        price_map=price_map,
        today=today,
        log_path=log_path,
    )
