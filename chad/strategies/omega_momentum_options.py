#!/usr/bin/env python3
"""
chad/strategies/omega_momentum_options.py

OMEGA_MOMENTUM_OPTIONS — CHAD's 15th strategy.

Intraday single-leg options momentum strategy. Buys calls on bullish
momentum, puts on bearish momentum. Uses real options chain cache when
available, falls back to Black-Scholes synthetic pricing when not.
Hard time exit by 3:45 PM ET. Fixed 50% profit target and 25% stop loss.

Entry triggers (both required):
  1) Price momentum: 0.3% 5-bar move + EMA slope + volume confirmation
  2) IV timing: VIX regime filter (skip if VIX > 40)

Universe: SPY, QQQ, AAPL, NVDA, MSFT (high-liquidity options only)
Timeframe: 1-minute bars with daily fallback
Market hours: 9:45 AM ET to 3:30 PM ET
Cooldown: 15 minutes per symbol
Max concurrent: 3 options positions
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from chad.types import (
    AssetClass,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)

from chad.strategies.options_pricing import (
    estimate_contract_price,
    estimate_iv_from_vix,
)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CHAINS_CACHE_PATH = RUNTIME_DIR / "options_chains_cache.json"

UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]

# Market hours in UTC (EDT = UTC-4)
ET_OFFSET_HOURS = -4  # EDT
MARKET_OPEN_ET_HOUR, MARKET_OPEN_ET_MIN = 9, 45   # 9:45 AM ET
MARKET_CLOSE_ET_HOUR, MARKET_CLOSE_ET_MIN = 15, 30  # 3:30 PM ET

# Momentum thresholds
MOMENTUM_THRESHOLD = 0.003       # 0.3% in 5 bars
VOLUME_MULT_THRESHOLD = 1.5      # 1.5x 20-bar average volume
VOLUME_STRONG_THRESHOLD = 2.0    # 2x for extra confidence
EMA_PERIOD = 10

# Position limits
COOLDOWN = timedelta(minutes=15)
MAX_OPEN_OPTIONS = 3

# Module-level state
_LAST_SIGNAL: Dict[str, datetime] = {}
_OPEN_COUNT: int = 0

# Chain cache TTL
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_omega_momentum_options_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.OMEGA_MOMENTUM_OPTIONS,
        enabled=True,
        target_universe=None,
        max_gross_exposure=None,
        notes="omega_momentum_options: intraday single-leg options momentum",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_market_hours(now: datetime) -> bool:
    """Check if current time is within the trading window (9:45 AM - 3:30 PM ET)."""
    et_hour = (now.hour + ET_OFFSET_HOURS) % 24
    et_min = now.minute

    # Convert to minutes since midnight for easy comparison
    et_minutes = et_hour * 60 + et_min
    open_minutes = MARKET_OPEN_ET_HOUR * 60 + MARKET_OPEN_ET_MIN
    close_minutes = MARKET_CLOSE_ET_HOUR * 60 + MARKET_CLOSE_ET_MIN

    return open_minutes <= et_minutes <= close_minutes


def _closes(bars: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for b in bars:
        try:
            if isinstance(b, dict):
                v = float(b.get("close", 0.0))
            else:
                v = float(getattr(b, "close", 0.0))
            if v > 0:
                out.append(v)
        except Exception:
            continue
    return out


def _volumes(bars: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for b in bars:
        try:
            if isinstance(b, dict):
                v = float(b.get("volume", 0.0))
            else:
                v = float(getattr(b, "volume", 0.0))
            out.append(max(0.0, v))
        except Exception:
            out.append(0.0)
    return out


def _ema(values: List[float], period: int) -> List[float]:
    """Compute EMA over a list of values."""
    if not values or period <= 0:
        return []
    alpha = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


def _vix_regime(vix: float) -> str:
    if vix < 15:
        return "low"
    if vix <= 30:
        return "normal"
    if vix <= 40:
        return "elevated"
    return "extreme"


def _load_chain_cache() -> Optional[Dict]:
    """Load options chain cache if fresh."""
    try:
        if not CHAINS_CACHE_PATH.exists():
            return None
        data = json.loads(CHAINS_CACHE_PATH.read_text())
        updated = data.get("updated_ts_utc", "")
        if updated:
            cache_time = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            age = (_now_utc() - cache_time).total_seconds()
            if age > CACHE_TTL_SECONDS:
                return None
        return data
    except Exception:
        return None


def _contract_from_cache(
    cache: Dict, symbol: str, direction: str, spot: float
) -> Optional[Dict]:
    """Try to extract a matching contract from the chain cache."""
    try:
        chains = cache.get("chains", {})
        chain = chains.get(symbol)
        if not chain:
            return None

        right = "C" if direction == "BUY_CALL" else "P"
        contracts = chain.get("contracts", [])
        if not contracts:
            return None

        # Find nearest OTM contract
        otm_pct = 0.02
        target_strike = spot * (1.0 + otm_pct) if right == "C" else spot * (1.0 - otm_pct)

        best = None
        best_dist = float("inf")
        for c in contracts:
            if c.get("right") != right:
                continue
            strike = float(c.get("strike", 0))
            if strike <= 0:
                continue
            dist = abs(strike - target_strike)
            if dist < best_dist:
                best_dist = dist
                best = c

        if best is None:
            return None

        return {
            "strike": float(best["strike"]),
            "expiry": best.get("expiry", ""),
            "right": right,
            "estimated_price": float(best.get("last", best.get("mid", 1.0))),
            "iv": float(best.get("iv", 0.20)),
            "delta": float(best.get("delta", 0.40 if right == "C" else -0.40)),
            "synthetic": False,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def _evaluate_symbol(
    symbol: str,
    bars_1m: List[Any],
    spot: float,
    vix: float,
    chain_cache: Optional[Dict],
    now: datetime,
) -> Optional[TradeSignal]:
    """Evaluate a single symbol for momentum entry."""
    global _OPEN_COUNT

    # Cooldown check
    last = _LAST_SIGNAL.get(symbol)
    if last and (now - last) < COOLDOWN:
        return None

    # Max concurrent check
    if _OPEN_COUNT >= MAX_OPEN_OPTIONS:
        return None

    closes = _closes(bars_1m)
    if len(closes) < 20:
        return None

    volumes = _volumes(bars_1m)

    # --- Trigger 1: Price Momentum ---
    recent = closes[-5:]
    momentum_pct = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0.0

    # EMA slope
    ema_vals = _ema(closes[-EMA_PERIOD:], EMA_PERIOD)
    if len(ema_vals) < 2:
        return None
    ema_slope_positive = ema_vals[-1] > ema_vals[-2]

    # Volume confirmation
    vol_recent = volumes[-20:] if len(volumes) >= 20 else volumes
    avg_vol = sum(vol_recent) / len(vol_recent) if vol_recent else 1.0
    current_vol = volumes[-1] if volumes else 0.0
    vol_mult = current_vol / avg_vol if avg_vol > 0 else 0.0
    volume_confirmed = vol_mult >= VOLUME_MULT_THRESHOLD

    # Direction determination
    direction = None
    if momentum_pct >= MOMENTUM_THRESHOLD and ema_slope_positive and volume_confirmed:
        direction = "BUY_CALL"
    elif momentum_pct <= -MOMENTUM_THRESHOLD and not ema_slope_positive and volume_confirmed:
        direction = "BUY_PUT"

    if direction is None:
        return None

    # --- Trigger 2: IV Timing ---
    regime = _vix_regime(vix)
    if regime == "extreme":
        return None  # VIX > 40, options too expensive

    # IV timing bias
    iv_bias = 0.0
    # Note: without yesterday's VIX we skip the day-over-day comparison

    # --- Contract specification ---
    synthetic = True
    contract = None

    if chain_cache:
        contract = _contract_from_cache(chain_cache, symbol, direction, spot)
        if contract:
            synthetic = False

    if contract is None:
        contract = estimate_contract_price(spot, direction, vix, symbol, dte=7)
        synthetic = True

    # --- Sizing ---
    if regime == "normal":
        base_contracts = 2
    else:  # low or elevated
        base_contracts = 1

    # --- Confidence ---
    confidence = 0.65
    confidence += min(0.20, abs(momentum_pct) * 30.0)
    if vol_mult >= VOLUME_STRONG_THRESHOLD:
        confidence += 0.05
    confidence += iv_bias
    confidence = max(0.60, min(0.90, confidence))

    # Record cooldown and open count
    _LAST_SIGNAL[symbol] = now
    _OPEN_COUNT += 1

    return TradeSignal(
        strategy=StrategyName.OMEGA_MOMENTUM_OPTIONS,
        symbol=symbol,
        side=SignalSide.BUY,
        size=float(base_contracts),
        confidence=round(confidence, 4),
        asset_class=AssetClass.OPTIONS,
        created_at=now,
        meta={
            "engine": "omega_momentum_options.v1",
            "option_right": contract["right"],
            "strike": contract["strike"],
            "expiry": contract["expiry"],
            "estimated_price": contract["estimated_price"],
            "contract_multiplier": 100,
            "synthetic_pricing": synthetic,
            "iv": contract["iv"],
            "delta": contract.get("delta", 0.0),
            "momentum_pct": round(momentum_pct, 6),
            "volume_multiple": round(vol_mult, 2),
            "vix_regime": regime,
            "vix": vix,
            "stop_loss_pct": 0.25,
            "take_profit_pct": 0.50,
            "time_exit_et": "15:45",
            "max_hold_bars": 390,
            "high_convexity": True,
            "sec_type": "OPT",
        },
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def omega_momentum_options_handler(ctx: Any) -> List[TradeSignal]:
    """
    Main strategy handler. Called by the strategy engine each cycle.

    Fail-closed: any exception returns [].
    """
    global _OPEN_COUNT
    try:
        now = _now_utc()

        # Market hours check
        if not _is_market_hours(now):
            LOG.debug("omega_momentum_options: outside market hours")
            return []

        # Get prices
        prices = getattr(ctx, "prices", {}) or {}
        if not prices:
            return []

        # Get VIX
        vix = prices.get("VIX") or prices.get("^VIX")
        if vix is None:
            return []
        vix = float(vix)
        if vix > 40:
            return []

        # Get 1m bars
        bars_1m = getattr(ctx, "bars_1m", {}) or {}

        # Load chain cache (best effort)
        chain_cache = _load_chain_cache()

        signals: List[TradeSignal] = []
        _OPEN_COUNT = 0  # Reset each cycle

        for symbol in UNIVERSE:
            spot = prices.get(symbol)
            if spot is None:
                continue
            spot = float(spot)
            if spot <= 0:
                continue

            sym_bars = bars_1m.get(symbol, [])
            sig = _evaluate_symbol(symbol, sym_bars, spot, vix, chain_cache, now)
            if sig is not None:
                signals.append(sig)

        return signals

    except Exception as exc:
        LOG.error("omega_momentum_options: unhandled error: %s", exc, exc_info=True)
        return []
