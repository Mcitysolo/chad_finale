#!/usr/bin/env python3
"""
chad/strategies/alpha_options.py

ALPHA_OPTIONS — Vertical Spread Options Strategy for CHAD.

Generates defined-risk vertical spreads (bull call / bear put) on SPY
based on directional signals from alpha and gamma strategies.

Signal Flow
-----------
1. Read strong directional signals from context (alpha/gamma outputs).
2. If bullish with high confidence: bull call spread on SPY.
3. If bearish with high confidence: bear put spread on SPY.
4. Use chain_provider for available strikes/expiries (from cache).
5. Use strike_selector to construct SpreadSpec.
6. Emit ONE BAG (combo) TradeSignal per spread — atomic execution.

Instruments
-----------
- SPY only for Phase 6a (most liquid options market).

Risk Model
----------
- Defined risk: max loss = spread width * 100 per contract.
- Position size = floor(equity * max_risk_per_trade_pct / max_loss).
- Max 4 open spreads at any time.
- No naked positions, no undefined risk.

Design
------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic given inputs. No IBKR calls in signal path.
- Uses cached chain data (from chain_provider file cache).
- Fail-closed: missing chain/price -> no signals.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from chad.types import (
    AssetClass,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)

from chad.options.chain_provider import OptionsChain
from chad.options.strike_selector import SpreadSpec, select_vertical_spread


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CHAINS_CACHE_PATH = RUNTIME_DIR / "options_chains_cache.json"


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaOptionsTuning:
    """Tuning parameters for ALPHA_OPTIONS strategy."""
    # Expiry targeting
    target_dte_min: int = 21
    target_dte_max: int = 45

    # Strike selection
    otm_offset_pct: float = 0.02
    spread_width_pct: float = 0.01

    # Risk sizing
    max_risk_per_trade_pct: float = 0.005  # 0.5% of equity per spread
    equity_fallback: float = 100_000.0

    # Confidence gate
    min_confidence: float = 0.70

    # Position limits
    max_open_spreads: int = 4

    # Signal sources
    signal_source_strategies: tuple = ("alpha", "gamma", "gamma_reversion")

    # Universe
    options_universe: tuple = ("SPY",)

    # Max hold — force exit after this many seconds (intraday options)
    max_hold_seconds: int = 3600  # force exit after 1 hour


DEFAULT_TUNING = AlphaOptionsTuning()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _get_mapping(obj: Any, attr: str) -> Mapping[str, Any]:
    try:
        m = getattr(obj, attr, None)
        if isinstance(m, dict):
            return m
        if m is not None and hasattr(m, "get"):
            return m  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _extract_equity(ctx: Any, fallback: float = 100_000.0) -> float:
    """Extract portfolio equity from context."""
    portfolio = getattr(ctx, "portfolio", None)
    if portfolio is not None:
        for field_name in ("total_equity", "equity", "net_liq", "cash"):
            val = _safe_float(getattr(portfolio, field_name, None), 0.0)
            if val > 0:
                return val
        extra = getattr(portfolio, "extra", None)
        if isinstance(extra, dict):
            eq = _safe_float(extra.get("equity"), 0.0)
            if eq > 0:
                return eq
    return fallback


def _extract_price(ctx: Any, symbol: str) -> float:
    """Extract latest price for a symbol from context."""
    # Try ticks
    ticks = _get_mapping(ctx, "ticks")
    tick = ticks.get(symbol)
    if tick is not None:
        price = _safe_float(getattr(tick, "price", None), 0.0)
        if price > 0:
            return price
        if isinstance(tick, dict):
            price = _safe_float(tick.get("price"), 0.0)
            if price > 0:
                return price

    # Try bars (last close)
    bars = _get_mapping(ctx, "bars")
    sym_bars = bars.get(symbol)
    if isinstance(sym_bars, (list, tuple)) and sym_bars:
        last = sym_bars[-1]
        if isinstance(last, dict):
            close = _safe_float(last.get("close"), 0.0)
            if close > 0:
                return close

    # Try prices dict
    prices = _get_mapping(ctx, "prices")
    p = _safe_float(prices.get(symbol), 0.0)
    if p > 0:
        return p

    return 0.0


# ---------------------------------------------------------------------------
# Chain loading from file cache
# ---------------------------------------------------------------------------

def _read_guard_positions() -> Dict[str, Any]:
    """Read raw position-guard state. Returns {} on any failure."""
    try:
        path = RUNTIME_DIR / "position_guard.json"
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _find_opening_bag_fill_for_strategy(
    strategy: str, symbol: str
) -> Optional[Dict[str, Any]]:
    """Return the most recent opening BAG paper fill for strategy+symbol.

    Thin wrapper around the writer-side helper so alpha_options can populate
    its max_hold_exit SELL signal with leg / debit context recovered from
    the daily fills log. Failure-soft — returns None on any I/O or import
    error so the writer's own fallback lookup still runs.
    """
    try:
        from chad.execution.paper_exec_evidence_writer import (
            _find_opening_bag_fill,
        )
    except Exception:
        return None
    try:
        return _find_opening_bag_fill(strategy, symbol)
    except Exception:
        return None


def _load_chain_from_cache(symbol: str) -> Optional[OptionsChain]:
    """
    Load an options chain from the file-based cache.

    The chain_provider daemon writes runtime/options_chains_cache.json.
    This function reads it without requiring an IBKR connection.
    """
    try:
        if not CHAINS_CACHE_PATH.is_file():
            return None
        raw = json.loads(CHAINS_CACHE_PATH.read_text(encoding="utf-8"))
        chains = raw.get("chains", {})
        chain_data = chains.get(symbol)
        if not isinstance(chain_data, dict):
            return None
        return OptionsChain.from_dict(chain_data)
    except Exception as _exc:
        LOG.warning("alpha_options: chain_cache_load_failed exc=%s", _exc)
        return None


# ---------------------------------------------------------------------------
# Directional signal extraction
# ---------------------------------------------------------------------------

def _extract_directional_signal(
    ctx: Any,
    symbol: str,
    tuning: AlphaOptionsTuning,
) -> Optional[Dict[str, Any]]:
    """
    Extract the strongest directional signal for a symbol from upstream strategies.

    Looks for signals from alpha, gamma, gamma_reversion in the context.
    Returns dict with 'direction' ('bullish'/'bearish') and 'confidence',
    or None if no strong signal found.
    """
    # Try strategy_signals (list of TradeSignal-like objects)
    signals = getattr(ctx, "strategy_signals", None)
    if not isinstance(signals, (list, tuple)):
        signals = getattr(ctx, "last_signals", None)
    if not isinstance(signals, (list, tuple)):
        signals = []

    best_signal: Optional[Dict[str, Any]] = None
    best_confidence = 0.0

    for sig in signals:
        # Get strategy name
        strategy = str(getattr(sig, "strategy", getattr(sig, "strategy_name", ""))).lower()
        if not any(src in strategy for src in tuning.signal_source_strategies):
            continue

        # Check symbol match
        sig_symbol = str(getattr(sig, "symbol", "")).strip().upper()
        if sig_symbol != symbol:
            continue

        # Get confidence
        confidence = _safe_float(getattr(sig, "confidence", 0.0), 0.0)
        if confidence < tuning.min_confidence:
            continue

        # Get direction
        side = str(getattr(sig, "side", "")).upper()
        if side == "BUY":
            direction = "bullish"
        elif side == "SELL":
            direction = "bearish"
        else:
            continue

        if confidence > best_confidence:
            best_confidence = confidence
            best_signal = {
                "direction": direction,
                "confidence": confidence,
                "source_strategy": strategy,
                "source_symbol": sig_symbol,
            }

    return best_signal


def _extract_directional_from_bars(
    ctx: Any,
    symbol: str,
    tuning: AlphaOptionsTuning,
) -> Optional[Dict[str, Any]]:
    """
    Derive a directional signal directly from price bars when no upstream
    strategy signals are available. Uses simple EMA cross.
    """
    bars = _get_mapping(ctx, "bars").get(symbol)
    if not isinstance(bars, (list, tuple)) or len(bars) < 30:
        return None

    closes = []
    for b in bars[-50:]:
        if isinstance(b, dict):
            c = _safe_float(b.get("close"), 0.0)
            if c > 0:
                closes.append(c)

    if len(closes) < 30:
        return None

    # EMA(12) vs EMA(26)
    def _ema(values: list, period: int) -> float:
        alpha = 2.0 / (period + 1.0)
        e = values[0]
        for v in values[1:]:
            e = alpha * v + (1.0 - alpha) * e
        return e

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    price = closes[-1]

    # Require clear trend
    if price > ema12 > ema26:
        trend_strength = abs(ema12 - ema26) / price
        conf = min(0.85, 0.65 + trend_strength * 10)
        if conf >= tuning.min_confidence:
            return {
                "direction": "bullish",
                "confidence": round(conf, 4),
                "source_strategy": "alpha_options_ema",
                "source_symbol": symbol,
            }
    elif price < ema12 < ema26:
        trend_strength = abs(ema12 - ema26) / price
        conf = min(0.85, 0.65 + trend_strength * 10)
        if conf >= tuning.min_confidence:
            return {
                "direction": "bearish",
                "confidence": round(conf, 4),
                "source_strategy": "alpha_options_ema",
                "source_symbol": symbol,
            }

    return None


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _build_spread_signals(
    *,
    spread: SpreadSpec,
    confidence: float,
    equity: float,
    tuning: AlphaOptionsTuning,
    now: datetime,
    source_info: Dict[str, Any],
) -> List[TradeSignal]:
    """
    Build ONE BAG (combo) TradeSignal from a SpreadSpec.

    Direction is encoded in the legs — the signal is always BUY the spread.
    Execution adapter resolves both legs atomically via IBKR combo order.
    """
    if spread.max_loss_per_contract <= 0:
        return []

    # Size: floor(risk_budget / max_loss_per_contract)
    risk_budget = equity * tuning.max_risk_per_trade_pct
    contracts = int(risk_budget / spread.max_loss_per_contract)
    if contracts <= 0:
        return []

    spread_id = str(uuid.uuid4())

    # Determine per-leg rights for meta
    if spread.spread_type == "BULL_CALL":
        long_right = "C"
        short_right = "C"
    elif spread.spread_type == "BEAR_PUT":
        long_right = "P"
        short_right = "P"
    else:
        long_right = spread.right
        short_right = spread.right

    signal = TradeSignal(
        strategy=StrategyName.ALPHA_OPTIONS,
        symbol=spread.symbol,
        side=SignalSide.BUY,
        size=float(contracts),
        confidence=float(confidence),
        asset_class=AssetClass.OPTIONS,
        created_at=now,
        meta={
            "engine": "alpha_options.v1",
            "spread_id": spread_id,
            "spread_type": spread.spread_type,
            "expiry": spread.expiry,
            "long_strike": spread.long_strike,
            "short_strike": spread.short_strike,
            "long_right": long_right,
            "short_right": short_right,
            "dte": spread.dte,
            "max_loss_per_contract": spread.max_loss_per_contract,
            "net_debit_estimate": spread.net_debit_estimate,
            "contracts": contracts,
            "required_asset_class": "options",
            "sec_type": "BAG",
            **source_info,
        },
    )

    return [signal]


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------

def build_alpha_options_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else safe default."""
    try:
        from chad.strategies.alpha_options_config import build_alpha_options_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.ALPHA_OPTIONS,
            enabled=True,
            target_universe=["SPY"],
            max_gross_exposure=0.15,
            notes="Vertical spread options engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def build_alpha_options_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    Build ALPHA_OPTIONS signals — vertical spreads on directional signals.

    Flow:
    1. For each symbol in universe, extract directional signal.
    2. If strong signal found, load options chain from cache.
    3. Construct vertical spread via strike_selector.
    4. Emit two linked TradeSignals (long + short leg).
    """
    tuning = DEFAULT_TUNING
    now = getattr(ctx, "now", None)
    if not isinstance(now, datetime):
        now = datetime.now(timezone.utc)

    equity = _extract_equity(ctx, tuning.equity_fallback)

    signals: List[TradeSignal] = []

    # ---- Max-hold exit check ----
    # Wrapped so a guard read failure never blocks normal signal generation.
    forced_exit_symbols: set = set()
    try:
        _now_dt = now if isinstance(now, datetime) else datetime.now(timezone.utc)
        _guard = _read_guard_positions()
        for _key, _entry in _guard.items():
            if not isinstance(_entry, dict):
                continue
            if not _entry.get("open"):
                continue
            if str(_entry.get("strategy", "")) != "alpha_options":
                continue
            _opened_raw = _entry.get("opened_at", "")
            if not _opened_raw:
                continue
            try:
                _opened_dt = datetime.fromisoformat(
                    str(_opened_raw).replace("Z", "+00:00")
                )
                _age_s = (_now_dt - _opened_dt).total_seconds()
            except Exception:
                continue
            if _age_s <= tuning.max_hold_seconds:
                continue
            _symbol = str(_entry.get("symbol", str(_key).split("|")[-1]))
            _qty = _safe_float(_entry.get("quantity", 1.0), 1.0) or 1.0

            # GAP-A001: include opening BAG context so the writer's
            # simulate_bag_paper_fill SELL-close branch can produce a
            # trusted close fill (with original_debit / reversed legs /
            # pnl_breakdown). Lookup is failure-soft — meta absent simply
            # means the writer will fall back to its own disk lookup or
            # mark the close pnl_untrusted.
            _exit_meta: Dict[str, Any] = {
                "engine": "alpha_options.v1",
                "reason": "max_hold_exit",
                "age_seconds": int(_age_s),
                "max_hold_seconds": int(tuning.max_hold_seconds),
                "exit": True,
                "sec_type": "BAG",
                "required_asset_class": "options",
            }
            try:
                _opener = _find_opening_bag_fill_for_strategy(
                    "alpha_options", _symbol
                )
                if isinstance(_opener, dict):
                    _opener_extra = _opener.get("extra") or {}
                    if isinstance(_opener_extra, dict):
                        _legs = _opener_extra.get("bag_legs")
                        if isinstance(_legs, list) and len(_legs) == 2:
                            _exit_meta["bag_legs"] = _legs
                            try:
                                _long_leg = next(
                                    (_l for _l in _legs
                                     if str(_l.get("action", "")).upper() == "BUY"),
                                    _legs[0],
                                )
                                _short_leg = next(
                                    (_l for _l in _legs
                                     if str(_l.get("action", "")).upper() == "SELL"),
                                    _legs[1],
                                )
                                _exit_meta["long_strike"] = _long_leg.get("strike")
                                _exit_meta["short_strike"] = _short_leg.get("strike")
                                _exit_meta["long_right"] = _long_leg.get("right")
                                _exit_meta["short_right"] = _short_leg.get("right")
                                _exit_meta["expiry"] = _long_leg.get("expiry")
                            except Exception:
                                pass
                        _net_debit = _opener_extra.get("net_debit") or _opener_extra.get(
                            "net_debit_estimate"
                        )
                        if _net_debit is not None:
                            _exit_meta["net_debit_estimate"] = _net_debit
                            _exit_meta["net_debit"] = _net_debit
            except Exception as _opener_err:
                LOG.debug(
                    "alpha_options: opener_lookup_failed symbol=%s exc=%s",
                    _symbol, _opener_err,
                )

            LOG.warning(
                "alpha_options_max_hold_exit symbol=%s age=%.0fs "
                "limit=%ds — emitting SELL (opening_debit=%s legs=%s)",
                _symbol, _age_s, tuning.max_hold_seconds,
                _exit_meta.get("net_debit_estimate"),
                "yes" if "bag_legs" in _exit_meta else "no",
            )
            signals.append(TradeSignal(
                strategy=StrategyName.ALPHA_OPTIONS,
                symbol=_symbol,
                side=SignalSide.SELL,
                size=float(_qty),
                confidence=1.0,
                asset_class=AssetClass.OPTIONS,
                created_at=now,
                meta=_exit_meta,
            ))
            forced_exit_symbols.add(_symbol)
    except Exception as _exc:
        LOG.debug("alpha_options: max_hold_check_failed exc=%s", _exc)

    for symbol in tuning.options_universe:
        sym = symbol.strip().upper()
        if sym in forced_exit_symbols:
            # Skip new BUY this cycle — exit signal already emitted.
            continue
        price = _extract_price(ctx, sym)
        if price <= 0:
            continue

        # Get directional signal
        dir_signal = _extract_directional_signal(ctx, sym, tuning)
        if dir_signal is None:
            dir_signal = _extract_directional_from_bars(ctx, sym, tuning)
        if dir_signal is None:
            LOG.debug(
                "alpha_options: sym=%s no directional signal "
                "(strategy_signals empty or EMA confidence < %.2f)",
                sym, tuning.min_confidence,
            )
            continue

        # Load chain from file cache
        chain = _load_chain_from_cache(sym)
        if chain is None:
            continue

        # Build spread
        spread = select_vertical_spread(
            chain=chain,
            current_price=price,
            direction=dir_signal["direction"],
            target_dte_min=tuning.target_dte_min,
            target_dte_max=tuning.target_dte_max,
            otm_offset_pct=tuning.otm_offset_pct,
            spread_width_pct=tuning.spread_width_pct,
        )
        if spread is None:
            continue

        # Build signals
        spread_signals = _build_spread_signals(
            spread=spread,
            confidence=dir_signal["confidence"],
            equity=equity,
            tuning=tuning,
            now=now,
            source_info={
                "source_strategy": dir_signal.get("source_strategy", ""),
                "source_direction": dir_signal.get("direction", ""),
            },
        )
        signals.extend(spread_signals)

    if not signals:
        LOG.info(
            "alpha_options: zero signals — "
            "chain_miss or no_directional or spread_sizing_fail "
            "(regime=%s)",
            getattr(ctx, "regime", "?"),
        )
    return signals


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def alpha_options_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """StrategyEngine-compatible handler for ALPHA_OPTIONS. Fail-closed."""
    try:
        cfg = build_alpha_options_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_alpha_options_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "AlphaOptionsTuning",
    "build_alpha_options_config",
    "build_alpha_options_signals",
    "alpha_options_handler",
]
