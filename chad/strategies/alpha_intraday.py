#!/usr/bin/env python3
"""
CHAD — AlphaIntraday (Delta high-convexity day trading brain)

14th strategy. Event-driven asymmetric payoff engine operating on 1-minute
bars with daily bar fallback. Three entry triggers (any one sufficient):
  1) Volatility explosion: current ATR >= 2x baseline ATR
  2) Momentum surge: accelerating 5-bar momentum aligned with 20-bar trend
  3) Mean reversion snap: Bollinger penetration + RSI extreme (equities only)

All signals tagged high_convexity=True with 1.5% stop / 4.5% target
(3:1 asymmetric reward-to-risk) and max_hold_bars=30.

Fail-closed: any exception or missing data returns [].
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from chad.types import (
    AssetClass,
    MarketContext,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)
from chad.utils.catalyst_gate import check_catalyst_gate
from chad.utils.liquidity import LiquidityClass, blocks_thin_entry
from chad.utils.risk_reward import passes_rr_gate
from chad.utils.rs_gate import RSGateResult, get_rs_adjustment
from chad.utils.session import session_decision

LOG = logging.getLogger(__name__)

UNIVERSE: List[str] = [
    "SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "BAC",
    "MES", "MNQ", "BTC-USD",
]

FUTURES = {"MES", "MNQ"}
CRYPTO = {"BTC-USD", "BTC", "ETH-USD", "ETH"}
EQUITIES_FOR_REVERSION = {"SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "BAC"}

COOLDOWN = timedelta(minutes=10)
_LAST_SIGNAL: Dict[str, datetime] = {}


def build_alpha_intraday_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.ALPHA_INTRADAY,
        enabled=True,
        target_universe=None,
        max_gross_exposure=None,
        notes="alpha_intraday: Delta high-convexity day trading brain",
    )


def _asset_class(sym: str) -> AssetClass:
    if sym in FUTURES:
        return AssetClass.FUTURES
    if sym in CRYPTO or sym.endswith("-USD"):
        return AssetClass.CRYPTO
    return AssetClass.EQUITY


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


def _highs_lows_closes(bars: Sequence[Any]):
    h, l, c = [], [], []
    for b in bars:
        try:
            if isinstance(b, dict):
                hi = float(b.get("high", 0.0))
                lo = float(b.get("low", 0.0))
                cl = float(b.get("close", 0.0))
            else:
                hi = float(getattr(b, "high", 0.0))
                lo = float(getattr(b, "low", 0.0))
                cl = float(getattr(b, "close", 0.0))
            if hi > 0 and lo > 0 and cl > 0:
                h.append(hi); l.append(lo); c.append(cl)
        except Exception:
            continue
    return h, l, c


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    try:
        if len(closes) < period + 1:
            return None
        trs: List[float] = []
        for i in range(len(closes) - period, len(closes)):
            if i == 0:
                continue
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)
    except Exception:
        return None


def _atr_window(highs, lows, closes, start_from_end: int, length: int) -> Optional[float]:
    # Compute ATR on window ending start_from_end bars from the end, of length `length`.
    try:
        n = len(closes)
        end_idx = n - start_from_end
        start_idx = end_idx - length
        if start_idx < 1:
            return None
        trs: List[float] = []
        for i in range(start_idx, end_idx):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return None
        return sum(trs) / len(trs)
    except Exception:
        return None


def _sma(values: List[float], period: int) -> Optional[float]:
    try:
        if len(values) < period:
            return None
        return sum(values[-period:]) / period
    except Exception:
        return None


def _std(values: List[float], period: int) -> Optional[float]:
    try:
        if len(values) < period:
            return None
        window = values[-period:]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        return var ** 0.5
    except Exception:
        return None


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    try:
        if len(closes) < period + 1:
            return None
        gains, losses = 0.0, 0.0
        for i in range(len(closes) - period, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains += diff
            else:
                losses -= diff
        if losses == 0:
            return 100.0
        rs = (gains / period) / (losses / period)
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return None


def _size_for(
    sym: str,
    confidence: float,
    *,
    atr: float = 0.0,
    tier_max_risk_usd: Optional[float] = None,
) -> float:
    if sym in ("MES", "MNQ") and atr > 0.0 and tier_max_risk_usd is not None:
        from chad.risk.futures_position_sizer import FUTURES_SPECS
        _spec = FUTURES_SPECS.get(sym)
        if _spec is not None:
            _stop_pts = atr * 2.0
            _risk_per = _stop_pts * _spec.point_value
            if _risk_per > 0:
                _contracts = int(float(tier_max_risk_usd) // _risk_per)
                if _contracts <= 0:
                    return 0.0
                return float(min(_contracts, 10))
    base = 5.0
    if sym == "MES":
        return min(2.0, max(1.0, round(base * confidence * 0.4)))
    if sym == "MNQ":
        return 1.0
    if _asset_class(sym) == AssetClass.CRYPTO:
        return min(0.1, max(0.01, round(base * confidence * 0.02, 3)))
    return min(15.0, max(1.0, round(base * confidence * 2.0, 2)))


def _cooldown_ok(sym: str, now: datetime) -> bool:
    last = _LAST_SIGNAL.get(sym)
    if last is None:
        return True
    return (now - last) >= COOLDOWN


def _build_signal(
    sym: str,
    side: SignalSide,
    confidence: float,
    trigger: str,
    timeframe: str,
    *,
    atr: float = 0.0,
    tier_max_risk_usd: Optional[float] = None,
    session_window: Optional[str] = None,
    primary_session_only: Optional[bool] = None,
) -> Optional[TradeSignal]:
    confidence = max(0.50, min(0.95, float(confidence)))
    size = _size_for(sym, confidence, atr=atr, tier_max_risk_usd=tier_max_risk_usd)
    if size <= 0.0:
        return None
    # Pre-entry R:R gate (entry-only, fail-open). Uses the strategy's fixed
    # stop_loss_pct / take_profit_pct as point-equivalents. With the current
    # 4.5 / 1.5 defaults this gate never blocks in normal operation — it is a
    # misconfiguration guard and future extension point.
    _stop_pts_pct = 1.5
    _target_pts_pct = 4.5
    if not passes_rr_gate(_target_pts_pct, _stop_pts_pct):
        return None
    # Pre-entry liquidity gate (entry-only, EQUITY/ETF, fail-open). Futures
    # (MES/MNQ) and crypto (*-USD) are excluded by AssetClass and pass with
    # liquidity_class=UNKNOWN. UNKNOWN classifications never block.
    _asset_cls_liq = _asset_class(sym)
    if _asset_cls_liq in (AssetClass.EQUITY, AssetClass.ETF):
        _liq_blocked, _liq_class, _liq_required_conf = blocks_thin_entry(
            sym,
            float(confidence),
        )
        if _liq_blocked:
            return None
    else:
        _liq_class = LiquidityClass.UNKNOWN
        _liq_required_conf = float(confidence)
    # Pre-entry catalyst gate (entry-only, EQUITY/ETF, fail-open). Futures
    # and crypto are excluded by AssetClass; missing/stale news_intel.json
    # yields an allowed=True unknown result.
    if _asset_cls_liq in (AssetClass.EQUITY, AssetClass.ETF):
        _cat = check_catalyst_gate(sym, side.value)
        if not _cat.allowed:
            return None
    else:
        _cat = None
    # Pre-entry relative-strength confidence modifier (entry-only,
    # EQUITY/ETF, fail-open). Futures (MES/MNQ) and crypto (*-USD) are
    # excluded by AssetClass. Missing/stale relative_strength.json yields
    # adjustment=0.0. Never hard-blocks.
    if _asset_cls_liq in (AssetClass.EQUITY, AssetClass.ETF):
        _rs_adj = get_rs_adjustment(sym, side.value)
        confidence = max(
            0.50,
            min(0.95, float(confidence) + _rs_adj.confidence_adjustment),
        )
    else:
        _rs_adj = RSGateResult(0.0, "unknown", None, None, "unknown")
    _point_value = {"MES": 5.0, "MNQ": 2.0}.get(sym, 0.0)
    _stop_pts = atr * 2.0 if atr > 0 else 0.0
    meta: Dict[str, Any] = {
        "high_convexity": True,
        "stop_loss_pct": _stop_pts_pct,
        "take_profit_pct": _target_pts_pct,
        "trigger": trigger,
        "setup_family": trigger,
        "max_hold_bars": 30,
        "timeframe": timeframe,
        "stop_distance_pts": round(_stop_pts, 6),
        "stop_distance_usd": round(_stop_pts * _point_value, 6),
        "tier_max_risk_usd": tier_max_risk_usd,
        "rr_ratio": round(_target_pts_pct / _stop_pts_pct, 4),
        "rr_gate": "PASSED",
        "liquidity_class": _liq_class.value,
        "liquidity_required_confidence": round(float(_liq_required_conf), 6),
        "catalyst_strength": (
            _cat.catalyst_strength if _cat is not None else "unknown"
        ),
        "catalyst_direction": (
            _cat.catalyst_direction if _cat is not None else "unknown"
        ),
        "catalyst_gate": (
            "PASSED" if _cat is None or _cat.allowed else "BLOCKED"
        ),
        "rs_class": _rs_adj.rs_class,
        "rs_vs_spy": _rs_adj.rs_vs_spy,
        "rs_excess_vs_spy_5d": _rs_adj.excess_vs_spy_5d,
        "rs_market_direction": _rs_adj.market_direction,
        "rs_confidence_adjustment": round(float(_rs_adj.confidence_adjustment), 6),
    }
    if primary_session_only is not None:
        meta["session_window"] = session_window
        meta["session_gate"] = "PASSED"
        meta["primary_session_only"] = bool(primary_session_only)
    return TradeSignal(
        strategy=StrategyName.ALPHA_INTRADAY,
        symbol=sym,
        side=side,
        size=size,
        confidence=confidence,
        asset_class=_asset_class(sym),
        meta=meta,
    )


def _evaluate_symbol(
    sym: str,
    bars: Sequence[Any],
    timeframe: str,
    *,
    tier_max_risk_usd: Optional[float] = None,
    session_window: Optional[str] = None,
    primary_session_only: Optional[bool] = None,
) -> Optional[TradeSignal]:
    highs, lows, closes = _highs_lows_closes(bars)
    if len(closes) < 45:
        return None

    # ATR for stop-distance sizing — also reused by the vol_explosion trigger.
    atr_for_sizing = _atr_window(highs, lows, closes, start_from_end=0, length=14) or 0.0

    # Scale parameters by timeframe
    if timeframe == "daily_fallback":
        vol_mult = 1.5
        mom_period = 3
        bb_pen = 0.015  # 1.5%
    else:
        vol_mult = 2.0
        mom_period = 5
        bb_pen = 0.003  # 0.3%

    # TRIGGER 1: Volatility explosion
    try:
        current_atr = _atr_window(highs, lows, closes, start_from_end=0, length=14)
        baseline_atr = _atr_window(highs, lows, closes, start_from_end=14, length=30)
        if current_atr and baseline_atr and baseline_atr > 0:
            ratio = current_atr / baseline_atr
            if ratio >= vol_mult:
                if closes[-1] > closes[-3]:
                    side = SignalSide.BUY
                elif closes[-1] < closes[-3]:
                    side = SignalSide.SELL
                else:
                    side = None
                if side is not None:
                    conf = min(0.95, 0.60 + (ratio - vol_mult) * 0.10)
                    return _build_signal(
                        sym, side, conf, "vol_explosion", timeframe,
                        atr=atr_for_sizing, tier_max_risk_usd=tier_max_risk_usd,
                        session_window=session_window,
                        primary_session_only=primary_session_only,
                    )
    except Exception:
        pass

    # TRIGGER 2: Momentum surge
    try:
        if len(closes) > max(mom_period, 20) and closes[-mom_period] > 0 and closes[-20] > 0:
            m_short = (closes[-1] - closes[-mom_period]) / closes[-mom_period]
            m_long = (closes[-1] - closes[-20]) / closes[-20]
            if (
                abs(m_short) >= 0.005
                and (m_short * m_long) > 0
                and abs(m_short) >= 1.5 * abs(m_long)
            ):
                side = SignalSide.BUY if m_short > 0 else SignalSide.SELL
                conf = min(0.90, 0.55 + abs(m_short) * 20.0)
                return _build_signal(
                    sym, side, conf, "momentum_surge", timeframe,
                    atr=atr_for_sizing, tier_max_risk_usd=tier_max_risk_usd,
                    session_window=session_window,
                    primary_session_only=primary_session_only,
                )
    except Exception:
        pass

    # TRIGGER 3: Mean reversion snap (equities only)
    try:
        if sym in EQUITIES_FOR_REVERSION:
            sma20 = _sma(closes, 20)
            std20 = _std(closes, 20)
            rsi = _rsi(closes, 14)
            if sma20 and std20 and rsi is not None and std20 > 0:
                upper = sma20 + 2.0 * std20
                lower = sma20 - 2.0 * std20
                px = closes[-1]
                if px < lower * (1.0 - bb_pen) and rsi < 25:
                    dist = abs(px - lower) / std20
                    conf = min(0.85, 0.58 + dist * 0.05)
                    return _build_signal(
                        sym, SignalSide.BUY, conf, "mean_reversion_snap", timeframe,
                        atr=atr_for_sizing, tier_max_risk_usd=tier_max_risk_usd,
                        session_window=session_window,
                        primary_session_only=primary_session_only,
                    )
                if px > upper * (1.0 + bb_pen) and rsi > 75:
                    dist = abs(px - upper) / std20
                    conf = min(0.85, 0.58 + dist * 0.05)
                    return _build_signal(
                        sym, SignalSide.SELL, conf, "mean_reversion_snap", timeframe,
                        atr=atr_for_sizing, tier_max_risk_usd=tier_max_risk_usd,
                        session_window=session_window,
                        primary_session_only=primary_session_only,
                    )
    except Exception:
        pass

    return None


def alpha_intraday_handler(ctx: MarketContext, *_args, **_kwargs) -> List[TradeSignal]:
    signals: List[TradeSignal] = []
    try:
        now = datetime.now(timezone.utc)
        bars_1m = getattr(ctx, "bars_1m", {}) or {}
        bars_daily = getattr(ctx, "bars", {}) or {}
        prices = getattr(ctx, "prices", {}) or {}
        _tier_profile = getattr(ctx, "tier_profile", None)
        tier_max_risk_usd = getattr(
            _tier_profile,
            "max_risk_per_trade_usd",
            None,
        )

        # Tier-aware session gate (entry-only, fail-open). When tier_profile
        # is absent the strategy preserves its pre-existing behavior.
        _primary_only = getattr(_tier_profile, "primary_session_only", None)
        _session_window: Optional[str] = None
        if _primary_only is not None:
            try:
                _decision = session_decision(
                    getattr(ctx, "now", None),
                    primary_session_only=bool(_primary_only),
                )
            except Exception as _exc:
                LOG.warning("alpha_intraday: session_decision failed=%s — fail-open", _exc)
                _decision = None
            if _decision is not None:
                _session_window = _decision.session_window
                if not _decision.entry_allowed:
                    LOG.info(
                        "alpha_intraday: session_gate_blocked reason=%s "
                        "primary_session_only=%s window=%s",
                        _decision.skip_reason, bool(_primary_only), _session_window,
                    )
                    return []

        for sym in UNIVERSE:
            try:
                # Skip zero price
                px = 0.0
                try:
                    px = float(prices.get(sym, 0.0) or 0.0)
                except Exception:
                    px = 0.0
                if px == 0.0 and sym not in bars_1m and sym not in bars_daily:
                    continue

                if not _cooldown_ok(sym, now):
                    continue

                timeframe = "1m"
                bars = bars_1m.get(sym)
                if not bars:
                    bars = bars_daily.get(sym)
                    timeframe = "daily_fallback"
                if not bars:
                    continue

                sig = _evaluate_symbol(
                    sym, bars, timeframe,
                    tier_max_risk_usd=tier_max_risk_usd,
                    session_window=_session_window,
                    primary_session_only=(
                        bool(_primary_only) if _primary_only is not None else None
                    ),
                )
                if sig is not None:
                    signals.append(sig)
                    _LAST_SIGNAL[sym] = now
            except Exception as _exc:
                LOG.warning("alpha_intraday: symbol=%s exception=%s", sym, _exc)
                continue
    except Exception:
        return []
    if not signals:
        LOG.info(
            "alpha_intraday: zero signals this cycle — "
            "conditions not met on all %d universe symbols "
            "(vol_mult/momentum/BB+RSI gates; regime=%s vix=%.2f)",
            len(UNIVERSE), getattr(ctx, "regime", "?"),
            getattr(ctx, "vix", 0.0),
        )
    return signals
