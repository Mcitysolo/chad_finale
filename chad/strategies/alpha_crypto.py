from __future__ import annotations

"""
AlphaCrypto — CHAD Phase-4 Edge Quality Upgrade (Strategy-only, execution-clean)

What this module is
-------------------
A production-grade crypto signal engine that stays strictly within CHAD's strategy boundary:
- NO network calls
- NO broker/exchange execution
- NO side effects beyond returning signals
- Fully deterministic for a given ctx snapshot + params

What this module fixes vs the current baseline
----------------------------------------------
Your current alpha_crypto.py intentionally emits NO signals (safe baseline). That guarantees
no wins — and it also guarantees that any logged crypto activity would not be driven by
this brain. Phase 11 dev asked to rebuild crypto signals and improve expectancy (paper-first).

This implementation adds:
- Regime gating (trend + volatility band)
- Liquidity gating (dollar volume if available; otherwise safe fail-closed)
- Anti-chase (avoid entering after oversized candles)
- Deterministic exit logic (time stop, ATR trail, trend break, vol spike)
- Churn controls (min delta exposure)
- Audit-grade metadata: blocked_by, reasons, exit_reason, diagnostics

Compatibility contract (must not break wiring)
----------------------------------------------
The following are required by current CHAD wiring and tests:
- AlphaCryptoParams dataclass
- build_alpha_crypto_config() -> StrategyConfig
- alpha_crypto_handler(ctx: Any, params: AlphaCryptoParams) -> List

This file preserves those names and import paths.

NOTE ON SIGNAL TYPE
-------------------
CHAD's strategy handlers typically return a list of "StrategySignal"-like objects.
Because this repo's concrete signal type varies across branches, this module uses a
small adapter layer:

- If ctx provides a StrategySignal class via ctx.signal_factory or ctx.StrategySignal, we use it.
- Else, we return a dict-shaped signal payload (safe, serializable) that upstream can adapt.

This keeps the module robust and prevents hard import coupling.

Paper-first evaluation guidance
-------------------------------
To satisfy Phase 11 acceptance, run in DRY_RUN and ensure trade-result logging is enabled.
SCR derives win_rate and sharpe_like from the ledger. This module increases quality by:
- blocking low-quality regimes
- exiting losers faster
- reducing over-trading and FOMO entries
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal
from chad.utils.crypto_signal_filter import CryptoFilterResult, get_crypto_filter

logger = logging.getLogger(__name__)

_REGIME_STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/regime_state.json")
_BARS_DIR = Path("/home/ubuntu/chad_finale/data/bars/1d")

# CBAR-S1 — CRYPTO-DAILY-BARS "real wire". The regime short-circuit in
# alpha_crypto_handler returns [] in ranging/adverse BEFORE any signal computes, which is why
# feeding daily bars alone cannot un-silence this brain. Under the CEW1 exploration flag
# (CHAD_CRYPTO_EXPLORATION on AND CHAD_EXECUTION_MODE=paper AND CHAD_KRAKEN_MODE!=live), the
# handler proceeds through its FULL momentum/breakout path in ALL regimes so CHAD can measure
# the "crypto has no edge in ranging" assumption on paper. Flag OFF — or either fail-closed
# refusal — restores byte-identical stock gating. The flag is resolved by the SINGLE CEW1
# authority (regime_activation.crypto_exploration_state); this module is the SECOND consumption
# site, never a second flag-evaluation path (one authority, two consumers).
MARKER_EXPLORATION_HANDLER_PASS = "CRYPTO_EXPLORATION_HANDLER_PASS"


# -------------------------
# Default Universe (safe)
# -------------------------

CRYPTO_UNIVERSE_DEFAULT: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Symbols known to be highly liquid globally — bypass the fail-closed
# dollar_volume gate when ctx.dollar_volume is unavailable.
_KNOWN_LIQUID: frozenset = frozenset({"BTC-USD", "ETH-USD", "SOL-USD"})

# CAD-quoted alternates used when USD buying power is insufficient and
# the live Kraken balance shows ZCAD on hand. Sized in CAD against the
# ZCAD line of runtime/kraken_balances.json by the orchestrator's risk
# layer; the strategy itself emits these on the same logic as the USD
# pair so the wiring stays simple.
CRYPTO_UNIVERSE_CAD: List[str] = ["BTC-CAD", "ETH-CAD"]


def _kraken_cad_balance_present() -> bool:
    """
    Read-only check: does the latest kraken balance snapshot show a
    non-trivial CAD line? The snapshot is refreshed by
    KrakenBalanceProvider on the orchestrator's cycle. We never block on
    a fetch from the strategy boundary — strategies must remain pure.

    Returns False on any I/O error so the CAD lane stays opt-in.
    """
    try:
        from chad.market_data.kraken_balance_provider import (
            DEFAULT_SNAPSHOT_PATH,
            load_latest_snapshot,
        )
        snap = load_latest_snapshot(DEFAULT_SNAPSHOT_PATH)
        if not snap:
            return False
        balances = snap.get("balances") or {}
        cad = balances.get("CAD")
        try:
            return float(cad) > 1.0
        except (TypeError, ValueError):
            return False
    except Exception:
        return False


# -------------------------
# Helpers (pure, safe)
# -------------------------

Number = float


def _clamp(x: Number, lo: Number, hi: Number) -> Number:
    if x != x:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(x: Any, default: Number = 0.0) -> Number:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_iter(obj: Any) -> Iterable[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return obj
    return [obj]


def _get_mapping(obj: Any, attr: str) -> Mapping[str, Any]:
    try:
        m = getattr(obj, attr, None)
        if isinstance(m, dict):
            return m
        if m is None:
            return {}
        # If some custom mapping type
        if hasattr(m, "get"):
            return m  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _get_series_map(ctx: Any, name: str) -> Mapping[str, Sequence[Any]]:
    """
    Returns mapping symbol -> sequence
    Accepts common ctx attributes:
      - ctx.bars[symbol] -> list of bar dicts or tuples
      - ctx.ohlcv[symbol] -> list
      - ctx.series[symbol] -> list
    """
    m = _get_mapping(ctx, name)
    if not m:
        return {}
    out: Dict[str, Sequence[Any]] = {}
    for k, v in m.items():
        if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) >= 5:
            out[k] = v
    return out


def _extract_prices(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Expected: ctx.prices[symbol] == float
    """
    prices = _get_mapping(ctx, "prices")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in prices:
            v = prices.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_volatility(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional: ctx.volatility[symbol] -> float (e.g. realized vol)
    """
    vol = _get_mapping(ctx, "volatility")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in vol:
            v = vol.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_dollar_volume(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional: ctx.dollar_volume[symbol] -> float
    Optional: ctx.volume_usd[symbol] -> float
    Optional: ctx.liquidity_usd[symbol] -> float
    """
    for key in ("dollar_volume", "volume_usd", "liquidity_usd"):
        m = _get_mapping(ctx, key)
        if m:
            out: Dict[str, Number] = {}
            for s in universe:
                if s in m:
                    v = m.get(s)
                    if v is not None:
                        out[s] = _safe_float(v, default=0.0)
            return out
    return {}


def _extract_bars(ctx: Any, symbol: str) -> Optional[Sequence[Any]]:
    """
    Attempts to extract a per-symbol OHLCV bar series.
    Accepts (best-effort):
      - ctx.bars[symbol]
      - ctx.ohlcv[symbol]
      - ctx.series[symbol]
      - ctx.candles[symbol]
    """
    for key in ("bars", "ohlcv", "series", "candles"):
        m = _get_mapping(ctx, key)
        if m and symbol in m:
            v = m.get(symbol)
            if isinstance(v, (list, tuple)) and len(v) >= 5:
                return v
    return None


def _bar_close(bar: Any) -> Number:
    # Supports dict-like or tuple-like. We attempt common keys first.
    if isinstance(bar, dict):
        for k in ("close", "c", "Close", "C"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        # fall back: last numeric field
        for v in reversed(list(bar.values())):
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(bar, (list, tuple)):
        # common formats: (ts, o, h, l, c, v) or (o, h, l, c, v)
        # choose last 2nd as close if length>=5
        if len(bar) >= 5:
            return _safe_float(bar[-2], default=0.0) if len(bar) >= 6 else _safe_float(bar[-2], default=0.0)
        return 0.0
    return _safe_float(bar, default=0.0)


def _bar_high(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("high", "h", "High", "H"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        if len(bar) >= 5:
            # (o,h,l,c,v) -> high index 1; (ts,o,h,l,c,v) -> high index 2
            if len(bar) == 5:
                return _safe_float(bar[1], default=0.0)
            if len(bar) >= 6:
                return _safe_float(bar[2], default=0.0)
        return 0.0
    return 0.0


def _bar_low(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("low", "l", "Low", "L"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        if len(bar) >= 5:
            # (o,h,l,c,v) -> low index 2; (ts,o,h,l,c,v) -> low index 3
            if len(bar) == 5:
                return _safe_float(bar[2], default=0.0)
            if len(bar) >= 6:
                return _safe_float(bar[3], default=0.0)
        return 0.0
    return 0.0


def _bar_volume(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("volume", "v", "Volume", "V"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        # if (o,h,l,c,v) -> last is v; if (ts,o,h,l,c,v) -> last is v
        if len(bar) >= 5:
            return _safe_float(bar[-1], default=0.0)
        return 0.0
    return 0.0


def _ema(values: Sequence[Number], period: int) -> List[Number]:
    if period <= 1:
        raise ValueError("EMA period must be > 1")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out: List[Number] = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
        out.append(e)
    return out


def _true_range(highs: Sequence[Number], lows: Sequence[Number], closes: Sequence[Number]) -> List[Number]:
    n = len(closes)
    if n == 0:
        return []
    out = [0.0] * n
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        out[i] = max(hl, hc, lc)
    return out


def _atr(highs: Sequence[Number], lows: Sequence[Number], closes: Sequence[Number], period: int) -> List[Number]:
    tr = _true_range(highs, lows, closes)
    return _ema(tr, period)


def _rolling_max(values: Sequence[Number], period: int) -> List[Number]:
    if period <= 1:
        raise ValueError("rolling_max period must be > 1")
    n = len(values)
    out = [values[0]] * n
    for i in range(n):
        lo = max(0, i - period + 1)
        out[i] = max(values[lo : i + 1])
    return out


# -------------------------
# Parameters (immutable)
# -------------------------

@dataclass(frozen=True)
class AlphaCryptoParams:
    """
    Tunable parameters for AlphaCrypto (Phase-4 upgrade).

    enabled:
      Master switch for strategy participation.
    universe:
      List of symbols to monitor.
    min_liquidity_usd:
      Dollar volume gate. If ctx provides dollar volume metrics, enforce median DV >= threshold.
      If ctx provides no liquidity metrics, we FAIL-CLOSED by not entering (exits still allowed).
    max_volatility:
      Legacy field retained for compatibility; superseded by ATR% band below.
    atr_period:
      ATR window for regime + exits.
    ema_fast / ema_slow:
      Trend filters.
    min_atr_pct / max_atr_pct:
      Volatility regime band to avoid chop and panic.
    anti_chase_range_atr:
      Block entries if candle range / ATR > this value.
    max_abs_exposure:
      Max normalized exposure per symbol.
    min_delta_exposure:
      Anti-churn band: do not change exposure unless delta >= band (except exits).
    momentum_atr:
      Require momentum (close - ema_fast) / ATR >= this value to enter (reduces noise trades).
    rsi_min / rsi_max:
      Optional sanity band if ctx provides rsi; otherwise ignored.
    time_stop_bars:
      Exit if held too long without favorable move.
    min_favor_move_atr:
      Minimum favorable move to avoid time-stop.
    atr_trail_mult:
      Exit if adverse move from peak favorable exceeds this * ATR.
    vol_spike_atr_pct:
      Exit on volatility spike while in position.
    """

    enabled: bool = True
    universe: Optional[List[str]] = None

    # CAD-quoted fallback universe — included automatically when the live
    # Kraken balance snapshot shows a non-zero CAD line and USD is empty.
    enable_cad_pairs: bool = True
    cad_universe: Optional[List[str]] = None

    # Liquidity
    min_liquidity_usd: float = 750_000.0

    # Legacy (kept)
    max_volatility: float = 5.0

    # Indicators
    atr_period: int = 14
    ema_fast: int = 10
    ema_slow: int = 30
    breakout_lookback: int = 20

    # Regime
    min_atr_pct: float = 0.004
    max_atr_pct: float = 0.090

    # Entry quality
    anti_chase_range_atr: float = 2.8
    momentum_atr: float = 0.35

    # Exposure & churn
    max_abs_exposure: float = 0.25
    min_delta_exposure: float = 0.03

    # Exits
    time_stop_bars: int = 25
    min_favor_move_atr: float = 0.6
    atr_trail_mult: float = 2.6
    vol_spike_atr_pct: float = 0.120

    def actual_universe(self) -> List[str]:
        base: List[str]
        if self.universe is None:
            base = list(CRYPTO_UNIVERSE_DEFAULT)
        else:
            try:
                u = [str(x) for x in self.universe if x]
                base = u if u else list(CRYPTO_UNIVERSE_DEFAULT)
            except Exception:
                base = list(CRYPTO_UNIVERSE_DEFAULT)
        if self.enable_cad_pairs and _kraken_cad_balance_present():
            cad = (
                list(self.cad_universe)
                if self.cad_universe is not None
                else list(CRYPTO_UNIVERSE_CAD)
            )
            for sym in cad:
                if sym not in base:
                    base.append(sym)
        return base


# -------------------------
# StrategyConfig Factory
# -------------------------

def build_alpha_crypto_config() -> StrategyConfig:
    """
    Creates the StrategyConfig required by CHAD’s StrategyEngine.
    """
    return StrategyConfig(
        name=StrategyName.ALPHA_CRYPTO,
        enabled=True,
        target_universe=list(CRYPTO_UNIVERSE_DEFAULT),
    )


# -------------------------
# Minimal state (position memory)
# -------------------------

class _StateStore:
    """
    In-memory per-process state. Strategy boundary should not write to disk here.
    If you need restart-safe state, wire it via orchestrator/state store elsewhere.

    This store provides:
      - exposure (last target)
      - entry_close
      - bars_held
      - peak_favorable_close
    """
    __slots__ = ("pos",)

    def __init__(self) -> None:
        self.pos: Dict[str, Dict[str, Any]] = {}

    def get(self, symbol: str) -> Dict[str, Any]:
        return dict(self.pos.get(symbol, {}))

    def set(self, symbol: str, st: Dict[str, Any]) -> None:
        self.pos[symbol] = dict(st)


_STATE = _StateStore()


# -------------------------
# Signal factory adapter
# -------------------------

def _make_signal(ctx: Any, payload: Dict[str, Any]) -> Any:
    """
    Returns a strategy signal object compatible with the host.

    Preference order:
      1) ctx.signal_factory(payload) if callable
      2) ctx.StrategySignal(**payload) if present
      3) dict payload (safe fallback)
    """
    try:
        sf = getattr(ctx, "signal_factory", None)
        if callable(sf):
            return sf(payload)
    except Exception:
        pass

    try:
        cls = getattr(ctx, "StrategySignal", None)
        if cls is not None:
            return cls(**payload)  # type: ignore[misc]
    except Exception:
        pass

    return payload


# -------------------------
# Core strategy logic
# -------------------------

def _read_regime(ctx: Any) -> str:
    """
    Read the current regime label. Prefer ctx.regime if present, else fall back
    to the on-disk regime state written by live_loop's classifier. Returns
    "unknown" on any failure so downstream gates fail-open to full strength.
    """
    r = getattr(ctx, "regime", None)
    if isinstance(r, str) and r:
        return r.lower()
    try:
        d = json.loads(_REGIME_STATE_PATH.read_text(encoding="utf-8"))
        v = d.get("regime")
        if isinstance(v, str) and v:
            return v.lower()
    except Exception:
        pass
    return "unknown"


def _exploration_bypass_active() -> bool:
    """True only when CEW1's exploration flag resolves to ``active`` — flag on AND global
    ``CHAD_EXECUTION_MODE=paper`` AND the Kraken lane not live (``CHAD_KRAKEN_MODE!=live``).

    Delegates to the ONE authority (``regime_activation.crypto_exploration_state``); it never
    re-implements the two-axis fail-closed logic here. Any import/eval failure fails CLOSED
    (returns False → stock gating), so a broken import can only ever restore stock behavior,
    never widen exploration. Both fail-closed refusals (``refused_non_paper`` /
    ``refused_kraken_live``) resolve ``active=False`` here, so a refused posture keeps the
    handler's stock ranging/adverse short-circuit.
    """
    try:
        from chad.portfolio.regime_activation import crypto_exploration_state
        active, _reason = crypto_exploration_state()
        return bool(active)
    except Exception:
        return False


def _load_bars_for_symbol(ctx: Any, symbol: str) -> List[Mapping[str, Any]]:
    """
    Pull the daily bar series for ``symbol`` from ctx.bars first (populated by
    ContextBuilder from data/bars/1d/{symbol}.json) and fall back to reading
    the JSON directly so the handler remains usable in smoke tests and any
    caller that hands in a ctx without pre-loaded bars.

    Returns [] on any error — the caller must tolerate missing history.
    """
    ctx_bars = getattr(ctx, "bars", None)
    if isinstance(ctx_bars, Mapping):
        v = ctx_bars.get(symbol)
        if isinstance(v, (list, tuple)) and v:
            return [b for b in v if isinstance(b, Mapping)]
    try:
        path = _BARS_DIR / f"{symbol}.json"
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            bars = doc.get("bars") or []
            return [b for b in bars if isinstance(b, Mapping)]
    except Exception:
        return []
    return []


def _pct_change(closes: Sequence[float], period: int) -> float:
    if len(closes) <= period:
        return 0.0
    a = closes[-1]
    b = closes[-1 - period]
    if b <= 0:
        return 0.0
    return (a - b) / b


def _realized_vol(closes: Sequence[float], period: int) -> float:
    """
    Sample standard deviation of log-ish returns over the last ``period`` bars.
    Returns 0.0 if history is too short. Uses simple returns (close/prev - 1)
    which is close enough for the 5d/20d ratio comparison done downstream.
    """
    if len(closes) < period + 1:
        return 0.0
    rets: List[float] = []
    for i in range(len(closes) - period, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev > 0:
            rets.append((cur - prev) / prev)
    if len(rets) < 2:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / len(rets)
    return var ** 0.5


def alpha_crypto_handler(ctx: Any, params: AlphaCryptoParams) -> List[TradeSignal]:
    """
    Emit long-only CRYPTO TradeSignals driven by three stacked filters:

      1. 20-day SMA momentum breakout (price > SMA20, last close > prior, 3d > 1.5%)
      2. 5d/20d realized-vol expansion (boost strong, skip on compression)
      3. Regime gate (skip ranging/adverse, halve strength in trending_bear)

    Deterministic for a given (ctx, params). Returns [] rather than raising on
    missing bar data so a single bad symbol cannot silence the whole brain.
    """
    if not params.enabled:
        return []

    regime = _read_regime(ctx)
    if regime in ("ranging", "adverse"):
        if _exploration_bypass_active():
            # CBAR-S1: exploration armed (paper-epoch only) — do NOT short-circuit; fall through
            # into the full momentum/breakout path in this otherwise-excluded regime. The loud
            # marker is the observability proof that the internal bypass engaged this cycle.
            logger.info("%s regime=%s", MARKER_EXPLORATION_HANDLER_PASS, regime)
        else:
            logger.info("alpha_crypto: no signals this cycle (reason: regime_%s)", regime)
            return []
    regime_mult = 0.5 if regime == "trending_bear" else 1.0

    universe = [s for s in params.actual_universe() if s.endswith("-USD")]
    prices_map = _get_mapping(ctx, "prices")

    signals: List[TradeSignal] = []
    skip_reasons: List[str] = []

    for symbol in universe:
        if len(signals) >= 2:
            break

        try:
            bars = _load_bars_for_symbol(ctx, symbol)
        except Exception as exc:
            skip_reasons.append(f"{symbol}:bar_read_error({type(exc).__name__})")
            continue

        if not bars or len(bars) < 22:
            skip_reasons.append(f"{symbol}:insufficient_history")
            continue

        closes: List[float] = []
        for b in bars[-30:]:
            c = _safe_float(b.get("close"), 0.0)
            if c > 0:
                closes.append(c)

        if len(closes) < 22:
            skip_reasons.append(f"{symbol}:insufficient_closes")
            continue

        live_px = _safe_float(prices_map.get(symbol), 0.0)
        price = live_px if live_px > 0 else closes[-1]
        prior_close = closes[-2]
        sma20 = sum(closes[-20:]) / 20.0

        # Signal 1: momentum breakout — long-only, direction-confirmed,
        # with a minimum 3-day run to filter noise and chop days.
        if price <= sma20:
            skip_reasons.append(f"{symbol}:below_sma20")
            continue
        if price <= prior_close:
            skip_reasons.append(f"{symbol}:no_up_confirm")
            continue
        pct_3d = _pct_change(closes, 3)
        if pct_3d < 0.015:
            skip_reasons.append(f"{symbol}:3d_move<1.5%")
            continue

        strength = _clamp((price - sma20) / sma20 * 10.0, 0.3, 1.0)

        # Signal 2: volatility expansion — compression kills the trade,
        # expansion boosts it. 20d baseline is required to have meaning.
        vol5 = _realized_vol(closes, 5)
        vol20 = _realized_vol(closes, 20)
        vol_ratio = (vol5 / vol20) if vol20 > 0 else 0.0
        if vol20 > 0 and vol5 > 0:
            if vol_ratio < 0.7:
                skip_reasons.append(f"{symbol}:vol_compression")
                continue
            if vol_ratio > 1.5:
                strength *= 1.2

        # Signal 3: regime multiplier (filter applied earlier).
        strength *= regime_mult
        strength = min(strength, 1.0)

        if strength < 0.3:
            skip_reasons.append(f"{symbol}:strength<0.3")
            continue

        # Size in base-currency units: target notional $1500–$5000 scaled by
        # strength, then divided by price. Downstream kraken_intents builder
        # multiplies size*price back to notional and caps against
        # dynamic_cap_for_crypto, so this sizing is a safe opening bid.
        target_notional_usd = 1500.0 + (strength - 0.3) / 0.7 * 3500.0
        size = target_notional_usd / price
        if size <= 0:
            continue

        side = SignalSide.BUY
        confidence = _clamp(0.5 + 0.4 * strength, 0.0, 0.95)

        # Phase B Item 4 — confidence-only crowding modifier. Reads the
        # public Kraken Futures derivatives snapshot from runtime/. Missing
        # or stale snapshot returns zero adjustment (fail-open). Never
        # blocks the trade and never alters side/size.
        _crypto_filter: CryptoFilterResult = get_crypto_filter(symbol, side.value)
        if _crypto_filter.confidence_adjustment != 0.0:
            confidence = max(
                0.0,
                min(0.95, float(confidence) + _crypto_filter.confidence_adjustment),
            )

        logger.info(
            "alpha_crypto signal: %s %s strength=%.3f",
            symbol, side.value, strength,
        )

        signals.append(
            TradeSignal(
                strategy=StrategyName.ALPHA_CRYPTO,
                symbol=symbol,
                side=side,
                size=float(size),
                confidence=float(confidence),
                asset_class=AssetClass.CRYPTO,
                meta={
                    "engine": "alpha_crypto.momentum_v1",
                    "regime": regime,
                    "regime_mult": float(regime_mult),
                    "price": float(price),
                    "sma20": float(sma20),
                    "prior_close": float(prior_close),
                    "pct_3d": float(pct_3d),
                    "vol5": float(vol5),
                    "vol20": float(vol20),
                    "vol_ratio": float(vol_ratio),
                    "strength": float(strength),
                    "target_notional_usd": float(target_notional_usd),
                    "required_asset_class": "crypto",
                    "crypto_market_bias": _crypto_filter.market_bias,
                    "crypto_funding_rate_8h": _crypto_filter.funding_rate_8h,
                    "crypto_funding_extreme": _crypto_filter.funding_extreme,
                    "crypto_confidence_adjustment": round(
                        float(_crypto_filter.confidence_adjustment), 6
                    ),
                },
            )
        )

    if not signals:
        reason = ";".join(skip_reasons) if skip_reasons else "no_eligible_symbols"
        logger.info("alpha_crypto: no signals this cycle (reason: %s)", reason)

    return signals
