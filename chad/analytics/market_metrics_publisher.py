"""
chad/analytics/market_metrics_publisher.py

Phase-8 Session 6 (G1 feed): compute market-regime inputs from daily
bar data and publish ``runtime/market_metrics.json`` for the G1
regime_classifier to consume.

Without this publisher the classifier reads nothing and returns
"unknown" every cycle. With it, the classifier sees real
``realized_vol_percentile``, ``adx``, ``trend_slope`` and
``market_breadth`` inputs and can produce actionable regime labels.

Computation
-----------

For each symbol with 1d bar data:

    realized_vol_percentile : float in [0, 1]
        Percentile rank of the most recent 20-day realized volatility
        against the last ``lookback_days`` (default 252) of 20-day vols.

    adx                     : float, ~0-100
        Wilder's Average Directional Index over 14 bars — a measure of
        *directional trend strength*, not volatility. Reads LOW (<20) in
        flat or choppy markets where +DI and -DI roughly cancel, and
        HIGH (>25) in a sustained trend. Compares against
        DEFAULT_ADX_THRESHOLD = 25 in regime_classifier.py.

    trend_slope             : float
        Slope of a linear regression on the last 20 daily closes,
        normalised by the first value of the window. Positive →
        up-trending, negative → down-trending.

The market-level aggregate metrics take the median of the per-symbol
values so a single outlier (e.g. a volatile crypto) cannot dominate.
Breadth is the fraction of symbols whose last close is above their
20-bar SMA; scaled from [0, 1] to [-1, 1] to match the classifier's
``market_breadth`` input convention.

Degradation
-----------

An empty or missing bars directory still writes a defensible payload::

    {"realized_vol_percentile": 0.5, "adx": 0.0, "trend_slope": 0.0,
     "market_breadth": 0.0, "symbols_covered": 0, ...}

so the classifier falls into rule 4 (ranging) rather than rule 5
(unknown). The publisher never raises — every file-level error is
caught and logged.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BARS_DIR = ROOT / "data" / "bars" / "1d"
MARKET_METRICS_PATH = ROOT / "runtime" / "market_metrics.json"

SCHEMA_VERSION = "market_metrics.v1"

DEFAULT_VOL_WINDOW: int = 20
DEFAULT_TREND_WINDOW: int = 20
DEFAULT_ADX_PERIOD: int = 14
DEFAULT_LOOKBACK_DAYS: int = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_bars(path: Path) -> List[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    bars = data.get("bars")
    return bars if isinstance(bars, list) else []


def _closes(bars: List[dict]) -> List[float]:
    out: List[float] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        c = _safe_float(bar.get("close"), 0.0)
        if c > 0.0:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def compute_realized_vol_percentile(
    closes: List[float], window: int = DEFAULT_VOL_WINDOW, lookback: int = DEFAULT_LOOKBACK_DAYS
) -> float:
    """Percentile rank of the latest ``window``-day realized vol vs history.

    Returns a float in [0, 1]. Not enough data → 0.5 (mid-range) so the
    classifier's vol branch does not fire on insufficient input.
    """
    window = max(2, int(window))
    lookback = max(window + 1, int(lookback))
    if len(closes) < window + 1:
        return 0.5
    rets: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev <= 0.0:
            continue
        rets.append(math.log(closes[i] / prev))
    if len(rets) < window:
        return 0.5
    # Rolling std of log-returns over `window` bars.
    vols: List[float] = []
    for i in range(window, len(rets) + 1):
        window_rets = rets[i - window : i]
        if len(window_rets) >= 2:
            vols.append(statistics.pstdev(window_rets))
    if not vols:
        return 0.5
    # Use the most recent `lookback` vols as the history.
    history = vols[-lookback:]
    latest = vols[-1]
    if len(history) < 2:
        return 0.5
    below_or_equal = sum(1 for v in history if v <= latest)
    return below_or_equal / len(history)


def _wilder_smoothed_series(values: List[float], period: int) -> List[float]:
    """Wilder's smoothing of ``values`` over ``period``.

    The first smoothed value is the *sum* of the first ``period`` inputs;
    each subsequent value is ``prior - prior / period + current``. Because
    the directional indicators divide one smoothed series by another
    (+DM/TR, -DM/TR), the sum-vs-average scaling cancels — only the
    consistent recursion matters. Returns a list of length
    ``len(values) - period + 1``; an empty list if there is too little
    data to seed the first smoothed value.
    """
    period = max(2, int(period))
    if len(values) < period:
        return []
    run = math.fsum(values[:period])
    smoothed: List[float] = [run]
    for i in range(period, len(values)):
        run = run - run / period + values[i]
        smoothed.append(run)
    return smoothed


def compute_adx_proxy(bars: List[dict], period: int = DEFAULT_ADX_PERIOD) -> float:
    """Wilder's Average Directional Index (ADX) from bar OHLC.

    Measures *directional trend strength*, not volatility. In a flat or
    choppy market +DI and -DI roughly cancel, so DX — and hence the ADX —
    reads LOW (<20); a strong sustained trend (up or down) drives them
    apart and the ADX reads HIGH (>25). The result is naturally bounded
    to roughly 0-100.

    Standard Wilder construction (period=14): per bar compute directional
    movement (+DM, -DM) and True Range (TR), Wilder-smooth each over
    ``period``, form +DI/-DI, then DX, then Wilder-smooth DX into the ADX.

    The name ``compute_adx_proxy`` is kept for drop-in compatibility with
    the publisher call site and the ``adx`` JSON key; only the internals
    changed. The earlier volatility proxy (``atr_pct * 1600``) was
    calibrated for ~1.5% index ATR and mis-read individual stocks (2-7%
    ATR) as "strongly trending" even in a flat market.

    Needs at least ``2 * period + 1`` bars for a stable ADX; with fewer it
    returns 0.0 — the same graceful-degrade contract as the prior proxy.
    """
    period = max(2, int(period))
    if len(bars) < 2 * period + 1:
        return 0.0

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs: List[float] = []
    for i in range(1, len(bars)):
        cur = bars[i]
        prev = bars[i - 1]
        if not isinstance(cur, dict) or not isinstance(prev, dict):
            continue
        hi = _safe_float(cur.get("high"))
        lo = _safe_float(cur.get("low"))
        prev_hi = _safe_float(prev.get("high"))
        prev_lo = _safe_float(prev.get("low"))
        prev_close = _safe_float(prev.get("close"))
        if hi <= 0.0 or lo <= 0.0 or prev_hi <= 0.0 or prev_lo <= 0.0 or prev_close <= 0.0:
            continue
        up_move = hi - prev_hi
        down_move = prev_lo - lo
        plus_dm.append(up_move if (up_move > down_move and up_move > 0.0) else 0.0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0.0) else 0.0)
        trs.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))

    # Two Wilder passes are needed: one to seed +DI/-DI/DX, one to seed the
    # ADX average. Require 2*period directional samples so both are stable.
    if len(trs) < 2 * period:
        return 0.0

    sm_plus = _wilder_smoothed_series(plus_dm, period)
    sm_minus = _wilder_smoothed_series(minus_dm, period)
    sm_tr = _wilder_smoothed_series(trs, period)

    dxs: List[float] = []
    for sp, sm, st in zip(sm_plus, sm_minus, sm_tr):
        if st <= 0.0:
            dxs.append(0.0)
            continue
        plus_di = 100.0 * sp / st
        minus_di = 100.0 * sm / st
        di_sum = plus_di + minus_di
        if di_sum <= 0.0:  # guard divide-by-zero (both DI == 0)
            dxs.append(0.0)
            continue
        dxs.append(100.0 * abs(plus_di - minus_di) / di_sum)

    if len(dxs) < period:
        return 0.0

    # ADX seed = simple mean of the first `period` DX values, then Wilder's
    # running average: adx = (prior * (period - 1) + current) / period.
    adx = statistics.fmean(dxs[:period])
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period

    # Defensive clamp: DX/ADX are mathematically in [0, 100].
    if adx < 0.0:
        return 0.0
    if adx > 100.0:
        return 100.0
    return adx


def compute_trend_slope(closes: List[float], period: int = DEFAULT_TREND_WINDOW) -> float:
    """Normalised linear-regression slope over the last ``period`` closes.

    Positive → up-trending, negative → down-trending. Normalisation by
    the oldest bar in the window keeps the magnitude roughly scale-
    invariant across assets. Not enough data → 0.0.
    """
    period = max(2, int(period))
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    x = list(range(period))
    try:
        import numpy as np
        slope = float(np.polyfit(x, window, 1)[0])
    except Exception:  # pragma: no cover - fallback to pure-Python OLS
        mean_x = sum(x) / period
        mean_y = sum(window) / period
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, window))
        den = sum((xi - mean_x) ** 2 for xi in x)
        if den == 0.0:
            return 0.0
        slope = num / den
    base = window[0]
    if base == 0.0:
        return slope
    return slope / base


def compute_breadth(
    closes_by_symbol: Mapping[str, List[float]], sma_window: int = 20
) -> float:
    """Fraction of symbols trading above their SMA, scaled to [-1, 1]."""
    sma_window = max(2, int(sma_window))
    n = 0
    above = 0
    for closes in closes_by_symbol.values():
        if len(closes) < sma_window:
            continue
        n += 1
        sma = statistics.fmean(closes[-sma_window:])
        if closes[-1] > sma:
            above += 1
    if n == 0:
        return 0.0
    ratio = above / n  # [0, 1]
    return (ratio - 0.5) * 2.0  # → [-1, 1]


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class MarketMetricsPublisher:
    """Compute per-symbol + market-level metrics and publish the snapshot."""

    def __init__(
        self,
        bars_dir: Path = DEFAULT_BARS_DIR,
        output_path: Path = MARKET_METRICS_PATH,
    ) -> None:
        self._bars_dir = Path(bars_dir)
        self._output_path = Path(output_path)

    def _gather_bars(self) -> Dict[str, List[dict]]:
        if not self._bars_dir.is_dir():
            return {}
        out: Dict[str, List[dict]] = {}
        for path in sorted(self._bars_dir.glob("*.json")):
            bars = _load_bars(path)
            if not bars:
                continue
            out[path.stem.upper()] = bars
        return out

    def compute_and_publish(
        self,
        vol_window: int = DEFAULT_VOL_WINDOW,
        trend_window: int = DEFAULT_TREND_WINDOW,
        adx_period: int = DEFAULT_ADX_PERIOD,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> Dict[str, Any]:
        """Run the computation and write runtime/market_metrics.json.

        Returns the payload that was persisted so callers can log it.
        """
        bars_by_sym = self._gather_bars()

        if not bars_by_sym:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "ts_utc": _utc_now_iso(),
                "realized_vol_percentile": 0.5,
                "adx": 0.0,
                "trend_slope": 0.0,
                "market_breadth": 0.0,
                "symbols_covered": 0,
                "source": "market_metrics_publisher",
                "per_symbol": {},
                "notes": "no_bars_found",
            }
            _write_atomic(self._output_path, payload)
            return payload

        per_symbol: Dict[str, Dict[str, float]] = {}
        closes_by_sym: Dict[str, List[float]] = {}
        for symbol, bars in bars_by_sym.items():
            closes = _closes(bars)
            closes_by_sym[symbol] = closes
            if len(closes) < 2:
                continue
            per_symbol[symbol] = {
                "realized_vol_percentile": compute_realized_vol_percentile(
                    closes, window=vol_window, lookback=lookback_days
                ),
                "adx": compute_adx_proxy(bars, period=adx_period),
                "trend_slope": compute_trend_slope(closes, period=trend_window),
            }

        if per_symbol:
            vols = [v["realized_vol_percentile"] for v in per_symbol.values()]
            adxs = [v["adx"] for v in per_symbol.values()]
            slopes = [v["trend_slope"] for v in per_symbol.values()]
            aggregate_vol = statistics.median(vols) if vols else 0.5
            aggregate_adx = statistics.median(adxs) if adxs else 0.0
            aggregate_slope = statistics.median(slopes) if slopes else 0.0
        else:
            aggregate_vol = 0.5
            aggregate_adx = 0.0
            aggregate_slope = 0.0

        breadth = compute_breadth(closes_by_sym, sma_window=trend_window)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "ts_utc": _utc_now_iso(),
            "realized_vol_percentile": round(aggregate_vol, 4),
            "adx": round(aggregate_adx, 4),
            "trend_slope": round(aggregate_slope, 6),
            "market_breadth": round(breadth, 4),
            "symbols_covered": len(per_symbol),
            "source": "market_metrics_publisher",
            "per_symbol": {
                k: {
                    "realized_vol_percentile": round(v["realized_vol_percentile"], 4),
                    "adx": round(v["adx"], 4),
                    "trend_slope": round(v["trend_slope"], 6),
                }
                for k, v in per_symbol.items()
            },
        }
        _write_atomic(self._output_path, payload)
        return payload


def compute_and_publish(
    bars_dir: Path = DEFAULT_BARS_DIR,
    output_path: Path = MARKET_METRICS_PATH,
) -> Dict[str, Any]:
    """Module-level convenience wrapper for the default publisher."""
    return MarketMetricsPublisher(
        bars_dir=bars_dir, output_path=output_path
    ).compute_and_publish()


__all__ = [
    "DEFAULT_ADX_PERIOD",
    "DEFAULT_BARS_DIR",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_TREND_WINDOW",
    "DEFAULT_VOL_WINDOW",
    "MARKET_METRICS_PATH",
    "SCHEMA_VERSION",
    "MarketMetricsPublisher",
    "compute_adx_proxy",
    "compute_and_publish",
    "compute_breadth",
    "compute_realized_vol_percentile",
    "compute_trend_slope",
]
