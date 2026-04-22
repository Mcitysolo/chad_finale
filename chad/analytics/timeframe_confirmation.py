"""
chad/analytics/timeframe_confirmation.py

Phase-8 Session 5 (S2): higher-timeframe bias attenuation for signal
confidence.

Reads data/bars/1d/{SYMBOL}.json and compares the latest close against
a 20-bar simple moving average. The resulting bias label is passed to
timeframe_confidence_multiplier() to produce a multiplicative attenuator
on intent confidence:

    intent side agrees with higher-TF bias   → 1.0    (no change)
    intent side DISAGREES                    → 0.6    (meaningful cut)
    neutral / missing data                   → 0.85   (mild haircut)

The multiplier is never below 0.1 — S2 must never zero confidence.
Missing bar data degrades gracefully to 'neutral' so unknown symbols
do not crash downstream callers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BARS_DIR = ROOT / "data" / "bars" / "1d"

# SMA lookback for the bias calculation. 20 daily bars ≈ one trading month.
DEFAULT_SMA_WINDOW: int = 20

# Multipliers — constants so tests can import them directly.
TF_AGREE_MULT: float = 1.0
TF_DISAGREE_MULT: float = 0.6
TF_NEUTRAL_MULT: float = 0.85
TF_MIN_MULT: float = 0.1

_VALID_BIASES = frozenset({"bullish", "bearish", "neutral"})


def _load_bars(symbol: str, bars_dir: Path) -> Optional[List[dict]]:
    if not symbol:
        return None
    candidate = bars_dir / f"{symbol}.json"
    if not candidate.is_file():
        # Try case variants — symbols in CHAD are usually uppercase.
        candidate = bars_dir / f"{symbol.upper()}.json"
        if not candidate.is_file():
            return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    bars = data.get("bars")
    if not isinstance(bars, list) or not bars:
        return None
    return bars


def _last_close(bars: List[dict]) -> Optional[float]:
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    try:
        return float(last.get("close"))
    except (TypeError, ValueError):
        return None


def _sma_close(bars: List[dict], window: int) -> Optional[float]:
    window = max(1, int(window))
    closes: List[float] = []
    for bar in bars[-window:]:
        if not isinstance(bar, dict):
            continue
        try:
            closes.append(float(bar.get("close")))
        except (TypeError, ValueError):
            continue
    if not closes:
        return None
    return sum(closes) / len(closes)


def get_higher_tf_bias(
    symbol: str,
    bars_dir: Path = DEFAULT_BARS_DIR,
    sma_window: int = DEFAULT_SMA_WINDOW,
) -> str:
    """Classify the higher-timeframe bias for a symbol.

    Returns one of 'bullish' / 'bearish' / 'neutral'. Neutral covers
    missing data, parse errors, and the boundary case where the last
    close equals the SMA.
    """
    bars = _load_bars(str(symbol or ""), Path(bars_dir))
    if not bars:
        return "neutral"
    last_close = _last_close(bars)
    sma = _sma_close(bars, sma_window)
    if last_close is None or sma is None:
        return "neutral"
    if last_close > sma:
        return "bullish"
    if last_close < sma:
        return "bearish"
    return "neutral"


def timeframe_confidence_multiplier(
    intent_side: Any,
    higher_tf_bias: Any,
) -> float:
    """Map (side, bias) to a confidence multiplier.

    Side can be 'BUY'/'SELL' (IBKR) or 'buy'/'sell' (Kraken). Any other
    value degrades to NEUTRAL so the multiplier is gentle rather than
    punitive.
    """
    side = str(intent_side or "").upper()
    bias = str(higher_tf_bias or "").lower()
    if bias not in _VALID_BIASES:
        return max(TF_MIN_MULT, TF_NEUTRAL_MULT)

    if bias == "neutral":
        return max(TF_MIN_MULT, TF_NEUTRAL_MULT)

    if side == "BUY":
        if bias == "bullish":
            return max(TF_MIN_MULT, TF_AGREE_MULT)
        if bias == "bearish":
            return max(TF_MIN_MULT, TF_DISAGREE_MULT)
    elif side == "SELL":
        if bias == "bearish":
            return max(TF_MIN_MULT, TF_AGREE_MULT)
        if bias == "bullish":
            return max(TF_MIN_MULT, TF_DISAGREE_MULT)

    # Unknown side → treat as neutral.
    return max(TF_MIN_MULT, TF_NEUTRAL_MULT)


__all__ = [
    "DEFAULT_BARS_DIR",
    "DEFAULT_SMA_WINDOW",
    "TF_AGREE_MULT",
    "TF_DISAGREE_MULT",
    "TF_NEUTRAL_MULT",
    "TF_MIN_MULT",
    "get_higher_tf_bias",
    "timeframe_confidence_multiplier",
]
