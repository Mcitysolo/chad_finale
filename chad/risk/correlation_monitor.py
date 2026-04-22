"""
chad/risk/correlation_monitor.py

Phase-8 Session 7 (R6): correlation-cluster exposure cap.

If the average pairwise correlation of the CURRENT + NEW open book
exceeds ``threshold`` (default 0.7), new position sizes are scaled
down proportionally::

    avg_corr = mean(|pearson(ret_i, ret_j)|) for i<j
    if avg_corr > threshold:
        size_multiplier = threshold / avg_corr     (clamped to [floor, 1])

The threshold is absolute: near-perfect negative correlation also
counts as cluster risk because CHAD sees it as "same trade" from a
risk perspective.

Edge cases
----------

  * 0 or 1 open symbols → multiplier=1.0 (nothing to cluster with)
  * Missing bar data for any symbol → dropped from the pairwise matrix;
    if fewer than 2 usable symbols remain → multiplier=1.0
  * NaN returns or constant series → the pair's correlation is skipped

The computation is pure — no I/O beyond bar reads, no state is held
between calls.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BARS_DIR = ROOT / "data" / "bars" / "1d"
DEFAULT_CONFIG_PATH = ROOT / "config" / "sizing_config.json"

DEFAULT_THRESHOLD: float = 0.7
DEFAULT_LOOKBACK_DAYS: int = 20
DEFAULT_FLOOR_MULTIPLIER: float = 0.1


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sub = data.get("correlation_monitor") if isinstance(data, dict) else None
    return sub if isinstance(sub, dict) else {}


def _load_closes(symbol: str, bars_dir: Path) -> List[float]:
    if not symbol:
        return []
    path = bars_dir / f"{str(symbol).upper()}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    bars = data.get("bars") if isinstance(data, dict) else None
    if not isinstance(bars, list):
        return []
    closes: List[float] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        c = _safe_float(bar.get("close"), 0.0)
        if c > 0.0:
            closes.append(c)
    return closes


def _log_returns(closes: Sequence[float], lookback: int) -> List[float]:
    lookback = max(2, int(lookback))
    if len(closes) < lookback + 1:
        return []
    tail = list(closes[-(lookback + 1):])
    out: List[float] = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        cur = tail[i]
        if prev <= 0.0 or cur <= 0.0:
            continue
        out.append(math.log(cur / prev))
    return out


def _pearson_correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Plain Pearson r — returns None when either series is constant or too short."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    ax = list(a[-n:])
    bx = list(b[-n:])
    mean_a = statistics.fmean(ax)
    mean_b = statistics.fmean(bx)
    num = sum((xa - mean_a) * (xb - mean_b) for xa, xb in zip(ax, bx))
    den_a = math.sqrt(sum((xa - mean_a) ** 2 for xa in ax))
    den_b = math.sqrt(sum((xb - mean_b) ** 2 for xb in bx))
    if den_a == 0.0 or den_b == 0.0:
        return None
    r = num / (den_a * den_b)
    # Clamp tiny numerical drift.
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    return r


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class CorrelationMonitor:
    """Correlation-cluster exposure cap for per-trade sizing."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        bars_dir: Path = DEFAULT_BARS_DIR,
        floor_multiplier: float = DEFAULT_FLOOR_MULTIPLIER,
    ) -> None:
        self.threshold = float(max(1e-4, min(1.0, threshold)))
        self.lookback_days = int(max(2, lookback_days))
        self._bars_dir = Path(bars_dir)
        self.floor_multiplier = float(max(0.0, min(1.0, floor_multiplier)))

    @classmethod
    def from_config(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
        bars_dir: Path = DEFAULT_BARS_DIR,
    ) -> "CorrelationMonitor":
        cfg = _load_config(config_path)
        return cls(
            threshold=_safe_float(cfg.get("threshold"), DEFAULT_THRESHOLD),
            lookback_days=int(cfg.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS),
            bars_dir=bars_dir,
            floor_multiplier=_safe_float(
                cfg.get("floor_multiplier"), DEFAULT_FLOOR_MULTIPLIER
            ),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def _returns_for(self, symbol: str) -> List[float]:
        return _log_returns(_load_closes(symbol, self._bars_dir), self.lookback_days)

    def compute_avg_pairwise_corr(self, symbols: Iterable[str]) -> Optional[float]:
        """Mean of |pairwise Pearson r| over all usable symbol pairs.

        Returns None when fewer than 2 symbols have usable data.
        """
        uniq = []
        seen = set()
        for s in symbols:
            key = str(s or "").upper()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(key)

        ret_by_sym: dict = {}
        for sym in uniq:
            rets = self._returns_for(sym)
            if len(rets) >= 2:
                ret_by_sym[sym] = rets

        usable = list(ret_by_sym.keys())
        if len(usable) < 2:
            return None

        corrs: List[float] = []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                a = ret_by_sym[usable[i]]
                b = ret_by_sym[usable[j]]
                r = _pearson_correlation(a, b)
                if r is None:
                    continue
                corrs.append(abs(r))
        if not corrs:
            return None
        return statistics.fmean(corrs)

    def get_size_multiplier(
        self,
        open_symbols: Iterable[str],
        new_symbol: str,
    ) -> float:
        """Return the proportional reduction for a new trade.

        Single-position book (0 or 1 open symbols) always returns 1.0 —
        one trade cannot cluster with itself.
        """
        open_list = [str(s or "").upper() for s in open_symbols if s]
        new_sym = str(new_symbol or "").upper()

        if len([s for s in open_list if s]) < 1:
            return 1.0
        # De-dup new into the combined universe.
        combined = list(open_list)
        if new_sym and new_sym not in open_list:
            combined.append(new_sym)
        if len(combined) < 2:
            return 1.0

        avg = self.compute_avg_pairwise_corr(combined)
        if avg is None or avg <= self.threshold:
            return 1.0
        raw = self.threshold / avg
        return max(self.floor_multiplier, min(1.0, raw))


__all__ = [
    "DEFAULT_BARS_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FLOOR_MULTIPLIER",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_THRESHOLD",
    "CorrelationMonitor",
]
