#!/usr/bin/env python3
"""
CHAD — Google Trends Signal Provider

Provides search-interest signals for trading symbols via Google Trends.
Used as a supplementary intelligence input for confidence adjustments.

Design:
- Caches results for 4 hours (Google rate-limits aggressively)
- Fails silently — never crashes advisory or orchestrator
- Writes state to runtime/trends_state.json for audit
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger("chad.intel.trends_provider")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
STATE_PATH = RUNTIME_DIR / "trends_state.json"
CACHE_TTL_SEC = 4 * 3600  # 4 hours

# Google Trends search terms for symbols that don't map directly
SYMBOL_SEARCH_TERMS: Dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq",
    "IWM": "Russell 2000",
    "MES": "S&P 500 futures",
    "MNQ": "Nasdaq futures",
    "MGC": "gold futures",
    "MCL": "oil futures",
    "GLD": "gold price",
    "TLT": "treasury bonds",
    "BTC": "Bitcoin",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "SVXY": "VIX volatility",
    "UVXY": "VIX volatility",
}

# Market-wide proxy symbols
MARKET_PROXIES = ["SPY", "QQQ", "BTC"]


@dataclass(frozen=True)
class TrendSignal:
    symbol: str
    current_interest: float
    avg_interest: float
    ratio: float
    signal: str  # HIGH, LOW, NEUTRAL


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        LOG.warning("Failed to write %s: %s", path, exc)


def _classify(ratio: float) -> str:
    if ratio > 1.5:
        return "HIGH"
    if ratio < 0.5:
        return "LOW"
    return "NEUTRAL"


class TrendsProvider:
    """
    Google Trends signal provider.

    All methods fail silently — returns empty dicts on any error.
    """

    def __init__(self, runtime_dir: Optional[Path] = None) -> None:
        self._runtime_dir = runtime_dir or RUNTIME_DIR
        self._state_path = self._runtime_dir / "trends_state.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._load_cache()

    def _load_cache(self) -> None:
        """Load persisted cache from disk."""
        state = _read_json(self._state_path)
        cached = state.get("cache", {})
        ts = state.get("cache_ts", 0.0)
        if isinstance(cached, dict) and isinstance(ts, (int, float)):
            self._cache = cached
            self._cache_ts = float(ts)

    def _save_state(self, signals: Dict[str, TrendSignal]) -> None:
        """Persist signals and cache to runtime/trends_state.json."""
        state = {
            "signals": {sym: asdict(sig) for sym, sig in signals.items()},
            "cache": self._cache,
            "cache_ts": self._cache_ts,
            "last_updated_utc": _utc_now_iso(),
        }
        _write_json(self._state_path, state)

    def _is_cache_fresh(self) -> bool:
        if not self._cache:
            return False
        return (time.time() - self._cache_ts) < CACHE_TTL_SEC

    def get_interest(
        self,
        symbols: list[str],
        timeframe: str = "today 3-m",
    ) -> Dict[str, TrendSignal]:
        """
        Get Google Trends interest for a list of symbols.

        Returns dict mapping symbol -> TrendSignal.
        Fails silently — returns empty dict on error.
        """
        if not symbols:
            return {}

        # Check cache
        if self._is_cache_fresh():
            cached_signals = {}
            all_cached = True
            for sym in symbols:
                cached = self._cache.get(sym)
                if cached and isinstance(cached, dict):
                    try:
                        cached_signals[sym] = TrendSignal(**cached)
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break
            if all_cached:
                LOG.debug("Returning cached trends for %s", symbols)
                return cached_signals

        # Fetch from Google Trends
        try:
            return self._fetch_trends(symbols, timeframe)
        except Exception as exc:
            LOG.warning("Google Trends fetch failed: %s", exc)
            return {}

    def _fetch_trends(
        self,
        symbols: list[str],
        timeframe: str,
    ) -> Dict[str, TrendSignal]:
        """Fetch trends data from Google Trends API."""
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

        signals: Dict[str, TrendSignal] = {}

        # Process in batches of 5 (Google Trends limit)
        search_terms = []
        term_to_symbol: Dict[str, str] = {}
        for sym in symbols:
            term = SYMBOL_SEARCH_TERMS.get(sym, sym)
            search_terms.append(term)
            term_to_symbol[term] = sym

        for i in range(0, len(search_terms), 5):
            batch = search_terms[i : i + 5]
            try:
                pytrends.build_payload(batch, timeframe=timeframe)
                df = pytrends.interest_over_time()

                if df is None or df.empty:
                    continue

                for term in batch:
                    if term not in df.columns:
                        continue
                    sym = term_to_symbol[term]
                    series = df[term]
                    avg_interest = float(series.mean())
                    # Current = average of last 7 data points (last week)
                    current_interest = float(series.tail(7).mean()) if len(series) >= 7 else avg_interest

                    ratio = current_interest / avg_interest if avg_interest > 0 else 1.0

                    sig = TrendSignal(
                        symbol=sym,
                        current_interest=round(current_interest, 2),
                        avg_interest=round(avg_interest, 2),
                        ratio=round(ratio, 2),
                        signal=_classify(ratio),
                    )
                    signals[sym] = sig
                    self._cache[sym] = asdict(sig)

            except Exception as exc:
                LOG.warning("Trends batch fetch failed for %s: %s", batch, exc)
                continue

        # Update cache timestamp and persist
        self._cache_ts = time.time()
        self._save_state(signals)

        return signals

    def get_market_interest(self) -> Dict[str, TrendSignal]:
        """
        Get market-wide interest using SPY, QQQ, BTC as proxies.

        Convenience method for the morning brief and intelligence layer.
        """
        return self.get_interest(MARKET_PROXIES)
