#!/usr/bin/env python3
"""
CHAD — Short Interest Provider

Reads short interest data from Finviz public pages.
Classifies short float into signal tiers and detects squeeze risk.

Design:
- Caches results for 6 hours (short interest changes slowly)
- Fails silently — never crashes advisory or orchestrator
- Writes state to runtime/short_interest.json for audit
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import urllib.error
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("chad.intel.short_interest_provider")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
STATE_PATH = RUNTIME_DIR / "short_interest.json"
CACHE_TTL_SEC = 6 * 3600  # 6 hours
REQUEST_DELAY_SEC = 1.5  # Rate limit between requests
REQUEST_TIMEOUT_SEC = 10

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Short float classification thresholds
LOW_THRESHOLD = 0.05       # < 5%
MODERATE_THRESHOLD = 0.15  # 5-15%
HIGH_THRESHOLD = 0.30      # 15-30%
# > 30% = EXTREME


@dataclass(frozen=True)
class ShortInterestSignal:
    symbol: str
    short_float_pct: float  # e.g. 0.032 = 3.2%
    signal: str  # LOW, MODERATE, HIGH, EXTREME
    squeeze_risk: bool  # True if HIGH/EXTREME + price uptrend
    ts_utc: str


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


def _classify_short_float(pct: float) -> str:
    if pct >= HIGH_THRESHOLD:
        return "EXTREME"
    if pct >= MODERATE_THRESHOLD:
        return "HIGH"
    if pct >= LOW_THRESHOLD:
        return "MODERATE"
    return "LOW"


def _parse_short_float(html: str) -> Optional[float]:
    """Extract Short Float percentage from Finviz HTML.

    Looks for patterns like 'Short Float</td>...<b>3.21%</b>' in the stats table.
    """
    # Try the table cell pattern used by Finviz
    patterns = [
        r"Short Float[^<]*</td>[^<]*<td[^>]*>[^<]*<b>([0-9]+\.?[0-9]*)%",
        r'Short Float["\s].*?([0-9]+\.?[0-9]*)%',
        r"Short Float.*?([0-9]+\.?[0-9]*)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) / 100.0  # Convert 3.2% -> 0.032
            except (ValueError, IndexError):
                continue
    return None


def _check_price_uptrend(symbol: str) -> bool:
    """Check if symbol has a recent price uptrend using price cache."""
    try:
        cache_path = RUNTIME_DIR / "price_cache.json"
        cache = _read_json(cache_path)
        val = cache.get(symbol)
        if isinstance(val, dict):
            chg = val.get("change_pct") or val.get("pct_change")
            if chg is not None:
                return float(chg) > 0
        return False
    except Exception:
        return False


class ShortInterestProvider:
    """
    Short interest data provider using Finviz public pages.

    All methods fail silently — returns empty dicts on any error.
    """

    def __init__(self, runtime_dir: Optional[Path] = None) -> None:
        self._runtime_dir = runtime_dir or RUNTIME_DIR
        self._state_path = self._runtime_dir / "short_interest.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._load_cache()

    def _load_cache(self) -> None:
        state = _read_json(self._state_path)
        cached = state.get("cache", {})
        ts = state.get("cache_ts", 0.0)
        if isinstance(cached, dict) and isinstance(ts, (int, float)):
            self._cache = cached
            self._cache_ts = float(ts)

    def _save_state(self, signals: Dict[str, ShortInterestSignal]) -> None:
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

    def get_short_interest(self, symbol: str) -> Optional[ShortInterestSignal]:
        """Get short interest for a single symbol. Fails silently."""
        result = self.get_batch_short_interest([symbol])
        return result.get(symbol)

    def get_batch_short_interest(
        self, symbols: List[str],
    ) -> Dict[str, ShortInterestSignal]:
        """Get short interest for multiple symbols. Fails silently."""
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
                        cached_signals[sym] = ShortInterestSignal(**cached)
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break
            if all_cached:
                LOG.debug("Returning cached short interest for %s", symbols)
                return cached_signals

        try:
            return self._fetch_short_interest(symbols)
        except Exception as exc:
            LOG.warning("Short interest fetch failed: %s", exc)
            return {}

    def _fetch_finviz_page(self, symbol: str) -> str:
        """Fetch Finviz quote page for a symbol."""
        url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def _fetch_short_interest(
        self, symbols: List[str],
    ) -> Dict[str, ShortInterestSignal]:
        """Fetch short interest from Finviz for all symbols."""
        signals: Dict[str, ShortInterestSignal] = {}

        for sym in symbols:
            try:
                html = self._fetch_finviz_page(sym)
                short_float = _parse_short_float(html)

                if short_float is None:
                    LOG.debug("Could not parse short float for %s", sym)
                    continue

                signal_label = _classify_short_float(short_float)
                uptrend = _check_price_uptrend(sym)
                squeeze = signal_label in ("HIGH", "EXTREME") and uptrend

                sig = ShortInterestSignal(
                    symbol=sym,
                    short_float_pct=round(short_float, 4),
                    signal=signal_label,
                    squeeze_risk=squeeze,
                    ts_utc=_utc_now_iso(),
                )
                signals[sym] = sig
                self._cache[sym] = asdict(sig)

            except Exception as exc:
                LOG.debug("Finviz fetch failed for %s: %s", sym, exc)
                continue

            # Rate limit between requests
            if sym != symbols[-1]:
                time.sleep(REQUEST_DELAY_SEC)

        self._cache_ts = time.time()
        self._save_state(signals)
        return signals
