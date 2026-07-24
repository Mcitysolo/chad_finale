#!/usr/bin/env python3
"""
chad/market_data/ibkr_historical_provider.py

IBKR Historical Bar Provider for CHAD.

Replaces Polygon daily bars backfill with direct IBKR reqHistoricalData.
Writes to data/bars/1d/ and data/bars/1m/ in the same format as
polygon_daily_bars_backfill.py for seamless consumer compatibility.

Usage:
    from ib_async import IB
    from chad.market_data.ibkr_historical_provider import IBKRHistoricalProvider
    from chad.execution.ibkr_client_ids import HISTORICAL_PROVIDER

    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=HISTORICAL_PROVIDER)
    provider = IBKRHistoricalProvider(ib)
    bars = provider.fetch_daily_bars('SPY', days=400)
    results = provider.backfill_universe(['SPY', 'QQQ', 'AAPL'])
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# PA-EP6b: reuse the bar-provider's true-UTC normalizer so backfills cannot
# re-introduce local-labeled bars. Import is clean — ibkr_bar_provider has no
# import-time side effects and does not import this module (no circular dep).
from chad.market_data.ibkr_bar_provider import _bar_ts_to_utc_iso

LOGGER = logging.getLogger("chad.market_data.ibkr_historical_provider")

REPO_ROOT = Path(__file__).resolve().parents[2]
BARS_DIR = REPO_ROOT / "data" / "bars"
UNIVERSE_PATH = REPO_ROOT / "config" / "universe.json"

# IBKR pacing: max 60 requests per 10 minutes; we use 1 req per 0.5s
PACING_DELAY_S = 0.5


_FUTURES_SPEC_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _load_futures_specs() -> Dict[str, Dict[str, str]]:
    """
    Load precise futures contract specs from config/universe.json.

    Returns dict keyed by symbol with exchange/currency/tradingClass fields.
    Cached after first load.
    """
    global _FUTURES_SPEC_CACHE
    if _FUTURES_SPEC_CACHE is not None:
        return _FUTURES_SPEC_CACHE
    specs: Dict[str, Dict[str, str]] = {}
    try:
        if UNIVERSE_PATH.is_file():
            obj = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
            for entry in obj.get("futures", []) or []:
                if not isinstance(entry, dict):
                    continue
                sym = str(entry.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                specs[sym] = {
                    "exchange": str(entry.get("exchange", "")).strip(),
                    "currency": str(entry.get("currency", "USD")).strip() or "USD",
                    "tradingClass": str(entry.get("tradingClass", "")).strip(),
                }
    except Exception as exc:
        LOGGER.warning("ibkr_historical_provider.universe_load_failed: %s", exc)
    _FUTURES_SPEC_CACHE = specs
    return specs


@dataclass(frozen=True)
class Bar:
    """Standard OHLCV bar."""
    ts_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class IBKRHistoricalProvider:
    """
    Fetches historical OHLCV bars from IBKR and writes to data/bars/.

    Output format matches polygon_daily_bars_backfill.py for consumer compatibility:
    {
        "symbol": "SPY",
        "source": "ibkr",
        "timeframe": "1d",
        "ts_utc": "...",
        "ttl_seconds": 86400,
        "bars": [{...}, ...]
    }
    """

    def __init__(
        self,
        ib: Any,
        *,
        bars_dir: Optional[Path] = None,
    ) -> None:
        self._ib = ib
        self._bars_dir = bars_dir or BARS_DIR

        # Request delayed data (free)
        try:
            self._ib.reqMarketDataType(3)
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Front-month expiry resolution
    # ---------------------------------------------------------------------------
    # W6A-2: the private ``_EXPIRY_SCHEDULE`` mirror that used to live here has
    # been DELETED. It was a byte-identical hand-maintained copy of
    # chad.market_data.futures_contract_resolver.EXPIRY_SCHEDULE, self-documented
    # as a "mirror copy" — two calendars that nothing kept in sync. Resolution
    # now delegates to that module, which is the single source of truth.
    #
    # Do not reintroduce a schedule table here; a named test
    # (test_w6a_no_private_expiry_schedule) asserts this class has no private
    # expiry table so the mirror cannot silently come back.

    # Per-symbol contract specs for explicit futures resolution.
    # ibkr_symbol: the root symbol IBKR uses (may differ from logical name, e.g. SIL->SI)
    # multiplier: contract size as string
    # exchange: primary exchange
    # currency: always USD
    _FUTURES_CONTRACT_SPECS: dict = {
        "MES":  {"ibkr_symbol": "MES", "multiplier": "5",     "exchange": "CME",   "currency": "USD"},
        "MNQ":  {"ibkr_symbol": "MNQ", "multiplier": "2",     "exchange": "CME",   "currency": "USD"},
        "MCL":  {"ibkr_symbol": "MCL", "multiplier": "100",   "exchange": "NYMEX", "currency": "USD"},
        "MGC":  {"ibkr_symbol": "MGC", "multiplier": "10",    "exchange": "COMEX", "currency": "USD"},
        "ZN":   {"ibkr_symbol": "ZN",  "multiplier": "1000",  "exchange": "CBOT",  "currency": "USD"},
        "ZB":   {"ibkr_symbol": "ZB",  "multiplier": "1000",  "exchange": "CBOT",  "currency": "USD"},
        "M6E":  {"ibkr_symbol": "M6E", "multiplier": "12500", "exchange": "CME",   "currency": "USD"},
        "SIL":  {"ibkr_symbol": "SI",  "multiplier": "1000",  "exchange": "COMEX", "currency": "USD"},
        "MYM":  {"ibkr_symbol": "MYM", "multiplier": "0.5",   "exchange": "CBOT",  "currency": "USD"},
        "M2K":  {"ibkr_symbol": "M2K", "multiplier": "5",     "exchange": "CME",   "currency": "USD"},
    }

    @staticmethod
    def _resolve_front_month(symbol: str, *, now: Optional[datetime] = None) -> Optional[str]:
        """
        Return the nearest active front-month expiry (YYYYMM) for ``symbol``.

        W6A-2: delegates to chad.market_data.futures_contract_resolver, the
        single source of truth. Returns None for symbols that module does not
        model, and None (not a stale month) when the resolver reports
        exhaustion — the caller treats None as "cannot resolve" and skips the
        symbol, which per-symbol error isolation then reports. The old
        behaviour returned an EXPIRED month in that case.
        """
        from chad.market_data.futures_contract_resolver import (
            ExpiryScheduleExhausted,
            resolve_contract_month,
        )

        try:
            return resolve_contract_month(symbol, now=now)
        except ExpiryScheduleExhausted as exc:
            LOGGER.error(
                "ibkr_historical.expiry_schedule_exhausted symbol=%s err=%s", symbol, exc
            )
            return None

    def _make_contract(self, symbol: str, sec_type: str = "STK") -> Any:
        from ib_async import Stock, Future

        sym = symbol.strip().upper()

        if sec_type == "FUT":
            spec = IBKRHistoricalProvider._FUTURES_CONTRACT_SPECS.get(sym)
            if spec:
                local_sym = spec.get("local_symbol")
                if local_sym:
                    contract = Future(
                        symbol=spec["ibkr_symbol"],
                        localSymbol=local_sym,
                        exchange=spec["exchange"],
                        currency=spec["currency"],
                        multiplier=spec["multiplier"],
                    )
                    LOGGER.info(
                        "ibkr_historical.contract_resolved symbol=%s local_symbol=%s multiplier=%s",
                        sym, local_sym, spec["multiplier"],
                    )
                    return contract

                expiry = IBKRHistoricalProvider._resolve_front_month(sym)
                if not expiry:
                    LOGGER.warning(
                        "ibkr_historical.no_expiry_found symbol=%s", sym
                    )
                    return Future(
                        symbol=spec["ibkr_symbol"],
                        exchange=spec["exchange"],
                        currency=spec["currency"],
                        tradingClass=sym,
                    )
                contract = Future(
                    symbol=spec["ibkr_symbol"],
                    lastTradeDateOrContractMonth=expiry,
                    exchange=spec["exchange"],
                    currency=spec["currency"],
                    multiplier=spec["multiplier"],
                    tradingClass=sym,
                )
                LOGGER.info(
                    "ibkr_historical.contract_resolved symbol=%s expiry=%s multiplier=%s",
                    sym, expiry, spec["multiplier"],
                )
                return contract

            # Fallback for futures not in the spec table: use universe.json specs
            specs = _load_futures_specs()
            spec_fallback = specs.get(sym, {})
            exchange = spec_fallback.get("exchange") or "CME"
            currency = spec_fallback.get("currency") or "USD"
            trading_class = spec_fallback.get("tradingClass") or sym
            contract = Future(
                symbol=sym,
                exchange=exchange,
                currency=currency,
                tradingClass=trading_class,
            )
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception as _qe:
                LOGGER.warning(
                    "ibkr_historical.qualify_failed symbol=%s err=%s", sym, _qe
                )
            return contract

        return Stock(sym, "SMART", "USD")

    def _duration_str(self, days: int) -> str:
        """Convert days to IBKR duration string."""
        if days <= 365:
            return f"{days} D"
        years = days // 365
        return f"{years} Y"

    def _parse_bars(self, ib_bars: Any, symbol: str) -> List[Bar]:
        """Parse IBKR bar objects into Bar dataclasses."""
        bars: List[Bar] = []
        for b in ib_bars:
            ts = _bar_ts_to_utc_iso(getattr(b, "date", ""))
            o = _safe_float(getattr(b, "open", 0))
            h = _safe_float(getattr(b, "high", 0))
            lo = _safe_float(getattr(b, "low", 0))
            c = _safe_float(getattr(b, "close", 0))
            v = _safe_float(getattr(b, "volume", 0))

            # Validate OHLC integrity
            if min(o, h, lo, c) <= 0 or h < lo:
                continue

            bars.append(Bar(
                ts_utc=ts,
                open=o, high=h, low=lo, close=c,
                volume=max(v, 0.0),
                symbol=symbol,
            ))
        return bars

    def _write_bars_file(
        self,
        symbol: str,
        bars: List[Bar],
        timeframe: str,
    ) -> Path:
        """Write bars to data/bars/{timeframe}/{symbol}.json atomically."""
        out_dir = self._bars_dir / timeframe
        out_path = out_dir / f"{symbol}.json"

        payload = {
            "bars": [b.to_dict() for b in bars],
            "symbol": symbol,
            "source": "ibkr",
            "timeframe": timeframe,
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": 86400 if timeframe == "1d" else 3600,
        }

        _atomic_write_json(out_path, payload)
        return out_path

    def fetch_daily_bars(
        self,
        symbol: str,
        days: int = 400,
        sec_type: str = "STK",
    ) -> List[Bar]:
        """
        Fetch daily OHLCV bars from IBKR.

        Writes to data/bars/1d/{symbol}.json.
        Returns list of Bar objects sorted ascending by timestamp.
        """
        sym = symbol.strip().upper()
        contract = self._make_contract(sym, sec_type)

        try:
            self._ib.qualifyContracts(contract)
        except Exception:
            pass

        ib_bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=self._duration_str(days),
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        bars = self._parse_bars(ib_bars, sym)
        if bars:
            self._write_bars_file(sym, bars, "1d")
            LOGGER.info("ibkr_historical.daily_fetched symbol=%s bars=%d", sym, len(bars))

        return bars

    def fetch_minute_bars(
        self,
        symbol: str,
        days: int = 5,
        sec_type: str = "STK",
    ) -> List[Bar]:
        """
        Fetch 1-minute OHLCV bars from IBKR.

        Writes to data/bars/1m/{symbol}.json.
        IBKR limits 1-min bars to ~5-7 days of history.
        """
        sym = symbol.strip().upper()
        contract = self._make_contract(sym, sec_type)

        try:
            self._ib.qualifyContracts(contract)
        except Exception:
            pass

        # IBKR max for 1-min bars is ~7 days
        capped_days = min(days, 7)

        ib_bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{capped_days} D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        bars = self._parse_bars(ib_bars, sym)
        if bars:
            self._write_bars_file(sym, bars, "1m")
            LOGGER.info("ibkr_historical.minute_fetched symbol=%s bars=%d", sym, len(bars))

        return bars

    def backfill_universe(
        self,
        symbols: List[str],
        days: int = 400,
        sec_type: str = "STK",
        include_minute: bool = False,
    ) -> Dict[str, int]:
        """
        Backfill daily bars for a list of symbols.

        Returns dict of symbol -> bar count written.
        Rate-limited to respect IBKR pacing (1 req per 0.5s).
        """
        results: Dict[str, int] = {}

        for sym in symbols:
            sym = sym.strip().upper()
            try:
                bars = self.fetch_daily_bars(sym, days=days, sec_type=sec_type)
                results[sym] = len(bars)
                LOGGER.info("ibkr_historical.backfill symbol=%s daily_bars=%d", sym, len(bars))
            except Exception as exc:
                LOGGER.warning("ibkr_historical.backfill_failed symbol=%s: %s", sym, exc)
                results[sym] = 0

            time.sleep(PACING_DELAY_S)

            if include_minute:
                try:
                    self.fetch_minute_bars(sym, days=5, sec_type=sec_type)
                except Exception as exc:
                    LOGGER.warning("ibkr_historical.minute_backfill_failed symbol=%s: %s", sym, exc)
                time.sleep(PACING_DELAY_S)

        total = sum(results.values())
        LOGGER.info("ibkr_historical.backfill_complete symbols=%d total_bars=%d", len(results), total)
        return results
