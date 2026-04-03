#!/usr/bin/env python3
"""
chad/options/chain_provider.py

Options chain data provider for CHAD.

Fetches and caches available option chains (expirations + strikes) via
IBKR reqSecDefOptParams. Provides TTL-based in-memory caching with
atomic file persistence for poll-based consumers.

Design
------
- Deterministic, fail-closed.
- In-memory cache with configurable TTL.
- Atomic file write to runtime/options_chains_cache.json.
- On IBKR failure: returns stale cache with warning log.
- Full typing, strict null safety.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

LOGGER = logging.getLogger("chad.options.chain_provider")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CACHE_FILE_PATH = RUNTIME_DIR / "options_chains_cache.json"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class IBLike(Protocol):
    """Minimal IB interface for chain fetching."""

    def reqSecDefOptParams(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list: ...

    def qualifyContracts(self, *contracts: Any) -> list: ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionsChain:
    """
    Cached option chain for a single underlying.

    Attributes
    ----------
    symbol : str
        Underlying symbol (e.g., "SPY").
    exchange : str
        Options exchange (e.g., "SMART", "CBOE").
    expirations : List[str]
        Available expiration dates, sorted ascending (YYYYMMDD format).
    strikes : List[float]
        Available strike prices, sorted ascending.
    ts_utc : str
        Timestamp when this chain was fetched.
    ttl_seconds : int
        Cache validity duration in seconds.
    """
    symbol: str
    exchange: str
    expirations: List[str]
    strikes: List[float]
    ts_utc: str
    ttl_seconds: int = 3600

    def is_expired(self) -> bool:
        """Check if this cached chain has exceeded its TTL."""
        try:
            ts = self.ts_utc.strip()
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            fetched = datetime.fromisoformat(ts)
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - fetched).total_seconds()
            return age > self.ttl_seconds
        except Exception:
            return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "expirations": self.expirations,
            "strikes": self.strikes,
            "ts_utc": self.ts_utc,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptionsChain":
        return cls(
            symbol=str(d.get("symbol", "")),
            exchange=str(d.get("exchange", "")),
            expirations=list(d.get("expirations", [])),
            strikes=[float(s) for s in d.get("strikes", [])],
            ts_utc=str(d.get("ts_utc", "")),
            ttl_seconds=int(d.get("ttl_seconds", 3600)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# IBKRChainProvider
# ---------------------------------------------------------------------------

class IBKRChainProvider:
    """
    Fetches and caches options chain data from IBKR.

    Uses reqSecDefOptParams to get available expirations and strikes
    for a given underlying. Caches results in memory and to disk.

    Parameters
    ----------
    ib : IBLike
        Connected ib_insync.IB instance (injected, not created).
    cache_ttl_seconds : int
        How long cached chains remain valid (default 3600 = 1 hour).
    cache_file_path : Path
        Path for persistent file cache.
    """

    def __init__(
        self,
        ib: Any,
        *,
        cache_ttl_seconds: int = 3600,
        cache_file_path: Optional[Path] = None,
    ) -> None:
        self._ib = ib
        self._ttl = max(60, cache_ttl_seconds)
        self._cache_file = cache_file_path or CACHE_FILE_PATH
        self._cache: Dict[str, OptionsChain] = {}

    def get_chain(
        self,
        symbol: str,
        sec_type: str = "STK",
    ) -> Optional[OptionsChain]:
        """
        Get the options chain for a symbol.

        Returns cached chain if fresh. On cache miss, fetches from IBKR.
        On IBKR failure, returns stale cache with warning.

        Parameters
        ----------
        symbol : str
            Underlying symbol (e.g., "SPY").
        sec_type : str
            Underlying security type ("STK" for equities).

        Returns
        -------
        Optional[OptionsChain]
            Chain data, or None if no data available at all.
        """
        sym = symbol.strip().upper()

        # Check in-memory cache
        cached = self._cache.get(sym)
        if cached is not None and not cached.is_expired():
            return cached

        # Fetch from IBKR
        try:
            chain = self._fetch_from_ibkr(sym, sec_type)
            if chain is not None:
                self._cache[sym] = chain
                self._write_cache_file()
                return chain
        except Exception as exc:
            LOGGER.warning(
                "chain_provider.fetch_failed",
                extra={"symbol": sym, "error": str(exc)},
            )

        # Fallback to stale cache
        if cached is not None:
            LOGGER.warning(
                "chain_provider.using_stale_cache",
                extra={"symbol": sym, "cache_ts": cached.ts_utc},
            )
            return cached

        return None

    def _fetch_from_ibkr(
        self,
        symbol: str,
        sec_type: str,
    ) -> Optional[OptionsChain]:
        """
        Fetch options chain from IBKR via reqSecDefOptParams.

        Requires the underlying to be qualified first to get its conId.
        """
        from ib_insync import Stock

        underlying = Stock(symbol, "SMART", "USD")
        qualified = self._ib.qualifyContracts(underlying)
        if not qualified:
            LOGGER.warning("chain_provider.qualify_failed", extra={"symbol": symbol})
            return None

        con_id = underlying.conId
        if not con_id:
            return None

        chains = self._ib.reqSecDefOptParams(
            underlyingSymbol=symbol,
            futFopExchange="",
            underlyingSecType=sec_type,
            underlyingConId=con_id,
        )

        if not chains:
            LOGGER.warning("chain_provider.no_chains", extra={"symbol": symbol})
            return None

        # Merge all chains (different exchanges) into one combined set
        all_expirations: Set[str] = set()
        all_strikes: Set[float] = set()
        exchange = ""

        for c in chains:
            exps = getattr(c, "expirations", set())
            stks = getattr(c, "strikes", set())
            exch = getattr(c, "exchange", "")

            all_expirations.update(str(e) for e in exps)
            all_strikes.update(float(s) for s in stks if _is_valid_strike(s))

            # Prefer SMART exchange
            if exch.upper() == "SMART" or not exchange:
                exchange = exch

        if not all_expirations or not all_strikes:
            return None

        chain = OptionsChain(
            symbol=symbol,
            exchange=exchange or "SMART",
            expirations=sorted(all_expirations),
            strikes=sorted(all_strikes),
            ts_utc=_utc_now_iso(),
            ttl_seconds=self._ttl,
        )

        LOGGER.info(
            "chain_provider.fetched",
            extra={
                "symbol": symbol,
                "expirations": len(chain.expirations),
                "strikes": len(chain.strikes),
            },
        )

        return chain

    def _write_cache_file(self) -> None:
        """Write all cached chains to disk atomically."""
        payload: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "chains": {
                sym: chain.to_dict()
                for sym, chain in self._cache.items()
            },
        }
        try:
            _atomic_write_json(self._cache_file, payload)
        except Exception as exc:
            LOGGER.warning(
                "chain_provider.cache_write_failed",
                extra={"error": str(exc)},
            )


def _is_valid_strike(s: Any) -> bool:
    try:
        v = float(s)
        return math.isfinite(v) and v > 0
    except Exception:
        return False


__all__ = [
    "OptionsChain",
    "IBKRChainProvider",
]
