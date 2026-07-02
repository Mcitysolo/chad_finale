"""
chad/market_data/kraken_balance_provider.py

KrakenBalanceProvider — read-only adapter that exposes the live Kraken
account balance to the rest of CHAD (advisory engine, daily report,
sizing logic for the alpha_crypto CAD lane, etc.).

Responsibilities
----------------
- Authenticated /0/private/Balance read using KrakenClientConfig.from_env()
- Normalization of Kraken's exchange-specific asset codes (XXBT, ZCAD, ...)
  into CHAD-canonical asset codes (BTC, CAD, ...)
- USD-equivalent valuation against a caller-supplied price map
- Throttled (default 5-minute) snapshot writer to runtime/kraken_balances.json
  for downstream consumers (advisory + daily report)

Design notes
------------
- This provider is read-only. It never places orders.
- Failures are non-fatal: get_balance() returns {} on error and the snapshot
  file records the error so the advisory engine can flag a stale read.
- The provider deliberately does not import any chad.execution modules to
  keep the dependency graph one-way (market_data -> exchanges, never the
  other direction).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig

LOGGER = logging.getLogger("chad.market_data.kraken_balance_provider")


# ---------------------------------------------------------------------------
# Asset code normalization
# ---------------------------------------------------------------------------
#
# Kraken's REST API returns exchange-specific asset codes:
#   XXBT -> BTC          (the leading 'X' is the legacy 'crypto' prefix)
#   XETH -> ETH
#   XLTC -> LTC
#   XDOG -> DOGE
#   ZUSD -> USD          (the leading 'Z' is the legacy 'fiat' prefix)
#   ZCAD -> CAD
#   ZEUR -> EUR
#   SOL  -> SOL          (newer assets keep their natural code)
#
# We normalize to CHAD-canonical codes (BTC, ETH, USD, CAD, ...) so the
# rest of the system never has to know about Kraken's prefixes.

_KRAKEN_ASSET_NORMALIZATION: Dict[str, str] = {
    "XXBT": "BTC",
    "XBT": "BTC",
    "XETH": "ETH",
    "XLTC": "LTC",
    "XXRP": "XRP",
    "XXDG": "DOGE",
    "XDOG": "DOGE",
    "ZUSD": "USD",
    "ZCAD": "CAD",
    "ZEUR": "EUR",
    "ZGBP": "GBP",
    "ZJPY": "JPY",
}


def _normalize_kraken_asset(code: str) -> str:
    if not code:
        return ""
    upper = str(code).strip().upper()
    return _KRAKEN_ASSET_NORMALIZATION.get(upper, upper)


# Symbol that prices are looked up under in the caller-supplied prices map
# for each canonical asset. The advisory and price-cache layers key prices
# by 'BTC-USD', 'ETH-USD', 'SOL-USD'.
_PRICE_LOOKUP_SYMBOL: Dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "LTC": "LTC-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
}

# Approximate FX rates used as a fallback when the caller has no fiat rate
# in the price map. These are intentionally conservative — the advisory
# layer should override with live FX when available.
_FIAT_USD_FALLBACK: Dict[str, float] = {
    "USD": 1.0,
    "CAD": 0.73,
    "EUR": 1.07,
    "GBP": 1.25,
    "JPY": 0.0066,
}


# ---------------------------------------------------------------------------
# Snapshot file
# ---------------------------------------------------------------------------

DEFAULT_SNAPSHOT_PATH = Path("/home/ubuntu/chad_finale/runtime/kraken_balances.json")
DEFAULT_REFRESH_INTERVAL_SECONDS = 300  # 5 minutes


@dataclass
class KrakenBalanceSnapshot:
    ts_utc: str
    ok: bool
    balances: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, str] = field(default_factory=dict)
    usd_equivalent: float = 0.0
    cad_equivalent: float = 0.0
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "ok": self.ok,
            "balances": dict(self.balances),
            "raw": dict(self.raw),
            "usd_equivalent": float(self.usd_equivalent),
            "cad_equivalent": float(self.cad_equivalent),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class KrakenBalanceProvider:
    """
    Read-only Kraken account balance provider.

    Usage
    -----
        provider = KrakenBalanceProvider()
        balances = provider.get_balance()                 # {'BTC': 0.0012, 'CAD': 252.85}
        usd_eq   = provider.get_usd_equivalent(prices)    # converts using prices + FX fallback
        cad_eq   = provider.get_cad_equivalent(prices)    # native CAD (crypto via C-ii constant)
        provider.maybe_refresh_snapshot(prices)           # writes runtime/kraken_balances.json
    """

    def __init__(
        self,
        cfg: Optional[KrakenClientConfig] = None,
        *,
        snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self._cfg = cfg
        self._client: Optional[KrakenClient] = None
        self._snapshot_path = Path(snapshot_path)
        self._refresh_interval = int(refresh_interval_seconds)
        self._last_refresh_ts: float = 0.0
        self._lock = threading.Lock()

    # -- low-level client lazy init -----------------------------------------

    def _get_client(self) -> KrakenClient:
        if self._client is not None:
            return self._client
        cfg = self._cfg or KrakenClientConfig.from_env()
        self._client = KrakenClient(cfg)
        return self._client

    # -- public API ---------------------------------------------------------

    def get_balance(self) -> Dict[str, float]:
        """
        Fetch the account balance and return a dict keyed by CHAD-canonical
        asset code (BTC, ETH, USD, CAD, ...) with float values.

        Returns an empty dict on any error (logged at WARNING).
        """
        try:
            client = self._get_client()
            raw = client.balance() or {}
        except Exception as exc:
            LOGGER.warning("kraken_balance_fetch_failed: %s", exc)
            return {}

        out: Dict[str, float] = {}
        for k, v in raw.items():
            asset = _normalize_kraken_asset(k)
            if not asset:
                continue
            try:
                amount = float(v)
            except (TypeError, ValueError):
                continue
            if amount == 0.0:
                # Skip dust-zero rows but keep tiny non-zero amounts.
                continue
            out[asset] = out.get(asset, 0.0) + amount
        return out

    def get_raw_balance(self) -> Dict[str, str]:
        """Return Kraken's untouched raw response (for audit/debug)."""
        try:
            client = self._get_client()
            raw = client.balance() or {}
            return {str(k): str(v) for k, v in raw.items()}
        except Exception as exc:
            LOGGER.warning("kraken_raw_balance_fetch_failed: %s", exc)
            return {}

    def get_usd_equivalent(self, prices: Mapping[str, float]) -> float:
        """
        Compute total USD-equivalent of all balances.

        prices is a mapping like the runtime price cache:
            {'BTC-USD': 71500.0, 'ETH-USD': 2200.0, 'SOL-USD': 83.0}

        Crypto assets are valued at their <ASSET>-USD price. Fiat assets
        are valued at their <FIAT>-USD FX rate from the prices map (key
        'CAD-USD', 'EUR-USD', etc.) or the conservative fallback table.
        Anything we cannot value is silently dropped from the total but
        is still returned by get_balance() for audit visibility.
        """
        balances = self.get_balance()
        return self._value_balances_in_usd(balances, prices)

    def _value_balances_in_usd(
        self,
        balances: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> float:
        total = 0.0
        for asset, qty in balances.items():
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                continue
            if qty_f == 0.0:
                continue

            # Crypto asset: look up '<ASSET>-USD'
            crypto_key = _PRICE_LOOKUP_SYMBOL.get(asset)
            if crypto_key is not None:
                p = prices.get(crypto_key)
                if p is not None:
                    try:
                        total += qty_f * float(p)
                        continue
                    except (TypeError, ValueError):
                        pass

            # Fiat asset: prefer caller-provided FX, fall back to constant table
            fx_key = f"{asset}-USD"
            fx = prices.get(fx_key)
            if fx is None:
                fx = _FIAT_USD_FALLBACK.get(asset)
            if fx is not None:
                try:
                    total += qty_f * float(fx)
                    continue
                except (TypeError, ValueError):
                    pass

            LOGGER.debug("kraken_balance_unvalued asset=%s qty=%s", asset, qty_f)

        return float(total)

    def get_cad_equivalent(self, prices: Mapping[str, float]) -> float:
        """
        Compute total CAD-equivalent of all balances (the base currency of the
        Kraken account, which is natively CAD).

        Native CAD cash is valued 1:1 with NO FX round-trip. The crypto sliver
        (BTC/ETH/SOL/...) is valued at its <ASSET>-USD price converted into CAD
        via the sanctioned USDCAD_CONVERSION_CONSTANT (the live USD.CAD feed is
        dark on the paper account; CHAD does not trade forex). This is the
        currency-honest rollup for the CAD base — it never values CAD in USD and
        never depends on the dead live FX feed.
        """
        balances = self.get_balance()
        return self._value_balances_in_cad(balances, prices)

    def _value_balances_in_cad(
        self,
        balances: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> float:
        total = 0.0
        for asset, qty in balances.items():
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                continue
            if qty_f == 0.0:
                continue

            # Native CAD cash: valued 1:1, NO FX.
            if asset == "CAD":
                total += qty_f * 1.0
                continue

            # Crypto asset: '<ASSET>-USD' price * USDCAD constant -> CAD.
            crypto_key = _PRICE_LOOKUP_SYMBOL.get(asset)
            if crypto_key is not None:
                p = prices.get(crypto_key)
                if p is not None:
                    try:
                        total += qty_f * float(p) * USDCAD_CONVERSION_CONSTANT
                        continue
                    except (TypeError, ValueError):
                        pass

            # Other fiat (USD/EUR/...): USD-per-fiat * USDCAD constant -> CAD.
            # (These legs are ~never present; handled for completeness.)
            fx_key = f"{asset}-USD"
            fx = prices.get(fx_key)
            if fx is None:
                fx = _FIAT_USD_FALLBACK.get(asset)
            if fx is not None:
                try:
                    total += qty_f * float(fx) * USDCAD_CONVERSION_CONSTANT
                    continue
                except (TypeError, ValueError):
                    pass

            LOGGER.debug("kraken_balance_unvalued_cad asset=%s qty=%s", asset, qty_f)

        return float(total)

    # -- snapshot writer ----------------------------------------------------

    def maybe_refresh_snapshot(
        self,
        prices: Mapping[str, float],
        *,
        force: bool = False,
    ) -> Optional[KrakenBalanceSnapshot]:
        """
        Refresh the on-disk snapshot if `refresh_interval_seconds` has
        elapsed since the last refresh (or always when force=True).

        Returns the snapshot that was written, or None if no refresh
        was due. Safe to call from any cycle — does not raise.
        """
        with self._lock:
            now = time.time()
            if (not force) and (now - self._last_refresh_ts) < self._refresh_interval:
                return None
            snap = self._build_snapshot(prices)
            self._write_snapshot(snap)
            self._last_refresh_ts = now
            return snap

    def write_snapshot(self, prices: Mapping[str, float]) -> KrakenBalanceSnapshot:
        """Force a snapshot write regardless of interval. Returns the snapshot."""
        with self._lock:
            snap = self._build_snapshot(prices)
            self._write_snapshot(snap)
            self._last_refresh_ts = time.time()
            return snap

    def _build_snapshot(self, prices: Mapping[str, float]) -> KrakenBalanceSnapshot:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            raw = self.get_raw_balance()
            balances: Dict[str, float] = {}
            for k, v in raw.items():
                asset = _normalize_kraken_asset(k)
                if not asset:
                    continue
                try:
                    amount = float(v)
                except (TypeError, ValueError):
                    continue
                if amount == 0.0:
                    continue
                balances[asset] = balances.get(asset, 0.0) + amount
            usd_eq = self._value_balances_in_usd(balances, prices)
            cad_eq = self._value_balances_in_cad(balances, prices)
            ok = bool(raw)
            err = None if ok else "empty_balance_response"
            return KrakenBalanceSnapshot(
                ts_utc=ts, ok=ok, balances=balances, raw=raw,
                usd_equivalent=usd_eq, cad_equivalent=cad_eq, error=err
            )
        except Exception as exc:
            return KrakenBalanceSnapshot(
                ts_utc=ts, ok=False, balances={}, raw={},
                usd_equivalent=0.0, cad_equivalent=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _write_snapshot(self, snap: KrakenBalanceSnapshot) -> None:
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._snapshot_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snap.to_json(), indent=2, default=str), encoding="utf-8")
            tmp.replace(self._snapshot_path)
        except Exception as exc:
            LOGGER.warning("kraken_snapshot_write_failed: %s", exc)


def load_latest_snapshot(
    path: Path = DEFAULT_SNAPSHOT_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Helper for advisory/report consumers: load the most recent snapshot
    from disk. Returns None if missing or unreadable.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


__all__ = [
    "KrakenBalanceProvider",
    "KrakenBalanceSnapshot",
    "DEFAULT_SNAPSHOT_PATH",
    "DEFAULT_REFRESH_INTERVAL_SECONDS",
    "load_latest_snapshot",
]
