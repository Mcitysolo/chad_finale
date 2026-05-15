from __future__ import annotations

"""
CHAD — Crypto Derivatives Intelligence Publisher (Phase B Item 4)

Fetches Kraken Futures public derivatives data (no auth) and publishes a
snapshot of funding rates, open interest, and crowding state to
``runtime/crypto_derivatives.json``.

Design principle
----------------
- Additive, fail-open, confidence modifier only.
- Public endpoint only — no credentials required.
- stdlib only — no broker, execution, or strategy imports.
- Atomic write — never partially overwrite a good runtime file.
- ``--dry-run`` prints JSON to stdout and writes nothing.
- On fetch failure the existing runtime file is preserved (unless --dry-run,
  which simply prints the error payload to stdout).

Endpoint
--------
``https://futures.kraken.com/derivatives/api/v3/tickers``

Symbol map
----------
Kraken perpetual contract symbols are mapped to CHAD canonical USD symbols:

    PF_XBTUSD -> BTC-USD
    PF_ETHUSD -> ETH-USD
    PF_SOLUSD -> SOL-USD
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

SCHEMA_VERSION = "crypto_derivatives.v1"
DEFAULT_TTL_SECONDS = 600
FUNDING_HIGH_THRESHOLD = 0.0001
FUNDING_EXTREME_THRESHOLD = 0.0003

_DEFAULT_ENDPOINT = "https://futures.kraken.com/derivatives/api/v3/tickers"
_DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
_DEFAULT_TIMEOUT_SECONDS = 10
_USER_AGENT = "CHAD/1.0 (crypto-derivatives-publisher)"

_PERP_SYMBOL_MAP: Dict[str, str] = {
    "PF_XBTUSD": "BTC-USD",
    "PF_ETHUSD": "ETH-USD",
    "PF_SOLUSD": "SOL-USD",
}


# ---------------------------------------------------------------------------
# Fetch layer (injectable)
# ---------------------------------------------------------------------------

def _default_fetch_tickers(
    endpoint: str = _DEFAULT_ENDPOINT,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> List[Mapping[str, Any]]:
    """Fetch the Kraken Futures tickers payload via the public endpoint."""
    req = urllib.request.Request(endpoint, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    tickers = data.get("tickers") if isinstance(data, dict) else None
    if not isinstance(tickers, list):
        return []
    return [t for t in tickers if isinstance(t, Mapping)]


# ---------------------------------------------------------------------------
# Pure transform layer (testable without network)
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Coerce to float; return None on missing/invalid."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN guard
        return None
    return v


def _classify_market_bias(funding_rate_8h: Optional[float]) -> str:
    if funding_rate_8h is None:
        return "unknown"
    if funding_rate_8h > FUNDING_EXTREME_THRESHOLD:
        return "long_crowded"
    if funding_rate_8h < -FUNDING_EXTREME_THRESHOLD:
        return "short_crowded"
    if funding_rate_8h > FUNDING_HIGH_THRESHOLD:
        return "long_leaning"
    return "balanced"


def _prior_open_interest_usd(prior_path: Path) -> Dict[str, float]:
    """Load prior open_interest_usd values from an existing runtime payload."""
    out: Dict[str, float] = {}
    try:
        doc = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    symbols = doc.get("symbols") if isinstance(doc, dict) else None
    if not isinstance(symbols, dict):
        return out
    for sym, entry in symbols.items():
        if not isinstance(entry, Mapping):
            continue
        oi = _safe_float(entry.get("open_interest_usd"))
        if oi is not None and oi > 0:
            out[str(sym)] = oi
    return out


def _build_symbol_entry(
    canonical: str,
    kraken_symbol: str,
    ticker: Optional[Mapping[str, Any]],
    prior_oi_usd_map: Mapping[str, float],
) -> Dict[str, Any]:
    if ticker is None:
        return {
            "kraken_symbol": kraken_symbol,
            "funding_rate_8h": None,
            "funding_rate_annualized": None,
            "open_interest_contracts": None,
            "open_interest_usd": None,
            "oi_change_pct": None,
            "vol_24h": None,
            "index_price": None,
            "bid": None,
            "ask": None,
            "last": None,
            "funding_extreme_long": False,
            "funding_extreme_short": False,
            "funding_elevated_long": False,
            "market_bias": "unknown",
            "data_available": False,
        }

    funding_rate_8h = _safe_float(ticker.get("fundingRate"))
    funding_rate_annualized: Optional[float]
    if funding_rate_8h is None:
        funding_rate_annualized = None
    else:
        funding_rate_annualized = funding_rate_8h * 3 * 365

    open_interest_contracts = _safe_float(ticker.get("openInterest"))
    index_price = _safe_float(ticker.get("indexPrice"))
    if index_price is None:
        index_price = _safe_float(ticker.get("markPrice"))
    if index_price is None:
        index_price = _safe_float(ticker.get("last"))

    open_interest_usd: Optional[float]
    if open_interest_contracts is None or index_price is None:
        open_interest_usd = None
    else:
        open_interest_usd = open_interest_contracts * index_price

    oi_change_pct: Optional[float]
    prior = prior_oi_usd_map.get(canonical)
    if (
        prior is not None
        and prior > 0
        and open_interest_usd is not None
    ):
        oi_change_pct = (open_interest_usd - prior) / prior
    else:
        oi_change_pct = None

    vol_24h = _safe_float(ticker.get("vol24h"))
    bid = _safe_float(ticker.get("bid"))
    ask = _safe_float(ticker.get("ask"))
    last = _safe_float(ticker.get("last"))

    bias = _classify_market_bias(funding_rate_8h)
    funding_extreme_long = bias == "long_crowded"
    funding_extreme_short = bias == "short_crowded"
    funding_elevated_long = bias == "long_leaning"

    return {
        "kraken_symbol": kraken_symbol,
        "funding_rate_8h": funding_rate_8h,
        "funding_rate_annualized": funding_rate_annualized,
        "open_interest_contracts": open_interest_contracts,
        "open_interest_usd": open_interest_usd,
        "oi_change_pct": oi_change_pct,
        "vol_24h": vol_24h,
        "index_price": index_price,
        "bid": bid,
        "ask": ask,
        "last": last,
        "funding_extreme_long": funding_extreme_long,
        "funding_extreme_short": funding_extreme_short,
        "funding_elevated_long": funding_elevated_long,
        "market_bias": bias,
        "data_available": True,
    }


def build_payload(
    tickers: List[Mapping[str, Any]],
    *,
    prior_oi_usd_map: Optional[Mapping[str, float]] = None,
    now_utc: Optional[datetime] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    fetch_ok: bool = True,
) -> Dict[str, Any]:
    """Build the full runtime payload from a list of Kraken Futures tickers.

    ``prior_oi_usd_map`` is used to compute ``oi_change_pct`` per symbol.
    ``fetch_ok=False`` forces ``status=error`` regardless of ticker contents.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if prior_oi_usd_map is None:
        prior_oi_usd_map = {}

    by_kraken: Dict[str, Mapping[str, Any]] = {}
    for t in tickers:
        sym = t.get("symbol")
        if isinstance(sym, str) and sym in _PERP_SYMBOL_MAP:
            by_kraken[sym] = t

    symbols: Dict[str, Dict[str, Any]] = {}
    available_count = 0
    long_crowded = 0
    short_crowded = 0
    for kraken_symbol, canonical in _PERP_SYMBOL_MAP.items():
        ticker = by_kraken.get(kraken_symbol)
        entry = _build_symbol_entry(canonical, kraken_symbol, ticker, prior_oi_usd_map)
        if entry["data_available"]:
            available_count += 1
            if entry["market_bias"] == "long_crowded":
                long_crowded += 1
            elif entry["market_bias"] == "short_crowded":
                short_crowded += 1
        symbols[canonical] = entry

    total_mapped = len(_PERP_SYMBOL_MAP)
    if not fetch_ok or available_count == 0:
        provider_status = "error"
        status = "error"
    elif available_count == total_mapped:
        provider_status = "real"
        status = "ok"
    else:
        provider_status = "partial"
        status = "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_seconds": int(ttl_seconds),
        "source": {
            "provider": "kraken_futures_public",
            "endpoint": "tickers",
            "provider_status": provider_status,
        },
        "status": status,
        "symbols": symbols,
        "summary": {
            "symbols_fetched": available_count,
            "long_crowded_count": long_crowded,
            "short_crowded_count": short_crowded,
        },
    }


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public publish entry point (used by CLI and tests)
# ---------------------------------------------------------------------------

def publish(
    *,
    runtime_dir: Path = _DEFAULT_RUNTIME_DIR,
    dry_run: bool = False,
    fetcher: Optional[Callable[[], List[Mapping[str, Any]]]] = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Fetch, build, and (optionally) write the payload.

    Returns ``(payload, written)``. ``written`` is False on dry-run or when
    a fetch error occurs and an existing runtime file is preserved.
    """
    target = runtime_dir / "crypto_derivatives.json"
    prior_map = _prior_open_interest_usd(target)

    fetch_ok = True
    tickers: List[Mapping[str, Any]]
    try:
        if fetcher is None:
            tickers = _default_fetch_tickers()
        else:
            tickers = fetcher()
    except Exception:
        fetch_ok = False
        tickers = []

    payload = build_payload(
        tickers,
        prior_oi_usd_map=prior_map,
        now_utc=now_utc,
        fetch_ok=fetch_ok,
    )

    if dry_run:
        return payload, False

    if not fetch_ok and target.exists():
        # Fail-open: keep last good snapshot.
        return payload, False

    _atomic_write_json(target, payload)
    return payload, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CHAD crypto derivatives publisher (Kraken Futures public).",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=_DEFAULT_RUNTIME_DIR,
        help="Directory to write crypto_derivatives.json into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout without writing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        payload, written = publish(
            runtime_dir=args.runtime_dir,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        # Fail-open: report failure on stderr and emit an error payload to
        # stdout so callers piping through jq still see something coherent.
        err_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        sys.stderr.write(f"crypto_derivatives_publisher: {err_payload['error']}\n")
        sys.stdout.write(json.dumps(err_payload))
        sys.stdout.write("\n")
        return 1

    if args.dry_run:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return 0

    if not written:
        sys.stderr.write(
            "crypto_derivatives_publisher: fetch failed, prior snapshot preserved\n"
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
