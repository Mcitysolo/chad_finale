from __future__ import annotations

"""
CHAD — Kraken Futures Public Intelligence Publisher (Phase C Item 1A).

Publishes a broad snapshot of Kraken Futures perpetual contract intelligence
(funding rates, open interest, prices, volume, crowding bias) sourced
exclusively from public Kraken Futures endpoints:

    https://futures.kraken.com/derivatives/api/v3/tickers
    https://futures.kraken.com/derivatives/api/v4/historicalfundingrates

Design principles
-----------------
- Read-only intelligence. No trading, no order placement, no execution imports.
- No private Kraken credentials. Public endpoints only.
- stdlib only. No broker, strategy, or execution module imports.
- Atomic write via tmp.<pid> + os.replace.
- Fail-open: on fetch error preserve the existing runtime file (unless --dry-run).
- Phase B ``crypto_derivatives.json`` covers BTC/ETH/SOL strategy-facing feed.
  This publisher covers the broader perpetual universe (~300 contracts) at
  ``runtime/kraken_futures_intel.json`` for intelligence/filtering.
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "kraken_futures_intel.v1"
DEFAULT_TTL_SECONDS = 600
KRAKEN_FUTURES_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
KRAKEN_FUTURES_HISTORICAL_FUNDING_URL = (
    "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
)

FUNDING_HIGH_THRESHOLD = 0.0001
FUNDING_EXTREME_THRESHOLD = 0.0003

_DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
_DEFAULT_TIMEOUT_SECONDS = 10
_USER_AGENT = "CHAD/1.0 (kraken-futures-intel-publisher)"
_DEFAULT_HISTORY_SYMBOLS = ("PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD")
_TOP_N = 10

_CANONICAL_PERP_MAP: Dict[str, str] = {
    "PF_XBTUSD": "BTC-USD",
    "PF_ETHUSD": "ETH-USD",
    "PF_SOLUSD": "SOL-USD",
}

_KRAKEN_BASE_NORMALIZATION: Dict[str, str] = {
    "XBT": "BTC",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_utc_now(now_utc: Optional[datetime] = None) -> str:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN guard
        return None
    return v


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    return None


def _fetch_json(url: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Fetch layer (injectable)
# ---------------------------------------------------------------------------

def fetch_tickers() -> List[Mapping[str, Any]]:
    data = _fetch_json(KRAKEN_FUTURES_TICKERS_URL)
    tickers = data.get("tickers") if isinstance(data, dict) else None
    if not isinstance(tickers, list):
        return []
    return [t for t in tickers if isinstance(t, Mapping)]


def fetch_historical_funding(symbol: str) -> Dict[str, Any]:
    qs = urllib.parse.urlencode({"symbol": symbol})
    url = f"{KRAKEN_FUTURES_HISTORICAL_FUNDING_URL}?{qs}"
    data = _fetch_json(url)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Symbol parsing
# ---------------------------------------------------------------------------

def map_perp_symbol(kraken_symbol: str) -> Optional[str]:
    if not isinstance(kraken_symbol, str):
        return None
    return _CANONICAL_PERP_MAP.get(kraken_symbol)


def parse_base_quote(kraken_symbol: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(kraken_symbol, str) or not kraken_symbol.startswith("PF_"):
        return (None, None)
    body = kraken_symbol[3:]
    # Kraken perps consistently end in a fiat / stable quote (USD or USDT).
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if body.endswith(quote) and len(body) > len(quote):
            base_raw = body[: -len(quote)]
            base = _KRAKEN_BASE_NORMALIZATION.get(base_raw, base_raw)
            return (base, quote)
    return (None, None)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_market_bias(funding_rate_8h: Optional[float]) -> str:
    if funding_rate_8h is None:
        return "unknown"
    if funding_rate_8h > FUNDING_EXTREME_THRESHOLD:
        return "long_crowded"
    if funding_rate_8h < -FUNDING_EXTREME_THRESHOLD:
        return "short_crowded"
    if funding_rate_8h > FUNDING_HIGH_THRESHOLD:
        return "long_leaning"
    return "balanced"


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_symbol_record(ticker: Mapping[str, Any]) -> Dict[str, Any]:
    kraken_symbol = ticker.get("symbol") if isinstance(ticker, Mapping) else None
    if not isinstance(kraken_symbol, str):
        kraken_symbol = ""

    mapped_symbol = map_perp_symbol(kraken_symbol)
    base, quote = parse_base_quote(kraken_symbol)

    funding_rate_8h = _safe_float(ticker.get("fundingRate"))
    funding_rate_prediction = _safe_float(ticker.get("fundingRatePrediction"))
    funding_rate_annualized: Optional[float]
    if funding_rate_8h is None:
        funding_rate_annualized = None
    else:
        funding_rate_annualized = funding_rate_8h * 3.0 * 365.0

    open_interest_contracts = _safe_float(ticker.get("openInterest"))
    index_price = _safe_float(ticker.get("indexPrice"))
    mark_price = _safe_float(ticker.get("markPrice"))
    last = _safe_float(ticker.get("last"))
    bid = _safe_float(ticker.get("bid"))
    ask = _safe_float(ticker.get("ask"))

    spread: Optional[float]
    if bid is not None and ask is not None:
        spread = ask - bid
    else:
        spread = None

    price_for_oi = index_price if index_price is not None else mark_price
    if price_for_oi is None:
        price_for_oi = last

    if open_interest_contracts is None or price_for_oi is None:
        open_interest_usd: Optional[float] = None
    else:
        open_interest_usd = open_interest_contracts * price_for_oi

    vol_24h = _safe_float(ticker.get("vol24h"))
    volume_quote_24h = _safe_float(ticker.get("volumeQuote"))
    change_24h = _safe_float(ticker.get("change24h"))
    post_only = _safe_bool(ticker.get("postOnly"))
    suspended = _safe_bool(ticker.get("suspended"))

    bias = classify_market_bias(funding_rate_8h)

    data_available = (
        funding_rate_8h is not None
        or open_interest_contracts is not None
        or last is not None
        or index_price is not None
    )

    return {
        "kraken_symbol": kraken_symbol,
        "mapped_symbol": mapped_symbol,
        "base": base,
        "quote": quote,
        "funding_rate_8h": funding_rate_8h,
        "funding_rate_annualized": funding_rate_annualized,
        "funding_rate_prediction": funding_rate_prediction,
        "open_interest_contracts": open_interest_contracts,
        "open_interest_usd": open_interest_usd,
        "index_price": index_price,
        "mark_price": mark_price,
        "last": last,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "vol_24h": vol_24h,
        "volume_quote_24h": volume_quote_24h,
        "change_24h": change_24h,
        "post_only": post_only,
        "suspended": suspended,
        "market_bias": bias,
        "funding_extreme_long": bias == "long_crowded",
        "funding_extreme_short": bias == "short_crowded",
        "funding_elevated_long": bias == "long_leaning",
        "data_available": data_available,
    }


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _top_n_by(
    symbols: Mapping[str, Mapping[str, Any]],
    field: str,
    n: int = _TOP_N,
) -> List[List[Any]]:
    items: List[Tuple[str, float]] = []
    for sym, rec in symbols.items():
        val = rec.get(field)
        if isinstance(val, (int, float)) and val == val and val > 0:
            items.append((sym, float(val)))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [[k, v] for k, v in items[:n]]


def build_payload(
    tickers: Sequence[Mapping[str, Any]],
    *,
    now_utc: Optional[datetime] = None,
    include_history: bool = False,
    history_symbols: Optional[Sequence[str]] = None,
    history_fetcher: Optional[Callable[[str], Mapping[str, Any]]] = None,
    max_symbols: int = 0,
    fetch_ok: bool = True,
) -> Dict[str, Any]:
    """Build the runtime payload from a list of Kraken Futures tickers.

    Only PF_* perpetual symbols are included.
    """
    perps_total = 0
    symbols: Dict[str, Dict[str, Any]] = {}

    for t in tickers:
        sym = t.get("symbol") if isinstance(t, Mapping) else None
        if not isinstance(sym, str) or not sym.startswith("PF_"):
            continue
        perps_total += 1
        if max_symbols and len(symbols) >= int(max_symbols):
            continue
        symbols[sym] = build_symbol_record(t)

    mapped_symbols: Dict[str, str] = {}
    long_crowded_count = 0
    short_crowded_count = 0
    suspended_count = 0
    post_only_count = 0

    for sym, rec in symbols.items():
        mapped = rec.get("mapped_symbol")
        if isinstance(mapped, str) and mapped:
            mapped_symbols[mapped] = sym
        bias = rec.get("market_bias")
        if bias == "long_crowded":
            long_crowded_count += 1
        elif bias == "short_crowded":
            short_crowded_count += 1
        if rec.get("suspended") is True:
            suspended_count += 1
        if rec.get("post_only") is True:
            post_only_count += 1

    top_open_interest_usd = _top_n_by(symbols, "open_interest_usd")
    top_volume_quote_24h = _top_n_by(symbols, "volume_quote_24h")

    history: Dict[str, Any] = {}
    if include_history:
        history_targets: Sequence[str]
        if history_symbols is None:
            history_targets = _DEFAULT_HISTORY_SYMBOLS
        else:
            history_targets = [s for s in history_symbols if isinstance(s, str) and s]
        for hsym in history_targets:
            if history_fetcher is None:
                continue
            try:
                hdata = history_fetcher(hsym)
            except Exception:
                continue
            if not isinstance(hdata, Mapping):
                continue
            rates = hdata.get("rates")
            if not isinstance(rates, list):
                continue
            latest = rates[-1] if rates else None
            history[hsym] = {
                "rates_count": len(rates),
                "latest": latest if isinstance(latest, Mapping) else None,
            }

    if not fetch_ok or not symbols:
        status = "error"
        provider_status = "error"
    elif perps_total > 0 and len(symbols) >= perps_total:
        status = "ok"
        provider_status = "real"
    else:
        status = "partial"
        provider_status = "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _iso_utc_now(now_utc),
        "ttl_seconds": int(DEFAULT_TTL_SECONDS),
        "status": status,
        "source": {
            "provider": "kraken_futures_public",
            "endpoint": "tickers",
            "provider_status": provider_status,
        },
        "symbols": symbols,
        "mapped_symbols": mapped_symbols,
        "summary": {
            "perps_total": int(perps_total),
            "symbols_published": int(len(symbols)),
            "mapped_symbols_count": int(len(mapped_symbols)),
            "long_crowded_count": int(long_crowded_count),
            "short_crowded_count": int(short_crowded_count),
            "suspended_count": int(suspended_count),
            "post_only_count": int(post_only_count),
            "top_open_interest_usd": top_open_interest_usd,
            "top_volume_quote_24h": top_volume_quote_24h,
        },
        "history": history,
    }


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.{os.getpid()}.",
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
# Publish entry point
# ---------------------------------------------------------------------------

def publish(
    runtime_dir: Path,
    *,
    dry_run: bool = False,
    include_history: bool = False,
    history_symbols: Optional[Sequence[str]] = None,
    max_symbols: int = 0,
    fetcher: Optional[Callable[[], Sequence[Mapping[str, Any]]]] = None,
    history_fetcher: Optional[Callable[[str], Mapping[str, Any]]] = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Fetch, build, and (optionally) write the payload.

    Returns ``(payload, written)``. ``written`` is False on dry-run or when
    fetch fails and an existing runtime file is preserved.
    """
    target = runtime_dir / "kraken_futures_intel.json"

    fetch_ok = True
    tickers: List[Mapping[str, Any]] = []
    try:
        if fetcher is None:
            tickers = list(fetch_tickers())
        else:
            result = fetcher()
            tickers = [t for t in result if isinstance(t, Mapping)]
    except Exception:
        fetch_ok = False
        tickers = []

    effective_history_fetcher: Optional[Callable[[str], Mapping[str, Any]]]
    if include_history:
        effective_history_fetcher = (
            history_fetcher if history_fetcher is not None else fetch_historical_funding
        )
    else:
        effective_history_fetcher = None

    payload = build_payload(
        tickers,
        now_utc=now_utc,
        include_history=include_history,
        history_symbols=history_symbols,
        history_fetcher=effective_history_fetcher,
        max_symbols=max_symbols,
        fetch_ok=fetch_ok,
    )

    if dry_run:
        return payload, False

    if not fetch_ok and target.exists():
        return payload, False

    _atomic_write_json(target, payload)
    return payload, True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CHAD Kraken Futures public intelligence publisher.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=_DEFAULT_RUNTIME_DIR,
        help="Directory to write kraken_futures_intel.json into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout without writing.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Limit number of perp symbols included (0 = all).",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also fetch historical funding for --history-symbols.",
    )
    parser.add_argument(
        "--history-symbols",
        type=str,
        default=",".join(_DEFAULT_HISTORY_SYMBOLS),
        help="Comma-separated list of Kraken perp symbols for history fetch.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    history_symbols = [
        s.strip() for s in str(args.history_symbols or "").split(",") if s.strip()
    ]

    try:
        payload, written = publish(
            runtime_dir=args.runtime_dir,
            dry_run=args.dry_run,
            include_history=bool(args.include_history),
            history_symbols=history_symbols,
            max_symbols=int(args.max_symbols or 0),
        )
    except Exception as exc:
        err_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        sys.stderr.write(f"kraken_futures_intel_publisher: {err_payload['error']}\n")
        sys.stdout.write(json.dumps(err_payload))
        sys.stdout.write("\n")
        return 1

    if args.dry_run:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        sys.stdout.write("\n")
        return 0

    if not written:
        sys.stderr.write(
            "kraken_futures_intel_publisher: fetch failed, prior snapshot preserved\n"
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
