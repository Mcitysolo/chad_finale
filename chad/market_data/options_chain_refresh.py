#!/usr/bin/env python3
"""
chad/market_data/options_chain_refresh.py

Pre-market options chain cache refresh service.

Connects to IBKR Gateway, fetches SPY option contract details via
reqContractDetails (filtered to ~10% of spot, nearest 2 expiries,
both calls and puts), and writes runtime/options_chains_cache.json
in the schema consumed by chad.strategies.alpha_options.

This is the only place in CHAD where reqContractDetails is permitted —
it runs once before market open, off the hot trading path.

CLI:
    python -m chad.market_data.options_chain_refresh
    python -m chad.market_data.options_chain_refresh --symbols SPY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CACHE_PATH = RUNTIME_DIR / "options_chains_cache.json"
BARS_1D_DIR = REPO_ROOT / "data" / "bars" / "1d"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 88
CONNECT_TIMEOUT_SEC = 20
CONTRACT_DETAILS_TIMEOUT_SEC = 30
PRICE_PCT_WINDOW = 0.10
# alpha_options.AlphaOptionsTuning targets DTE 21-45. With MAX_EXPIRIES=2 the
# refresher only ever kept the two nearest (weekly/daily) expirations, which
# never intersect the 21-45 window, so alpha_options could never build a spread.
# 20 covers ~3 months of weeklies + quarterlies, comfortably straddling 21-45.
MAX_EXPIRIES = 20
# Ignore expirations more than ~90 days out — keeps the cache bounded and
# excludes LEAPs that alpha_options will never select.
MAX_EXPIRY_DTE = 90
CACHE_TTL_SECONDS = 3600


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("[%Y-%m-%dT%H:%M:%SZ]")


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


def _spot_price_from_bars(symbol: str) -> float | None:
    """Fallback: read latest close from data/bars/1d/<symbol>.json."""
    try:
        bars_path = BARS_1D_DIR / f"{symbol}.json"
        d = json.loads(bars_path.read_text())
        bars = d.get("bars", [])
        if bars:
            last = bars[-1]
            close = last.get("close") or last.get("c") or last.get("Close")
            if close:
                _log(
                    f"spot_price_from_bars symbol={symbol} price={close} "
                    f"ts={last.get('ts_utc','?')}"
                )
                return float(close)
    except Exception as e:
        _log(f"spot_price_from_bars_failed symbol={symbol} err={e}")
    return None


def _fetch_spy_price(ib: Any, symbol: str) -> float:
    """Fetch current price via market data snapshot, fall back to bar data."""
    from ib_insync import Stock

    stock = Stock(symbol, "SMART", "USD")
    try:
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            raise RuntimeError(f"qualifyContracts failed for {symbol}")
    except Exception as exc:
        _log(f"qualifyContracts_failed symbol={symbol} err={exc}")
        bar_price = _spot_price_from_bars(symbol)
        if bar_price and bar_price > 0:
            return bar_price
        raise

    try:
        ticker = ib.reqMktData(stock, "", False, False)
        ib.sleep(2.0)
    except Exception as exc:
        _log(f"reqMktData_failed symbol={symbol} err={exc}")
        bar_price = _spot_price_from_bars(symbol)
        if bar_price and bar_price > 0:
            return bar_price
        raise

    price = 0.0
    for attr in ("last", "close", "marketPrice"):
        try:
            v = getattr(ticker, attr, None)
            if callable(v):
                v = v()
            if v is not None and float(v) > 0 and float(v) == float(v):
                price = float(v)
                break
        except Exception:
            continue

    if price <= 0:
        midpoint = None
        try:
            bid = float(getattr(ticker, "bid", 0.0) or 0.0)
            ask = float(getattr(ticker, "ask", 0.0) or 0.0)
            if bid > 0 and ask > 0:
                midpoint = (bid + ask) / 2.0
        except Exception:
            midpoint = None
        if midpoint and midpoint > 0:
            price = midpoint

    try:
        ib.cancelMktData(stock)
    except Exception:
        pass

    if price <= 0:
        _log(
            f"live_spot_unavailable symbol={symbol} "
            f"(likely no market data subscription / Error 10089), "
            f"falling back to bar close"
        )
        bar_price = _spot_price_from_bars(symbol)
        if bar_price and bar_price > 0:
            return bar_price
        raise RuntimeError(f"could not resolve spot price for {symbol}")

    return price


def _fetch_chain_via_contract_details(
    ib: Any,
    symbol: str,
    spot: float,
) -> Tuple[List[str], List[float], str]:
    """
    Use reqContractDetails to enumerate SPY option contracts within
    +/- PRICE_PCT_WINDOW of spot, restricted to the nearest MAX_EXPIRIES
    expiries. Returns (expirations, strikes, exchange).
    """
    from ib_insync import Option

    template = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth="",
        strike=0.0,
        right="",
        exchange="SMART",
        currency="USD",
    )

    # ISSUE-50 fix: bound reqContractDetails with the
    # CONTRACT_DETAILS_TIMEOUT_SEC declared above. The synchronous
    # ib.reqContractDetails call has no internal timeout and was
    # hanging the whole refresh service when IBKR Gateway stalled.
    try:
        details = ib.run(
            asyncio.wait_for(
                ib.reqContractDetailsAsync(template),
                timeout=CONTRACT_DETAILS_TIMEOUT_SEC,
            )
        )
    except asyncio.TimeoutError:
        _log(
            f"options_chain_reqcontractdetails_timeout symbol={symbol} "
            f"timeout={CONTRACT_DETAILS_TIMEOUT_SEC}s — skipping"
        )
        return [], [], "SMART"
    if not details:
        raise RuntimeError(f"reqContractDetails returned empty for {symbol}")

    lo = spot * (1.0 - PRICE_PCT_WINDOW)
    hi = spot * (1.0 + PRICE_PCT_WINDOW)

    all_expiries_set = set()
    for det in details:
        c = getattr(det, "contract", None)
        if c is None:
            continue
        exp = getattr(c, "lastTradeDateOrContractMonth", "") or ""
        if exp:
            all_expiries_set.add(str(exp))

    # Cap by DTE so the cache stays bounded even if IBKR returns LEAPs.
    today = datetime.now(timezone.utc).date()
    bounded: List[str] = []
    for exp_str in sorted(all_expiries_set):
        try:
            exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte < 0 or dte > MAX_EXPIRY_DTE:
            continue
        bounded.append(exp_str)
    nearest_expiries = bounded[:MAX_EXPIRIES]
    nearest_set = set(nearest_expiries)

    strikes_set = set()
    exchange = ""
    for det in details:
        c = getattr(det, "contract", None)
        if c is None:
            continue
        exp = str(getattr(c, "lastTradeDateOrContractMonth", "") or "")
        if exp not in nearest_set:
            continue
        right = str(getattr(c, "right", "") or "").upper()
        if right not in ("C", "P"):
            continue
        try:
            strike = float(getattr(c, "strike", 0.0) or 0.0)
        except Exception:
            continue
        if strike <= 0 or strike < lo or strike > hi:
            continue
        strikes_set.add(strike)
        if not exchange:
            exch = str(getattr(c, "exchange", "") or "")
            if exch:
                exchange = exch

    return (
        sorted(nearest_expiries),
        sorted(strikes_set),
        exchange or "SMART",
    )


def _refresh_symbol(ib: Any, symbol: str) -> Dict[str, Any]:
    """Build a single chain dict in the schema alpha_options expects."""
    spot = _fetch_spy_price(ib, symbol)
    expirations, strikes, exchange = _fetch_chain_via_contract_details(
        ib, symbol, spot
    )
    if not expirations or not strikes:
        raise RuntimeError(
            f"empty chain for {symbol}: expirations={len(expirations)} "
            f"strikes={len(strikes)}"
        )

    return {
        "symbol": symbol,
        "exchange": exchange,
        "expirations": expirations,
        "strikes": strikes,
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": CACHE_TTL_SECONDS,
    }


def _atomic_write(cache_path: Path, cache_doc: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(cache_path) + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(cache_doc, indent=2) + chr(10),
        encoding="utf-8",
    )
    os.replace(tmp, cache_path)


def run(symbols: Sequence[str]) -> int:
    try:
        from ib_insync import IB
    except Exception as exc:
        _log(f"ERROR ib_insync import failed: {exc}")
        return 1

    ib = IB()
    try:
        try:
            ib.connect(
                IBKR_HOST,
                IBKR_PORT,
                clientId=88,
                timeout=CONNECT_TIMEOUT_SEC,
                readonly=True,
            )
        except Exception as exc:
            _log(f"ERROR IBKR connect failed host={IBKR_HOST} port={IBKR_PORT} err={exc}")
            return 1

        if not ib.isConnected():
            _log(f"ERROR IBKR not connected after connect()")
            return 1

        def _on_ib_error(reqId, errorCode, errorString, contract):
            if errorCode == 10089:
                _log(
                    f"ibkr_error_10089 reqId={reqId} "
                    f"msg='market data subscription required' — "
                    f"will fall back to bar data for spot price"
                )
            elif errorCode in (2104, 2106, 2158, 2107, 2119, 2100):
                # market data farm connectivity info — benign
                return
            else:
                _log(
                    f"ibkr_error reqId={reqId} code={errorCode} "
                    f"msg={errorString}"
                )

        try:
            ib.errorEvent += _on_ib_error
        except Exception:
            pass

        chains: Dict[str, Any] = {}
        success = 0
        failure = 0

        for raw_sym in symbols:
            sym = raw_sym.strip().upper()
            if not sym:
                continue
            try:
                chain = _refresh_symbol(ib, sym)
                chains[sym] = chain
                _log(
                    f"{sym}: {len(chain['strikes'])} strikes, "
                    f"{len(chain['expirations'])} expiries written"
                )
                success += 1
            except Exception as exc:
                _log(f"{sym}: FAILED err={exc}")
                failure += 1

        if not chains:
            _log(f"SUMMARY success={success} failure={failure} (no chains written)")
            return 1

        cache_doc = {
            "ts_utc": _utc_now_iso(),
            "chains": chains,
        }

        try:
            _atomic_write(CACHE_PATH, cache_doc)
        except Exception as exc:
            _log(f"ERROR atomic write failed path={CACHE_PATH} err={exc}")
            return 1

        _log(f"SUMMARY success={success} failure={failure}")
        return 0 if failure == 0 else 1

    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CHAD pre-market options chain cache refresh"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Underlying symbols to refresh (default: SPY)",
    )
    args = parser.parse_args()

    try:
        return run(args.symbols)
    except Exception as exc:
        _log(f"ERROR unhandled exception: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
