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
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CACHE_PATH = RUNTIME_DIR / "options_chains_cache.json"
BARS_1D_DIR = REPO_ROOT / "data" / "bars" / "1d"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 88
CONNECT_TIMEOUT_SEC = 20
DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC = 30
OPTIONS_CHAIN_TIMEOUT_ENV = "CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS"
# PR-04: bounded retry/backoff. The headline 2026-05-25 failure ("SPY=
# timeout_after_30.0s") was a single-attempt run that gave up on the first
# slow IBKR farm response. Multiple short retries with backoff lets us
# absorb transient stalls without changing the failure-loud contract:
# every attempt is logged, and when *every* attempt fails we still write
# the empty-cache-with-error-field that R17/R18 already monitor.
OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV = "CHAD_OPTIONS_CHAIN_REFRESH_ATTEMPTS"
OPTIONS_CHAIN_REFRESH_BACKOFF_ENV = "CHAD_OPTIONS_CHAIN_REFRESH_BACKOFF_SECONDS"
DEFAULT_REFRESH_ATTEMPTS = 3
DEFAULT_REFRESH_BACKOFF_SECONDS = 5.0
MAX_REFRESH_ATTEMPTS = 10  # safety cap
FAILURE_ARTIFACT_NAME = "options_chain_refresh_failure.json"
FAILURE_ARTIFACT_SCHEMA = "options_chain_refresh_failure.v1"
SERVICE_ENTRYPOINT = "python -m chad.market_data.options_chain_refresh"
PRICE_PCT_WINDOW = 0.10
# Phase B Item 6: cache schema version. v1 (implicit) = expirations + strikes
# only. v2 adds per-symbol spot_price so the synthetic Greeks publisher can
# avoid re-fetching the underlying. The shape of existing fields is preserved
# verbatim — v1 readers ignore unknown keys and continue working.
OPTIONS_CHAIN_CACHE_SCHEMA = "options_chain_cache.v2"
# alpha_options.AlphaOptionsTuning targets DTE 21-45. With MAX_EXPIRIES=2 the
# refresher only ever kept the two nearest (weekly/daily) expirations, which
# never intersect the 21-45 window, so alpha_options could never build a spread.
# 20 covers ~3 months of weeklies + quarterlies, comfortably straddling 21-45.
MAX_EXPIRIES = 20
# Ignore expirations more than ~90 days out — keeps the cache bounded and
# excludes LEAPs that alpha_options will never select.
MAX_EXPIRY_DTE = 90
CACHE_TTL_SECONDS = 3600


class OptionsChainTimeoutError(TimeoutError):
    """Raised when an IBKR options-chain metadata fetch exceeds its bound."""

    def __init__(self, *, symbol: str, timeout_seconds: float) -> None:
        super().__init__(
            f"options chain refresh timed out for {symbol} "
            f"after {timeout_seconds}s"
        )
        self.symbol = symbol
        self.timeout_seconds = float(timeout_seconds)


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


def _resolve_int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    """Read a positive bounded int env var, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        _log(
            f"WARNING env_invalid name={name} value={raw!r} reason=non_numeric "
            f"— falling back to default {default}"
        )
        return default
    if v < lo or v > hi:
        _log(
            f"WARNING env_invalid name={name} value={raw!r} reason=out_of_range "
            f"[{lo},{hi}] — falling back to default {default}"
        )
        return default
    return v


def _resolve_float_env(
    name: str, default: float, *, lo: float, hi: float
) -> float:
    """Read a positive finite float env var, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _log(
            f"WARNING env_invalid name={name} value={raw!r} reason=non_numeric "
            f"— falling back to default {default}"
        )
        return default
    if not math.isfinite(v) or v < lo or v > hi:
        _log(
            f"WARNING env_invalid name={name} value={raw!r} "
            f"reason=non_finite_or_out_of_range — falling back to default {default}"
        )
        return default
    return v


def _resolve_refresh_attempts() -> int:
    return _resolve_int_env(
        OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV,
        DEFAULT_REFRESH_ATTEMPTS,
        lo=1,
        hi=MAX_REFRESH_ATTEMPTS,
    )


def _resolve_refresh_backoff_seconds() -> float:
    return _resolve_float_env(
        OPTIONS_CHAIN_REFRESH_BACKOFF_ENV,
        DEFAULT_REFRESH_BACKOFF_SECONDS,
        lo=0.0,
        hi=600.0,
    )


def _resolve_contract_details_timeout() -> float:
    """
    Resolve the per-call timeout (seconds) for the IBKR options-chain
    metadata fetch. Honors ``CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS`` and
    falls back safely to ``DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC`` when
    the env var is unset, empty, non-numeric, non-finite, or non-positive.
    """
    default = float(DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC)
    raw = os.environ.get(OPTIONS_CHAIN_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        _log(
            f"WARNING options_chain_timeout_invalid env={OPTIONS_CHAIN_TIMEOUT_ENV} "
            f"value={raw!r} reason=non_numeric — falling back to default {default}s"
        )
        return default
    if not math.isfinite(v) or v <= 0:
        _log(
            f"WARNING options_chain_timeout_invalid env={OPTIONS_CHAIN_TIMEOUT_ENV} "
            f"value={raw!r} reason=non_positive_or_non_finite — "
            f"falling back to default {default}s"
        )
        return default
    return v


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
    from ib_async import Stock

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
    from ib_async import Option

    template = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth="",
        strike=0.0,
        right="",
        exchange="SMART",
        currency="USD",
    )

    # ISSUE-50 fix: bound the IBKR options-chain metadata fetch with a
    # configurable timeout (CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS, default
    # 30s). The synchronous ib.reqContractDetails call has no internal
    # timeout and was hanging the whole refresh service when IBKR
    # Gateway / data servers stalled. On timeout we raise a typed error
    # so the caller can log a WARNING and write a valid empty/error
    # cache artifact instead of leaving the service hung.
    timeout_sec = _resolve_contract_details_timeout()
    try:
        details = ib.run(
            asyncio.wait_for(
                ib.reqContractDetailsAsync(template),
                timeout=timeout_sec,
            )
        )
    except asyncio.TimeoutError as exc:
        _log(
            f"WARNING options_chain_reqcontractdetails_timeout symbol={symbol} "
            f"timeout={timeout_sec}s — IBKR data servers unresponsive, aborting symbol"
        )
        raise OptionsChainTimeoutError(
            symbol=symbol, timeout_seconds=timeout_sec
        ) from exc
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

    # Phase B Item 6 (v2): persist the spot used for strike-window filtering
    # so the downstream Greeks publisher can compute synthetic deltas/theos
    # without re-fetching the underlying. Additive; older readers ignore it.
    return {
        "symbol": symbol,
        "exchange": exchange,
        "expirations": expirations,
        "strikes": strikes,
        "spot_price": float(spot) if spot and spot > 0 else None,
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": CACHE_TTL_SECONDS,
    }


def _read_last_successful_ts(cache_path: Path) -> Optional[str]:
    """Best-effort read of the prior cache's ``ts_utc`` for failure lineage."""
    try:
        if not cache_path.is_file():
            return None
        d = json.loads(cache_path.read_text(encoding="utf-8"))
        chains = d.get("chains") if isinstance(d, dict) else None
        ts = d.get("ts_utc") if isinstance(d, dict) else None
        if isinstance(chains, dict) and chains and isinstance(ts, str):
            return ts
    except Exception:
        return None
    return None


def _classify_blocked_reason(symbol_errors: Dict[str, str]) -> str:
    """Infer the operator-facing blocked_reason from collected per-symbol
    errors. The label is informational — health-monitor rules R17/R18 do
    not branch on it — but it gives operators an immediate read on whether
    the failure is local code, IBKR farm degradation, or contract data.
    """
    if not symbol_errors:
        return "unknown"
    reasons = list(symbol_errors.values())
    if all("timeout_after" in r for r in reasons):
        return "ibkr_contract_details_unresponsive"
    if all("could not resolve spot price" in r for r in reasons):
        return "ibkr_spot_unavailable"
    if all("qualifyContracts" in r for r in reasons):
        return "ibkr_qualify_contracts_failed"
    return "mixed_failures"


def _write_failure_artifact(
    *,
    symbol_errors: Dict[str, str],
    symbol_attempts: Dict[str, List[Dict[str, Any]]],
    max_attempts: int,
    timeout_seconds: float,
    backoff_seconds: float,
    last_successful_ts: Optional[str] = None,
) -> None:
    """Emit runtime/options_chain_refresh_failure.json (Pattern E).

    ``last_successful_ts`` should be captured by the caller BEFORE the empty
    cache is written; if omitted we still try to recover it from disk.
    """
    if last_successful_ts is None:
        last_successful_ts = _read_last_successful_ts(CACHE_PATH)
    blocked_reason = _classify_blocked_reason(symbol_errors)
    payload: Dict[str, Any] = {
        "schema_version": FAILURE_ARTIFACT_SCHEMA,
        "ts_utc": _utc_now_iso(),
        "status": "failed",
        "provider": "ibkr_contract_details",
        "service_entrypoint": SERVICE_ENTRYPOINT,
        "max_attempts": int(max_attempts),
        "per_call_timeout_seconds": float(timeout_seconds),
        "backoff_seconds": float(backoff_seconds),
        "symbol_errors": dict(symbol_errors),
        "attempts": {k: list(v) for k, v in symbol_attempts.items()},
        "error_type": "all_symbols_failed",
        "error_message": "all_symbols_failed: "
        + "; ".join(f"{s}={r}" for s, r in symbol_errors.items()),
        "last_successful_ts": last_successful_ts,
        "blocked_reason": blocked_reason,
    }
    fp = RUNTIME_DIR / FAILURE_ARTIFACT_NAME
    _atomic_write(fp, payload)
    _log(
        f"failure_artifact_written path={fp} "
        f"blocked_reason={blocked_reason} last_successful_ts={last_successful_ts}"
    )


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
        from ib_async import IB
    except Exception as exc:
        _log(f"ERROR ib_async import failed: {exc}")
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
        symbol_errors: Dict[str, str] = {}
        symbol_attempts: Dict[str, List[Dict[str, Any]]] = {}
        success = 0
        failure = 0

        max_attempts = _resolve_refresh_attempts()
        backoff_seconds = _resolve_refresh_backoff_seconds()
        timeout_seconds = _resolve_contract_details_timeout()
        _log(
            f"refresh_config attempts={max_attempts} "
            f"backoff_seconds={backoff_seconds} "
            f"per_call_timeout_seconds={timeout_seconds}"
        )

        for raw_sym in symbols:
            sym = raw_sym.strip().upper()
            if not sym:
                continue
            attempts_log: List[Dict[str, Any]] = []
            symbol_attempts[sym] = attempts_log
            last_error: str = ""
            chain: Optional[Dict[str, Any]] = None  # type: ignore[assignment]
            for attempt_idx in range(1, max_attempts + 1):
                attempt_ts = _utc_now_iso()
                try:
                    chain = _refresh_symbol(ib, sym)
                    attempts_log.append(
                        {
                            "attempt": attempt_idx,
                            "ts_utc": attempt_ts,
                            "result": "success",
                            "strikes": len(chain.get("strikes") or []),
                            "expirations": len(chain.get("expirations") or []),
                        }
                    )
                    _log(
                        f"{sym}: attempt={attempt_idx}/{max_attempts} "
                        f"result=success strikes={len(chain['strikes'])} "
                        f"expiries={len(chain['expirations'])}"
                    )
                    break
                except OptionsChainTimeoutError as exc:
                    last_error = f"timeout_after_{exc.timeout_seconds}s"
                    attempts_log.append(
                        {
                            "attempt": attempt_idx,
                            "ts_utc": attempt_ts,
                            "result": "timeout",
                            "error_type": "OptionsChainTimeoutError",
                            "timeout_seconds": float(exc.timeout_seconds),
                        }
                    )
                    _log(
                        f"{sym}: attempt={attempt_idx}/{max_attempts} "
                        f"result=timeout err={exc}"
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempts_log.append(
                        {
                            "attempt": attempt_idx,
                            "ts_utc": attempt_ts,
                            "result": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    _log(
                        f"{sym}: attempt={attempt_idx}/{max_attempts} "
                        f"result=error err={exc}"
                    )
                # Backoff before next attempt (skip after the last).
                if attempt_idx < max_attempts and backoff_seconds > 0:
                    _log(
                        f"{sym}: backing off {backoff_seconds}s before "
                        f"attempt {attempt_idx + 1}/{max_attempts}"
                    )
                    try:
                        ib.sleep(backoff_seconds)
                    except Exception:
                        # Fall back to plain wait if the IB stub lacks sleep.
                        import time as _time

                        _time.sleep(backoff_seconds)

            if chain is not None:
                chains[sym] = chain
                success += 1
            else:
                symbol_errors[sym] = last_error or "no_attempts_succeeded"
                _log(
                    f"{sym}: FAILED after attempts={max_attempts} "
                    f"last_error={last_error!r}"
                )
                failure += 1

        if not chains:
            # Snapshot the prior cache's ts_utc BEFORE we overwrite it, so the
            # failure artifact can quote the last successful refresh.
            last_successful_ts = _read_last_successful_ts(CACHE_PATH)
            error_msg = (
                "all_symbols_failed: "
                + (
                    "; ".join(
                        f"{s}={r}" for s, r in symbol_errors.items()
                    )
                    or "no_symbols_processed"
                )
            )
            cache_doc = {
                "schema_version": OPTIONS_CHAIN_CACHE_SCHEMA,
                "ts_utc": _utc_now_iso(),
                "chains": {},
                "error": error_msg,
            }
            try:
                _atomic_write(CACHE_PATH, cache_doc)
                _log(
                    f"WARNING wrote empty options_chains_cache with error field: "
                    f"{error_msg}"
                )
            except Exception as exc:
                _log(
                    f"ERROR atomic write of empty cache failed path={CACHE_PATH} "
                    f"err={exc}"
                )
            # Pattern E — emit a structured failure artifact so operators see
            # attempts, blocked_reason, and the last_successful_ts of the
            # prior chain without parsing the cache error string.
            try:
                _write_failure_artifact(
                    symbol_errors=symbol_errors,
                    symbol_attempts=symbol_attempts,
                    max_attempts=max_attempts,
                    timeout_seconds=timeout_seconds,
                    backoff_seconds=backoff_seconds,
                    last_successful_ts=last_successful_ts,
                )
            except Exception as exc:
                _log(
                    f"ERROR failure-artifact write failed path={FAILURE_ARTIFACT_NAME} "
                    f"err={exc}"
                )
            _log(
                f"SUMMARY success={success} failure={failure} "
                f"(empty cache + failure artifact written)"
            )
            return 1

        cache_doc = {
            "schema_version": OPTIONS_CHAIN_CACHE_SCHEMA,
            "ts_utc": _utc_now_iso(),
            "chains": chains,
        }
        if symbol_errors:
            cache_doc["error"] = "partial: " + "; ".join(
                f"{s}={r}" for s, r in symbol_errors.items()
            )

        try:
            _atomic_write(CACHE_PATH, cache_doc)
        except Exception as exc:
            _log(f"ERROR atomic write failed path={CACHE_PATH} err={exc}")
            return 1

        # When the run is healthy (no symbol_errors at all) clear any prior
        # failure artifact so the operator surface reflects current truth.
        if not symbol_errors:
            try:
                fp = RUNTIME_DIR / FAILURE_ARTIFACT_NAME
                if fp.is_file():
                    fp.unlink()
            except Exception:
                pass

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
