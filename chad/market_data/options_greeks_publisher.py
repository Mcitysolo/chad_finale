#!/usr/bin/env python3
"""
chad/market_data/options_greeks_publisher.py

Phase B Item 6 — synthetic options Greeks publisher (metadata only).

Reads runtime/options_chains_cache.json (schema_version
options_chain_cache.v2), data/bars/1d/VIX.json, and runtime/price_cache.json
and emits runtime/options_greeks.json containing per-symbol /
per-expiration / per-strike synthetic delta + theoretical price estimates.

The output is metadata only — it is read by chad.utils.options_greeks_gate
to annotate alpha_options TradeSignal.meta. No sizing logic, no broker
calls, no IBKR imports, stdlib only. Failure-soft at every layer.

CLI:
    python -m chad.market_data.options_greeks_publisher
    python -m chad.market_data.options_greeks_publisher --dry-run
    python -m chad.market_data.options_greeks_publisher --runtime-dir /tmp/foo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime"
BARS_1D_DIR = REPO_ROOT / "data" / "bars" / "1d"

GREEKS_SCHEMA_VERSION = "options_greeks.v1"
GREEKS_FILE_NAME = "options_greeks.json"
CHAIN_CACHE_FILE_NAME = "options_chains_cache.json"
PRICE_CACHE_FILE_NAME = "price_cache.json"
DEFAULT_TTL_SECONDS = 3600

MAX_EXPIRIES_PER_SYMBOL = 3
NEAR_ATM_STRIKES_PER_SIDE = 3  # 3 below + ATM + 3 above = up to 7
MIN_T_YEARS = 1.0 / 365.0 / 24.0  # ~one hour floor to avoid div-by-zero

# Bounds for delta clipping (per spec). Values are clamped before writing.
CALL_DELTA_MIN = 0.01
CALL_DELTA_MAX = 0.99
PUT_DELTA_MIN = -0.99
PUT_DELTA_MAX = -0.01


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_vix(bars_dir: Path) -> Optional[float]:
    raw = _read_json(bars_dir / "VIX.json")
    if not isinstance(raw, dict):
        return None
    bars = raw.get("bars")
    if not isinstance(bars, list) or not bars:
        return None
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    for key in ("close", "c", "Close"):
        v = last.get(key)
        try:
            if v is None:
                continue
            f = float(v)
            if math.isfinite(f) and f > 0:
                return f
        except Exception:
            continue
    return None


def _price_from_price_cache(price_cache: Optional[Dict[str, Any]], symbol: str) -> Optional[float]:
    if not isinstance(price_cache, dict):
        return None
    prices = price_cache.get("prices")
    if not isinstance(prices, dict):
        return None
    try:
        v = prices.get(symbol)
        if v is None:
            return None
        f = float(v)
        if math.isfinite(f) and f > 0:
            return f
    except Exception:
        return None
    return None


def _parse_expiry_to_date(expiry: str) -> Optional[datetime]:
    s = str(expiry or "").strip()
    if not s:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _pick_near_atm_strikes(strikes: List[float], spot: float) -> List[float]:
    """Return up to (2*NEAR_ATM_STRIKES_PER_SIDE + 1) strikes nearest spot."""
    if not strikes or spot <= 0:
        return []
    valid: List[float] = []
    for s in strikes:
        try:
            f = float(s)
            if math.isfinite(f) and f > 0:
                valid.append(f)
        except Exception:
            continue
    if not valid:
        return []
    valid_sorted = sorted(valid)
    # ATM = closest strike to spot
    atm_idx = min(range(len(valid_sorted)), key=lambda i: abs(valid_sorted[i] - spot))
    lo = max(0, atm_idx - NEAR_ATM_STRIKES_PER_SIDE)
    hi = min(len(valid_sorted), atm_idx + NEAR_ATM_STRIKES_PER_SIDE + 1)
    return valid_sorted[lo:hi]


def _atm_strike(strikes: List[float], spot: float) -> Optional[float]:
    if not strikes or spot <= 0:
        return None
    return min(strikes, key=lambda s: abs(s - spot))


def _compute_greeks_for_strike(
    *,
    spot: float,
    strike: float,
    T: float,
    iv: float,
    r: float,
) -> Dict[str, Optional[float]]:
    """
    Compute synthetic call/put delta + theo price via Black-Scholes.

    Returns a dict shaped for the published schema. gamma/theta are not
    exposed by chad.strategies.options_pricing (only price + delta), so
    they are emitted as None — the schema explicitly permits this.
    """
    # Lazy import to keep this module stdlib-only at import time and to
    # tolerate options_pricing being absent (failure-soft).
    try:
        from chad.strategies.options_pricing import (
            _norm_cdf,
            black_scholes_price,
        )
    except Exception:
        _norm_cdf = None  # type: ignore[assignment]
        black_scholes_price = None  # type: ignore[assignment]

    call_delta: Optional[float] = None
    put_delta: Optional[float] = None
    call_theo: Optional[float] = None
    put_theo: Optional[float] = None

    if spot > 0 and strike > 0 and T > 0 and iv > 0 and _norm_cdf is not None:
        try:
            sqrt_T = math.sqrt(T)
            d1 = (
                math.log(spot / strike) + (r + 0.5 * iv * iv) * T
            ) / (iv * sqrt_T)
            cd = _norm_cdf(d1)
            call_delta = max(CALL_DELTA_MIN, min(CALL_DELTA_MAX, float(cd)))
            put_delta = max(PUT_DELTA_MIN, min(PUT_DELTA_MAX, float(cd - 1.0)))
        except Exception:
            call_delta = None
            put_delta = None

    if black_scholes_price is not None and spot > 0 and strike > 0 and T > 0 and iv > 0:
        try:
            call_theo = round(float(black_scholes_price(spot, strike, T, r, iv, "C")), 4)
            put_theo = round(float(black_scholes_price(spot, strike, T, r, iv, "P")), 4)
        except Exception:
            call_theo = None
            put_theo = None

    return {
        "call_delta": round(call_delta, 6) if call_delta is not None else None,
        "put_delta": round(put_delta, 6) if put_delta is not None else None,
        "call_gamma": None,
        "put_gamma": None,
        "call_theta": None,
        "put_theta": None,
        "call_theo_price": call_theo,
        "put_theo_price": put_theo,
    }


def _resolve_iv(vix: Optional[float], symbol: str) -> float:
    """Resolve annualized IV from VIX with options_pricing helper when present."""
    if vix is None or vix <= 0:
        return 0.20  # conservative default ~20% if VIX missing
    try:
        from chad.strategies.options_pricing import estimate_iv_from_vix
        return float(estimate_iv_from_vix(float(vix), symbol))
    except Exception:
        # Fallback to a simple VIX-based estimate
        return max(0.10, float(vix) / 100.0)


def _resolve_rate() -> float:
    try:
        from chad.strategies.options_pricing import RISK_FREE_RATE
        return float(RISK_FREE_RATE)
    except Exception:
        return 0.05


def _build_symbol_block(
    *,
    symbol: str,
    chain: Dict[str, Any],
    vix: Optional[float],
    price_cache: Optional[Dict[str, Any]],
    now_utc: datetime,
    r: float,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Build the symbol-level Greeks block. Returns (block, expiries_processed,
    strikes_processed). On error/insufficient data, returns a stub with
    data_available=False (still valid schema).
    """
    spot: Optional[float] = None
    raw_spot = chain.get("spot_price")
    try:
        if raw_spot is not None:
            f = float(raw_spot)
            if math.isfinite(f) and f > 0:
                spot = f
    except Exception:
        spot = None

    if spot is None:
        spot = _price_from_price_cache(price_cache, symbol)

    if spot is None or spot <= 0:
        return (
            {
                "underlying_price": None,
                "vix": vix,
                "expirations": {},
                "data_available": False,
                "reason": "no_spot_price",
            },
            0,
            0,
        )

    expirations_in = chain.get("expirations") or []
    strikes_in = chain.get("strikes") or []
    if not isinstance(expirations_in, list) or not isinstance(strikes_in, list):
        return (
            {
                "underlying_price": spot,
                "vix": vix,
                "expirations": {},
                "data_available": False,
                "reason": "malformed_chain",
            },
            0,
            0,
        )

    near_atm = _pick_near_atm_strikes([float(s) for s in strikes_in], spot)
    atm = _atm_strike(near_atm, spot)
    iv = _resolve_iv(vix, symbol)

    # Filter and sort expirations by DTE (skip expired / unparseable).
    today = now_utc.date()
    parsed: List[Tuple[int, str]] = []
    for exp in expirations_in:
        d = _parse_expiry_to_date(str(exp))
        if d is None:
            continue
        dte = (d.date() - today).days
        if dte < 0:
            continue
        parsed.append((dte, str(exp)))
    parsed.sort(key=lambda x: x[0])
    parsed = parsed[:MAX_EXPIRIES_PER_SYMBOL]

    exp_blocks: Dict[str, Any] = {}
    expiries_processed = 0
    strikes_processed = 0

    for dte, exp_key in parsed:
        T = max(MIN_T_YEARS, dte / 365.0)
        strikes_block: Dict[str, Any] = {}
        atm_call_delta: Optional[float] = None
        atm_put_delta: Optional[float] = None
        for strike in near_atm:
            greeks = _compute_greeks_for_strike(
                spot=spot, strike=strike, T=T, iv=iv, r=r,
            )
            try:
                moneyness = round((spot - strike) / spot, 6) if spot > 0 else 0.0
            except Exception:
                moneyness = 0.0
            is_atm = atm is not None and abs(strike - atm) < 1e-9
            entry = {
                **greeks,
                "moneyness": moneyness,
                "near_atm": bool(is_atm),
                "source": "synthetic",
            }
            strikes_block[f"{strike:.4f}".rstrip("0").rstrip(".") or "0"] = entry
            strikes_processed += 1
            if is_atm:
                atm_call_delta = entry.get("call_delta")
                atm_put_delta = entry.get("put_delta")
        exp_blocks[exp_key] = {
            "days_to_expiry": int(dte),
            "strikes": strikes_block,
            "atm_call_delta": atm_call_delta,
            "atm_put_delta": atm_put_delta,
        }
        expiries_processed += 1

    return (
        {
            "underlying_price": round(float(spot), 6),
            "vix": vix,
            "expirations": exp_blocks,
            "data_available": bool(exp_blocks),
        },
        expiries_processed,
        strikes_processed,
    )


def build_greeks_payload(
    *,
    runtime_dir: Path,
    bars_dir: Path = BARS_1D_DIR,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Pure builder — read inputs, return payload dict. Never raises."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    chain_path = runtime_dir / CHAIN_CACHE_FILE_NAME
    price_cache_path = runtime_dir / PRICE_CACHE_FILE_NAME

    chain_doc = _read_json(chain_path)
    price_cache = _read_json(price_cache_path)
    vix = _latest_vix(bars_dir)
    r = _resolve_rate()

    chain_ts_utc: Optional[str] = None
    chains: Dict[str, Any] = {}
    provider_status = "approximated"
    overall_status = "ok"

    if not isinstance(chain_doc, dict):
        provider_status = "error"
        overall_status = "error"
    else:
        try:
            ts = chain_doc.get("ts_utc")
            if isinstance(ts, str) and ts:
                chain_ts_utc = ts
        except Exception:
            chain_ts_utc = None
        raw_chains = chain_doc.get("chains")
        if isinstance(raw_chains, dict):
            chains = raw_chains
        else:
            provider_status = "error"
            overall_status = "error"

    symbols_out: Dict[str, Any] = {}
    expiries_total = 0
    strikes_total = 0
    symbols_processed = 0
    any_missing = False

    for symbol, chain in chains.items():
        if not isinstance(chain, dict):
            continue
        block, expiries_processed, strikes_processed = _build_symbol_block(
            symbol=symbol,
            chain=chain,
            vix=vix,
            price_cache=price_cache,
            now_utc=now_utc,
            r=r,
        )
        symbols_out[symbol] = block
        symbols_processed += 1
        expiries_total += expiries_processed
        strikes_total += strikes_processed
        if not block.get("data_available"):
            any_missing = True

    if overall_status == "ok" and any_missing and symbols_processed > 0:
        provider_status = "partial"
        overall_status = "partial"
    if symbols_processed == 0 and overall_status == "ok":
        # No chains at all — treat as partial (file present but empty).
        provider_status = "partial"
        overall_status = "partial"

    return {
        "schema_version": GREEKS_SCHEMA_VERSION,
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "source": {
            "provider": "synthetic_black_scholes_vix",
            "chain_cache_ts_utc": chain_ts_utc,
            "provider_status": provider_status,
        },
        "symbols": symbols_out,
        "summary": {
            "symbols_processed": int(symbols_processed),
            "greeks_source": "approximated",
            "expiries_processed": int(expiries_total),
            "strikes_processed": int(strikes_total),
        },
        "status": overall_status,
    }


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(*, runtime_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Build payload and either print (dry-run) or atomically write."""
    payload = build_greeks_payload(runtime_dir=runtime_dir)
    if dry_run:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        sys.stdout.flush()
        return payload
    _atomic_write(runtime_dir / GREEKS_FILE_NAME, payload)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CHAD synthetic options Greeks publisher (Phase B Item 6)"
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help="Runtime directory (default: <repo>/runtime)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload to stdout; do not write file.",
    )
    args = parser.parse_args(argv)

    try:
        payload = run(runtime_dir=args.runtime_dir, dry_run=args.dry_run)
    except Exception as exc:
        # Fail-open: emit a minimal valid-error payload to stdout so callers
        # never see a non-JSON exit.
        err = {
            "schema_version": GREEKS_SCHEMA_VERSION,
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "source": {
                "provider": "synthetic_black_scholes_vix",
                "chain_cache_ts_utc": None,
                "provider_status": "error",
            },
            "symbols": {},
            "summary": {
                "symbols_processed": 0,
                "greeks_source": "approximated",
                "expiries_processed": 0,
                "strikes_processed": 0,
            },
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.dry_run:
            sys.stdout.write(json.dumps(err, indent=2) + "\n")
        return 1

    status = str(payload.get("status", "error"))
    return 0 if status in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
