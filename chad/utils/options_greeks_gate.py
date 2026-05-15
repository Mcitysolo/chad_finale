#!/usr/bin/env python3
"""
chad/utils/options_greeks_gate.py

Phase B Item 6 — read-only Greeks metadata lookup.

alpha_options calls ``get_option_greeks(symbol, expiry, strike, option_type)``
to annotate TradeSignal.meta with synthetic delta + theoretical price hints.
This module never raises and never participates in sizing or execution.

If runtime/options_greeks.json is missing, stale, malformed, or lacks the
requested entry, a default-shaped ``GreeksResult`` is returned:
  - call (right="C"): delta = +0.5, source="default"
  - put  (right="P"): delta = -0.5, source="default"

The TTL is read from the file (``ttl_seconds``) with a 7200s fallback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime"
GREEKS_FILE_NAME = "options_greeks.json"

GREEKS_FILE_TTL = 7200
DEFAULT_CALL_DELTA = 0.5
DEFAULT_PUT_DELTA = -0.5


@dataclass(frozen=True)
class GreeksResult:
    delta: float
    gamma: Optional[float]
    theta: Optional[float]
    theo_price: Optional[float]
    source: str
    near_atm: bool


def _normalize_option_type(option_type: str) -> Optional[str]:
    s = str(option_type or "").strip().lower()
    if s in ("c", "call"):
        return "C"
    if s in ("p", "put"):
        return "P"
    return None


def _default_result(option_type: str) -> GreeksResult:
    right = _normalize_option_type(option_type)
    if right == "P":
        delta = DEFAULT_PUT_DELTA
    else:
        # Default to call delta when option_type is unrecognized — keeps
        # the lookup failure-soft (caller still gets a usable value).
        delta = DEFAULT_CALL_DELTA
    return GreeksResult(
        delta=float(delta),
        gamma=None,
        theta=None,
        theo_price=None,
        source="default",
        near_atm=False,
    )


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_greeks_doc(runtime_dir: Path) -> Optional[Dict[str, Any]]:
    path = runtime_dir / GREEKS_FILE_NAME
    try:
        if not path.is_file():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return None
    except Exception:
        return None

    ts = _parse_ts(doc.get("ts_utc"))
    if ts is None:
        return None
    try:
        ttl_raw = doc.get("ttl_seconds")
        ttl = int(ttl_raw) if ttl_raw is not None else GREEKS_FILE_TTL
        if ttl <= 0:
            ttl = GREEKS_FILE_TTL
    except Exception:
        ttl = GREEKS_FILE_TTL

    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age > ttl:
        return None
    return doc


def _nearest_strike_block(
    strikes: Dict[str, Any], strike: float
) -> Optional[Dict[str, Any]]:
    """Return the strike entry whose numeric key is closest to ``strike``."""
    best_key: Optional[str] = None
    best_diff = math.inf
    for k, v in strikes.items():
        try:
            kf = float(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        diff = abs(kf - strike)
        if diff < best_diff:
            best_diff = diff
            best_key = k
    if best_key is None:
        return None
    block = strikes.get(best_key)
    return block if isinstance(block, dict) else None


def _exact_strike_block(
    strikes: Dict[str, Any], strike: float
) -> Optional[Dict[str, Any]]:
    for k, v in strikes.items():
        try:
            kf = float(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        if abs(kf - strike) < 1e-9:
            return v
    return None


def _extract_for_right(
    block: Dict[str, Any], right: str
) -> GreeksResult:
    prefix = "call" if right == "C" else "put"
    try:
        delta_raw = block.get(f"{prefix}_delta")
        delta = float(delta_raw) if delta_raw is not None else (
            DEFAULT_CALL_DELTA if right == "C" else DEFAULT_PUT_DELTA
        )
    except Exception:
        delta = DEFAULT_CALL_DELTA if right == "C" else DEFAULT_PUT_DELTA
    gamma = block.get(f"{prefix}_gamma")
    theta = block.get(f"{prefix}_theta")
    theo = block.get(f"{prefix}_theo_price")
    source = str(block.get("source") or "synthetic")
    near_atm = bool(block.get("near_atm", False))
    return GreeksResult(
        delta=float(delta),
        gamma=(float(gamma) if isinstance(gamma, (int, float)) else None),
        theta=(float(theta) if isinstance(theta, (int, float)) else None),
        theo_price=(float(theo) if isinstance(theo, (int, float)) else None),
        source=source,
        near_atm=near_atm,
    )


def get_option_greeks(
    symbol: str,
    expiry: str,
    strike: float,
    option_type: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> GreeksResult:
    """
    Look up a single option's synthetic Greeks. Fail-open.

    Lookup order:
      1. file missing/stale/unreadable -> default
      2. symbol missing                 -> default
      3. expiry missing                 -> default
      4. exact strike match             -> use exact
      5. otherwise                      -> nearest strike in same expiry
    """
    right = _normalize_option_type(option_type)
    if right is None:
        return _default_result(option_type)

    rdir = runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_DIR
    doc = _load_greeks_doc(rdir)
    if doc is None:
        return _default_result(option_type)

    try:
        strike_f = float(strike)
    except Exception:
        return _default_result(option_type)

    symbols = doc.get("symbols")
    if not isinstance(symbols, dict):
        return _default_result(option_type)
    sym_block = symbols.get(str(symbol))
    if not isinstance(sym_block, dict):
        return _default_result(option_type)

    expirations = sym_block.get("expirations")
    if not isinstance(expirations, dict):
        return _default_result(option_type)
    exp_block = expirations.get(str(expiry))
    if not isinstance(exp_block, dict):
        return _default_result(option_type)

    strikes = exp_block.get("strikes")
    if not isinstance(strikes, dict) or not strikes:
        return _default_result(option_type)

    block = _exact_strike_block(strikes, strike_f)
    if block is None:
        block = _nearest_strike_block(strikes, strike_f)
    if block is None:
        return _default_result(option_type)

    return _extract_for_right(block, right)


__all__ = [
    "GreeksResult",
    "get_option_greeks",
    "GREEKS_FILE_TTL",
    "DEFAULT_CALL_DELTA",
    "DEFAULT_PUT_DELTA",
]
