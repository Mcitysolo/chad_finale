from __future__ import annotations

"""
CHAD — Crypto Signal Filter (Phase B Item 4)

Confidence-only crowding modifier for crypto signals. Reads the runtime
snapshot written by ``chad.market_data.crypto_derivatives_publisher`` and
returns a confidence adjustment plus market-bias metadata.

This module is **never** a hard block. The worst it can do is shave
confidence by a fixed amount when crowding is detected on the same side as
the proposed trade.

Fail-open contract
------------------
- Missing runtime file -> zero adjustment, ``market_bias="unknown"``.
- Stale runtime file (older than ``ttl_seconds``) -> zero adjustment.
- Unreadable / malformed payload -> zero adjustment.
- Symbol absent from snapshot -> zero adjustment.

Rules
-----
- BUY  + ``long_crowded``  -> -0.20
- SELL + ``short_crowded`` -> -0.20
- BUY  + ``long_leaning``  -> -0.05
- everything else          ->  0.00
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

CROWDED_PENALTY = 0.20
LEANING_PENALTY = 0.05
CRYPTO_DERIV_TTL = 1200

_DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
_RUNTIME_FILENAME = "crypto_derivatives.json"

_BIAS_CROWDED_VALUES = {"long_crowded", "short_crowded"}


@dataclass(frozen=True)
class CryptoFilterResult:
    """Outcome of the crypto crowding filter — confidence modifier only."""

    confidence_adjustment: float
    market_bias: str
    funding_rate_8h: Optional[float]
    funding_extreme: bool


_FAIL_OPEN_RESULT = CryptoFilterResult(
    confidence_adjustment=0.0,
    market_bias="unknown",
    funding_rate_8h=None,
    funding_extreme=False,
)


def _parse_ts(ts_value: Any) -> Optional[float]:
    if not isinstance(ts_value, str) or not ts_value:
        return None
    raw = ts_value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _load_payload(runtime_dir: Path) -> Optional[Mapping[str, Any]]:
    path = runtime_dir / _RUNTIME_FILENAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _normalize_side(signal_side: str) -> str:
    return signal_side.strip().upper()


def get_crypto_filter(
    symbol: str,
    signal_side: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> CryptoFilterResult:
    """Return the confidence-only crypto crowding modifier for ``symbol``.

    Fails open (zero adjustment, ``market_bias="unknown"``) on any missing
    data, stale snapshot, or unparseable payload.
    """
    if not isinstance(symbol, str) or not symbol:
        return _FAIL_OPEN_RESULT
    if not isinstance(signal_side, str) or not signal_side:
        return _FAIL_OPEN_RESULT

    base_dir = runtime_dir if runtime_dir is not None else _DEFAULT_RUNTIME_DIR
    payload = _load_payload(base_dir)
    if payload is None:
        return _FAIL_OPEN_RESULT

    ttl_seconds = payload.get("ttl_seconds")
    try:
        ttl_value = float(ttl_seconds)
        if ttl_value <= 0:
            ttl_value = float(CRYPTO_DERIV_TTL)
    except (TypeError, ValueError):
        ttl_value = float(CRYPTO_DERIV_TTL)

    ts_epoch = _parse_ts(payload.get("ts_utc"))
    if ts_epoch is None:
        return _FAIL_OPEN_RESULT
    now_epoch = time.time()
    if now_epoch - ts_epoch > ttl_value:
        return _FAIL_OPEN_RESULT

    symbols = payload.get("symbols")
    if not isinstance(symbols, Mapping):
        return _FAIL_OPEN_RESULT

    key = _normalize_symbol(symbol)
    entry = symbols.get(key)
    if not isinstance(entry, Mapping):
        return _FAIL_OPEN_RESULT

    market_bias_raw = entry.get("market_bias")
    market_bias = (
        market_bias_raw if isinstance(market_bias_raw, str) and market_bias_raw else "unknown"
    )

    funding_raw = entry.get("funding_rate_8h")
    try:
        funding_rate_8h = float(funding_raw) if funding_raw is not None else None
    except (TypeError, ValueError):
        funding_rate_8h = None

    side = _normalize_side(signal_side)
    adjustment = 0.0
    if side == "BUY" and market_bias == "long_crowded":
        adjustment = -CROWDED_PENALTY
    elif side == "SELL" and market_bias == "short_crowded":
        adjustment = -CROWDED_PENALTY
    elif side == "BUY" and market_bias == "long_leaning":
        adjustment = -LEANING_PENALTY

    funding_extreme = market_bias in _BIAS_CROWDED_VALUES

    return CryptoFilterResult(
        confidence_adjustment=float(adjustment),
        market_bias=market_bias,
        funding_rate_8h=funding_rate_8h,
        funding_extreme=funding_extreme,
    )
