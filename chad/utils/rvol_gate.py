"""Phase B Item 3 — Pre-entry RVOL confidence modifier.

Reads ``runtime/volume_scan.json`` and returns a small, bounded
confidence adjustment for the entry-only alpha gates. This module never
hard-blocks a trade. Missing/stale/unreadable files yield a zero
adjustment so the strategies keep trading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
RVOL_FILENAME = "volume_scan.json"
RVOL_FILE_TTL: int = 600
RVOL_HIGH_BOOST: float = 0.05
RVOL_LOW_PENALTY: float = 0.05


@dataclass(frozen=True)
class RvolGateResult:
    confidence_adjustment: float
    rvol: Optional[float]
    rvol_class: str


def _fail_open() -> RvolGateResult:
    return RvolGateResult(
        confidence_adjustment=0.0,
        rvol=None,
        rvol_class="unavailable",
    )


def _parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_stale(payload: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    ttl = payload.get("ttl_seconds")
    try:
        ttl_s = int(ttl) if ttl is not None else RVOL_FILE_TTL
    except Exception:
        ttl_s = RVOL_FILE_TTL
    if ttl_s <= 0:
        return True
    ts = _parse_iso_z(str(payload.get("ts_utc") or ""))
    if ts is None:
        return True
    cur = now or datetime.now(timezone.utc)
    age = (cur - ts).total_seconds()
    return age > ttl_s


def _opt_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def get_rvol_adjustment(
    symbol: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> RvolGateResult:
    """Return a small confidence adjustment based on RVOL.

    Rules:
      - rvol_class="high" -> +RVOL_HIGH_BOOST
      - rvol_class="low"  -> -RVOL_LOW_PENALTY
      - all other classes (above/normal/unavailable/unknown) -> 0.0

    Fail-open on any error path. Never hard-blocks.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _fail_open()

    rd = runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_DIR
    path = Path(rd) / RVOL_FILENAME

    try:
        if not path.is_file():
            return _fail_open()
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return _fail_open()
    if not isinstance(payload, dict):
        return _fail_open()
    if _is_stale(payload):
        return _fail_open()

    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return _fail_open()
    rec = symbols.get(sym)
    if not isinstance(rec, dict):
        return _fail_open()

    rvol_class = str(rec.get("rvol_class") or "unavailable").strip().lower()
    if rvol_class not in ("high", "above", "normal", "low", "unavailable"):
        rvol_class = "unavailable"
    rvol = _opt_float(rec.get("rvol"))

    adjustment = 0.0
    if rvol_class == "high":
        adjustment = RVOL_HIGH_BOOST
    elif rvol_class == "low":
        adjustment = -RVOL_LOW_PENALTY

    return RvolGateResult(
        confidence_adjustment=adjustment,
        rvol=rvol,
        rvol_class=rvol_class,
    )


__all__ = [
    "RvolGateResult",
    "RVOL_HIGH_BOOST",
    "RVOL_LOW_PENALTY",
    "RVOL_FILE_TTL",
    "get_rvol_adjustment",
]
