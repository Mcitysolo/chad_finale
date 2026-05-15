"""Phase B Item 2 — Pre-entry relative-strength confidence modifier.

Reads ``runtime/relative_strength.json`` and returns a small, bounded
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
RS_FILENAME = "relative_strength.json"
RS_FILE_TTL: int = 90000
RS_CONFIDENCE_PENALTY: float = 0.10


@dataclass(frozen=True)
class RSGateResult:
    confidence_adjustment: float
    rs_class: str
    rs_vs_spy: Optional[float]
    excess_vs_spy_5d: Optional[float]
    market_direction: str


def _fail_open() -> RSGateResult:
    return RSGateResult(
        confidence_adjustment=0.0,
        rs_class="unknown",
        rs_vs_spy=None,
        excess_vs_spy_5d=None,
        market_direction="unknown",
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
        ttl_s = int(ttl) if ttl is not None else RS_FILE_TTL
    except Exception:
        ttl_s = RS_FILE_TTL
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


def get_rs_adjustment(
    symbol: str,
    signal_side: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> RSGateResult:
    """Return a small confidence adjustment based on relative strength.

    Rules (all else returns 0.0):
      - BUY + rs_class="weak" + market_direction="up" -> -RS_CONFIDENCE_PENALTY
      - SELL + rs_class="strong" + market_direction="down" -> -RS_CONFIDENCE_PENALTY

    Fail-open on any error path. Never hard-blocks.
    """
    sym = (symbol or "").strip().upper()
    side = (signal_side or "").strip().upper()
    if not sym:
        return _fail_open()

    rd = runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_DIR
    path = Path(rd) / RS_FILENAME

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

    market_direction = str(payload.get("market_direction") or "unknown").strip().lower()
    if market_direction not in ("up", "down", "flat", "unknown"):
        market_direction = "unknown"

    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return RSGateResult(
            confidence_adjustment=0.0,
            rs_class="unknown",
            rs_vs_spy=None,
            excess_vs_spy_5d=None,
            market_direction=market_direction,
        )
    rec = symbols.get(sym)
    if not isinstance(rec, dict):
        return RSGateResult(
            confidence_adjustment=0.0,
            rs_class="unknown",
            rs_vs_spy=None,
            excess_vs_spy_5d=None,
            market_direction=market_direction,
        )

    rs_class = str(rec.get("rs_class") or "unknown").strip().lower()
    if rs_class not in ("strong", "neutral", "weak", "unknown"):
        rs_class = "unknown"
    rs_vs_spy = _opt_float(rec.get("rs_vs_spy"))
    excess_vs_spy = _opt_float(rec.get("excess_vs_spy_5d"))

    adjustment = 0.0
    if side == "BUY" and rs_class == "weak" and market_direction == "up":
        adjustment = -RS_CONFIDENCE_PENALTY
    elif side == "SELL" and rs_class == "strong" and market_direction == "down":
        adjustment = -RS_CONFIDENCE_PENALTY

    return RSGateResult(
        confidence_adjustment=adjustment,
        rs_class=rs_class,
        rs_vs_spy=rs_vs_spy,
        excess_vs_spy_5d=excess_vs_spy,
        market_direction=market_direction,
    )


__all__ = [
    "RSGateResult",
    "RS_CONFIDENCE_PENALTY",
    "RS_FILE_TTL",
    "get_rs_adjustment",
]
