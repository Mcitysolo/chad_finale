"""Phase B Item 1 — Pre-entry catalyst gate.

Reads ``runtime/news_intel.json`` and decides whether a brand-new entry
signal is allowed given the latest catalyst direction. Fails open on every
error path: missing file, stale TTL, malformed JSON, unknown symbol, low
or absent catalyst all return ``allowed=True``.

The gate is wired only from strategies that respect the entry-only contract
(see chad/strategies/alpha.py and chad/strategies/alpha_intraday.py). It
must never be consulted on exits, stop-loss reductions, or position cuts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
NEWS_INTEL_FILENAME = "news_intel.json"
DEFAULT_TTL_SECONDS = 3600


@dataclass(frozen=True)
class CatalystGateResult:
    allowed: bool
    catalyst_strength: str
    catalyst_direction: str
    block_reason: Optional[str]


def _fail_open_unknown() -> CatalystGateResult:
    return CatalystGateResult(
        allowed=True,
        catalyst_strength="unknown",
        catalyst_direction="unknown",
        block_reason=None,
    )


def _fail_open_none() -> CatalystGateResult:
    return CatalystGateResult(
        allowed=True,
        catalyst_strength="none",
        catalyst_direction="none",
        block_reason=None,
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
        ttl_s = int(ttl) if ttl is not None else DEFAULT_TTL_SECONDS
    except Exception:
        ttl_s = DEFAULT_TTL_SECONDS
    if ttl_s <= 0:
        return True
    ts = _parse_iso_z(str(payload.get("ts_utc") or ""))
    if ts is None:
        return True
    cur = now or datetime.now(timezone.utc)
    age = (cur - ts).total_seconds()
    return age > ttl_s


def check_catalyst_gate(
    symbol: str,
    signal_side: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> CatalystGateResult:
    sym = (symbol or "").strip().upper()
    side = (signal_side or "").strip().upper()
    if not sym:
        return _fail_open_unknown()

    rd = runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_DIR
    path = Path(rd) / NEWS_INTEL_FILENAME

    try:
        if not path.is_file():
            return _fail_open_unknown()
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return _fail_open_unknown()
    if not isinstance(payload, dict):
        return _fail_open_unknown()

    if _is_stale(payload):
        return _fail_open_unknown()

    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return _fail_open_none()
    rec = symbols.get(sym)
    if not isinstance(rec, dict):
        return _fail_open_none()

    has_catalyst = bool(rec.get("has_catalyst"))
    strength = str(rec.get("catalyst_strength") or "none").strip().lower()
    direction = str(rec.get("catalyst_direction") or "none").strip().lower()
    confirmed_gate_relevant = bool(rec.get("confirmed_gate_relevant", False))

    # Without explicit headline confirmation (provider ticker tags alone are
    # not enough), never block trades — old payloads missing the field fail
    # open by default.
    if not confirmed_gate_relevant:
        return CatalystGateResult(
            allowed=True,
            catalyst_strength=strength,
            catalyst_direction=direction,
            block_reason=None,
        )

    if not has_catalyst:
        return CatalystGateResult(
            allowed=True,
            catalyst_strength=strength,
            catalyst_direction=direction,
            block_reason=None,
        )

    if strength in ("low", "none", "unknown"):
        return CatalystGateResult(
            allowed=True,
            catalyst_strength=strength,
            catalyst_direction=direction,
            block_reason=None,
        )

    if direction in ("neutral", "none", "unknown"):
        return CatalystGateResult(
            allowed=True,
            catalyst_strength=strength,
            catalyst_direction=direction,
            block_reason=None,
        )

    if strength == "high":
        if direction == "bullish" and side == "SELL":
            return CatalystGateResult(
                allowed=False,
                catalyst_strength=strength,
                catalyst_direction=direction,
                block_reason="high_catalyst_bullish_opposes_bearish",
            )
        if direction == "bearish" and side == "BUY":
            return CatalystGateResult(
                allowed=False,
                catalyst_strength=strength,
                catalyst_direction=direction,
                block_reason="high_catalyst_bearish_opposes_bullish",
            )
    elif strength == "medium":
        if direction == "bullish" and side == "SELL":
            return CatalystGateResult(
                allowed=False,
                catalyst_strength=strength,
                catalyst_direction=direction,
                block_reason="medium_catalyst_bullish_opposes_bearish",
            )
        if direction == "bearish" and side == "BUY":
            return CatalystGateResult(
                allowed=False,
                catalyst_strength=strength,
                catalyst_direction=direction,
                block_reason="medium_catalyst_bearish_opposes_bullish",
            )

    return CatalystGateResult(
        allowed=True,
        catalyst_strength=strength,
        catalyst_direction=direction,
        block_reason=None,
    )


__all__ = ["CatalystGateResult", "check_catalyst_gate"]
