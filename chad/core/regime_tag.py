#!/usr/bin/env python3
"""
CHAD — Phase 10 Regime Tagging (SSOT Parity)

Purpose
-------
Provide a single, authoritative, deterministic way to tag trade ledger records with
a regime label derived from runtime sensor artifacts (Market Radar).

This module exists to satisfy Phase 10 SSOT + Additions requirements:
- Regime-tagged results persisted in the immutable (append-only) ledger
- Deterministic scoring (no ML required)
- Fail-closed behavior (unknown when stale/missing)

Source of truth
---------------
Primary:  runtime/macro_state.json  -> "risk_label" with TTL freshness.
Optional: runtime/regime_state.json -> "regime" (if present + fresh), used as a
          secondary hint when macro_state is missing/unknown.

Contract
--------
- Never returns None / null.
- Returns one of: "risk_on" | "risk_off" | "neutral" | "unknown"
- Enforces TTL freshness using ts_utc + ttl_seconds.
- Uses atomic, read-only file access; no writes.
- Contains a small, thread-safe cache keyed by file mtime to minimize disk reads.

Intended usage
--------------
Call resolve_regime_label(now_utc=...) inside each ledger writer at the moment a
TradeResult is constructed.

Example:
    from datetime import datetime, timezone
    from chad.core.regime_tag import resolve_regime_label

    regime = resolve_regime_label(now_utc=datetime.now(timezone.utc), runtime_dir=ROOT / "runtime")
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

RegimeLabel = Literal["risk_on", "risk_off", "neutral", "unknown"]


@dataclass(frozen=True)
class RegimeResolution:
    label: RegimeLabel
    source: str  # e.g. "macro_state" | "regime_state" | "unknown"
    ts_utc: Optional[str]
    ttl_seconds: Optional[int]
    stale: bool
    reason: str  # concise diagnostic, safe to log into ledgers


_CACHE_LOCK = threading.Lock()
# Cache entries keyed by absolute path; each stores (mtime_ns, resolution)
_CACHE: Dict[str, Tuple[int, RegimeResolution]] = {}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # Fail-closed: naive time treated as UTC to avoid locale ambiguity.
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_ts_utc(ts: str) -> Optional[datetime]:
    """
    Parse SSOT-style ts_utc, usually "YYYY-MM-DDTHH:MM:SSZ".
    Returns timezone-aware UTC datetime or None if invalid.
    """
    s = (ts or "").strip()
    if not s:
        return None

    # Fast path for canonical Z format
    try:
        if s.endswith("Z") and len(s) == 20:
            # 2026-02-10T21:14:33Z
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        pass

    # Robust fallback: ISO 8601 variants
    try:
        # fromisoformat doesn’t understand trailing 'Z' in older Pythons; normalize.
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        else:
            s2 = s
        dt = datetime.fromisoformat(s2)
        return _to_utc(dt)
    except Exception:
        return None


def _is_fresh(*, now_utc: datetime, ts_utc: datetime, ttl_seconds: int) -> bool:
    """
    TTL freshness check. Fail-closed on clock skew.
    """
    if ttl_seconds <= 0:
        return False
    age = (now_utc - ts_utc).total_seconds()
    if age < 0:
        # Clock skew / time travel => treat stale.
        return False
    return age <= float(ttl_seconds)


def _read_json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Read JSON dict from disk. Never raises.
    Returns (obj_or_none, reason).
    """
    try:
        if not path.is_file():
            return None, "missing"
        raw = path.read_text(encoding="utf-8", errors="strict")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None, "not_a_dict"
        return obj, "ok"
    except Exception as e:
        return None, f"read_error:{type(e).__name__}"


def _resolve_from_macro_state(*, now_utc: datetime, runtime_dir: Path) -> RegimeResolution:
    path = runtime_dir / "macro_state.json"
    obj, why = _read_json_file(path)
    if obj is None:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=None,
            ttl_seconds=None,
            stale=True,
            reason=f"macro_state:{why}",
        )

    ts_s = str(obj.get("ts_utc") or "").strip()
    ttl = int(obj.get("ttl_seconds") or 0)
    risk = str(obj.get("risk_label") or "").strip()

    ts_dt = _parse_ts_utc(ts_s)
    if ts_dt is None:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=ts_s or None,
            ttl_seconds=ttl if ttl > 0 else None,
            stale=True,
            reason="macro_state:bad_ts_utc",
        )

    fresh = _is_fresh(now_utc=now_utc, ts_utc=ts_dt, ttl_seconds=ttl)
    if not fresh:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=ts_s,
            ttl_seconds=ttl,
            stale=True,
            reason="macro_state:stale",
        )

    if risk in {"risk_on", "risk_off", "neutral"}:
        return RegimeResolution(
            label=risk,  # type: ignore[return-value]
            source="macro_state",
            ts_utc=ts_s,
            ttl_seconds=ttl,
            stale=False,
            reason="macro_state:fresh",
        )

    return RegimeResolution(
        label="unknown",
        source="unknown",
        ts_utc=ts_s,
        ttl_seconds=ttl,
        stale=False,
        reason="macro_state:fresh_but_unknown_label",
    )


def _resolve_from_regime_state(*, now_utc: datetime, runtime_dir: Path) -> RegimeResolution:
    """
    Optional secondary source. Only used if present and fresh.
    We accept either a top-level "regime" key or "label" key if implemented.
    """
    path = runtime_dir / "regime_state.json"
    obj, why = _read_json_file(path)
    if obj is None:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=None,
            ttl_seconds=None,
            stale=True,
            reason=f"regime_state:{why}",
        )

    ts_s = str(obj.get("ts_utc") or "").strip()
    ttl = int(obj.get("ttl_seconds") or 0)
    raw = (obj.get("regime") if "regime" in obj else obj.get("label"))
    label = str(raw or "").strip()

    ts_dt = _parse_ts_utc(ts_s)
    if ts_dt is None:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=ts_s or None,
            ttl_seconds=ttl if ttl > 0 else None,
            stale=True,
            reason="regime_state:bad_ts_utc",
        )

    fresh = _is_fresh(now_utc=now_utc, ts_utc=ts_dt, ttl_seconds=ttl)
    if not fresh:
        return RegimeResolution(
            label="unknown",
            source="unknown",
            ts_utc=ts_s,
            ttl_seconds=ttl,
            stale=True,
            reason="regime_state:stale",
        )

    # Map any non-standard regime labels into our locked enum if possible.
    if label in {"risk_on", "risk_off", "neutral"}:
        return RegimeResolution(
            label=label,  # type: ignore[return-value]
            source="regime_state",
            ts_utc=ts_s,
            ttl_seconds=ttl,
            stale=False,
            reason="regime_state:fresh",
        )

    return RegimeResolution(
        label="unknown",
        source="unknown",
        ts_utc=ts_s,
        ttl_seconds=ttl,
        stale=False,
        reason="regime_state:fresh_but_unknown_label",
    )


def resolve_regime(
    *,
    now_utc: datetime,
    runtime_dir: Path,
    allow_regime_state_fallback: bool = True,
) -> RegimeResolution:
    """
    Resolve regime deterministically with TTL freshness enforcement.

    Order:
      1) macro_state.json (risk_label)
      2) regime_state.json (optional)
      3) unknown

    Includes a small mtime-based cache to reduce disk reads.
    """
    now_utc = _to_utc(now_utc)
    runtime_dir = runtime_dir.resolve()

    macro_path = (runtime_dir / "macro_state.json").resolve()
    regime_path = (runtime_dir / "regime_state.json").resolve()

    # Cache key uses both files; macro dominates, but a change in either invalidates.
    # We key on a synthetic key to keep cache simple.
    key = f"{macro_path}|{regime_path}"

    def _mtime_ns(p: Path) -> int:
        try:
            st = p.stat()
            return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        except Exception:
            return -1

    macro_m = _mtime_ns(macro_path)
    regime_m = _mtime_ns(regime_path)
    combined_m = (macro_m << 1) ^ regime_m  # deterministic combine

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == combined_m:
            return cached[1]

    # Compute fresh
    res = _resolve_from_macro_state(now_utc=now_utc, runtime_dir=runtime_dir)
    if res.label == "unknown" and allow_regime_state_fallback:
        res2 = _resolve_from_regime_state(now_utc=now_utc, runtime_dir=runtime_dir)
        if res2.label != "unknown":
            res = res2

    with _CACHE_LOCK:
        _CACHE[key] = (combined_m, res)
    return res


def resolve_regime_label(
    *,
    now_utc: datetime,
    runtime_dir: Optional[Path] = None,
) -> RegimeLabel:
    """
    Convenience wrapper returning only the locked enum label.
    """
    if runtime_dir is None:
        # Default runtime dir is SSOT-pinned by systemd in this build.
        runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime"))
    return resolve_regime(now_utc=now_utc, runtime_dir=runtime_dir).label


def attach_regime_tag(
    payload: Dict[str, Any],
    *,
    now_utc: datetime,
    runtime_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Pure helper: returns a shallow-copied payload with payload["regime"] set to a non-null label.
    Safe to use in ledger writers that build dict payloads.
    """
    out = dict(payload or {})
    out["regime"] = resolve_regime_label(now_utc=now_utc, runtime_dir=runtime_dir)
    return out
