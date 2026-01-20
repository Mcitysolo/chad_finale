"""
chad.utils.runtime_json

World-class runtime JSON utilities for CHAD.

CSB requirements (canonical)
----------------------------
Runtime JSON files are CHAD's shared memory and must be:
- Human-inspectable
- Written atomically (tmp -> fsync -> rename)
- Freshness-checked using (ts_utc, ttl_seconds)
- Treated as unsafe when stale/missing/corrupt (fail closed)

This module provides:
- Atomic runtime JSON writes
- Automatic injection of ts_utc + ttl_seconds for state files
- Safe reads with schema-light validation
- Freshness checks (staleness gating)

Security
--------
- Never store secrets in runtime JSON.
- Runtime JSON must be safe to read without leaking credentials.

"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def _parse_ts_utc(s: Any) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    st = s.strip()
    if st.endswith("Z"):
        st = st[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(st).astimezone(timezone.utc)
    except Exception:
        return None


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class Freshness:
    ok: bool
    age_seconds: float
    ttl_seconds: float
    ts_utc: str
    reason: str


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Atomic write:
      write to <path>.tmp, flush, fsync, then os.replace

    Guarantees readers never see partial JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")

    payload = stable_json_dumps(obj) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_runtime_state_json(
    path: Path,
    state: Dict[str, Any],
    *,
    ttl_seconds: int,
    inject_ts: bool = True,
) -> Dict[str, Any]:
    """
    Write a runtime STATE JSON file with ts_utc + ttl_seconds injected.

    Returns the final object written (with injected fields).
    """
    out = dict(state)
    if inject_ts:
        out["ts_utc"] = utc_now_iso()
    out["ttl_seconds"] = int(ttl_seconds)
    atomic_write_json(path, out)
    return out


def write_runtime_config_json(path: Path, cfg: Dict[str, Any]) -> None:
    """
    Write a runtime CONFIG JSON file atomically.

    Configs are not freshness-gated by default and typically do not include ttl_seconds.
    """
    atomic_write_json(path, dict(cfg))


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Safe JSON read. Returns None if missing/corrupt/non-object.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def read_runtime_state_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Freshness]:
    """
    Read a runtime STATE JSON and compute freshness.

    If missing/corrupt:
      - returns (None, Freshness(ok=False,...))
    """
    obj = read_json(path)
    if obj is None:
        return None, Freshness(
            ok=False,
            age_seconds=float("inf"),
            ttl_seconds=0.0,
            ts_utc="",
            reason="missing_or_corrupt",
        )

    ts = _parse_ts_utc(obj.get("ts_utc"))
    ttl = obj.get("ttl_seconds")

    if ts is None or not isinstance(ttl, (int, float)):
        return obj, Freshness(
            ok=False,
            age_seconds=float("inf"),
            ttl_seconds=float(ttl) if isinstance(ttl, (int, float)) else 0.0,
            ts_utc=str(obj.get("ts_utc", "")),
            reason="missing_ts_or_ttl",
        )

    now = utc_now()
    age = (now - ts).total_seconds()
    ttl_f = float(ttl)

    if age <= ttl_f:
        return obj, Freshness(ok=True, age_seconds=age, ttl_seconds=ttl_f, ts_utc=ts.isoformat().replace("+00:00", "Z"), reason="fresh")
    return obj, Freshness(ok=False, age_seconds=age, ttl_seconds=ttl_f, ts_utc=ts.isoformat().replace("+00:00", "Z"), reason="stale")


def is_fresh(path: Path) -> bool:
    """
    Convenience: True if runtime state exists and is fresh.
    """
    _, fr = read_runtime_state_json(path)
    return fr.ok
