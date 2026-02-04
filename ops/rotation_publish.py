#!/usr/bin/env python3
"""
CHAD Rotation Rules Layer — bootstrap publisher (Production, advisory-only)

File: ops/rotation_publish.py

Outputs:
  /home/ubuntu/CHAD FINALE/reports/rotation/ROTATION_<ts>.json
Optional pointer:
  /home/ubuntu/CHAD FINALE/runtime/rotation_state.json

SSOT v4.2 contract:
- Rotation Rules produce tilts + blocks as advisory proposals, never direct execution.
- Outputs must be bounded and fail-closed if inputs are stale/unknown.
- Artifacts logged under reports/rotation/ and referenced by the system. :contentReference[oaicite:0]{index=0}

Bootstrap Strategy (honest + bounded)
-------------------------------------
Inputs:
- runtime/macro_state.json (risk_label)
- runtime/event_risk.json (severity/windows)
- runtime/sector_rotation.json (bootstrap ranks currently empty)

Outputs:
- tilt suggestions at the sleeve level (growth_momo vs hedge) with bounded delta_weight
- symbol_blocks empty by default (until earnings/calendar provider exists)
- notes explain bootstrap logic

No broker calls. No secrets. Advisory-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Paths / Config
# -----------------------------

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
REPORTS_DIR = Path(os.environ.get("CHAD_ROTATION_REPORTS_DIR", "/home/ubuntu/CHAD FINALE/reports/rotation"))

MACRO_PATH = RUNTIME_DIR / "macro_state.json"
EVENT_RISK_PATH = RUNTIME_DIR / "event_risk.json"
SECTOR_ROT_PATH = RUNTIME_DIR / "sector_rotation.json"

ROTATION_STATE_PATH = RUNTIME_DIR / "rotation_state.json"

TTL_SECONDS = int(os.environ.get("CHAD_ROTATION_TTL_SECONDS", "3600"))

# Hard bounds (bootstrap): never tilt more than 5% in one step
MAX_ABS_TILT = float(os.environ.get("CHAD_ROTATION_MAX_ABS_TILT", "0.05"))

# If true, write rotation_state.json pointer
WRITE_POINTER = os.environ.get("CHAD_ROTATION_WRITE_POINTER", "1").lower() in ("1", "true", "yes", "on")


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_dict:{path}")
    return obj


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_str(d: Dict[str, Any], k: str) -> str:
    v = d.get(k)
    return str(v) if v is not None else ""


# -----------------------------
# Rotation logic (bootstrap)
# -----------------------------

def compute_bootstrap_tilts(macro: Dict[str, Any], event_risk: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (regime_label, tilts list)
    Tilts are bounded and advisory.
    """
    risk_label = _safe_str(macro, "risk_label").strip().lower()
    sev = _safe_str(event_risk, "severity").strip().lower()

    # Default neutral
    regime = "neutral"
    tilts: List[Dict[str, Any]] = []

    # If event risk is high/unknown, fail-closed => increase hedge, reduce growth
    if sev in ("high", "unknown"):
        regime = "risk_off"
        tilts.append({"sleeve": "growth_momo", "delta_weight": clamp(-MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "event_risk_high_or_unknown"})
        tilts.append({"sleeve": "hedge", "delta_weight": clamp(MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "event_risk_high_or_unknown"})
        return regime, tilts

    # Macro-driven
    if risk_label == "risk_off":
        regime = "risk_off"
        tilts.append({"sleeve": "growth_momo", "delta_weight": clamp(-MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "macro_risk_off"})
        tilts.append({"sleeve": "hedge", "delta_weight": clamp(MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "macro_risk_off"})
    elif risk_label == "risk_on":
        regime = "risk_on"
        tilts.append({"sleeve": "growth_momo", "delta_weight": clamp(MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "macro_risk_on"})
        tilts.append({"sleeve": "hedge", "delta_weight": clamp(-MAX_ABS_TILT, -MAX_ABS_TILT, MAX_ABS_TILT), "reason": "macro_risk_on"})
    else:
        regime = "neutral"
        tilts.append({"sleeve": "growth_momo", "delta_weight": 0.0, "reason": "macro_neutral"})
        tilts.append({"sleeve": "hedge", "delta_weight": 0.0, "reason": "macro_neutral"})

    return regime, tilts


def build_rotation_payload() -> Dict[str, Any]:
    ts = utc_now_iso()

    # Fail-closed if any required file is missing/unreadable
    try:
        macro = read_json_dict(MACRO_PATH)
        event_risk = read_json_dict(EVENT_RISK_PATH)
        sector_rot = read_json_dict(SECTOR_ROT_PATH)
    except Exception as exc:
        return {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "schema_version": "rotation.v1",
            "regime": "unknown",
            "tilts": [],
            "symbol_blocks": [],
            "notes": f"fail_closed_missing_inputs:{type(exc).__name__}",
            "inputs": {
                "macro_state": {"path": str(MACRO_PATH), "exists": MACRO_PATH.is_file()},
                "event_risk": {"path": str(EVENT_RISK_PATH), "exists": EVENT_RISK_PATH.is_file()},
                "sector_rotation": {"path": str(SECTOR_ROT_PATH), "exists": SECTOR_ROT_PATH.is_file()},
            },
        }

    regime, tilts = compute_bootstrap_tilts(macro, event_risk)

    return {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "schema_version": "rotation.v1",
        "regime": regime,
        "tilts": tilts,
        "symbol_blocks": [],  # bootstrap: empty
        "notes": "bootstrap rotation: bounded sleeve tilts derived from macro_state + event_risk; no sector ranks yet",
        "inputs": {
            "macro_state": {"ts_utc": macro.get("ts_utc"), "risk_label": macro.get("risk_label")},
            "event_risk": {"ts_utc": event_risk.get("ts_utc"), "severity": event_risk.get("severity")},
            "sector_rotation": {"ts_utc": sector_rot.get("ts_utc"), "provider_status": sector_rot.get("provider_status")},
        },
    }


def main() -> int:
    payload = build_rotation_payload()
    ts_compact = utc_now_compact()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"ROTATION_{ts_compact}.json"
    atomic_write_json(out, payload)

    # pointer
    if WRITE_POINTER:
        pointer = {
            "ts_utc": payload.get("ts_utc"),
            "ttl_seconds": payload.get("ttl_seconds"),
            "latest_rotation_path": str(out),
            "latest_rotation_sha256": sha256_hex(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
            "schema_version": "rotation_state.v1",
        }
        atomic_write_json(ROTATION_STATE_PATH, pointer)

    print(json.dumps({"ok": True, "out": str(out), "ts_utc": payload.get("ts_utc")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
