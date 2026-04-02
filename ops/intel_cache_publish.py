#!/usr/bin/env python3
"""
ops/intel_cache_publish.py

PHASE 11 ADDITIONS — Intel Router Publisher (Production-Grade)
--------------------------------------------------------------
This publisher creates the SSOT-required "Intel Cache" that brains consume.

Contract
--------
- Brains do NOT call vendor APIs directly.
- A central publisher normalizes data -> cached runtime artifacts:
    runtime/intel_cache/*.json
- Each cache file includes:
    - as_of_utc
    - ttl_seconds
    - staleness_flag (fresh|stale|unknown)
    - provider_id
    - payload (summary only)
- If required intel is stale/unknown, the system must TIGHTEN (never loosen).

This publisher:
- Reads config/intel_profiles.json (BUDGET/STANDARD/PREMIUM) to determine required sensors.
- Reads config/providers_allowlist.json to validate provider ids + TTL policies.
- Uses existing "radar publishers" outputs as inputs (macro_state/event_risk/earnings_state/sector_rotation).
  (No vendor scraping. No secrets.)
- Produces:
    runtime/intel_cache/intel_cache_state.json  (global summary)
    runtime/intel_cache/macro_state.json
    runtime/intel_cache/event_risk.json
    runtime/intel_cache/earnings_state.json
    runtime/intel_cache/sector_rotation.json

Design Guarantees
-----------------
- Atomic writes (tmp -> fsync -> replace).
- Deterministic outputs.
- Fail-closed: missing inputs => staleness_flag=unknown; required_missing_count increments.
- No network calls by default (bootstrap via existing runtime files).
- Strict JSON validity always (single object, newline terminated).

Env
---
- CHAD_RUNTIME_DIR (default /home/ubuntu/chad_finale/runtime)
- CHAD_CONFIG_DIR  (default /home/ubuntu/chad_finale/config)
- CHAD_INTEL_PROFILE (default from intel_profiles.json -> BUDGET)
- CHAD_INTEL_CACHE_TTL_SECONDS (default 300)

Exit codes
----------
0 success (even if some sensors stale; that is part of contract)
2 config error (cannot parse profiles/allowlist)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Paths / Env
# -----------------------------

DEFAULT_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_ROOT / "runtime"))).resolve()
CONFIG_DIR = Path(os.environ.get("CHAD_CONFIG_DIR", str(DEFAULT_ROOT / "config"))).resolve()

INTEL_CACHE_DIR = RUNTIME_DIR / "intel_cache"
STATE_PATH = INTEL_CACHE_DIR / "intel_cache_state.json"

PROFILES_PATH = CONFIG_DIR / "intel_profiles.json"
ALLOWLIST_PATH = CONFIG_DIR / "providers_allowlist.json"

DEFAULT_CACHE_TTL = int(os.environ.get("CHAD_INTEL_CACHE_TTL_SECONDS", "300"))


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

    # fsync directory for crash-safety (best-effort)
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


def safe_read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        return read_json_dict(path)
    except Exception:
        return None


def parse_ts_utc(ts: Any) -> Optional[int]:
    """
    Parse YYYY-MM-DDTHH:MM:SSZ to epoch seconds. Returns None on failure.
    """
    try:
        s = str(ts).strip()
        if not s.endswith("Z"):
            return None
        # naive parsing to avoid datetime import overhead; deterministic
        y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
        hh = int(s[11:13]); mm = int(s[14:16]); ss = int(s[17:19])
        # timegm for UTC
        return int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone
    except Exception:
        return None


def freshness_flag(ts_utc: Any, ttl_seconds: int) -> Tuple[str, Optional[float]]:
    """
    Return (flag, age_seconds).
      flag: fresh|stale|unknown
    """
    now = int(time.time())
    epoch = parse_ts_utc(ts_utc)
    if epoch is None or ttl_seconds <= 0:
        return "unknown", None
    age = float(now - epoch)
    if age < 0:
        return "unknown", age
    if age <= ttl_seconds:
        return "fresh", age
    return "stale", age


# -----------------------------
# Config models
# -----------------------------

@dataclass(frozen=True)
class SensorSpec:
    name: str
    path: str
    ttl_seconds: int

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "SensorSpec":
        n = str(obj.get("name") or "").strip()
        p = str(obj.get("path") or "").strip()
        ttl = int(obj.get("ttl_seconds") or 0)
        if not n or not p or ttl <= 0:
            raise ValueError(f"invalid_sensor_spec:{obj}")
        return cls(name=n, path=p, ttl_seconds=ttl)


@dataclass(frozen=True)
class IntelProfile:
    profile: str
    required: List[SensorSpec]
    optional: List[SensorSpec]
    on_required_stale: str

    @classmethod
    def from_profile_obj(cls, profile: str, obj: Dict[str, Any]) -> "IntelProfile":
        req_raw = obj.get("required_sensors") or []
        opt_raw = obj.get("optional_sensors") or []
        if not isinstance(req_raw, list) or not isinstance(opt_raw, list):
            raise ValueError("invalid_profile_lists")
        required = [SensorSpec.from_obj(x) for x in req_raw if isinstance(x, dict)]
        optional = [SensorSpec.from_obj(x) for x in opt_raw if isinstance(x, dict)]
        on_stale = str(obj.get("on_required_stale") or "TIGHTEN_ONLY").strip().upper()
        return cls(profile=profile, required=required, optional=optional, on_required_stale=on_stale)


def load_profiles() -> Dict[str, IntelProfile]:
    cfg = read_json_dict(PROFILES_PATH)
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("intel_profiles_missing_profiles")
    out: Dict[str, IntelProfile] = {}
    for k, v in profiles.items():
        if not isinstance(v, dict):
            continue
        out[str(k).strip().upper()] = IntelProfile.from_profile_obj(str(k).strip().upper(), v)
    if not out:
        raise ValueError("intel_profiles_empty")
    return out


def select_profile(profiles: Dict[str, IntelProfile]) -> IntelProfile:
    cfg = read_json_dict(PROFILES_PATH)
    defaults = cfg.get("defaults") or {}
    default_profile = str((defaults.get("default_profile") or "BUDGET")).strip().upper()
    env_name = str((defaults.get("active_profile_env") or "CHAD_INTEL_PROFILE")).strip()
    chosen = str(os.environ.get(env_name, default_profile)).strip().upper()
    return profiles.get(chosen) or profiles[default_profile] if default_profile in profiles else list(profiles.values())[0]


def load_allowlist_provider_ids() -> List[str]:
    cfg = read_json_dict(ALLOWLIST_PATH)
    providers = cfg.get("providers")
    if not isinstance(providers, list):
        raise ValueError("providers_allowlist_missing_providers")
    ids: List[str] = []
    for p in providers:
        if isinstance(p, dict):
            pid = str(p.get("provider_id") or "").strip()
            if pid:
                ids.append(pid)
    return ids


# -----------------------------
# Cache building
# -----------------------------

def build_cache_record(*, sensor: SensorSpec, src_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create normalized cache record, summary-only.
    """
    as_of = utc_now_iso()
    if src_obj is None:
        return {
            "schema_version": "intel_cache_record.v1",
            "sensor": sensor.name,
            "provider_id": "unknown",
            "as_of_utc": as_of,
            "ttl_seconds": int(sensor.ttl_seconds),
            "staleness_flag": "unknown",
            "age_seconds": None,
            "source_path": sensor.path,
            "payload": {},
            "notes": "source_missing_or_invalid"
        }

    ts = src_obj.get("ts_utc")
    ttl = int(src_obj.get("ttl_seconds") or sensor.ttl_seconds)
    flag, age = freshness_flag(ts, ttl)

    # payload is summary-only; we copy the source object but drop anything that looks like secrets
    # (these runtime publishers should not contain secrets; we still keep this conservative)
    payload = dict(src_obj)
    for k in list(payload.keys()):
        if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower():
            payload.pop(k, None)

    provider_id = "bootstrap"
    src = src_obj.get("source")
    if isinstance(src, dict) and src.get("provider"):
        provider_id = str(src.get("provider")).strip() or provider_id

    return {
        "schema_version": "intel_cache_record.v1",
        "sensor": sensor.name,
        "provider_id": provider_id,
        "as_of_utc": as_of,
        "ttl_seconds": int(ttl),
        "staleness_flag": flag,
        "age_seconds": age,
        "source_path": sensor.path,
        "payload": payload,
        "notes": "copied_from_runtime_publisher"
    }


def publish() -> int:
    # Ensure cache dir exists
    INTEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles()
    prof = select_profile(profiles)
    allow_ids = set(load_allowlist_provider_ids())

    # Build required + optional caches
    records: Dict[str, Dict[str, Any]] = {}
    required_missing = 0
    required_stale = 0
    required_unknown = 0

    def handle(sensor: SensorSpec, required: bool) -> None:
        nonlocal required_missing, required_stale, required_unknown
        src_path = (DEFAULT_ROOT / sensor.path) if not Path(sensor.path).is_absolute() else Path(sensor.path)
        # allow relative paths from repo root
        if not src_path.is_absolute():
            src_path = (DEFAULT_ROOT / src_path).resolve()

        src_obj = safe_read_json_dict(src_path)
        rec = build_cache_record(sensor=sensor, src_obj=src_obj)

        # Enforce provider allowlist if provider_id is known and non-bootstrap
        pid = str(rec.get("provider_id") or "").strip()
        if pid and pid not in ("bootstrap", "unknown") and pid not in allow_ids:
            rec["staleness_flag"] = "unknown"
            rec["notes"] = f"provider_not_allowlisted:{pid}"
            rec["payload"] = {}

        records[sensor.name] = rec

        if required:
            flag = str(rec.get("staleness_flag") or "unknown")
            if src_obj is None:
                required_missing += 1
            if flag == "stale":
                required_stale += 1
            elif flag == "unknown":
                required_unknown += 1

        # Write per-sensor cache file (stable name)
        out_path = INTEL_CACHE_DIR / f"{sensor.name}.json"
        atomic_write_json(out_path, rec)

    for s in prof.required:
        handle(s, required=True)
    for s in prof.optional:
        handle(s, required=False)

    # Global state summary
    state = {
        "schema_version": "intel_cache_state.v1",
        "ts_utc": utc_now_iso(),
        "ttl_seconds": int(DEFAULT_CACHE_TTL),
        "active_profile": prof.profile,
        "policy": {
            "on_required_stale": prof.on_required_stale,
            "tighten_only": True
        },
        "required": [s.name for s in prof.required],
        "optional": [s.name for s in prof.optional],
        "required_missing_count": int(required_missing),
        "required_stale_count": int(required_stale),
        "required_unknown_count": int(required_unknown),
        "notes": "Brains must consume runtime/intel_cache/*.json; stale/unknown required intel implies tighten-only."
    }
    atomic_write_json(STATE_PATH, state)
    return 0


def main() -> int:
    try:
        return publish()
    except Exception as exc:  # noqa: BLE001
        # Config parse errors should be visible; still return non-zero.
        err = {
            "schema_version": "intel_cache_state.v1",
            "ts_utc": utc_now_iso(),
            "ttl_seconds": int(DEFAULT_CACHE_TTL),
            "active_profile": "unknown",
            "required_missing_count": None,
            "required_stale_count": None,
            "required_unknown_count": None,
            "error": str(exc),
            "notes": "publish_failed_config_or_runtime_error"
        }
        INTEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(STATE_PATH, err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
