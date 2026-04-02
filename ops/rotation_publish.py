#!/usr/bin/env python3
"""
CHAD Rotation Rules Layer — config-driven publisher (Production, advisory-only)

File: ops/rotation_publish.py

Outputs:
  /home/ubuntu/chad_finale/reports/rotation/ROTATION_<ts>.json
Optional pointer:
  /home/ubuntu/chad_finale/runtime/rotation_state.json

SSOT v4.2 contract:
- Rotation Rules produce tilts + blocks as advisory proposals, never direct execution.
- Outputs must be bounded and fail-closed if inputs are stale/unknown.
- Artifacts logged under reports/rotation/ and referenced by the system.

Config contract:
- Reads config/rotation_rules.json when present.
- Preserves prior bootstrap behavior by default.
- No broker calls. No secrets. Advisory-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
DEFAULT_REPORTS_DIR = Path("/home/ubuntu/chad_finale/reports/rotation")
DEFAULT_CONFIG_PATH = Path("/home/ubuntu/chad_finale/config/rotation_rules.json")


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


def _safe_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_rotation_rules() -> Dict[str, Any]:
    runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
    reports_dir = Path(os.environ.get("CHAD_ROTATION_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
    config_path = Path(os.environ.get("CHAD_ROTATION_RULES_PATH", str(DEFAULT_CONFIG_PATH)))

    base: Dict[str, Any] = {
        "schema_version": "rotation_rules.v1",
        "enabled": True,
        "advisory_only": True,
        "write_pointer": _safe_bool(os.environ.get("CHAD_ROTATION_WRITE_POINTER", "1"), True),
        "ttl_seconds": _safe_int(os.environ.get("CHAD_ROTATION_TTL_SECONDS", "3600"), 3600),
        "fail_closed_on_missing_inputs": True,
        "fail_closed_event_risk_severities": ["high", "unknown"],
        "max_abs_tilt": _safe_float(os.environ.get("CHAD_ROTATION_MAX_ABS_TILT", "0.05"), 0.05),
        "inputs": {
            "macro_state_path": str(runtime_dir / "macro_state.json"),
            "event_risk_path": str(runtime_dir / "event_risk.json"),
            "sector_rotation_path": str(runtime_dir / "sector_rotation.json"),
        },
        "output": {
            "reports_dir": str(reports_dir),
            "pointer_path": str(runtime_dir / "rotation_state.json"),
        },
        "regime_rules": {
            "event_risk_override": {
                "growth_momo": -0.05,
                "hedge": 0.05,
                "reason": "event_risk_high_or_unknown",
            },
            "risk_off": {
                "growth_momo": -0.05,
                "hedge": 0.05,
                "reason": "macro_risk_off",
            },
            "risk_on": {
                "growth_momo": 0.05,
                "hedge": -0.05,
                "reason": "macro_risk_on",
            },
            "neutral": {
                "growth_momo": 0.0,
                "hedge": 0.0,
                "reason": "macro_neutral",
            },
        },
        "symbol_blocks": [],
        "notes": [
            "Bootstrap fallback contract active.",
            "No broker calls. Advisory-only.",
        ],
        "_meta": {
            "config_path": str(config_path),
            "config_loaded": False,
        },
    }

    if not config_path.is_file():
        return base

    raw = read_json_dict(config_path)

    cfg = dict(base)
    cfg["schema_version"] = str(raw.get("schema_version", base["schema_version"]))
    cfg["enabled"] = _safe_bool(raw.get("enabled"), True)
    cfg["advisory_only"] = _safe_bool(raw.get("advisory_only"), True)
    cfg["write_pointer"] = _safe_bool(raw.get("write_pointer"), base["write_pointer"])
    cfg["ttl_seconds"] = _safe_int(raw.get("ttl_seconds"), base["ttl_seconds"])
    cfg["fail_closed_on_missing_inputs"] = _safe_bool(raw.get("fail_closed_on_missing_inputs"), True)
    cfg["fail_closed_event_risk_severities"] = list(
        raw.get("fail_closed_event_risk_severities", base["fail_closed_event_risk_severities"])
    )
    cfg["max_abs_tilt"] = abs(_safe_float(raw.get("max_abs_tilt"), base["max_abs_tilt"]))

    raw_inputs = raw.get("inputs", {})
    if not isinstance(raw_inputs, dict):
        raw_inputs = {}
    cfg["inputs"] = {
        "macro_state_path": str(raw_inputs.get("macro_state_path", base["inputs"]["macro_state_path"])),
        "event_risk_path": str(raw_inputs.get("event_risk_path", base["inputs"]["event_risk_path"])),
        "sector_rotation_path": str(raw_inputs.get("sector_rotation_path", base["inputs"]["sector_rotation_path"])),
    }

    raw_output = raw.get("output", {})
    if not isinstance(raw_output, dict):
        raw_output = {}
    cfg["output"] = {
        "reports_dir": str(raw_output.get("reports_dir", base["output"]["reports_dir"])),
        "pointer_path": str(raw_output.get("pointer_path", base["output"]["pointer_path"])),
    }

    raw_regime_rules = raw.get("regime_rules", {})
    if not isinstance(raw_regime_rules, dict):
        raw_regime_rules = {}
    regime_rules: Dict[str, Dict[str, Any]] = {}
    for key, fallback_rule in base["regime_rules"].items():
        candidate = raw_regime_rules.get(key, {})
        if not isinstance(candidate, dict):
            candidate = {}
        regime_rules[key] = {
            "growth_momo": clamp(
                _safe_float(candidate.get("growth_momo"), fallback_rule["growth_momo"]),
                -cfg["max_abs_tilt"],
                cfg["max_abs_tilt"],
            ),
            "hedge": clamp(
                _safe_float(candidate.get("hedge"), fallback_rule["hedge"]),
                -cfg["max_abs_tilt"],
                cfg["max_abs_tilt"],
            ),
            "reason": str(candidate.get("reason", fallback_rule["reason"])),
        }
    cfg["regime_rules"] = regime_rules

    symbol_blocks = raw.get("symbol_blocks", [])
    cfg["symbol_blocks"] = symbol_blocks if isinstance(symbol_blocks, list) else []

    notes = raw.get("notes", [])
    cfg["notes"] = notes if isinstance(notes, list) else base["notes"]

    cfg["_meta"] = {
        "config_path": str(config_path),
        "config_loaded": True,
    }
    return cfg


def build_tilts_from_rule(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    reason = str(rule.get("reason", "rotation_rule"))
    return [
        {
            "sleeve": "growth_momo",
            "delta_weight": float(rule.get("growth_momo", 0.0)),
            "reason": reason,
        },
        {
            "sleeve": "hedge",
            "delta_weight": float(rule.get("hedge", 0.0)),
            "reason": reason,
        },
    ]


def compute_tilts(
    macro: Dict[str, Any],
    event_risk: Dict[str, Any],
    rules: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], str]:
    risk_label = _safe_str(macro, "risk_label").strip().lower()
    sev = _safe_str(event_risk, "severity").strip().lower()

    fail_closed_sev = {
        str(x).strip().lower()
        for x in rules.get("fail_closed_event_risk_severities", ["high", "unknown"])
    }
    regime_rules = rules["regime_rules"]

    if sev in fail_closed_sev:
        return "risk_off", build_tilts_from_rule(regime_rules["event_risk_override"]), "event_risk_override"

    if risk_label == "risk_off":
        return "risk_off", build_tilts_from_rule(regime_rules["risk_off"]), "risk_off"
    if risk_label == "risk_on":
        return "risk_on", build_tilts_from_rule(regime_rules["risk_on"]), "risk_on"
    return "neutral", build_tilts_from_rule(regime_rules["neutral"]), "neutral"


def build_rotation_payload() -> Dict[str, Any]:
    ts = utc_now_iso()
    rules = load_rotation_rules()

    macro_path = Path(rules["inputs"]["macro_state_path"])
    event_risk_path = Path(rules["inputs"]["event_risk_path"])
    sector_rot_path = Path(rules["inputs"]["sector_rotation_path"])

    ttl_seconds = _safe_int(rules.get("ttl_seconds"), 3600)

    if not _safe_bool(rules.get("enabled"), True):
        return {
            "ts_utc": ts,
            "ttl_seconds": ttl_seconds,
            "schema_version": "rotation.v1",
            "regime": "disabled",
            "tilts": [],
            "symbol_blocks": [],
            "notes": "rotation_rules_disabled",
            "inputs": {
                "macro_state": {"path": str(macro_path), "exists": macro_path.is_file()},
                "event_risk": {"path": str(event_risk_path), "exists": event_risk_path.is_file()},
                "sector_rotation": {"path": str(sector_rot_path), "exists": sector_rot_path.is_file()},
            },
            "config": {
                "schema_version": rules.get("schema_version"),
                "advisory_only": _safe_bool(rules.get("advisory_only"), True),
                "config_path": rules["_meta"]["config_path"],
                "config_loaded": rules["_meta"]["config_loaded"],
            },
        }

    try:
        macro = read_json_dict(macro_path)
        event_risk = read_json_dict(event_risk_path)
        sector_rot = read_json_dict(sector_rot_path)
    except Exception as exc:
        return {
            "ts_utc": ts,
            "ttl_seconds": ttl_seconds,
            "schema_version": "rotation.v1",
            "regime": "unknown",
            "tilts": [],
            "symbol_blocks": [],
            "notes": f"fail_closed_missing_inputs:{type(exc).__name__}",
            "inputs": {
                "macro_state": {"path": str(macro_path), "exists": macro_path.is_file()},
                "event_risk": {"path": str(event_risk_path), "exists": event_risk_path.is_file()},
                "sector_rotation": {"path": str(sector_rot_path), "exists": sector_rot_path.is_file()},
            },
            "config": {
                "schema_version": rules.get("schema_version"),
                "advisory_only": _safe_bool(rules.get("advisory_only"), True),
                "config_path": rules["_meta"]["config_path"],
                "config_loaded": rules["_meta"]["config_loaded"],
            },
        }

    regime, tilts, rule_key = compute_tilts(macro, event_risk, rules)

    notes = [
        f"rotation_rule_applied:{rule_key}",
        f"config_loaded:{rules['_meta']['config_loaded']}",
    ]
    for n in rules.get("notes", []):
        notes.append(str(n))

    return {
        "ts_utc": ts,
        "ttl_seconds": ttl_seconds,
        "schema_version": "rotation.v1",
        "regime": regime,
        "tilts": tilts,
        "symbol_blocks": list(rules.get("symbol_blocks", [])),
        "notes": "; ".join(notes),
        "inputs": {
            "macro_state": {"ts_utc": macro.get("ts_utc"), "risk_label": macro.get("risk_label")},
            "event_risk": {"ts_utc": event_risk.get("ts_utc"), "severity": event_risk.get("severity")},
            "sector_rotation": {"ts_utc": sector_rot.get("ts_utc"), "provider_status": sector_rot.get("provider_status")},
        },
        "config": {
            "schema_version": rules.get("schema_version"),
            "advisory_only": _safe_bool(rules.get("advisory_only"), True),
            "max_abs_tilt": _safe_float(rules.get("max_abs_tilt"), 0.05),
            "config_path": rules["_meta"]["config_path"],
            "config_loaded": rules["_meta"]["config_loaded"],
        },
    }


def main() -> int:
    payload = build_rotation_payload()
    rules = load_rotation_rules()

    reports_dir = Path(rules["output"]["reports_dir"])
    pointer_path = Path(rules["output"]["pointer_path"])
    write_pointer = _safe_bool(rules.get("write_pointer"), True)

    ts_compact = utc_now_compact()
    reports_dir.mkdir(parents=True, exist_ok=True)

    out = reports_dir / f"ROTATION_{ts_compact}.json"
    atomic_write_json(out, payload)

    if write_pointer:
        pointer = {
            "ts_utc": payload.get("ts_utc"),
            "ttl_seconds": payload.get("ttl_seconds"),
            "latest_rotation_path": str(out),
            "latest_rotation_sha256": sha256_hex(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            ),
            "schema_version": "rotation_state.v1",
        }
        atomic_write_json(pointer_path, pointer)

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "ts_utc": payload.get("ts_utc"),
                "config_loaded": rules["_meta"]["config_loaded"],
                "config_path": rules["_meta"]["config_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
