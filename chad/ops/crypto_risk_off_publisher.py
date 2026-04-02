#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from chad.utils.runtime_json import read_runtime_state_json, write_runtime_state_json

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _runtime_dir() -> Path:
    env = os.getenv("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = os.getenv("CHAD_ROOT", "").strip()
    if root:
        return (Path(root).expanduser().resolve() / "runtime").resolve()
    return (Path.cwd() / "runtime").resolve()

@dataclass(frozen=True)
class RiskOffPayload:
    ts_utc: str
    ttl_seconds: int
    risk_off: bool
    risk_label: str
    reasons: list[str]
    sources: Dict[str, Any]
    notes: str

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main() -> int:
    ap = argparse.ArgumentParser(description="Publish runtime/crypto_risk_off.json (TTL, fail-safe).")
    ap.add_argument("--ttl-seconds", type=int, default=int(os.getenv("CHAD_CRYPTO_RISK_OFF_TTL", "300")), help="TTL for crypto_risk_off.json")
    ap.add_argument("--out", default="", help="Optional override output path (default: <runtime>/crypto_risk_off.json)")
    args = ap.parse_args()

    runtime = _runtime_dir()
    out_path = Path(args.out).expanduser().resolve() if args.out else (runtime / "crypto_risk_off.json")

    ts = _utc_now()
    ttl = int(args.ttl_seconds) if int(args.ttl_seconds) > 0 else 300

    reasons: list[str] = []
    sources: Dict[str, Any] = {}

    # Source 1: macro_state risk_label (preferred)
    macro = read_runtime_state_json(runtime / "macro_state.json")  # may be None
    risk_label = "unknown"
    if isinstance(macro, dict):
        risk_label = str(macro.get("risk_label") or "unknown")
        sources["macro_state"] = {"present": True, "ts_utc": macro.get("ts_utc"), "ttl_seconds": macro.get("ttl_seconds"), "risk_label": risk_label}
    else:
        sources["macro_state"] = {"present": False}

    # Source 2: full_execution_cycle_last.json (debug / evidence)
    cycle = _read_json(runtime / "full_execution_cycle_last.json")
    if isinstance(cycle, dict):
        counts = cycle.get("counts") if isinstance(cycle.get("counts"), dict) else {}
        summary = cycle.get("summary") if isinstance(cycle.get("summary"), dict) else {}

        sources["full_execution_cycle_last"] = {
            "present": True,
            "ts_utc": summary.get("now_iso") or cycle.get("ts_utc"),
            "raw_signals": counts.get("raw_signals"),
            "routed_signals": counts.get("routed_signals"),
            "orders_count": counts.get("orders_count"),
            "execution_mode": summary.get("execution_mode"),
        }
    else:
        sources["full_execution_cycle_last"] = {"present": False}

    # Determination rule (tighten-only / fail-safe):
    # - If macro says risk_off -> risk_off=True
    # - If macro is missing/unknown -> default risk_off=True (fail-safe for crypto)
    if risk_label == "risk_off":
        risk_off = True
        reasons.append("macro:risk_off")
    elif risk_label in ("risk_on", "neutral"):
        risk_off = False
        reasons.append(f"macro:{risk_label}")
    else:
        risk_off = True
        reasons.append("macro:unknown_or_missing=>fail_safe_risk_off")

    payload = RiskOffPayload(
        ts_utc=ts,
        ttl_seconds=ttl,
        risk_off=risk_off,
        risk_label=risk_label,
        reasons=reasons,
        sources=sources,
        notes="Publisher is tighten-only. When unknown, defaults to risk_off=true for safety. Intended for CHAD AlphaCrypto gating/tilt.",
    )

    ok = write_runtime_state_json(out_path, asdict(payload), ttl_seconds=ttl, inject_ts=False)
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
