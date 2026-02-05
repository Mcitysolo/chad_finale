#!/usr/bin/env python3
"""
CHAD Phase 12 — Self-Regulation Governor Publisher (paper-first, tighten-only)

File: ops/governor_publish.py

Outputs:
- runtime/governor_state.json                (pointer, current posture)
- reports/governor/GOVERNOR_<ts>.json        (full snapshot)

Purpose
-------
The Governor is a safety layer that can only TIGHTEN.
It never enables live trading, never loosens risk, never places orders.

It inspects:
- SCR shadow state (state/paper_only/sizing_factor + stats)
- LiveGate snapshot
- stop_state
- event_risk severity
- macro risk label
- live_readiness pointer (if present)

Then produces:
- governor_mode: NORMAL | TIGHTEN | PAUSE | DENY_ALL
- recommended actions (advisory)
- reasons list (auditable)
- ready_for_live flag (copied from live_readiness if present)

Fail-closed:
- Any unreadable inputs => more conservative mode.

No secrets, no broker calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import Request, urlopen

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime")).resolve()
REPORTS_DIR = Path(os.environ.get("CHAD_REPORTS_DIR", "/home/ubuntu/CHAD FINALE/reports")).resolve()
OUT_DIR = REPORTS_DIR / "governor"

POINTER_PATH = RUNTIME_DIR / "governor_state.json"

LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")
SHADOW_URL = os.environ.get("CHAD_SHADOW_URL", "http://127.0.0.1:9618/shadow")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "4.0"))

STOP_PATH = RUNTIME_DIR / "stop_state.json"
EVENT_RISK_PATH = RUNTIME_DIR / "event_risk.json"
MACRO_PATH = RUNTIME_DIR / "macro_state.json"
LIVE_READY_PTR = RUNTIME_DIR / "live_readiness.json"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_hex(data)

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

    return digest


def read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_dict:{path}")
    return obj


def http_get_json(url: str) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "chad-governor/1.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("http_json_not_dict")
    return obj


def safe_load(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return read_json_dict(path)
    except Exception:
        return default


def stop_state() -> Tuple[bool, str]:
    s = safe_load(STOP_PATH, {})
    if bool(s.get("stop", False)):
        return True, str(s.get("reason") or "stop_true")
    return False, str(s.get("reason") or "stop_false")


def event_severity() -> str:
    e = safe_load(EVENT_RISK_PATH, {})
    return str(e.get("severity") or "unknown").strip().lower()


def macro_label() -> str:
    m = safe_load(MACRO_PATH, {})
    return str(m.get("risk_label") or "unknown").strip().lower()


def live_ready_flag() -> Tuple[bool, str]:
    p = safe_load(LIVE_READY_PTR, {})
    if not p:
        return False, "live_readiness_missing"
    return bool(p.get("ready_for_live", False)), "live_readiness_present"


def decide_mode(stop: bool, scr_paper_only: bool, scr_state: str, sev: str) -> Tuple[str, list[str], list[dict]]:
    reasons: list[str] = []
    actions: list[dict] = []

    if stop:
        reasons.append("STOP=true => DENY_ALL")
        actions.append({"action": "deny_all", "reason": "stop_engaged"})
        return "DENY_ALL", reasons, actions

    # If SCR is PAUSED or paper_only, governor must keep system conservative
    if scr_state.upper() == "PAUSED":
        reasons.append("SCR state=PAUSED => PAUSE")
        actions.append({"action": "pause", "reason": "scr_paused"})
        return "PAUSE", reasons, actions

    if scr_paper_only:
        reasons.append("SCR paper_only=true => TIGHTEN")
        actions.append({"action": "tighten", "reason": "scr_paper_only"})
        return "TIGHTEN", reasons, actions

    # Event risk based tightening
    if sev in ("high", "unknown"):
        reasons.append(f"event_risk={sev} => TIGHTEN")
        actions.append({"action": "tighten", "reason": f"event_risk_{sev}"})
        return "TIGHTEN", reasons, actions

    reasons.append("all checks normal => NORMAL")
    actions.append({"action": "normal", "reason": "baseline"})
    return "NORMAL", reasons, actions


def main() -> int:
    ts = utc_now_iso()
    ts_compact = utc_now_compact()

    # Pull live snapshots (fail-closed)
    try:
        shadow = http_get_json(SHADOW_URL)
    except Exception as exc:
        shadow = {"error": f"shadow_fetch_failed:{type(exc).__name__}"}

    try:
        live_gate = http_get_json(LIVE_GATE_URL)
    except Exception as exc:
        live_gate = {"error": f"live_gate_fetch_failed:{type(exc).__name__}"}

    stop, stop_reason = stop_state()
    sev = event_severity()
    macro = macro_label()
    ready_for_live, ready_note = live_ready_flag()

    sh = shadow.get("shadow") if isinstance(shadow.get("shadow"), dict) else {}
    scr_state = str(sh.get("state") or "UNKNOWN")
    scr_paper_only = bool(sh.get("paper_only", True))

    governor_mode, reasons, actions = decide_mode(stop, scr_paper_only, scr_state, sev)

    payload = {
        "ts_utc": ts,
        "schema_version": "governor.v1",
        "governor_mode": governor_mode,
        "reasons": reasons,
        "actions": actions,
        "inputs": {
            "stop": {"stop": stop, "reason": stop_reason},
            "event_risk_severity": sev,
            "macro_risk_label": macro,
            "scr": {"state": scr_state, "paper_only": scr_paper_only},
            "ready_for_live": {"value": ready_for_live, "note": ready_note},
        },
        "shadow_snapshot": shadow,
        "live_gate_snapshot": live_gate,
        "notes": "tighten-only self-regulation scaffold; no live changes",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"GOVERNOR_{ts_compact}.json"
    digest = atomic_write_json(report_path, payload)

    pointer = {
        "ts_utc": ts,
        "schema_version": "governor_state.v1",
        "latest_report_path": str(report_path),
        "latest_report_sha256": f"sha256:{digest}",
        "governor_mode": governor_mode,
        "ready_for_live": bool(ready_for_live),
    }
    atomic_write_json(POINTER_PATH, pointer)

    print(json.dumps({"ok": True, "governor_mode": governor_mode, "report_path": str(report_path), "pointer_path": str(POINTER_PATH), "ts_utc": ts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
