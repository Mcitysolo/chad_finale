from __future__ import annotations

"""
chad/core/decision_trace_heartbeat.py

CHAD DecisionTrace Heartbeat (Phase 9B/9C)

Purpose
-------
This is a timer-driven *oneshot* that:
1) Probes the LiveGate endpoint (source of truth for trading permissions),
2) Writes a heartbeat snapshot to runtime/decision_trace_heartbeat.json (atomic),
3) Sends Telegram alerts if LiveGate is unreachable or returns invalid JSON.

Why it matters (CSB alignment)
------------------------------
Phase 9C requires trace continuity: every real-world decision chain must be auditable.
This heartbeat provides a periodic, machine-readable proof that CHAD's LiveGate
evaluation was reachable and what it returned at time T (even when no trades occur).

Safety
------
- No broker calls.
- No order placement.
- Only reads HTTP endpoint + writes local runtime artifact.
- Alerts are best-effort and dedupe-aware.

Environment
-----------
- LIVE_GATE_URL (optional; default http://127.0.0.1:9618/live-gate)
- DECISION_TRACE_HEARTBEAT_OUT (optional; default runtime/decision_trace_heartbeat.json)
- DECISION_TRACE_HEARTBEAT_TIMEOUT_SECONDS (optional; default 2.0)

Telegram
--------
Uses TELEGRAM_BOT_TOKEN + TELEGRAM_ALLOWED_CHAT_ID.
If missing, attempts to load /etc/chad/telegram.env (same as the bot service).

Run
---
python -m chad.core.decision_trace_heartbeat
"""

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.utils.telegram_notify import NotifyError, notify

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"
DEFAULT_OUT_PATH = RUNTIME_DIR / "decision_trace_heartbeat.json"
DEFAULT_ENV_FILE = Path("/etc/chad/telegram.env")

DEFAULT_LIVE_GATE_URL = "http://127.0.0.1:9618/live-gate"
DEFAULT_TIMEOUT_S = 2.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _env(name: str, default: str) -> str:
    v = str(os.environ.get(name) or "").strip()
    return v if v else default


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_path(name: str, default: Path) -> Path:
    raw = str(os.environ.get(name) or "").strip()
    return Path(raw) if raw else default


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_env_file_if_missing(env_file: Path = DEFAULT_ENV_FILE) -> None:
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_ALLOWED_CHAT_ID"):
        return
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_CHAT_ID"} and v:
            os.environ.setdefault(k, v)


def _alert(msg: str, *, severity: str, dedupe_key: str) -> bool:
    try:
        _load_env_file_if_missing()
        return bool(notify(msg, severity=severity, dedupe_key=dedupe_key, raise_on_fail=False))
    except NotifyError:
        return False
    except Exception:
        return False


@dataclass(frozen=True)
class LiveGateResult:
    ok: bool
    payload: Optional[Dict[str, Any]]
    error: Optional[str]
    latency_ms: float


def _fetch_live_gate(url: str, timeout_s: float) -> LiveGateResult:
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=float(timeout_s)) as r:
            raw = r.read().decode("utf-8", errors="replace")
        j = json.loads(raw)
        if not isinstance(j, dict):
            return LiveGateResult(ok=False, payload=None, error="live_gate_json_not_object", latency_ms=(time.time() - t0) * 1000.0)
        return LiveGateResult(ok=True, payload=j, error=None, latency_ms=(time.time() - t0) * 1000.0)
    except Exception as exc:
        return LiveGateResult(ok=False, payload=None, error=f"{type(exc).__name__}: {exc}", latency_ms=(time.time() - t0) * 1000.0)


def main() -> int:
    now = _utc_now()
    out_path = _env_path("DECISION_TRACE_HEARTBEAT_OUT", DEFAULT_OUT_PATH)
    url = _env("LIVE_GATE_URL", DEFAULT_LIVE_GATE_URL)
    timeout_s = _env_float("DECISION_TRACE_HEARTBEAT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_S)

    res = _fetch_live_gate(url, timeout_s=timeout_s)

    payload: Dict[str, Any] = {
        "ts_utc": _iso(now),
        "ok": bool(res.ok),
        "live_gate_url": url,
        "latency_ms": float(res.latency_ms),
        "error": res.error,
        "live_gate": res.payload,
    }

    # Alerts: only on failure to fetch/parse.
    alert_sent = False
    if not res.ok:
        alert_sent = _alert(
            f"DecisionTrace Heartbeat: LiveGate unreachable/invalid. url={url} err={res.error} latency_ms={res.latency_ms:.1f}",
            severity="critical",
            dedupe_key="decision_trace_livegate_down",
        )
    payload["alert_sent"] = bool(alert_sent)
    _atomic_write_json(out_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
