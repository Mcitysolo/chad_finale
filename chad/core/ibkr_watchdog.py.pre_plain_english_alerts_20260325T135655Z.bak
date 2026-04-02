from __future__ import annotations

"""
chad/core/ibkr_watchdog.py

CHAD IBKR Watchdog (Phase 9B Alerting)

Purpose
-------
Runs as a timer-driven oneshot to alert when IBKR health degrades.

Source of truth
---------------
Reads runtime/ibkr_status.json written by chad.core.ibkr_healthcheck (and/or related
health writers). This watchdog does NOT talk to IBKR directly.

Alert policy (default)
----------------------
Triggers Telegram alerts when any of the following are true:
- ok == False
- consecutive_failures >= max_failures (default 1)
- last_ok_age_seconds >= max_age_seconds (derived from ttl_seconds or default)

Safety / reliability
--------------------
- Never places orders.
- Never modifies trading state.
- Never raises to systemd unless the script itself is broken; failures are reported
  in the output JSON and exit code remains 0 (watchdogs must not crash-loop).
- Telegram notifier is fail-safe and dedupe-aware.

Environment
-----------
Uses TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_CHAT_ID. If missing, this module will
attempt to load /etc/chad/telegram.env (same file used by chad-telegram-bot.service).

Config env (optional)
---------------------
- IBKR_WATCHDOG_MAX_AGE_SECONDS          (default: max(3*ttl_seconds, 300))
- IBKR_WATCHDOG_MAX_FAILURES            (default: 1)
- IBKR_WATCHDOG_DEDUPE_TTL_SECONDS      (default: 900)  # forwarded via TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS if not set
- IBKR_WATCHDOG_SILENT                  (default: 0)    # 1 disables Telegram sends (still logs)
- IBKR_WATCHDOG_STATE_PATH              (default: runtime/ibkr_status.json)
- IBKR_WATCHDOG_OUTPUT_PATH             (default: runtime/ibkr_watchdog_last.json)

CLI
---
python -m chad.core.ibkr_watchdog
"""
import math
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.utils.telegram_notify import NotifyError, notify

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"
DEFAULT_STATE_PATH = RUNTIME_DIR / "ibkr_status.json"
DEFAULT_OUT_PATH = RUNTIME_DIR / "ibkr_watchdog_last.json"
DEFAULT_ENV_FILE = Path("/etc/chad/telegram.env")


def _utc_now_unix() -> float:
    return float(time.time())


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(x)
    except Exception:
        return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


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


def _load_env_file_if_missing(env_file: Path = DEFAULT_ENV_FILE) -> List[str]:
    """
    If TELEGRAM_* vars are missing, load them from /etc/chad/telegram.env.
    Returns a list of notes describing what happened.
    """
    notes: List[str] = []
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_ALLOWED_CHAT_ID"):
        return notes

    if not env_file.is_file():
        notes.append(f"telegram_env_missing:{env_file}")
        return notes

    try:
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_CHAT_ID"} and v:
                os.environ.setdefault(k, v)
        notes.append(f"telegram_env_loaded:{env_file}")
    except Exception as exc:
        notes.append(f"telegram_env_load_failed:{type(exc).__name__}:{exc}")

    return notes


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if not path.is_file():
            return None, f"missing:{path}"
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None, f"invalid_json_object:{path}"
        return obj, None
    except Exception as exc:
        return None, f"read_failed:{path}:{type(exc).__name__}:{exc}"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class IBKRStatus:
    ok: bool
    consecutive_failures: int
    last_ok_at_unix: float
    ts_utc_unix: float
    ttl_seconds: int

    @property
    def last_ok_age_seconds(self) -> float:
        if self.last_ok_at_unix <= 0:
            return float("inf")
        return max(0.0, self.ts_utc_unix - self.last_ok_at_unix)


def _parse_status(obj: Dict[str, Any]) -> IBKRStatus:
    ok = _safe_bool(obj.get("ok", False), False)
    failures = _safe_int(obj.get("consecutive_failures", 0), 0)

    # Prefer explicit unix timestamps if present.
    last_ok_at = _safe_float(obj.get("last_ok_at", 0.0), 0.0)
    ts_unix = _safe_float(obj.get("ts_unix", 0.0), 0.0)

    # Fallback: parse ISO timestamps emitted by ibkr_healthcheck (ts_utc / server_time_iso).
    def _iso_to_unix(v: Any) -> float:
        s = str(v or "").strip()
        if not s:
            return 0.0
        try:
            # Accept "Z" suffix.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            from datetime import datetime

            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                # Treat naive as UTC
                from datetime import timezone

                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return 0.0

    if ts_unix <= 0.0:
        ts_unix = _iso_to_unix(obj.get("ts_utc")) or _utc_now_unix()

    # If last_ok_at isn't present, derive it:
    # - when ok==true, treat this sample time as last known OK
    # - otherwise leave as 0.0 so watchdog can mark stale appropriately
    if last_ok_at <= 0.0 and ok:
        last_ok_at = float(ts_unix)

    ttl = _safe_int(obj.get("ttl_seconds", 0), 0)
    return IBKRStatus(
        ok=ok,
        consecutive_failures=failures,
        last_ok_at_unix=float(last_ok_at),
        ts_utc_unix=float(ts_unix),
        ttl_seconds=ttl,
    )


def _derive_thresholds(st: IBKRStatus) -> Tuple[float, int]:
    max_failures = max(1, _env_int("IBKR_WATCHDOG_MAX_FAILURES", 1))

    # If ttl is known, allow up to 3x ttl before declaring stale; otherwise 300s default.
    derived_age = 0.0
    if st.ttl_seconds > 0:
        derived_age = float(st.ttl_seconds) * 3.0
    max_age = _env_float("IBKR_WATCHDOG_MAX_AGE_SECONDS", max(derived_age, 300.0))

    return float(max_age), int(max_failures)


def _send_alerts(st: IBKRStatus, *, max_age: float, max_failures: int) -> List[Dict[str, Any]]:
    """
    Returns list of alert events produced (whether sent or suppressed).
    """
    events: List[Dict[str, Any]] = []
    silent = _safe_bool(os.environ.get("IBKR_WATCHDOG_SILENT", "0"), False)

    # Align dedupe TTL with watchdog TTL unless overridden.
    dedupe_ttl = _env_int("IBKR_WATCHDOG_DEDUPE_TTL_SECONDS", 0)
    if dedupe_ttl > 0 and not os.environ.get("TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS"):
        os.environ["TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS"] = str(dedupe_ttl)

    def emit(*, msg: str, severity: str, dedupe_key: str) -> None:
        sent = False
        err = ""
        if silent:
            events.append({"event": "alert_suppressed_silent", "severity": severity, "dedupe_key": dedupe_key, "message": msg})
            return
        try:
            sent = notify(msg, severity=severity, dedupe_key=dedupe_key, raise_on_fail=False)
        except NotifyError as exc:
            err = str(exc)
            sent = False
        events.append({"event": "alert_attempt", "severity": severity, "dedupe_key": dedupe_key, "sent": bool(sent), "error": err, "message": msg})

    # Hard down
    if not st.ok:
        emit(
            msg=f"IBKR WATCHDOG: ibkr_status.ok=false (failures={st.consecutive_failures}, last_ok_age_s={st.last_ok_age_seconds:.1f})",
            severity="critical",
            dedupe_key="ibkr_down",
        )

    # Stale
    if st.last_ok_age_seconds >= float(max_age):
        emit(
            msg=f"IBKR WATCHDOG: stale heartbeat (last_ok_age_s={st.last_ok_age_seconds:.1f} >= {max_age:.1f}) failures={st.consecutive_failures}",
            severity="critical",
            dedupe_key="ibkr_stale",
        )

    # Failures threshold
    if st.consecutive_failures >= int(max_failures) and st.consecutive_failures > 0:
        emit(
            msg=f"IBKR WATCHDOG: consecutive_failures={st.consecutive_failures} >= {max_failures} (last_ok_age_s={st.last_ok_age_seconds:.1f})",
            severity="warn",
            dedupe_key="ibkr_failures",
        )

    return events


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv  # reserved

    notes: List[str] = []
    notes.extend(_load_env_file_if_missing(DEFAULT_ENV_FILE))

    state_path = _env_path("IBKR_WATCHDOG_STATE_PATH", DEFAULT_STATE_PATH)
    out_path = _env_path("IBKR_WATCHDOG_OUTPUT_PATH", DEFAULT_OUT_PATH)

    raw, err = _read_json(state_path)
    now_unix = _utc_now_unix()

    if raw is None:
        payload: Dict[str, Any] = {
            "ts_unix": now_unix,
            "ok": False,
            "reason": "no_state",
            "error": err,
            "notes": notes,
            "state_path": str(state_path),
            "alerts": [],
        }
        # If no state file, that itself is an alert condition (feed stale / service down).
        try:
            events = []
            silent = _safe_bool(os.environ.get("IBKR_WATCHDOG_SILENT", "0"), False)
            if not silent:
                events = _send_alerts(
                    IBKRStatus(ok=False, consecutive_failures=9999, last_ok_at_unix=0.0, ts_utc_unix=now_unix, ttl_seconds=0),
                    max_age=300.0,
                    max_failures=1,
                )
            payload["alerts"] = events
        except Exception:
            pass
        _atomic_write_json(out_path, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    st = _parse_status(raw)
    max_age, max_failures = _derive_thresholds(st)

    alerts = _send_alerts(st, max_age=max_age, max_failures=max_failures)

    payload2: Dict[str, Any] = {
        "ts_unix": now_unix,
        "state_path": str(state_path),
        "ok": bool(st.ok),
        "consecutive_failures": int(st.consecutive_failures),
        "last_ok_at": float(st.last_ok_at_unix),
        "last_ok_age_seconds": float(st.last_ok_age_seconds) if math.isfinite(st.last_ok_age_seconds) else 1e18,
        "ttl_seconds": int(st.ttl_seconds),
        "thresholds": {"max_age_seconds": float(max_age), "max_failures": int(max_failures)},
        "alerts": alerts,
        "notes": notes,
    }

    _atomic_write_json(out_path, payload2)
    print(json.dumps(payload2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
