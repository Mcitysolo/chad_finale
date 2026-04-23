#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen

RUNTIME = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime"))

# Canonical SCR runtime artifact (SSOT)
OUT_SCR = RUNTIME / "scr_state.json"

# Back-compat + unified runtime truth for anything that still reads runtime/shadow_state.json
OUT_SHADOW = RUNTIME / "shadow_state.json"

SHADOW_URL = os.environ.get("CHAD_SHADOW_URL", "http://127.0.0.1:9618/shadow")
TTL_SECONDS = int(os.environ.get("CHAD_SCR_TTL_SECONDS", "180"))


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Best-effort fsync of directory entry for crash-safety
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def _fetch_shadow() -> Dict[str, Any]:
    req = Request(SHADOW_URL, headers={"User-Agent": "chad-scr-sync/1.2"})
    with urlopen(req, timeout=3) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or "shadow" not in obj:
        raise RuntimeError("invalid_shadow_payload")
    return obj


def _build_payloads(*, ts: str, shadow: Dict[str, Any], ok: bool, err: str | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (scr_state_payload, shadow_state_payload)
    - scr_state.json: SSOT canonical SCR artifact (includes source, schema_version)
    - shadow_state.json: compatibility artifact for LiveGate fallback + operator surfaces
    """
    state = shadow.get("state") if ok else "UNKNOWN"
    paper_only = bool(shadow.get("paper_only")) if ok else True
    sizing_factor = float(shadow.get("sizing_factor") or 0.0) if ok else 0.0
    reasons = list(shadow.get("reasons") or [])
    stats = dict(shadow.get("stats") or {})

    if not ok and err:
        reasons = [f"scr_sync_error:{err}"]
        stats = {}

    scr_payload: Dict[str, Any] = {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "state": state,
        "paper_only": paper_only,
        "sizing_factor": sizing_factor,
        "reasons": reasons,
        "stats": stats,
        "source": {"url": SHADOW_URL},
        "schema_version": "scr_state.v1",
    }

    # Keep this minimal and compatible with existing readers that expect these keys.
    shadow_payload: Dict[str, Any] = {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "state": state,
        "paper_only": paper_only,
        "sizing_factor": sizing_factor,
        "reasons": reasons,
        "stats": stats,
    }

    return scr_payload, shadow_payload


def main() -> int:
    ts = _utc_now_iso()
    try:
        obj = _fetch_shadow()
        shadow = obj.get("shadow") or {}
        scr_payload, shadow_payload = _build_payloads(ts=ts, shadow=shadow, ok=True, err=None)

        _atomic_write_json(OUT_SCR, scr_payload)
        _atomic_write_json(OUT_SHADOW, shadow_payload)

        # Fire Telegram alert on SCR state transitions (WARMUP → CAUTIOUS →
        # CONFIDENT → PAUSED). Uses runtime/scr_last_notified_state.json as
        # sentinel so we only fire on transitions, not every sync cycle.
        try:
            import sys
            _REPO_ROOT = Path(__file__).resolve().parents[1]
            if str(_REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(_REPO_ROOT))
            from chad.utils.telegram_notify import check_and_send_scr_milestone
            _scr_state_now = str(scr_payload.get("state", "") or "").upper()
            _eff = int((scr_payload.get("stats") or {}).get("effective_trades") or 0)
            if _scr_state_now:
                check_and_send_scr_milestone(_scr_state_now, _eff)
        except Exception:
            pass

        print(
            json.dumps(
                {"ok": True, "out_scr": str(OUT_SCR), "out_shadow": str(OUT_SHADOW), "ts_utc": ts},
                sort_keys=True,
            )
        )
        return 0

    except Exception as e:
        scr_payload, shadow_payload = _build_payloads(ts=ts, shadow={}, ok=False, err=f"{type(e).__name__}:{e}")

        _atomic_write_json(OUT_SCR, scr_payload)
        _atomic_write_json(OUT_SHADOW, shadow_payload)

        print(
            json.dumps(
                {"ok": False, "error": str(e), "out_scr": str(OUT_SCR), "out_shadow": str(OUT_SHADOW), "ts_utc": ts},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
