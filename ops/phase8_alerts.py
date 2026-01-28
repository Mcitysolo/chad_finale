#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

RUNTIME = Path("/home/ubuntu/CHAD FINALE/runtime")
STATE_PATH = RUNTIME / "phase8_alert_state.json"
LIVEGATE_URL = "http://127.0.0.1:9618/live-gate"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _http_json(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "chad-phase8-alerts"})
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("live_gate not a dict")
    return obj


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _atomic_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _telegram_send(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        # No secrets configured => silent no-op (still write state)
        return False

    import urllib.parse

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": "true"}
    ).encode("utf-8")

    req = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=10.0) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    j = json.loads(body)
    return bool(j.get("ok"))


@dataclass(frozen=True)
class Signature:
    allow_exits_only: bool
    allow_ibkr_paper: bool
    allow_ibkr_live: bool
    operator_mode: str
    key_reasons: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allow_exits_only": self.allow_exits_only,
            "allow_ibkr_paper": self.allow_ibkr_paper,
            "allow_ibkr_live": self.allow_ibkr_live,
            "operator_mode": self.operator_mode,
            "key_reasons": list(self.key_reasons),
        }


def _extract_signature(lg: Dict[str, Any]) -> Signature:
    reasons = lg.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = []

    # Keep only Phase-8-relevant reason headers (stable + low-noise)
    keep_prefixes = ("RECONCILIATION_", "FEED_","STOP","OperatorIntent=")
    key = []
    for r in reasons:
        if not isinstance(r, str):
            continue
        if r.startswith(keep_prefixes):
            key.append(r)
    return Signature(
        allow_exits_only=bool(lg.get("allow_exits_only")),
        allow_ibkr_paper=bool(lg.get("allow_ibkr_paper")),
        allow_ibkr_live=bool(lg.get("allow_ibkr_live")),
        operator_mode=str(lg.get("operator_mode") or ""),
        key_reasons=tuple(key),
    )


def main() -> int:
    now = _utc_now()
    lg = _http_json(LIVEGATE_URL)

    sig = _extract_signature(lg)
    prev = _read_json(STATE_PATH)

    prev_sig = None
    if prev and isinstance(prev.get("signature"), dict):
        prev_sig = json.dumps(prev["signature"], sort_keys=True)

    cur_sig = json.dumps(sig.as_dict(), sort_keys=True)

    changed = prev_sig != cur_sig

    payload = {
        "ts_utc": now,
        "changed": bool(changed),
        "signature": sig.as_dict(),
    }
    _atomic_write(STATE_PATH, payload)

    if changed:
        text = (
            "CHAD Phase 8 Safety State Changed\\n"
            f"ts_utc: {now}\\n"
            f"allow_exits_only: {sig.allow_exits_only}\\n"
            f"allow_ibkr_paper: {sig.allow_ibkr_paper}\\n"
            f"allow_ibkr_live: {sig.allow_ibkr_live}\\n"
        )
        if sig.key_reasons:
            text += "reasons:\\n- " + "\\n- ".join(sig.key_reasons)

        _telegram_send(text)

    # Always print machine-readable result for journal
    print(json.dumps({"ts_utc": now, "changed": bool(changed), "signature": sig.as_dict()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
