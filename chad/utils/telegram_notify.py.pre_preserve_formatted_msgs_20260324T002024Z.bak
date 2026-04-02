from __future__ import annotations

"""
chad/utils/telegram_notify.py

Production-grade Telegram notifier for CHAD (Phase 9B).

Why this exists
---------------
CHAD runs watchdog/timer services (IBKR health, IBKR watchdog, ledger watcher,
DecisionTrace heartbeat, Shadow snapshot). These must be able to send alerts
out-of-band in a deterministic, auditable way.

This module provides a small, hardened notifier:
- Read-only side effects: sends Telegram messages ONLY.
- Strict env configuration: requires TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_CHAT_ID.
- Fail-safe: never raises to callers unless explicitly requested.
- Retries with exponential backoff + jitter for transient network errors.
- Rate-limits duplicate messages via an optional "dedupe_key" file in runtime/.

Environment
-----------
- TELEGRAM_BOT_TOKEN (required)
- TELEGRAM_ALLOWED_CHAT_ID (required)  # we reuse existing naming, treat as destination chat

Optional environment
--------------------
- TELEGRAM_NOTIFY_TIMEOUT_SECONDS (default 6.0)
- TELEGRAM_NOTIFY_MAX_RETRIES (default 3)
- TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS (default 900)  # 15 minutes

Usage
-----
from chad.utils.telegram_notify import notify

notify("IBKR DOWN: ...", severity="critical", dedupe_key="ibkr_down")
"""

import json
import os
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"


@dataclass(frozen=True)
class NotifyConfig:
    token: str
    chat_id: int
    timeout_s: float
    max_retries: int
    dedupe_ttl_s: int


class NotifyError(RuntimeError):
    pass


def _env(name: str) -> str:
    v = str(os.environ.get(name) or "").strip()
    return v


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def load_config() -> NotifyConfig:
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        raise NotifyError("Missing TELEGRAM_BOT_TOKEN")

    chat_raw = _env("TELEGRAM_ALLOWED_CHAT_ID")
    if not chat_raw:
        raise NotifyError("Missing TELEGRAM_ALLOWED_CHAT_ID (destination chat)")
    try:
        chat_id = int(chat_raw)
    except Exception as exc:
        raise NotifyError(f"Invalid TELEGRAM_ALLOWED_CHAT_ID: {chat_raw!r}") from exc

    timeout_s = _env_float("TELEGRAM_NOTIFY_TIMEOUT_SECONDS", 6.0)
    max_retries = max(0, _env_int("TELEGRAM_NOTIFY_MAX_RETRIES", 3))
    dedupe_ttl_s = max(0, _env_int("TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS", 900))

    return NotifyConfig(
        token=token,
        chat_id=chat_id,
        timeout_s=float(timeout_s),
        max_retries=int(max_retries),
        dedupe_ttl_s=int(dedupe_ttl_s),
    )


def _dedupe_path(dedupe_key: str) -> Path:
    safe = "".join(ch for ch in dedupe_key if ch.isalnum() or ch in ("-", "_", ".")).strip("._-")
    if not safe:
        safe = "dedupe"
    return RUNTIME_DIR / f"telegram_dedupe_{safe}.json"


def _dedupe_allows(cfg: NotifyConfig, dedupe_key: str) -> bool:
    """
    Returns True if we should send, False if suppressed by TTL.
    """
    if cfg.dedupe_ttl_s <= 0:
        return True
    p = _dedupe_path(dedupe_key)
    try:
        if not p.is_file():
            return True
        raw = json.loads(p.read_text(encoding="utf-8"))
        last_ts = float(raw.get("last_sent_unix", 0.0))
        age = time.time() - last_ts
        return age >= float(cfg.dedupe_ttl_s)
    except Exception:
        # If dedupe file is corrupt, do not suppress alerts.
        return True


def _dedupe_mark(dedupe_key: str) -> None:
    p = _dedupe_path(dedupe_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_sent_unix": time.time()}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def _post_send_message(cfg: NotifyConfig, *, text: str) -> Tuple[bool, str]:
    """
    Returns (ok, error_text). Never raises.
    """
    url = f"https://api.telegram.org/bot{cfg.token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": str(cfg.chat_id),
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.timeout_s)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if 200 <= int(resp.status) < 300:
                return True, ""
            return False, f"http_status={resp.status} body={body[:300]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def notify(
    message: str,
    *,
    severity: str = "info",
    dedupe_key: Optional[str] = None,
    raise_on_fail: bool = False,
) -> bool:
    """
    Send a Telegram alert. Returns True if sent, False if suppressed or failed.

    - severity is used only for message prefixing.
    - dedupe_key suppresses repeats within TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS.
    - raise_on_fail makes this raise NotifyError on failures (default False).
    """
    cfg = load_config()

    sev = str(severity).strip().lower() or "info"
    prefix = {
        "info": "ℹ️",
        "warn": "⚠️",
        "warning": "⚠️",
        "critical": "🚨",
        "error": "🚨",
    }.get(sev, "ℹ️")

    text = f"{prefix} CHAD {sev.upper()}: {str(message).strip()}"
    if not text.strip():
        return False

    if dedupe_key:
        if not _dedupe_allows(cfg, dedupe_key):
            return False

    # Retry with exponential backoff + jitter
    attempt = 0
    last_err = ""
    while True:
        ok, err = _post_send_message(cfg, text=text)
        if ok:
            if dedupe_key:
                _dedupe_mark(dedupe_key)
            return True

        last_err = err
        if attempt >= cfg.max_retries:
            if raise_on_fail:
                raise NotifyError(f"Telegram notify failed after retries: {last_err}")
            return False

        # backoff: 0.35, 0.7, 1.4, ... + jitter
        base = 0.35 * (2 ** attempt)
        sleep_s = base + random.random() * 0.25
        time.sleep(min(5.0, float(sleep_s)))
        attempt += 1
