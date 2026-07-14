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
import logging
import os
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"

_NOTIFY_LOGGER = logging.getLogger("chad.telegram_notify")


@dataclass(frozen=True)
class NotifyConfig:
    token: str
    chat_id: int
    timeout_s: float
    max_retries: int
    dedupe_ttl_s: int


class NotifyError(RuntimeError):
    pass


class DeliveryStatus(str, Enum):
    """Disposition of a single notify attempt.

    The historical ``notify() -> bool`` collapsed FOUR distinct outcomes onto
    ``False`` (config error, empty message, dedupe suppression, transport
    failure). That conflation is exactly what latched
    ``chad-service-alert@*`` into systemd ``failed`` on 2026-06-27: a flapping
    service fired ``OnFailure`` repeatedly, the FIRST alert delivered, and every
    duplicate WITHIN the 15-minute dedupe TTL returned ``False`` — which the
    caller mapped to ``EXIT_TELEGRAM_FAILED`` (exit 4). A *successfully
    suppressed duplicate* is NOT a delivery failure; this enum lets callers tell
    the difference.
    """

    SENT = "sent"                          # message delivered to Telegram
    SUPPRESSED_DEDUPE = "suppressed_dedupe"  # recent duplicate; intentionally not sent
    EMPTY_MESSAGE = "empty_message"        # nothing to send
    CONFIG_ERROR = "config_error"          # missing/invalid env (handler misconfig)
    TRANSPORT_ERROR = "transport_error"    # HTTP/network failure after retries


@dataclass(frozen=True)
class NotifyOutcome:
    """Structured result of :func:`notify_detailed`."""

    status: DeliveryStatus
    error: Optional[str] = None

    @property
    def sent(self) -> bool:
        return self.status is DeliveryStatus.SENT

    @property
    def suppressed(self) -> bool:
        return self.status is DeliveryStatus.SUPPRESSED_DEDUPE

    @property
    def failed(self) -> bool:
        """True only for genuine delivery failures (config or transport).

        Dedupe suppression and empty messages are NOT failures.
        """
        return self.status in (
            DeliveryStatus.CONFIG_ERROR,
            DeliveryStatus.TRANSPORT_ERROR,
        )


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


def notify_detailed(
    message: str,
    *,
    severity: str = "info",
    dedupe_key: Optional[str] = None,
) -> NotifyOutcome:
    """Send a Telegram alert and return a structured :class:`NotifyOutcome`.

    Unlike :func:`notify` (which returns a bare bool), this distinguishes the
    four ways a send can *not* deliver: a config error (missing/invalid env),
    an empty message, a dedupe suppression (a recent identical alert already
    went out — NOT a failure), and a transport/HTTP failure after retries.

    Never raises: config errors are reported via ``DeliveryStatus.CONFIG_ERROR``
    rather than propagated, so a caller can audit the disposition instead of
    crashing. (The legacy :func:`notify` wrapper re-raises to preserve its
    historical contract.)
    """
    try:
        cfg = load_config()
    except NotifyError as exc:
        return NotifyOutcome(DeliveryStatus.CONFIG_ERROR, str(exc))

    sev = str(severity).strip().lower() or "info"
    prefix = {
        "info": "ℹ️",
        "warn": "⚠️",
        "warning": "⚠️",
        "critical": "🚨",
        "error": "🚨",
    }.get(sev, "ℹ️")

    raw_message = str(message).strip()
    if not raw_message:
        return NotifyOutcome(DeliveryStatus.EMPTY_MESSAGE, "empty_message")

    # Preserve already-formatted operator-facing messages as-is.
    # Example: "ℹ️ CHAD Daily Update"
    if raw_message.startswith(("ℹ️", "⚠️", "🚨")):
        text = raw_message
    else:
        text = f"{prefix} CHAD {sev.upper()}: {raw_message}"

    if dedupe_key and not _dedupe_allows(cfg, dedupe_key):
        # A recent identical alert already delivered within the TTL window.
        # This is a successful no-op, NOT a delivery failure.
        return NotifyOutcome(DeliveryStatus.SUPPRESSED_DEDUPE, None)

    # Retry with exponential backoff + jitter
    attempt = 0
    last_err = ""
    while True:
        ok, err = _post_send_message(cfg, text=text)
        if ok:
            if dedupe_key:
                _dedupe_mark(dedupe_key)
            return NotifyOutcome(DeliveryStatus.SENT, None)

        last_err = err
        if attempt >= cfg.max_retries:
            return NotifyOutcome(DeliveryStatus.TRANSPORT_ERROR, last_err)

        # backoff: 0.35, 0.7, 1.4, ... + jitter
        base = 0.35 * (2 ** attempt)
        sleep_s = base + random.random() * 0.25
        time.sleep(min(5.0, float(sleep_s)))
        attempt += 1


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
    - raise_on_fail makes this raise NotifyError on transport failures.

    Behaviour is unchanged from the pre-``notify_detailed`` implementation: a
    missing/invalid config raises ``NotifyError`` (historical contract), an
    empty or dedupe-suppressed message returns ``False`` without raising, and a
    transport failure returns ``False`` (or raises when ``raise_on_fail``).
    Callers that need to distinguish suppression from failure — notably the
    service-failure alert handler — should use :func:`notify_detailed`.
    """
    outcome = notify_detailed(message, severity=severity, dedupe_key=dedupe_key)
    if outcome.status is DeliveryStatus.CONFIG_ERROR:
        # Historical contract: load_config() raised out of notify().
        raise NotifyError(outcome.error or "notify config error")
    if outcome.sent:
        return True
    if outcome.status is DeliveryStatus.TRANSPORT_ERROR and raise_on_fail:
        raise NotifyError(f"Telegram notify failed after retries: {outcome.error}")
    return False


# =============================================================================
# Specialized alert helpers
#
# These are the operator-facing push notifications that fire from the live
# loop, risk subsystems, and scr-sync service. Each wraps `notify()` so the
# existing retry/dedupe/config machinery applies. All helpers are fail-safe:
# they never raise on transport failures — alerting is supplementary to
# logging and must never block execution.
# =============================================================================

def _send_raw_telegram(text: str, *, dedupe_key: Optional[str] = None) -> bool:
    """
    Send a pre-formatted operator message without the "CHAD INFO:" prefix
    that `notify()` injects for generic strings. The notifier already has
    a bypass for messages starting with recognized severity emojis.
    """
    try:
        msg = text.strip()
        if not msg.startswith(("ℹ️", "⚠️", "\U0001f6a8")):
            # Prefix with info-severity marker so notify() leaves it intact.
            msg = "ℹ️ " + msg
        return notify(msg, severity="info", dedupe_key=dedupe_key)
    except Exception as exc:
        _NOTIFY_LOGGER.warning("telegram_notify_failed err=%s", exc)
        return False


def send_trade_alert(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    strategy: str,
    notional: float,
    is_live: bool,
) -> bool:
    """
    Instant Telegram notification on every submitted order.

    Never raises — alert failure must not block execution.
    """
    try:
        mode = "\U0001f534 LIVE" if is_live else "\U0001f4cb PAPER"
        side_norm = (side or "").upper()
        side_emoji = "\U0001f4c8" if side_norm in ("BUY", "LONG") else "\U0001f4c9"
        qty_display = f"{quantity:.4f}".rstrip("0").rstrip(".") or "0"
        msg = (
            f"{side_emoji} {mode}\n"
            f"{side_norm} {qty_display} {symbol} @ ${price:,.2f}\n"
            f"Strategy: {strategy or 'unknown'}\n"
            f"Notional: ${notional:,.0f}"
        )
        return _send_raw_telegram(msg, dedupe_key=f"trade_{symbol}_{strategy}_{side}")
    except Exception as exc:
        _NOTIFY_LOGGER.warning("send_trade_alert_failed symbol=%s err=%s", symbol, exc)
        return False


_SCR_SENTINEL_PATH = RUNTIME_DIR / "scr_last_notified_state.json"


def check_and_send_scr_milestone(current_state: str, effective_trades: int) -> bool:
    """
    Fires a Telegram alert when SCR transitions to a new state.

    Uses runtime/scr_last_notified_state.json as a sentinel so the alert
    only triggers on transitions, not every sync cycle. Transitions to
    CAUTIOUS, CONFIDENT, and PAUSED are announced; other states (e.g.
    WARMUP, UNKNOWN) are tracked silently.
    """
    try:
        state_norm = str(current_state or "").strip().upper()
        if not state_norm:
            return False

        last_state = "WARMUP"
        try:
            if _SCR_SENTINEL_PATH.is_file():
                raw = json.loads(_SCR_SENTINEL_PATH.read_text(encoding="utf-8"))
                last_state = str(raw.get("state", "WARMUP")).strip().upper() or "WARMUP"
        except Exception:
            last_state = "WARMUP"

        sent = False
        if state_norm != last_state:
            milestones = {
                "CAUTIOUS": (
                    "⚡ SCR advanced to CAUTIOUS — sizing factor increased. "
                    f"({effective_trades} clean trades logged)"
                ),
                "CONFIDENT": (
                    "\U0001f680 SCR advanced to CONFIDENT — full sizing unlocked. "
                    f"({effective_trades} clean trades logged)"
                ),
                "PAUSED": (
                    "⏸️ SCR entered PAUSED — performance dropped below "
                    "threshold. Sizing cut to minimum."
                ),
                "WARMUP": (
                    "📉 SCR DEGRADED → WARMUP\n"
                    "System has dropped below CAUTIOUS thresholds. "
                    "Sizing reduced to 10%."
                ),
                "CAUTIOUS_RECOVERY": None,  # handled below
            }
            msg = milestones.get(state_norm)
            if msg:
                sent = _send_raw_telegram(msg, dedupe_key=f"scr_milestone_{state_norm}")

            if last_state == "PAUSED" and state_norm in ("CAUTIOUS", "CONFIDENT"):
                _recovery_msg = (
                    f"✅ SCR RECOVERED from PAUSED → {state_norm}\n"
                    f"Sizing restored to "
                    f"{'25%' if state_norm == 'CAUTIOUS' else '100%'}."
                )
                notify(_recovery_msg, severity="info",
                       dedupe_key=f"scr_recovery_{state_norm}")

            try:
                _SCR_SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = _SCR_SENTINEL_PATH.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(
                        {
                            "state": state_norm,
                            "effective_trades": int(effective_trades),
                            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tmp.replace(_SCR_SENTINEL_PATH)
            except Exception as exc:
                _NOTIFY_LOGGER.warning("scr_sentinel_write_failed err=%s", exc)

        return sent
    except Exception as exc:
        _NOTIFY_LOGGER.warning("scr_milestone_failed err=%s", exc)
        return False


def send_stop_bus_alert(reason: str) -> bool:
    """Telegram alert when the STOP bus trips — all trading halted."""
    try:
        msg = (
            "\U0001f6d1 STOP BUS TRIGGERED\n"
            f"Reason: {reason or 'unspecified'}\n"
            "All trading halted. Check runtime/stop_bus.json"
        )
        return _send_raw_telegram(msg, dedupe_key="stop_bus_triggered")
    except Exception as exc:
        _NOTIFY_LOGGER.warning("stop_bus_alert_failed err=%s", exc)
        return False


def send_edge_decay_alert(strategy: str, consecutive_losses: int) -> bool:
    """Telegram alert when a strategy is halted by the edge-decay monitor."""
    try:
        msg = (
            f"⚠️ EDGE DECAY — {strategy}\n"
            f"{consecutive_losses} consecutive losses. "
            "Strategy paused pending recovery."
        )
        return _send_raw_telegram(msg, dedupe_key=f"edge_decay_{strategy}")
    except Exception as exc:
        _NOTIFY_LOGGER.warning("edge_decay_alert_failed strategy=%s err=%s", strategy, exc)
        return False


def send_drawdown_alert(drawdown_pct: float, threshold_pct: float) -> bool:
    """Telegram alert when current drawdown breaches its threshold."""
    try:
        msg = (
            "\U0001f4c9 DRAWDOWN ALERT\n"
            f"Current drawdown: {drawdown_pct:.1f}%\n"
            f"Threshold: {threshold_pct:.1f}%"
        )
        return _send_raw_telegram(msg, dedupe_key="drawdown_threshold")
    except Exception as exc:
        _NOTIFY_LOGGER.warning("drawdown_alert_failed err=%s", exc)
        return False
