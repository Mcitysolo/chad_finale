#!/usr/bin/env python3
"""
CHAD — Telegram Coach Bot (Read-Only, Production)

What this bot is
----------------
A read-only “coach” interface for CHAD that:
- Explains system posture (why blocked / risk posture / readiness)
- Shows operator-safe observability snapshots (op/* endpoints)
- Shows portfolio proposal endpoints (portfolio/* endpoints)
- Provides price + AI research (ai/price, ai/research) when available
- Supports simple natural-language queries (best-effort, no execution)

Hard safety guarantees
----------------------
- NEVER places orders
- NEVER flips CHAD_MODE or operator intent
- NEVER writes runtime/config
- Only reads:
  - HTTP endpoints on CHAD backend (default http://127.0.0.1:9618)
  - Optional local runtime files for extra operator context (read-only)

Operational requirements
------------------------
Environment variables (required):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_ALLOWED_CHAT_ID   (single chat id allowed; integer)

Optional:
- CHAD_BACKEND_BASE_URL      (default http://127.0.0.1:9618)
- TELEGRAM_DISABLE_OPENAI    (1/true to disable)
- OPENAI_API_KEY             (optional; if absent, the bot still works)

The service in your system runs:
  python3 -m chad.utils.telegram_bot

So keep the module importable and side-effect free on import.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from telegram import ParseMode, Update
from telegram.ext import (
    CallbackContext,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)

# Optional OpenAI (bot is fully functional without it)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore


LOGGER_NAME = "chad.telegram_bot"
LOGGER = logging.getLogger(LOGGER_NAME)

TELEGRAM_MAX_CHARS = 3900  # keep margin under Telegram 4096
HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "12"))
HTTP_CONNECT_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_CONNECT_TIMEOUT_S", "4"))
DEFAULT_BASE_URL = "http://127.0.0.1:9618"


# =============================================================================
# Logging + env utilities
# =============================================================================

def _init_logger() -> None:
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _load_env_file(path: str) -> None:
    """
    Best-effort KEY=VALUE loader without overwriting existing env vars.
    """
    p = Path(path)
    if not p.is_file():
        return
    try:
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("env_file_load_failed path=%s err=%s", path, exc)


def _env(name: str, required: bool = True) -> Optional[str]:
    v = os.environ.get(name)
    if required and not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _get_backend_base_url() -> str:
    base = (os.environ.get("CHAD_BACKEND_BASE_URL") or "").strip()
    if not base:
        return DEFAULT_BASE_URL
    return base.rstrip("/")


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# Text helpers (safe for Telegram)
# =============================================================================

def _clip(s: str, limit: int = TELEGRAM_MAX_CHARS) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 20)] + "\n…(truncated)…"


def _chunks(s: str, limit: int = TELEGRAM_MAX_CHARS) -> List[str]:
    s = s or ""
    if len(s) <= limit:
        return [s]
    out: List[str] = []
    buf = s
    while buf:
        out.append(buf[:limit])
        buf = buf[limit:]
    return out


def _codeblock(s: str) -> str:
    # Telegram MarkdownV2 escaping is painful; use HTML <pre> for stability.
    # We keep ParseMode.HTML and wrap with <pre>.
    return f"<pre>{_escape_html(s)}</pre>"


def _escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _bullets(items: List[str], max_items: int = 10) -> str:
    out = []
    for x in items[:max_items]:
        out.append(f"• {x}")
    if len(items) > max_items:
        out.append(f"• …(+{len(items) - max_items} more)")
    return "\n".join(out)


# =============================================================================
# Backend HTTP client (read-only)
# =============================================================================

class BackendClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendClientConfig:
    base_url: str
    timeout_s: float = HTTP_TIMEOUT_S
    connect_timeout_s: float = HTTP_CONNECT_TIMEOUT_S


class BackendClient:
    """
    Thin, robust client. Only GET/POST read-only endpoints.

    Used endpoints:
    - Operator Surface V2: /op/status /op/why_blocked /op/risk_explain /op/perf_snapshot /op/readiness /op/what_if_caps
    - Core gateway: /live-gate /risk-state /shadow /health
    - Portfolio surface: /portfolio/active /portfolio/targets/{PROFILE} /portfolio/rebalance/latest?profile=...
    - AI: /ai/price (GET) /ai/research (POST)
    """

    def __init__(self, cfg: BackendClientConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._url(path)
        try:
            resp = self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=(self.cfg.connect_timeout_s, self.cfg.timeout_s),
            )
        except requests.RequestException as exc:
            raise BackendClientError(f"network_error {path}: {exc}") from exc

        # Allow 200 only (stable for operator surfaces)
        if resp.status_code != 200:
            detail = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    detail = str(j.get("detail") or j)
                else:
                    detail = str(j)
            except Exception:
                detail = (resp.text or "").strip()
            raise BackendClientError(f"http_{resp.status_code} {path}: {detail[:300]}")

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise BackendClientError(f"bad_json {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise BackendClientError(f"bad_shape {path}: expected dict")
        return data

    # --- Operator Surface V2 ---
    def op_status(self) -> Dict[str, Any]:
        return self._request("GET", "/op/status")

    def op_readiness(self) -> Dict[str, Any]:
        return self._request("GET", "/op/readiness")

    def op_why_blocked(self) -> Dict[str, Any]:
        return self._request("GET", "/op/why_blocked")

    def op_risk_explain(self) -> Dict[str, Any]:
        return self._request("GET", "/op/risk_explain")

    def op_perf_snapshot(self) -> Dict[str, Any]:
        return self._request("GET", "/op/perf_snapshot")

    def op_what_if_caps(self, *, equity: float, daily_risk_fraction: float) -> Dict[str, Any]:
        return self._request("GET", "/op/what_if_caps", params={"equity": equity, "daily_risk_fraction": daily_risk_fraction})

    # --- Core gateway ---
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def live_gate(self) -> Dict[str, Any]:
        return self._request("GET", "/live-gate")

    def risk_state(self) -> Dict[str, Any]:
        return self._request("GET", "/risk-state")

    def shadow(self) -> Dict[str, Any]:
        return self._request("GET", "/shadow")

    # --- Portfolio surface ---
    def portfolio_active(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/active")

    def portfolio_targets(self, profile: str) -> Dict[str, Any]:
        return self._request("GET", f"/portfolio/targets/{profile.strip().upper()}")

    def portfolio_rebalance_latest(self, profile: str) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/rebalance/latest", params={"profile": profile.strip().upper()})

    # --- AI ---
    def ai_price(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/ai/price", params={"symbol": symbol.strip().upper()})

    def ai_research(self, symbol: str, question: str, timeframe: str = "1m") -> Dict[str, Any]:
        payload = {
            "symbol": symbol.strip().upper(),
            "scenario_timeframe": timeframe,
            "question": question.strip() or "High-level macro and risk context.",
        }
        return self._request("POST", "/ai/research", json_body=payload)


# =============================================================================
# Optional Teaching Mode (OpenAI) — safe degradation
# =============================================================================

@dataclass(frozen=True)
class OpenAIConfig:
    intent_model: str = os.environ.get("OPENAI_INTENT_MODEL", "gpt-4.1-mini")
    coach_model: str = os.environ.get("OPENAI_COACH_MODEL", "gpt-4.1-mini")


def _openai_enabled() -> bool:
    if _env_bool("TELEGRAM_DISABLE_OPENAI", default=False):
        return False
    return OpenAI is not None


def _get_openai_client() -> Optional[Any]:
    if not _openai_enabled():
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        _load_env_file("/etc/chad/openai.env")
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        return OpenAI()
    except Exception:
        return None


def _teach_brief(text: str) -> str:
    """
    Best-effort “teaching mode” rewrite:
    - Short
    - Beginner-friendly
    - Ends with "Why this matters: …"
    If OpenAI not available, returns original.
    """
    client = _get_openai_client()
    if client is None:
        return text

    cfg = OpenAIConfig()
    prompt = (
        "Rewrite this for a beginner in 3-4 sentences. "
        "Be direct, no hype, no trading advice. "
        "End with: 'Why this matters: ...'\n\n"
        f"TEXT:\n{text}"
    )
    try:
        resp = client.chat.completions.create(
            model=cfg.coach_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or text
    except Exception:
        return text


# =============================================================================
# Telegram guard + message send
# =============================================================================

def _allowed_chat_id() -> int:
    # best-effort load from /etc/chad/telegram.env
    if not os.environ.get("TELEGRAM_ALLOWED_CHAT_ID"):
        _load_env_file("/etc/chad/telegram.env")
    v = _env("TELEGRAM_ALLOWED_CHAT_ID", required=True)
    try:
        return int(str(v).strip())
    except Exception as exc:
        raise RuntimeError(f"Invalid TELEGRAM_ALLOWED_CHAT_ID: {v!r}") from exc


def _is_allowed(update: Update) -> bool:
    try:
        allowed = _allowed_chat_id()
    except Exception:
        return False
    chat = update.effective_chat
    if chat is None:
        return False
    return int(chat.id) == int(allowed)


def _send(update: Update, context: CallbackContext, text: str, *, html: bool = True) -> None:
    if update.effective_chat is None:
        return
    chunks = _chunks(text)
    for part in chunks:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_clip(part),
            parse_mode=ParseMode.HTML if html else None,
            disable_web_page_preview=True,
        )


# =============================================================================
# Formatting: operator + risk + portfolio
# =============================================================================

def _fmt_kv(title: str, kv: List[Tuple[str, Any]]) -> str:
    lines = [f"<b>{_escape_html(title)}</b>"]
    for k, v in kv:
        lines.append(f"{_escape_html(str(k))}: {_escape_html(str(v))}")
    return "\n".join(lines)


def _fmt_status_min(op_status: Dict[str, Any]) -> str:
    # stable, compact operator status
    ts = op_status.get("ts_utc", "")
    failed = (op_status.get("failed_units") or {}).get("failed_units") if isinstance(op_status.get("failed_units"), dict) else []
    failed_count = len(failed) if isinstance(failed, list) else 0

    timers = op_status.get("timers") if isinstance(op_status.get("timers"), dict) else {}
    t_lines = []
    for name, st in timers.items():
        if not isinstance(st, dict):
            continue
        enabled = ((st.get("enabled") or {}).get("state")) if isinstance(st.get("enabled"), dict) else st.get("enabled")
        active = ((st.get("active") or {}).get("state")) if isinstance(st.get("active"), dict) else st.get("active")
        t_lines.append(f"{name}: enabled={enabled} active={active}")

    runtime_files = op_status.get("runtime_files") if isinstance(op_status.get("runtime_files"), dict) else {}
    # show 6 key files only
    key_files = ["feed_state", "positions_snapshot", "reconciliation_state", "dynamic_caps", "scr_state", "tier_state"]
    f_lines = []
    for k in key_files:
        meta = runtime_files.get(k)
        if not isinstance(meta, dict):
            f_lines.append(f"{k}: missing_meta")
            continue
        exists = bool(meta.get("exists"))
        ts_utc = meta.get("ts_utc") or ""
        f_lines.append(f"{k}: exists={exists} ts={ts_utc}")

    msg = "\n".join(
        [
            f"<b>CHAD STATUS</b> (ts={_escape_html(str(ts))})",
            "",
            "<b>Timers</b>",
            _escape_html("\n".join(t_lines) if t_lines else "none"),
            "",
            f"<b>Failed units</b>: {failed_count}",
            "",
            "<b>Runtime artifacts</b>",
            _escape_html("\n".join(f_lines)),
        ]
    )
    return msg


def _fmt_why_blocked(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary") or ""
    # keep it short and human
    s = "<b>WHY BLOCKED</b>\n\n" + _escape_html(str(summary))
    return s


def _fmt_risk_explain(payload: Dict[str, Any]) -> str:
    posture = payload.get("posture_summary") or ""
    why = payload.get("why_blocked_summary") or []
    caps = payload.get("strategy_caps") or []

    lines = ["<b>RISK EXPLAIN</b>", "", _escape_html(str(posture)), ""]
    if isinstance(why, list) and why:
        lines.append("<b>Main blockers</b>")
        lines.append(_escape_html(_bullets([str(x) for x in why], max_items=8)))
        lines.append("")
    if isinstance(caps, list) and caps:
        lines.append("<b>Strategy caps (share of portfolio_risk_cap)</b>")
        for row in caps[:10]:
            if not isinstance(row, dict):
                continue
            strat = row.get("strategy")
            pct = row.get("cap_pct_of_portfolio_risk_cap")
            cap = row.get("cap_usd")
            lines.append(_escape_html(f"{strat}: ${cap:.2f} ({pct:.4f}%)" if isinstance(cap, (int, float)) and isinstance(pct, (int, float)) else f"{strat}: {cap} / {pct}"))
    return "\n".join(lines)


def _fmt_perf_snapshot(payload: Dict[str, Any]) -> str:
    narrative = payload.get("operator_narrative") or ""
    scr = payload.get("scr") or {}
    stats = (scr.get("stats") if isinstance(scr, dict) else {}) or {}
    lines = ["<b>PERF SNAPSHOT</b>", ""]
    if narrative:
        lines.append(_escape_html(str(narrative)))
        lines.append("")
    if isinstance(stats, dict) and stats:
        keys = ["total_trades", "effective_trades", "win_rate", "sharpe_like", "max_drawdown", "total_pnl"]
        lines.append("<b>SCR stats</b>")
        for k in keys:
            if k in stats:
                lines.append(_escape_html(f"{k}: {stats.get(k)}"))
    return "\n".join(lines)


def _fmt_portfolio_active(payload: Dict[str, Any]) -> str:
    if not payload.get("ok"):
        return "<b>PORTFOLIO ACTIVE</b>\n\n" + _escape_html(str(payload))
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    total = payload.get("total_notional_proxy")
    lines = ["<b>PORTFOLIO ACTIVE</b>"]
    lines.append(_escape_html(f"total_notional_proxy≈{total}"))
    lines.append("")
    for p in positions[:20]:
        if not isinstance(p, dict):
            continue
        sym = p.get("symbol")
        qty = p.get("qty")
        notional = p.get("notional_proxy")
        avg = p.get("avg_cost")
        lines.append(_escape_html(f"{sym}: qty={qty} notional≈{notional} avg_cost={avg}"))
    return "\n".join(lines)


def _fmt_portfolio_targets(payload: Dict[str, Any]) -> str:
    if not payload.get("ok"):
        return "<b>PORTFOLIO TARGETS</b>\n\n" + _escape_html(str(payload))
    prof = payload.get("profile")
    targets = payload.get("targets") if isinstance(payload.get("targets"), list) else []
    lines = [f"<b>PORTFOLIO TARGETS</b> ({_escape_html(str(prof))})", ""]
    for t in targets[:25]:
        if not isinstance(t, dict):
            continue
        sym = t.get("symbol")
        w = t.get("weight")
        lines.append(_escape_html(f"{sym}: {w:.4f}" if isinstance(w, (int, float)) else f"{sym}: {w}"))
    return "\n".join(lines)


def _fmt_portfolio_rebalance(payload: Dict[str, Any]) -> str:
    if not payload.get("ok"):
        return "<b>REBALANCE PREVIEW</b>\n\n" + _escape_html(str(payload))
    prof = payload.get("profile")
    diffs = payload.get("diffs") if isinstance(payload.get("diffs"), list) else []
    lines = [f"<b>REBALANCE PREVIEW</b> ({_escape_html(str(prof))})", ""]
    for d in diffs[:25]:
        if not isinstance(d, dict):
            continue
        sym = d.get("symbol")
        cur = d.get("current_weight")
        tgt = d.get("target_weight")
        delta = d.get("delta_weight")
        if isinstance(cur, (int, float)) and isinstance(tgt, (int, float)) and isinstance(delta, (int, float)):
            lines.append(_escape_html(f"{sym}: current={cur:.4f} target={tgt:.4f} delta={delta:+.4f}"))
        else:
            lines.append(_escape_html(f"{sym}: {d}"))
    notes = payload.get("notes")
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("<b>Notes</b>")
        lines.append(_escape_html(_bullets([str(x) for x in notes], max_items=6)))
    return "\n".join(lines)


# =============================================================================
# Command handlers
# =============================================================================

def _client() -> BackendClient:
    return BackendClient(BackendClientConfig(base_url=_get_backend_base_url()))


def cmd_ping(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    _send(update, context, f"<b>PONG</b> {_escape_html(_utc_now_iso())}")


def cmd_help(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    msg = textwrap.dedent(
        """
        <b>CHAD Coach — commands</b>

        <b>System</b>
        • /ping
        • /status          (operator snapshot)
        • /readiness       (operator gate checklist)
        • /why_blocked     (why live is blocked)
        • /risk            (risk posture + caps)
        • /perf            (SCR performance snapshot)
        • /live_gate       (raw LiveGate snapshot)

        <b>Portfolio (read-only)</b>
        • /portfolio_active
        • /portfolio_targets BALANCED
        • /portfolio_rebalance BALANCED

        <b>Caps “what-if”</b>
        • /caps 20000 0.05   (equity, daily_risk_fraction)

        <b>Prices / Research</b>
        • /price AAPL
        • /ai_research AAPL what’s the macro risk?

        You can also ask in plain English:
        “why is it blocked?”, “risk posture”, “portfolio targets balanced”, “rebalance balanced”, “price of AAPL”
        """
    ).strip()
    _send(update, context, msg)


def cmd_status(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().op_status()
        _send(update, context, _fmt_status_min(data))
    except Exception as exc:
        _send(update, context, "<b>STATUS ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_readiness(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().op_readiness()
        ok = bool(data.get("ok"))
        blockers = data.get("blockers") if isinstance(data.get("blockers"), list) else []
        warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
        msg = [f"<b>READINESS</b> ok={_escape_html(str(ok))}", ""]
        if blockers:
            msg.append("<b>Blockers</b>")
            msg.append(_escape_html(_bullets([str(x) for x in blockers], max_items=10)))
            msg.append("")
        if warnings:
            msg.append("<b>Warnings</b>")
            msg.append(_escape_html(_bullets([str(x) for x in warnings], max_items=10)))
        if not blockers and not warnings:
            msg.append(_escape_html("No blockers/warnings reported by /op/readiness."))
        _send(update, context, "\n".join(msg))
    except Exception as exc:
        _send(update, context, "<b>READINESS ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_why_blocked(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().op_why_blocked()
        _send(update, context, _fmt_why_blocked(data))
    except Exception as exc:
        _send(update, context, "<b>WHY_BLOCKED ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_risk(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().op_risk_explain()
        _send(update, context, _fmt_risk_explain(data))
    except Exception as exc:
        _send(update, context, "<b>RISK ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_perf(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().op_perf_snapshot()
        _send(update, context, _fmt_perf_snapshot(data))
    except Exception as exc:
        _send(update, context, "<b>PERF ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_live_gate(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().live_gate()
        # keep small: show key fields + first reasons
        reasons = data.get("reasons") if isinstance(data.get("reasons"), list) else []
        msg = [
            "<b>LIVE GATE</b>",
            "",
            _escape_html(f"allow_ibkr_paper={data.get('allow_ibkr_paper')}"),
            _escape_html(f"allow_ibkr_live={data.get('allow_ibkr_live')}"),
            _escape_html(f"allow_exits_only={data.get('allow_exits_only')}"),
            _escape_html(f"operator_mode={data.get('operator_mode')}"),
            "",
        ]
        if reasons:
            msg.append("<b>Reasons</b>")
            msg.append(_escape_html(_bullets([str(x) for x in reasons], max_items=8)))
        _send(update, context, "\n".join(msg))
    except Exception as exc:
        _send(update, context, "<b>LIVE_GATE ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_caps(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    # /caps EQUITY DRF
    try:
        args = context.args or []
        if len(args) < 2:
            _send(update, context, "<b>USAGE</b>\n\n/caps 20000 0.05")
            return
        equity = float(args[0])
        drf = float(args[1])
        data = _client().op_what_if_caps(equity=equity, daily_risk_fraction=drf)
        caps = data.get("strategy_caps") if isinstance(data.get("strategy_caps"), list) else []
        lines = [
            "<b>WHAT-IF CAPS</b>",
            _escape_html(f"equity={equity} drf={drf}"),
            _escape_html(f"portfolio_risk_cap={data.get('portfolio_risk_cap')}"),
            "",
        ]
        for row in caps[:20]:
            if not isinstance(row, dict):
                continue
            lines.append(_escape_html(f"{row.get('strategy')}: cap=${row.get('cap_usd')} (frac={row.get('fraction')})"))
        _send(update, context, "\n".join(lines))
    except Exception as exc:
        _send(update, context, "<b>CAPS ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_portfolio_active(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    try:
        data = _client().portfolio_active()
        _send(update, context, _fmt_portfolio_active(data))
    except Exception as exc:
        _send(update, context, "<b>PORTFOLIO ACTIVE ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_portfolio_targets(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    args = context.args or []
    if not args:
        _send(update, context, "<b>USAGE</b>\n\n/portfolio_targets BALANCED")
        return
    profile = str(args[0]).strip().upper()
    try:
        data = _client().portfolio_targets(profile)
        _send(update, context, _fmt_portfolio_targets(data))
    except Exception as exc:
        _send(update, context, "<b>PORTFOLIO TARGETS ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_portfolio_rebalance(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    args = context.args or []
    if not args:
        _send(update, context, "<b>USAGE</b>\n\n/portfolio_rebalance BALANCED")
        return
    profile = str(args[0]).strip().upper()
    try:
        data = _client().portfolio_rebalance_latest(profile)
        _send(update, context, _fmt_portfolio_rebalance(data))
    except Exception as exc:
        _send(update, context, "<b>REBALANCE ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_price(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    args = context.args or []
    if not args:
        _send(update, context, "<b>USAGE</b>\n\n/price AAPL")
        return
    sym = str(args[0]).strip().upper()
    try:
        data = _client().ai_price(sym)
        _send(update, context, "<b>PRICE</b>\n\n" + _escape_html(json.dumps(data, indent=2, sort_keys=True)))
    except Exception as exc:
        _send(update, context, "<b>PRICE ERROR</b>\n\n" + _escape_html(str(exc)))


def cmd_ai_research(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    args = context.args or []
    if not args:
        _send(update, context, "<b>USAGE</b>\n\n/ai_research AAPL what’s the macro risk?")
        return
    sym = str(args[0]).strip().upper()
    question = " ".join(args[1:]).strip() or "High-level macro and risk context."
    try:
        data = _client().ai_research(sym, question)
        # keep it readable
        text = json.dumps(data, indent=2, sort_keys=True)
        _send(update, context, "<b>AI RESEARCH</b>\n\n" + _escape_html(_clip(text, 3600)))
    except Exception as exc:
        _send(update, context, "<b>AI RESEARCH ERROR</b>\n\n" + _escape_html(str(exc)))


# =============================================================================
# Natural language handler (best-effort routing)
# =============================================================================

_RX_PRICE = re.compile(r"\bprice\b|\bquote\b", re.IGNORECASE)
_RX_SYMBOL = re.compile(r"\b([A-Z]{1,5})\b")


def _nl_route(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()

    if "why" in low and ("blocked" in low or "stuck" in low):
        return "why_blocked"
    if "readiness" in low or "ready" in low:
        return "readiness"
    if "risk" in low or "caps" in low:
        return "risk"
    if "perf" in low or "performance" in low or "win rate" in low:
        return "perf"
    if "live gate" in low or "live-gate" in low or "livegate" in low:
        return "live_gate"
    if "portfolio" in low and "active" in low:
        return "portfolio_active"
    if "targets" in low and "portfolio" in low:
        return "portfolio_targets"
    if "rebalance" in low:
        return "portfolio_rebalance"
    if _RX_PRICE.search(t):
        return "price"
    if low in ("status", "health"):
        return "status"
    return "unknown"


def handle_free_text(update: Update, context: CallbackContext) -> None:
    if not _is_allowed(update):
        return
    text = (update.message.text if update.message else "") or ""
    route = _nl_route(text)

    # Attempt to extract a profile/symbol where needed
    profile = "BALANCED"
    sym = None
    m = _RX_SYMBOL.search(text.upper())
    if m:
        sym = m.group(1)

    try:
        if route == "why_blocked":
            cmd_why_blocked(update, context)
            return
        if route == "readiness":
            cmd_readiness(update, context)
            return
        if route == "risk":
            cmd_risk(update, context)
            return
        if route == "perf":
            cmd_perf(update, context)
            return
        if route == "live_gate":
            cmd_live_gate(update, context)
            return
        if route == "portfolio_active":
            cmd_portfolio_active(update, context)
            return
        if route == "portfolio_targets":
            # try: “targets balanced”
            if "balanced" in text.lower():
                profile = "BALANCED"
            cmd_portfolio_targets(update, _fake_context_args(context, [profile]))
            return
        if route == "portfolio_rebalance":
            if "balanced" in text.lower():
                profile = "BALANCED"
            cmd_portfolio_rebalance(update, _fake_context_args(context, [profile]))
            return
        if route == "price" and sym:
            cmd_price(update, _fake_context_args(context, [sym]))
            return
        if route == "status":
            cmd_status(update, context)
            return

        # fallback: short teaching response, no trading advice
        msg = _teach_brief(
            "I can help with CHAD status, why it’s blocked, risk posture, portfolio previews, and prices. "
            "Try: /status, /why_blocked, /risk, /portfolio_rebalance BALANCED, /price AAPL."
        )
        _send(update, context, _escape_html(msg))
    except Exception as exc:
        _send(update, context, "<b>ERROR</b>\n\n" + _escape_html(str(exc)))


def _fake_context_args(context: CallbackContext, args: List[str]) -> CallbackContext:
    # python-telegram-bot v13 stores args on context, so we can temporarily override.
    context.args = args
    return context


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    _init_logger()

    # best-effort load env files
    _load_env_file("/etc/chad/telegram.env")
    _load_env_file("/etc/chad/openai.env")

    token = _env("TELEGRAM_BOT_TOKEN", required=True)
    _ = _allowed_chat_id()  # validates

    base_url = _get_backend_base_url()
    LOGGER.info("telegram_bot_start ts=%s base_url=%s", _utc_now_iso(), base_url)

    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler("ping", cmd_ping))
    dp.add_handler(CommandHandler("help", cmd_help))

    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("readiness", cmd_readiness))
    dp.add_handler(CommandHandler("why_blocked", cmd_why_blocked))
    dp.add_handler(CommandHandler("risk", cmd_risk))
    dp.add_handler(CommandHandler("perf", cmd_perf))
    dp.add_handler(CommandHandler("live_gate", cmd_live_gate))

    dp.add_handler(CommandHandler("caps", cmd_caps))

    dp.add_handler(CommandHandler("portfolio_active", cmd_portfolio_active))
    dp.add_handler(CommandHandler("portfolio_targets", cmd_portfolio_targets))
    dp.add_handler(CommandHandler("portfolio_rebalance", cmd_portfolio_rebalance))

    dp.add_handler(CommandHandler("price", cmd_price))
    dp.add_handler(CommandHandler("ai_research", cmd_ai_research))

    # Free text
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_free_text))

    # Start polling
    updater.start_polling(drop_pending_updates=True)
    updater.idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
