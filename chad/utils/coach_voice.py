from __future__ import annotations

"""
chad/utils/coach_voice.py — COACH-VOICE-L1 human alert formatter.

Why this exists
---------------
CHAD's Telegram pushes historically read like machine output: unit names,
rule ids, raw second-counts, JSON paths. The operator wants a calm, plain-
English trading-coach voice instead. This module is the presentation layer
that turns the *facts* a sender already has into a short, friendly message.

Every formatted alert answers four questions, in order, in <=6 short lines:

  1. What happened     — the bottom line, no codes.
  2. Why it matters    — plain English.
  3. What CHAD did      — the automatic action, or "nothing needed doing".
  4. Do you need to act — usually "No action needed" + the next step.

An optional 5th line carries a one-line micro-lesson (or a term is defined
inline in <=10 words when it first appears).

Design constraints (COACH-VOICE-L1)
-----------------------------------
- TEMPLATED, never an LLM. Deterministic templates keyed by ``kind`` and
  driven by the real artifact/finding fields. No network, no API keys.
- Three verbosity modes:
    * SIMPLE   (default) — friendly text only. NEVER shows codes.
    * STANDARD           — SIMPLE + one extra plain-English context line.
    * PRO                — SIMPLE + a trailing raw marker/metric line
                           (unit name, rule id, evidence, path, raw seconds).
- Mode is persisted in ``config/coach_voice.json`` (operator-editable) and
  overridden by the ``CHAD_COACH_MODE`` environment variable. A per-call
  ``mode`` argument beats both (used by tests).
- Unknown ``kind`` values fall through to a safe generic template that still
  answers the four questions; it only exposes the artifact path in PRO.
- ``format_alert`` never raises — alerting is supplementary and must never
  crash a caller. Any internal error degrades to the generic template.

This layer is presentation only: it does not decide *whether* an alert fires,
does not touch dedupe keys, and does not read or write any runtime artifact.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
COACH_CONFIG_PATH = ROOT / "config" / "coach_voice.json"

_LOG = logging.getLogger("chad.coach_voice")

VALID_MODES: Tuple[str, ...] = ("SIMPLE", "STANDARD", "PRO")
DEFAULT_MODE = "SIMPLE"
MAX_LINES = 6

# Severity → leading emoji. All three glyphs are recognised by the notifier's
# "already formatted" bypass (telegram_notify.notify_detailed / _send_raw_telegram),
# so a coach message that starts with one is delivered verbatim rather than
# re-prefixed with "CHAD WARNING:".
_SEVERITY_EMOJI = {
    "critical": "\U0001f6a8",  # 🚨
    "high": "\U0001f6a8",
    "error": "\U0001f6a8",
    "warning": "⚠️",  # ⚠️
    "warn": "⚠️",
    "medium": "⚠️",
    "info": "ℹ️",  # ℹ️
    "low": "ℹ️",
    "notice": "ℹ️",
}


def _emoji_for(severity: str) -> str:
    return _SEVERITY_EMOJI.get(str(severity or "").strip().lower(), "ℹ️")


# ── mode resolution ───────────────────────────────────────────────────────────
def _norm_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    m = str(mode).strip().upper()
    return m if m in VALID_MODES else None


def _config_mode() -> Optional[str]:
    """Read the persisted operator mode from config/coach_voice.json, if valid."""
    try:
        raw = json.loads(COACH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _norm_mode(raw.get("mode")) if isinstance(raw, dict) else None


def resolve_mode() -> str:
    """Resolve the active mode: CHAD_COACH_MODE env > config file > default.

    A per-call ``mode`` argument to :func:`format_alert` takes precedence over
    this (handled in :func:`format_alert`); this function covers the
    no-explicit-mode path used by the wired-in senders.
    """
    env = _norm_mode(os.environ.get("CHAD_COACH_MODE"))
    if env:
        return env
    cfg = _config_mode()
    if cfg:
        return cfg
    return DEFAULT_MODE


# ── humanizers ────────────────────────────────────────────────────────────────
def humanize_duration(seconds: Any) -> str:
    """Turn a raw second count into a plain phrase, e.g. 5877963 -> '68 days'.

    Rounds to the largest natural unit so the operator reads '68 days', not
    '5877963s'. Always singular/plural-correct. Returns 'an unknown time' when
    the input is not a usable number.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "an unknown time"
    if s < 0:
        s = -s
    minute, hour, day = 60.0, 3600.0, 86400.0
    if s < minute:
        n, unit = int(round(s)), "second"
    elif s < hour:
        n, unit = int(round(s / minute)), "minute"
    elif s < day:
        n, unit = int(round(s / hour)), "hour"
    else:
        n, unit = int(round(s / day)), "day"
    if n <= 0:
        n = 1
    return f"{n} {unit}" + ("s" if n != 1 else "")


def humanize_name(token: Any) -> str:
    """Turn a code-y identifier into a readable label.

    'omega_macro' -> 'Omega Macro'; 'alpha_intraday' -> 'Alpha Intraday'.
    Strips a leading 'chad-' and a trailing '.service'/'.timer'/'.json'.
    """
    t = str(token or "").strip()
    if not t:
        return "one of CHAD's parts"
    for suffix in (".service", ".timer", ".socket", ".json"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    if t.startswith("chad-"):
        t = t[len("chad-") :]
    t = t.replace("-", " ").replace("_", " ").strip()
    return " ".join(w.capitalize() for w in t.split()) or "one of CHAD's parts"


# ── friendly lookup tables ────────────────────────────────────────────────────
# unit stem (no chad- / no .service) -> (friendly noun, what it does, impact while down)
_UNIT_FRIENDLY: Dict[str, Tuple[str, str, str]] = {
    "ibkr-bar-provider": (
        "price-data feed",
        "it supplies the latest market prices",
        "some strategies pause until prices refresh",
    ),
    "ibkr-price-refresh": (
        "price refresher",
        "it keeps quoted prices current",
        "prices may be briefly stale",
    ),
    "orchestrator": (
        "main decision engine",
        "it decides how much each strategy may trade",
        "sizing falls back to safe defaults",
    ),
    "live-loop": (
        "trading loop",
        "it runs each trading cycle",
        "no new trades run until it recovers",
    ),
    "reconciliation-publisher": (
        "position-checker",
        "it double-checks our positions against the broker",
        "position reports pause briefly",
    ),
    "options-chain-refresh": (
        "options-data refresher",
        "it loads the options prices we need",
        "options strategies wait for fresh data",
    ),
    "regime-classifier-refresh": (
        "market-mood reader",
        "it labels the current market conditions",
        "strategy selection may lag slightly",
    ),
    "kraken-ws": (
        "crypto price feed",
        "it streams live crypto prices",
        "crypto strategies pause until it reconnects",
    ),
    "telegram-bot": (
        "messaging service",
        "it sends and receives these updates",
        "some messages may be delayed",
    ),
    "dashboard": (
        "dashboard",
        "it shows the live status page",
        "the status page may be unavailable",
    ),
    "backend": (
        "background service",
        "it serves data to the dashboard",
        "the status page may show stale data",
    ),
    "health-monitor": (
        "self-check service",
        "it watches CHAD's own health",
        "self-checks pause briefly",
    ),
    "daily-bars-refresh": (
        "daily-history updater",
        "it refreshes end-of-day price history",
        "history-based signals may lag a day",
    ),
    "ibkr-daily-bars-refresh": (
        "daily-history updater",
        "it refreshes end-of-day price history",
        "history-based signals may lag a day",
    ),
    "service-alert": (
        "alerting service",
        "it sends you failure notifications",
        "some alerts may be delayed",
    ),
    "ibgateway": (
        "broker gateway",
        "it connects CHAD to the broker",
        "broker actions pause until it reconnects",
    ),
    "metrics-server": (
        "metrics service",
        "it publishes monitoring numbers",
        "monitoring dashboards may be stale",
    ),
}

# runtime filename -> (friendly noun, what-it-is <=10 words, impact / reassurance)
_FEED_FRIENDLY: Dict[str, Tuple[str, str, str]] = {
    "var_state.json": (
        "VaR risk report",
        "it estimates your worst-case daily loss",
        "Trading is unaffected; this report is informational.",
    ),
    "regime_state.json": (
        "market-mood reading",
        "it labels the current market conditions",
        "Strategy selection may lag until it refreshes.",
    ),
    "price_cache.json": (
        "price cache",
        "it holds the latest quoted prices",
        "Some prices may be briefly stale.",
    ),
    "dynamic_caps.json": (
        "sizing limits file",
        "it sets how much each strategy may trade",
        "Sizing falls back to safe defaults.",
    ),
    "reconciliation_state.json": (
        "position-check report",
        "it compares our positions with the broker",
        "Position checks may be briefly delayed.",
    ),
    "kraken_prices.json": (
        "crypto price feed",
        "it holds live crypto prices",
        "Crypto strategies may pause briefly.",
    ),
    "regime_booster.json": (
        "regime booster file",
        "it nudges sizing with the market mood",
        "Sizing nudges may be briefly stale.",
    ),
    "choppy_regime_state.json": (
        "choppy-market reading",
        "it flags choppy, hard-to-trade conditions",
        "Choppy-market protection may lag slightly.",
    ),
    "macro_state.json": (
        "big-picture market file",
        "it tracks the broad market backdrop",
        "The macro backdrop may be briefly stale.",
    ),
}

# stop-bus trigger token -> plain phrase
_STOP_REASON_FRIENDLY: Dict[str, str] = {
    "broker_latency": "the broker connection slowed down",
    "latency": "the broker connection slowed down",
    "daily_loss_limit": "today's losses reached the safety limit",
    "daily_loss": "today's losses reached the safety limit",
    "drawdown": "the account fell past its safety limit",
    "max_drawdown": "the account fell past its safety limit",
    "exposure": "positions grew past the safety limit",
    "max_exposure": "positions grew past the safety limit",
    "manual": "someone set a manual stop",
    "operator": "someone set a manual stop",
    "kill_switch": "an emergency stop was pressed",
}


def _friendly_unit(unit: Any) -> Tuple[str, str, str]:
    """Return (friendly noun, role, impact) for a systemd unit name."""
    stem = str(unit or "").strip()
    for suffix in (".service", ".timer", ".socket"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    if stem.startswith("chad-"):
        stem = stem[len("chad-") :]
    if stem in _UNIT_FRIENDLY:
        return _UNIT_FRIENDLY[stem]
    return (
        "one of CHAD's background services",
        "it runs a supporting job",
        "the affected job pauses until it recovers",
    )


def _friendly_reason(reason: Any) -> str:
    """Translate a stop-bus reason (possibly comma-joined codes) to plain text."""
    raw = str(reason or "").strip()
    if not raw:
        return "a safety check tripped"
    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", raw) if t.strip()]
    phrases: List[str] = []
    for tok in tokens:
        phrase = _STOP_REASON_FRIENDLY.get(tok)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        return "a safety check tripped"
    if len(phrases) == 1:
        return phrases[0]
    return "; ".join(phrases[:2])


# ── card model ────────────────────────────────────────────────────────────────
@dataclass
class _Card:
    """The four-question payload for one alert, plus optional extra lines."""

    severity: str
    happened: str  # Q1
    why: str       # Q2
    did: str       # Q3
    act: str       # Q4
    standard_context: Optional[str] = None  # STANDARD-only 5th line (no codes)
    pro: Optional[str] = None               # PRO-only raw marker/metric line


def _assemble(card: _Card, mode: str) -> str:
    """Render a :class:`_Card` into the final message text for ``mode``."""
    emoji = _emoji_for(card.severity)
    head = card.happened.strip()
    if not head.startswith((emoji, "ℹ️", "⚠️", "\U0001f6a8")):
        head = f"{emoji} {head}"

    lines: List[str] = [head, card.why.strip(), card.did.strip(), card.act.strip()]
    if mode == "STANDARD" and card.standard_context:
        lines.append(card.standard_context.strip())
    elif mode == "PRO" and card.pro:
        lines.append(card.pro.strip())

    lines = [ln for ln in lines if ln]
    if len(lines) > MAX_LINES:
        # Never drop the final (action or raw) line; trim the middle.
        lines = lines[: MAX_LINES - 1] + [lines[-1]]
    return "\n".join(lines)


# ── templates ─────────────────────────────────────────────────────────────────
def _tpl_service_failure(facts: Dict[str, Any], mode: str) -> _Card:
    unit = facts.get("failed_unit") or facts.get("unit") or ""
    severity = str(facts.get("severity") or "HIGH")
    status = str(facts.get("active_unit_status") or "unknown")
    friendly, role, impact = _friendly_unit(unit)

    if _looks_like_test_or_manual(facts):
        return _Card(
            severity="info",
            happened=f"Quick note: a health check just ran for CHAD's {friendly}.",
            why="This looks like a test or manual run, not a real failure.",
            did="Nothing was wrong, so nothing needed doing.",
            act="No action needed.",
            standard_context="I only send these when a check is triggered by hand.",
            pro=f"raw: unit={unit} state={status} journal_lines=0",
        )

    return _Card(
        severity=severity,
        happened=f"One of CHAD's background services stopped: the {friendly}.",
        why=f"That service matters because {role}; while it's down, {impact}.",
        did="CHAD saved a full report and the service is set to restart on its own.",
        act="No action needed right now — I'll flag it again if it keeps failing.",
        standard_context="A one-off blip usually self-heals; repeated failures get escalated.",
        pro=f"raw: unit={unit} state={status} artifact={facts.get('artifact_path', '?')}",
    )


def _looks_like_test_or_manual(facts: Dict[str, Any]) -> bool:
    """Heuristic: an empty journal tail on a still-healthy unit reads as a
    test/manual invocation rather than a genuine crash.

    A real ``OnFailure`` fires with the unit in ``failed``/``inactive`` and a
    populated journal tail. If ``journalctl`` returned nothing AND read cleanly
    AND the unit is not in a failed state, the alert was almost certainly run
    by hand or as a test.
    """
    tail = facts.get("journal_tail")
    has_tail = bool(tail) if isinstance(tail, (list, tuple)) else bool(str(tail or "").strip())
    if has_tail:
        return False
    if facts.get("journal_error"):
        return False
    status = str(facts.get("active_unit_status") or "").strip().lower()
    return status not in ("failed", "inactive", "activating", "auto-restart", "deactivating")


def _tpl_feed_stale(facts: Dict[str, Any], mode: str) -> _Card:
    feed = _feed_name(facts)
    age_seconds = _feed_age_seconds(facts)
    severity = str(facts.get("severity") or "WARNING")
    remedy_action = str(facts.get("remedy_action") or "").strip().lower()
    missing = "missing" in str(facts.get("title") or "").lower()

    friendly, what_is, impact = _FEED_FRIENDLY.get(
        feed,
        ("data feed", "it supplies numbers CHAD uses", "Signals that use it may be slightly behind."),
    )
    age_phrase = humanize_duration(age_seconds) if age_seconds is not None else "a long time"

    # The VaR case is special: there is no scheduler for var_state.json, so it
    # will simply stay stale until one is built — the operator must be told this
    # is expected and harmless, not a fault to chase.
    is_var = feed == "var_state.json"

    if missing:
        happened = f"Heads up — CHAD's {friendly} isn't being produced right now."
        why = f"That report ({what_is}) is missing, so it can't be read."
    else:
        happened = f"Heads up — one of CHAD's reports is out of date: the {friendly}."
        if is_var:
            why = (
                f"The {friendly} ({what_is}) hasn't updated in {age_phrase} — "
                "its scheduler was never built."
            )
        else:
            why = f"The {friendly} ({what_is}) hasn't updated in {age_phrase}."

    if remedy_action == "restart_feed_publisher":
        did = "CHAD is restarting the feed automatically to refresh it."
        act = "No action needed unless you keep seeing this."
    elif is_var:
        did = impact + " Nothing needed doing."
        act = "No action needed — a fix is already on this week's list."
    else:
        did = impact + " CHAD has flagged it for review."
        act = "No action needed yet — I'll nudge you if it doesn't clear."

    return _Card(
        severity=severity,
        happened=happened,
        why=why,
        did=did,
        act=act,
        standard_context="I'll only nudge you again if the situation actually changes.",
        pro=(
            f"raw: feed={feed} age_seconds={age_seconds} "
            f"rule={facts.get('rule_id', '?')} evidence={facts.get('evidence', '')}"
        ),
    )


def _feed_name(facts: Dict[str, Any]) -> str:
    """Best-effort extract of the runtime filename from finding fields."""
    if facts.get("feed"):
        return str(facts["feed"]).strip()
    for field in ("evidence", "title", "description"):
        text = str(facts.get(field) or "")
        m = re.search(r"([a-z0-9_]+\.json)", text)
        if m:
            return m.group(1)
    return ""


def _feed_age_seconds(facts: Dict[str, Any]) -> Optional[float]:
    """Best-effort extract of the feed age in seconds from finding fields."""
    val = facts.get("age_seconds", facts.get("age"))
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    for field in ("evidence", "title", "description"):
        text = str(facts.get(field) or "")
        m = re.search(r"age[=\s]*(\d+)\s*s", text) or re.search(r"(\d+)\s*s\s*old", text)
        if m:
            return float(m.group(1))
    return None


def _tpl_stop_bus(facts: Dict[str, Any], mode: str) -> _Card:
    reason = _friendly_reason(facts.get("reason"))
    return _Card(
        severity="critical",
        happened="CHAD has paused all trading — a safety stop was triggered.",
        why=f"This is a protective halt because {reason}. No new trades will be placed.",
        did="CHAD stopped trading on its own to protect the account — nothing was lost by pausing.",
        act="No action needed yet; this often clears by itself within a few minutes.",
        standard_context="If it stays stopped for long, I'll let you know it needs a look.",
        pro=f"raw: reason={facts.get('reason', '')} artifact=runtime/stop_bus.json",
    )


def _tpl_drawdown(facts: Dict[str, Any], mode: str) -> _Card:
    dd = _as_float(facts.get("drawdown_pct"))
    threshold = _as_float(facts.get("threshold_pct"), 5.0)
    dd_abs = abs(dd) if dd is not None else None
    dd_phrase = f"{dd_abs:.1f}%" if dd_abs is not None else "past its safety mark"
    thr_phrase = f"{abs(threshold):.0f}%" if threshold is not None else "its watch level"
    return _Card(
        severity="warning",
        happened=f"The account is down {dd_phrase} from its recent high.",
        why=(
            f"That's past the {thr_phrase} level CHAD watches — 'drawdown' just means "
            "how far below the recent peak we are."
        ),
        did="CHAD flagged this; its separate risk limits stay fully in force.",
        act="No action needed — this is an early heads-up, not an emergency.",
        standard_context="Watch for another update only if it deepens from here.",
        pro=(
            f"raw: drawdown_pct={facts.get('drawdown_pct')} "
            f"threshold_pct={facts.get('threshold_pct')}"
        ),
    )


def _tpl_edge_decay(facts: Dict[str, Any], mode: str) -> _Card:
    strat = humanize_name(facts.get("strategy"))
    try:
        losses = int(facts.get("consecutive_losses") or 0)
    except (TypeError, ValueError):
        losses = 0
    loss_phrase = (
        f"{losses} losing trades in a row" if losses > 0 else "a run of losing trades"
    )
    return _Card(
        severity="warning",
        happened=f"CHAD paused one of its strategies: {strat}.",
        why=(
            f"It just had {loss_phrase}, so CHAD benched it — 'edge decay' means "
            "a strategy that stopped working as well as before."
        ),
        did="CHAD paused that one strategy automatically; the others keep trading normally.",
        act="No action needed — CHAD will bring it back if it recovers.",
        standard_context="Benching a weak strategy protects the account's overall results.",
        pro=f"raw: strategy={facts.get('strategy', '')} consecutive_losses={losses}",
    )


def _tpl_health_finding(facts: Dict[str, Any], mode: str) -> _Card:
    """Generic health-monitor finding (the notify-only, non-feed cases)."""
    severity = str(facts.get("severity") or "WARNING")
    title = str(facts.get("title") or "CHAD noticed something worth a look").strip()
    description = str(facts.get("description") or "").strip()
    # Keep the human why short: prefer the description's first sentence.
    why = _first_sentence(description) or "It's worth a quick look, but not urgent."
    return _Card(
        severity=severity,
        happened=f"CHAD's self-check flagged something: {_soften(title)}.",
        why=why,
        did="CHAD logged it and is keeping an eye on it.",
        act="No action needed yet — I'll escalate if it gets worse.",
        standard_context="These self-checks run automatically around the clock.",
        pro=(
            f"raw: rule={facts.get('rule_id', '?')} severity={severity} "
            f"evidence={facts.get('evidence', '')}"
        ),
    )


def _tpl_health_autofix(facts: Dict[str, Any], mode: str) -> _Card:
    """A health finding that was auto-remediated."""
    severity = str(facts.get("severity") or "WARNING")
    title = str(facts.get("title") or "a small issue").strip()
    result = str(facts.get("remedy_result") or facts.get("result") or "").strip()
    return _Card(
        severity=severity,
        happened=f"CHAD spotted and fixed something on its own: {_soften(title)}.",
        why="A routine self-check found it and applied the standard fix.",
        did="CHAD already applied the fix — no manual step was needed.",
        act="No action needed — this is just a heads-up that it happened.",
        standard_context="CHAD only auto-fixes changes on its pre-approved safe list.",
        pro=f"raw: rule={facts.get('rule_id', '?')} action={result}",
    )


def _tpl_health_analysis(facts: Dict[str, Any], mode: str) -> _Card:
    """The optional Claude health-analysis narrative."""
    analysis = str(facts.get("analysis") or facts.get("text") or "").strip()
    why = _first_sentence(analysis) or "It's a routine review of how things are running."
    return _Card(
        severity="info",
        happened="CHAD ran its regular health review.",
        why=why,
        did="CHAD logged the full review for the record.",
        act="No action needed — this is a routine summary.",
        standard_context="These reviews run automatically and are informational.",
        pro=f"raw: analysis={analysis[:180]}",
    )


def _tpl_market_closed(facts: Dict[str, Any], mode: str) -> _Card:
    """Market-closed / outside-regular-hours notice.

    No CHAD sender currently pushes this kind; the template is provided so that
    if one is added later it reads in the coach voice without further work.
    """
    return _Card(
        severity="info",
        happened="The market is closed right now, so CHAD is holding off on new trades.",
        why="Trading outside regular hours is riskier, so CHAD waits for the open.",
        did="CHAD paused new entries until the market reopens — nothing is wrong.",
        act="No action needed — trading resumes automatically at the open.",
        standard_context="Existing safety stops still apply while the market is closed.",
        pro=f"raw: {facts.get('detail', 'market_closed')}",
    )


def _tpl_generic(facts: Dict[str, Any], mode: str) -> _Card:
    """Safe fallback for an unknown ``kind``. Still answers the four questions;
    exposes the artifact path only in PRO."""
    headline = str(
        facts.get("headline") or facts.get("title") or "CHAD has an update for you."
    ).strip()
    detail = str(facts.get("detail") or facts.get("description") or "").strip()
    why = _first_sentence(detail) or "It's informational."
    return _Card(
        severity=str(facts.get("severity") or "info"),
        happened=_soften(headline),
        why=why,
        did="CHAD logged this for the record.",
        act="No action needed unless you see it repeat.",
        standard_context="This is a general status update from CHAD.",
        pro=f"raw: kind={facts.get('_kind', 'unknown')} artifact={facts.get('artifact_path', '')}",
    )


def _drift_kind_phrase(identity: str) -> str:
    """Human phrase for one 'kind|SYMBOL' drift identity (W3B-4)."""
    kind, _, symbol = str(identity).partition("|")
    sym = symbol or "a position"
    phrases = {
        "qty_mismatch": f"{sym} (share counts differ)",
        "broker_untracked_position": f"{sym} (the broker holds it, CHAD isn't tracking it)",
        "phantom_guard_entry": f"{sym} (CHAD tracks it, the broker shows none)",
    }
    return phrases.get(kind, f"{sym} ({kind})")


def _tpl_position_drift(facts: Dict[str, Any], mode: str) -> _Card:
    """W3B-4: new book-vs-broker drift appeared (transition alert)."""
    appeared = [str(x) for x in (facts.get("appeared") or [])]
    listed = "; ".join(_drift_kind_phrase(i) for i in appeared[:6]) or "one or more positions"
    more = f" (+{len(appeared) - 6} more)" if len(appeared) > 6 else ""
    n = len(appeared) or "some"
    return _Card(
        severity="warning",
        happened=f"CHAD's book and the broker's book disagree on {n} position(s).",
        why=f"Affected: {listed}{more}. This is a bookkeeping mismatch, not a trade loss.",
        did=(
            "CHAD recorded the mismatch in its reconciliation report; going live "
            "stays blocked while any mismatch stands."
        ),
        act=(
            "If you recently traded these symbols yourself, no action is needed — "
            "CHAD reconciles them on its own. Otherwise this deserves a look."
        ),
        standard_context="CHAD re-checks every few minutes and will confirm when it clears.",
        pro=f"raw: appeared={appeared} counts_by_kind={facts.get('counts_by_kind', {})} artifact=runtime/position_guard_drift.json",
    )


def _tpl_position_drift_resolved(facts: Dict[str, Any], mode: str) -> _Card:
    """W3B-4: previously alerted book-vs-broker drift cleared (all-clear)."""
    resolved = [str(x) for x in (facts.get("resolved") or [])]
    listed = "; ".join(_drift_kind_phrase(i) for i in resolved[:6]) or "the earlier mismatch"
    more = f" (+{len(resolved) - 6} more)" if len(resolved) > 6 else ""
    still = int(facts.get("still_active") or 0)
    act = (
        "Nothing to do — this is the all-clear."
        if still == 0
        else f"Nothing to do for these; {still} other mismatch(es) still stand."
    )
    return _Card(
        severity="info",
        happened="Good news — a book-vs-broker mismatch CHAD flagged earlier has cleared.",
        why=f"Now matching again: {listed}{more}.",
        did="CHAD confirmed the books agree on these positions again.",
        act=act,
        standard_context="No trading was affected by the mismatch itself.",
        pro=f"raw: resolved={resolved} artifact=runtime/position_guard_drift.json",
    )


_TEMPLATES: Dict[str, Callable[[Dict[str, Any], str], _Card]] = {
    "service_failure": _tpl_service_failure,
    "feed_stale": _tpl_feed_stale,
    "stop_bus": _tpl_stop_bus,
    "drawdown": _tpl_drawdown,
    "edge_decay": _tpl_edge_decay,
    "health_finding": _tpl_health_finding,
    "health_autofix": _tpl_health_autofix,
    "health_analysis": _tpl_health_analysis,
    "market_closed": _tpl_market_closed,
    "position_drift": _tpl_position_drift,
    "position_drift_resolved": _tpl_position_drift_resolved,
}


# ── small text helpers ────────────────────────────────────────────────────────
def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_sentence(text: str) -> str:
    """First sentence of a description, trimmed of trailing evidence noise."""
    text = str(text or "").strip()
    if not text:
        return ""
    # Cut at the first sentence boundary; keep it short and human.
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    sentence = (m.group(1) if m else text).strip()
    if len(sentence) > 160:
        sentence = sentence[:157].rstrip() + "..."
    return sentence


def _soften(text: str) -> str:
    """Strip a trailing period and any parenthetical code tail from a title so
    it slots mid-sentence, e.g. 'Feed STALE (notify): var_state.json (…)' ->
    'Feed stale'. Only used by generic/health templates where a raw title may
    leak; the SIMPLE-safety scrub below removes any residual codes."""
    t = str(text or "").strip().rstrip(".")
    # Drop a trailing "(…)" that usually carries codes/paths.
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    # Drop a leading "SCREAMING:" style prefix token if present.
    return t or "an update"


# Substrings that must never appear in SIMPLE/STANDARD output (codes/paths).
_CODE_PATTERNS = (
    re.compile(r"\.service\b"),
    re.compile(r"\.timer\b"),
    re.compile(r"\bchad-[a-z0-9-]+"),
    re.compile(r"[a-z0-9_]+\.json\b"),
    re.compile(r"\bR\d{1,3}[a-z]?\b"),          # rule ids R02, R17b
    re.compile(r"\braw:\s"),                     # the PRO marker prefix
    re.compile(r"\bruntime/"),
    re.compile(r"\breports/"),
    re.compile(r"\d{4,}\s*s\b"),                 # raw second counts like 5877963s
    re.compile(r"evidence="),
)


def _scrub_codes(text: str) -> str:
    """Defensive last line of defence: if a template leaked a code into a
    non-PRO render (e.g. a raw title echoed verbatim), redact it so SIMPLE's
    'never shows codes' contract holds regardless of caller-supplied facts."""
    scrubbed = text
    scrubbed = re.sub(r"\b(chad-[a-z0-9-]+)\.(service|timer|socket)\b", "a background service", scrubbed)
    scrubbed = re.sub(r"\b[a-z0-9_]+\.json\b", "a data file", scrubbed)
    scrubbed = re.sub(r"\bR\d{1,3}[a-z]?\b", "a self-check", scrubbed)
    scrubbed = re.sub(r"\b\d{4,}\s*s\b", "a while", scrubbed)
    scrubbed = re.sub(r"\s*evidence=\S+", "", scrubbed)
    scrubbed = re.sub(r"\bruntime/\S+", "an internal file", scrubbed)
    return scrubbed


# ── public API ────────────────────────────────────────────────────────────────
def format_alert(kind: str, facts: Optional[Dict[str, Any]] = None, mode: Optional[str] = None) -> str:
    """Format one operator alert as a calm, plain-English coach message.

    Parameters
    ----------
    kind:
        The alert kind. Known kinds: ``service_failure``, ``feed_stale``,
        ``stop_bus``, ``drawdown``, ``edge_decay``, ``health_finding``,
        ``health_autofix``, ``health_analysis``, ``market_closed``. Any other
        value uses the safe generic template.
    facts:
        The real fields the sender already has (artifact/finding values). Missing
        keys degrade gracefully.
    mode:
        ``SIMPLE`` / ``STANDARD`` / ``PRO``. When omitted or invalid, resolved
        from ``CHAD_COACH_MODE`` then ``config/coach_voice.json`` then SIMPLE.

    Returns a message of at most six short lines. Never raises.
    """
    facts = dict(facts or {})
    facts.setdefault("_kind", kind)
    active_mode = _norm_mode(mode) or resolve_mode()

    try:
        builder = _TEMPLATES.get(str(kind or "").strip().lower(), _tpl_generic)
        card = builder(facts, active_mode)
        text = _assemble(card, active_mode)
    except Exception as exc:  # never let presentation crash a sender
        _LOG.warning("coach_voice_format_failed kind=%s err=%s", kind, exc)
        try:
            text = _assemble(_tpl_generic(facts, active_mode), active_mode)
        except Exception:
            return "ℹ️ CHAD has an update for you.\nIt's informational.\nNo action needed."

    # SIMPLE/STANDARD must never show codes, even if a caller passed a raw
    # title/description straight through into a template line.
    if active_mode != "PRO":
        text = _scrub_codes(text)
    return text


# ── sender adapters (keep the wire-in thin) ───────────────────────────────────
def kind_for_finding(finding: Any) -> str:
    """Map a health-monitor Finding to a coach ``kind``.

    Feed staleness/missing (R02) → ``feed_stale``; a remediated (non-notify)
    finding → ``health_autofix``; everything else → ``health_finding``.
    """
    rule_id = str(getattr(finding, "rule_id", "") or "")
    title = str(getattr(finding, "title", "") or "").lower()
    remedy_action = str(getattr(finding, "remedy_action", "") or "").strip().lower()
    if rule_id == "R02" or title.startswith(("feed stale", "feed missing")):
        return "feed_stale"
    if remedy_action and remedy_action != "notify":
        return "health_autofix"
    return "health_finding"


def facts_from_finding(finding: Any, *, remedy_result: Optional[str] = None) -> Dict[str, Any]:
    """Flatten a health-monitor Finding into a plain facts dict for
    :func:`format_alert`. Presentation-only: no runtime reads."""
    return {
        "rule_id": getattr(finding, "rule_id", ""),
        "severity": getattr(finding, "severity", "WARNING"),
        "title": getattr(finding, "title", ""),
        "description": getattr(finding, "description", ""),
        "evidence": getattr(finding, "evidence", ""),
        "remedy_type": getattr(finding, "remedy_type", ""),
        "remedy_action": getattr(finding, "remedy_action", ""),
        "remedy_result": remedy_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COACH-VOICE-L2 — conversational answers, refusals, and mode persistence.
#
# CV2 adds a *question-and-answer* presentation surface on top of CV1's alert
# formatter, reusing the SAME voice: the same mode resolution, the same emoji
# map, the same humanizers, and the same SIMPLE-never-shows-codes scrub. There
# is ONE voice definition — this section only adds new renderers, it does not
# fork the tone.
#
# Division of labour (mirrors CV1):
#   * FACTS come from ``chad/utils/coach_intents.py`` (the read-only intent
#     layer that reads runtime files). This module NEVER reads a runtime
#     artifact — it only renders the ``facts`` dict it is handed.
#   * PRESENTATION lives here.
#
# The one write this module owns is :func:`set_config_mode`, which persists the
# operator's verbosity choice to ``config/coach_voice.json`` — the coach's own
# state, not a runtime/trading artifact.
# ═══════════════════════════════════════════════════════════════════════════════

# Friendly display name per mode (for acknowledgements).
_MODE_DISPLAY = {"SIMPLE": "simple", "STANDARD": "normal", "PRO": "pro (technical)"}


def set_config_mode(mode: Any) -> bool:
    """Persist the verbosity ``mode`` to ``config/coach_voice.json``.

    Returns True on a successful write of a valid mode, False otherwise. Only
    the ``mode`` field is changed; ``schema_version`` / ``_comment`` and any
    other keys are preserved. This is the coach's *own* configuration — it is
    not a runtime or trading artifact, and it never touches order/risk/service
    state. Never raises.
    """
    m = _norm_mode(mode)
    if not m:
        return False
    try:
        try:
            raw = json.loads(COACH_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}
        raw.setdefault("schema_version", "coach_voice.v1")
        raw["mode"] = m
        tmp = COACH_CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, COACH_CONFIG_PATH)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("coach_voice_set_mode_failed mode=%s err=%s", mode, exc)
        return False


# ── number / money humanizers (kept scrub-safe: comma grouping breaks the
#    4+ digit run the SIMPLE scrub looks for, and no trailing 's') ────────────────
def _fmt_money(value: Any, currency: str = "CAD") -> str:
    """Format a money value. Pass ``currency=""`` to render WITHOUT a currency
    label — used for values whose currency basis is not established (e.g. the
    SCR raw paper-P&L sum, which the money-truth layer intentionally leaves
    uncurrencied). Attaching a false 'CAD'/'USD' to such a value is a currency-
    provenance overclaim, CHAD's #1 bug class."""
    v = _as_float(value)
    if v is None:
        return "an unknown amount"
    ccy = str(currency or "").upper()
    neg = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000:
        body = f"{a / 1_000_000:.2f}M"
    elif a >= 10_000:
        body = f"{a / 1_000:.0f}K"
    else:
        body = f"{a:,.0f}"
    suffix = f" {ccy}" if ccy else ""
    return f"{neg}${body}{suffix}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    v = _as_float(value)
    if v is None:
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def _fmt_signed(value: Any, digits: int = 2) -> str:
    v = _as_float(value)
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}"


def _fmt_qty(value: Any) -> str:
    v = _as_float(value)
    if v is None:
        return "?"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.2f}"


# ── staleness + source citation (shared by every answer template) ──────────────
def _staleness_line(facts: Dict[str, Any]) -> Optional[str]:
    """Return a plain-English 'this data is behind' disclosure, or None.

    Honours the CV2 STALENESS RULE: whenever the intent layer flags any source
    as older than its TTL, we say so in words and give the age — we never
    present stale data as current. No filename leaks (SIMPLE-safe).
    """
    if not facts.get("_stale"):
        return None
    age = facts.get("_stale_age_seconds")
    label = str(facts.get("_stale_label") or "part of this reading").strip()
    if age is not None:
        return (
            f"⚠️ Heads up: {label} is about {humanize_duration(age)} old, "
            "so treat it as a little behind, not live."
        )
    return f"⚠️ Heads up: {label} may be out of date, so treat it as a little behind."


def _sources_pro_line(facts: Dict[str, Any]) -> Optional[str]:
    """PRO-only provenance line citing each runtime source + its age."""
    sources = facts.get("_sources")
    if not isinstance(sources, (list, tuple)) or not sources:
        return None
    parts: List[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        path = str(s.get("path", "?"))
        if not s.get("present", True):
            parts.append(f"{path}(missing)")
            continue
        age = s.get("age_seconds")
        age_txt = f"{int(age)}s" if isinstance(age, (int, float)) else "?"
        flag = "STALE" if s.get("stale") else "ok"
        parts.append(f"{path}(age={age_txt},{flag})")
    if not parts:
        return None
    return "raw sources: " + "; ".join(parts)


def _humanize_reason_word(reason: Any) -> str:
    return _friendly_reason(reason)


# ── conversational answer templates (one per allow-listed intent) ──────────────
def _ans_status_overview(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    equity = f.get("equity_cad")
    scr_state = str(f.get("scr_state") or "unknown").upper()
    sizing = _as_float(f.get("sizing_factor"))
    win_rate = f.get("win_rate")
    pnl = f.get("total_pnl")
    eff = f.get("effective_trades")
    total = f.get("total_trades")
    n_pos = int(f.get("open_positions_count") or 0)
    fills = int(f.get("fills_today") or 0)
    drift = int(f.get("drift_count") or 0)

    posture = {
        "CONFIDENT": "trading at full size",
        "CAUTIOUS": "trading carefully at reduced size",
        "WARMUP": "still warming up (small, learning-mode size)",
        "PAUSED": "paused for now",
        "RECOVERY": "easing back in after a rough patch",
    }.get(scr_state, "running")

    head = "ℹ️ Here's where we stand right now."
    equity_txt = _fmt_money(equity, "CAD") if equity is not None else "not available yet"
    body = [
        f"Balance is about {equity_txt}, and CHAD is {posture}.",
    ]
    perf_bits = []
    if win_rate is not None:
        perf_bits.append(f"win rate {_fmt_pct(win_rate)}")
    if pnl is not None:
        # SCR paper P&L is a raw, currency-unnormalized paper sum — render it
        # without a currency label rather than mislabel it CAD/USD.
        perf_bits.append(f"paper P&L {_fmt_money(pnl, '')}")
    if eff is not None:
        perf_bits.append(f"{int(eff)} scored trades so far")
    if perf_bits:
        body.append("So far: " + ", ".join(perf_bits) + ".")
    # COACH-TRUTH-FIX T1: the headline trade count is the canonical SCR
    # effective_trades (the scoreable subset). The much larger raw paper
    # rollup (total_trades) must NEVER be shown as "scored trades" — disclose
    # the non-scoreable remainder explicitly so the two are never conflated.
    not_scoreable = None
    if total is not None and eff is not None:
        try:
            not_scoreable = max(0, int(total) - int(eff))
        except (TypeError, ValueError):
            not_scoreable = None
    if not_scoreable:
        body.append(
            f"On top of that there are {not_scoreable:,} practice fills "
            "(not yet scoreable) — those don't count toward the rating."
        )
    # "Broker-confirmed" qualifies POSITIONS only; fills/drift are separate.
    pos_txt = (
        "no broker-confirmed open positions" if n_pos == 0
        else f"{n_pos} broker-confirmed open position" + ("s" if n_pos != 1 else "")
    )
    body.append(
        pos_txt[0].upper() + pos_txt[1:] + ". "
        + f"Today: {fills} fill" + ("s" if fills != 1 else "")
        + f", and {drift} position mismatch" + ("es" if drift != 1 else "") + " to watch."
    )
    body.append("No action needed — this is just the lay of the land.")
    standard = None
    if mode == "STANDARD":
        sz = f"{sizing:.2f}x" if sizing is not None else "its usual"
        standard = f"CHAD is sizing new trades at {sz} of normal while in {posture.split(' (')[0]}."
        body.append(standard)
    pro = (
        f"raw: scr_state={scr_state} sizing_factor={sizing} win_rate={win_rate} "
        f"total_pnl={pnl} effective_trades={eff} total_trades={total} "
        f"open_positions={n_pos} fills_today={fills} drift_count={drift}"
    )
    return head, body, pro


def _ans_why_paused(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    primary = str(f.get("primary_reason") or "none")
    stop_reason = _humanize_reason_word(f.get("stop_reason"))
    scr_state = str(f.get("scr_state") or "unknown").upper()

    if primary == "halted":
        head = "⚠️ CHAD has paused trading on purpose."
        body = [
            f"It's a protective stop because {stop_reason}. No new trades go out while it's on.",
            "CHAD did this itself to protect the account — nothing was lost by pausing.",
            "No action needed yet; these usually clear on their own within a few minutes.",
        ]
    elif primary == "market_closed":
        head = "ℹ️ Nothing's wrong — the stock market is just closed right now."
        body = [
            "CHAD holds new stock trades until regular hours, because off-hours trading is riskier.",
            "It paused new entries on its own; existing safety limits still apply.",
            "No action needed — it starts up again automatically at the open.",
        ]
    elif primary == "regime_adverse":
        head = "ℹ️ CHAD is sitting on its hands because conditions look poor."
        body = [
            "The market read is 'adverse', so CHAD silenced its strategies for now.",
            "That's a deliberate wait-for-better-conditions stance, not a fault.",
            "No action needed — it re-activates when the market read improves.",
        ]
    elif primary in ("warmup", "paused"):
        head = "ℹ️ CHAD is being cautious, not blocked."
        if scr_state == "WARMUP":
            body = [
                "It's still in warm-up: it needs more scored trades before it sizes up.",
                "So it trades small on purpose while it builds a track record.",
            ]
        else:
            body = [
                "Its self-rating dipped, so it pulled sizing back to stay safe.",
                "It keeps trading small until the numbers recover.",
            ]
        body.append("No action needed — this eases up on its own as results improve.")
    else:
        head = "ℹ️ Good news — nothing is blocking CHAD right now."
        body = [
            "No safety stop is active, the market is open, and conditions are workable.",
            "CHAD is trading within its normal paper-mode limits.",
            "No action needed.",
        ]
    standard = None
    if mode == "STANDARD":
        rth = f.get("rth_open")
        rth_txt = "open" if rth else ("closed" if rth is False else "unknown")
        standard = f"Context: safety-stop={'on' if f.get('halted') else 'off'}, market={rth_txt}, self-rating={scr_state}."
        body.append(standard)
    pro = (
        f"raw: primary_reason={primary} halted={f.get('halted')} stop_reason={f.get('stop_reason')!r} "
        f"rth_open={f.get('rth_open')} regime={f.get('regime')} scr_state={scr_state} "
        f"ready_for_live={f.get('ready_for_live')}"
    )
    return head, body, pro


def _ans_position_detail(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    symbol = str(f.get("symbol") or "that symbol").upper()
    broker_qty = _as_float(f.get("broker_qty"))
    broker_side = str(f.get("broker_side") or "").upper()
    chad_lots = f.get("chad_lots") or []
    verdict = str(f.get("exit_verdict") or "").upper()
    reason = str(f.get("exit_reason") or "")

    if not f.get("has_position"):
        head = f"ℹ️ You have no open {symbol} position right now."
        body = [
            "Neither the broker nor CHAD's own book shows any shares in it.",
            "No action needed.",
        ]
        pro = f"raw: symbol={symbol} broker_qty=None chad_lots=0"
        return head, body, pro

    head = f"ℹ️ Here's your {symbol} exposure."
    body: List[str] = []
    if broker_qty is not None:
        dir_word = "long" if (broker_side == "BUY" or broker_qty > 0) else "short"
        body.append(f"Broker shows {_fmt_qty(abs(broker_qty))} shares {dir_word} (confirmed by IBKR).")
    else:
        body.append("The broker hasn't confirmed a position in it under CHAD's tracking.")
    if chad_lots:
        lot_txt = ", ".join(
            f"{humanize_name(l.get('strategy'))} {_fmt_qty(l.get('quantity'))}" for l in chad_lots[:4]
        )
        body.append(f"CHAD's strategies tracking it: {lot_txt}.")
    if verdict == "WOULD_CLOSE":
        body.append("CHAD's exit-watch says this one is at a level it would trim/close (shadow only — it hasn't).")
    elif verdict == "HOLD":
        body.append("CHAD's exit-watch says: hold — no exit condition met yet.")
    elif verdict.startswith("SKIP"):
        body.append("CHAD's exit-watch is standing aside on this one (not an equity it manages, or unconfirmed).")
    body.append("No action needed — I only read and explain positions, I don't change them.")
    standard = None
    if mode == "STANDARD" and f.get("exit_price") is not None:
        standard = (
            f"Last look: price {f.get('exit_price')}, trailing-stop near {f.get('exit_atr_stop')}, "
            f"held {f.get('exit_age_days')} of {f.get('exit_max_hold')} days."
        )
        body.append(standard)
    pro = (
        f"raw: symbol={symbol} broker_qty={broker_qty} broker_side={broker_side} "
        f"chad_lots={len(chad_lots)} exit_verdict={verdict or None} exit_reason={reason or None}"
    )
    return head, body, pro


def _ans_system_health(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    failed = int(f.get("failed_count") or 0)
    stale_feeds = int(f.get("stale_feeds") or 0)
    delivery_ok = f.get("alert_delivery_ok")
    recent = f.get("recent_failures") or []

    if failed == 0 and stale_feeds == 0:
        head = "ℹ️ CHAD looks healthy."
        body = ["No failed background jobs, and its key data feeds are all fresh."]
    else:
        head = "⚠️ A couple of things are worth a glance, but nothing urgent."
        bits = []
        if failed:
            recent_units = ", ".join(humanize_name(r.get("unit")) for r in recent[:3])
            bits.append(f"{failed} background job" + ("s" if failed != 1 else "") + f" hiccupped recently ({recent_units})")
        if stale_feeds:
            bits.append(f"{stale_feeds} data feed" + ("s" if stale_feeds != 1 else "") + " running behind")
        body = ["; ".join(bits).capitalize() + "."]
        body.append("These are set to self-heal; CHAD re-flags them only if they keep failing.")
    if delivery_ok is False:
        body.append("Note: the last alert may not have been delivered — worth checking the messaging service.")
    elif delivery_ok is True:
        body.append("Alerts are being delivered normally.")
    body.append("No action needed unless you keep seeing the same thing.")
    standard = None
    if mode == "STANDARD":
        standard = f"Checked: {len(recent)} recent failure record(s), key feeds' freshness, and last alert delivery."
        body.append(standard)
    pro = (
        f"raw: failed_count={failed} stale_feeds={stale_feeds} "
        f"alert_delivery_ok={delivery_ok} recent={[r.get('unit') for r in recent[:5]]}"
    )
    return head, body, pro


def _ans_explain_last_alert(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    if not f.get("has_alert"):
        head = "ℹ️ There are no recent alerts to explain."
        body = ["CHAD hasn't raised anything lately — quiet is good.", "No action needed."]
        return head, body, "raw: has_alert=false"
    # Reuse CV1's alert voice for the actual body, then wrap with a lead-in.
    artifact = f.get("artifact") or {}
    inner = format_alert("service_failure", artifact, mode=mode)
    head = "ℹ️ Here's the most recent alert, in plain English:"
    body = [inner]
    pro = f"raw: artifact_path={artifact.get('artifact_path', '?')}"
    return head, body, pro


def _ans_regime_and_roster(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    regime = str(f.get("regime") or "unknown")
    allowed = f.get("active_now") or f.get("allowed") or []
    halted = f.get("halted") or []
    regime_word = {
        "ranging": "range-bound / sideways",
        "trending": "trending",
        "volatile": "volatile / choppy",
        "adverse": "adverse (stand-aside)",
    }.get(regime.lower(), regime)

    head = f"ℹ️ Right now the market reads as {regime_word}."
    if allowed:
        names = ", ".join(humanize_name(s) for s in allowed[:8])
        body = [f"Strategies cleared to trade in this regime: {names}."]
    else:
        body = ["No strategies are cleared to trade in this regime — CHAD is standing aside."]
    if halted:
        h_txt = ", ".join(f"{humanize_name(h.get('strategy'))}" for h in halted[:5])
        body.append(f"Benched for now (recent losing streaks): {h_txt}.")
    body.append("No action needed — CHAD rotates these automatically as the market shifts.")
    standard = None
    if mode == "STANDARD":
        conf = f.get("regime_confidence")
        standard = f"Regime confidence is {_fmt_pct(conf) if conf is not None else 'n/a'}; roster follows the regime-activation matrix."
        body.append(standard)
    pro = (
        f"raw: regime={regime} confidence={f.get('regime_confidence')} "
        f"allowed={list(allowed)} halted={[h.get('strategy') for h in halted]}"
    )
    return head, body, pro


def _ans_generic(f: Dict[str, Any], mode: str) -> Tuple[str, List[str], Optional[str]]:
    head = "ℹ️ I can tell you how CHAD is doing, why it paused, a position, its health, the last alert, or who's trading."
    body = [
        "Ask me something like “how are we doing?” or “why did CHAD stop?”",
        "No action needed.",
    ]
    return head, body, "raw: intent=unknown"


_ANSWER_TEMPLATES: Dict[str, Callable[[Dict[str, Any], str], Tuple[str, List[str], Optional[str]]]] = {
    "status_overview": _ans_status_overview,
    "why_paused": _ans_why_paused,
    "why_blocked": _ans_why_paused,  # alias
    "position_detail": _ans_position_detail,
    "system_health": _ans_system_health,
    "explain_last_alert": _ans_explain_last_alert,
    "regime_and_roster": _ans_regime_and_roster,
}


def format_answer(intent: str, facts: Optional[Dict[str, Any]] = None, *, mode: Optional[str] = None) -> str:
    """Render a read-only intent answer as a calm, plain-English coach message.

    ``facts`` is produced by :mod:`chad.utils.coach_intents` (the read-only
    intent layer). This function performs NO runtime reads — it only renders.
    Staleness disclosure (the ``_stale`` / ``_stale_age_seconds`` / ``_stale_label``
    keys) and PRO source citation (``_sources``) are handled uniformly here so
    every intent obeys the CV2 staleness rule. Never raises.
    """
    facts = dict(facts or {})
    active_mode = _norm_mode(mode) or resolve_mode()

    try:
        builder = _ANSWER_TEMPLATES.get(str(intent or "").strip().lower(), _ans_generic)
        head, body, pro = builder(facts, active_mode)
    except Exception as exc:  # never let presentation crash the bot
        _LOG.warning("coach_voice_answer_failed intent=%s err=%s", intent, exc)
        head, body, pro = _ans_generic(facts, active_mode)

    lines: List[str] = [head] + [b for b in (body or []) if b]

    # A source that could not be read is surfaced honestly (never faked).
    if facts.get("_error"):
        lines.append(
            "⚠️ I couldn't read one of the underlying files, so this may be incomplete."
        )

    stale_line = _staleness_line(facts)
    if stale_line:
        lines.append(stale_line)

    if active_mode == "PRO":
        if pro:
            lines.append(pro)
        src = _sources_pro_line(facts)
        if src:
            lines.append(src)

    text = "\n".join(ln.strip() for ln in lines if ln and ln.strip())
    if active_mode != "PRO":
        text = _scrub_codes(text)
    return text


# ── refusal path (zero-authority coach) ────────────────────────────────────────
# topic key -> (what the coach can't do, what an operator would do instead)
_REFUSAL_TOPICS: Dict[str, Tuple[str, str]] = {
    "trade": (
        "place, change, or close a trade",
        "an operator does that in the broker/execution layer — I only read and explain.",
    ),
    "service": (
        "start, stop, or restart any of CHAD's services",
        "that's an operator action on the server; I don't control services.",
    ),
    "config": (
        "change a setting, risk cap, or configuration",
        "config changes go through a reviewed Pending Action that a human applies.",
    ),
    "go_live": (
        "switch CHAD to live trading or flip its readiness flag",
        "live activation is gated and operator-only — I can't set it.",
    ),
    "generic": (
        "do that",
        "actions like that are operator-only; I'm read-and-explain only.",
    ),
}


def format_refusal(topic: Optional[str] = None, *, mode: Optional[str] = None, detail: Optional[str] = None) -> str:
    """Render the polite, fixed refusal for an out-of-authority request.

    The coach has zero execution authority: it interprets and presents only.
    Any request to *act* (trade / service control / config / go-live) returns
    this — 'I can tell you about it, but I can't do it' — with what an operator
    would do instead. Never raises, never has side effects.
    """
    active_mode = _norm_mode(mode) or resolve_mode()
    cant, instead = _REFUSAL_TOPICS.get(str(topic or "generic"), _REFUSAL_TOPICS["generic"])
    lines = [
        f"ℹ️ I can tell you about it, but I can't {cant}.",
        f"I'm a read-only coach — {instead}",
        "Happy to explain what's going on instead — just ask.",
    ]
    if active_mode == "PRO" and detail:
        lines.append(f"raw: refused_topic={topic} matched={detail}")
    text = "\n".join(lines)
    if active_mode != "PRO":
        text = _scrub_codes(text)
    return text


def format_mode_ack(mode: Any, ok: bool = True) -> str:
    """Confirm a verbosity mode switch in the coach voice."""
    m = _norm_mode(mode)
    if not ok or not m:
        return (
            "ℹ️ I couldn't change how I talk just now, so I'll keep my current style.\n"
            "You can try: 'talk simple', 'talk normal', or 'talk pro'."
        )
    disp = _MODE_DISPLAY.get(m, m.lower())
    blurb = {
        "SIMPLE": "plain and friendly, no jargon",
        "STANDARD": "plain English with a bit more context",
        "PRO": "detailed, with the raw numbers and sources",
    }.get(m, "")
    return (
        f"ℹ️ Done — I'll talk in {disp} mode from now on ({blurb}).\n"
        "Ask me anything, like 'how are we doing?'"
    )
