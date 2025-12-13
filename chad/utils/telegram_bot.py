#!/usr/bin/env python3
"""
Telegram Coach Bot for CHAD
Phase 8 – Price Engine + Risk View + Research + Teaching Mode (Read-Only)

This bot provides:
    * /ping            – liveness check
    * /help            – command summary
    * /shadow          – SCR risk state (rewritten for beginners)
    * /price SYMBOL    – live price lookup via CHAD API Gateway
    * /ai_research SYMBOL [question...]
                       – macro/risk research via /ai/research
    * natural language:
          “price of AAPL?”, “AAPL quote”, “what’s TSLA doing today?”,
          “what’s the macro picture for SPY?”, “explain today’s risk posture”

SAFETY:
    * 100% read-only:
        - Never sends orders.
        - Never flips CHAD_MODE.
        - Never modifies config or state.
    * Will NOT answer “what should I buy/sell?” with trades; only explains
      data and patterns.

ENGINEERING STANDARDS:
    * Structured logging
    * Robust HTTP handling
    * Timeout protection
    * Graceful OpenAI degradation
    * Clean, beginner-friendly answers taught via Teaching Mode:
        - Short, direct, max 3–4 sentences
        - Always ends with: “Why this matters: …”
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
)

from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence
from chad.analytics.shadow_formatting import format_shadow_summary
from chad.core.mode import get_chad_mode, is_live_mode_enabled

# ======================================================================
# Logging & basic helpers
# ======================================================================

LOGGER_NAME = "chad.telegram_bot"
LOGGER = logging.getLogger(LOGGER_NAME)


def _init_logger() -> logging.Logger:
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return LOGGER


def _env(name: str, required: bool = True) -> Optional[str]:
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_backend_base_url() -> str:
    base = os.environ.get("CHAD_BACKEND_BASE_URL", "").strip()
    if not base:
        return "http://127.0.0.1:9618"
    return base.rstrip("/")


# ======================================================================
# OpenAI client & config (Teaching Mode)
# ======================================================================

@dataclass(frozen=True)
class OpenAIConfig:
    intent_model: str = os.environ.get("OPENAI_INTENT_MODEL", "gpt-4.1-mini")
    coach_model: str = os.environ.get("OPENAI_COACH_MODEL", "gpt-4.1-mini")


_OPENAI_CFG = OpenAIConfig()
_openai_client: Optional[OpenAI] = None


def _load_env_file(path: str) -> None:
    """Best-effort KEY=VALUE loader without overwriting existing env vars."""
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load env file %s: %s", path, exc)


def _get_openai_client() -> Optional[OpenAI]:
    """
    Lazily instantiate OpenAI client, loading OPENAI_API_KEY from
    /etc/chad/openai.env if needed.

    Returns None if key is not available or initialization fails.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if not os.environ.get("OPENAI_API_KEY"):
        _load_env_file("/etc/chad/openai.env")

    if not os.environ.get("OPENAI_API_KEY"):
        LOGGER.warning(
            "OPENAI_API_KEY not set; Teaching Mode will fall back to raw text."
        )
        return None

    try:
        _openai_client = OpenAI()
        return _openai_client
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to initialize OpenAI client: %s", exc)
        return None


# ======================================================================
# Backend client (CHAD API Gateway)
# ======================================================================

class BackendClientError(RuntimeError):
    """Error raised when the CHAD backend returns an error or bad JSON."""


class BackendClient:
    """
    Thin HTTP client for CHAD backend endpoints.

    Endpoints used:
        - /ai/research  (AI research scenarios)
        - /ai/price     (unified price snapshots)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 15.0) -> None:
        self.base_url = base_url or _get_backend_base_url()
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _check_json(self, resp: requests.Response, context: str) -> Dict[str, Any]:
        if resp.status_code != 200:
            try:
                err_json = resp.json()
            except Exception:
                err_json = {}
            detail = err_json.get("detail") if isinstance(err_json, dict) else resp.text
            raise BackendClientError(
                f"{context} returned {resp.status_code}: {detail}"
            )
        try:
            data: Dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise BackendClientError(f"{context} JSON decode error: {exc}") from exc
        return data

    def ai_research(self, symbol: str, question: str, timeframe: str = "1m") -> Dict[str, Any]:
        payload = {
            "symbol": symbol.upper().strip(),
            "scenario_timeframe": timeframe,
            "question": question.strip() or "High-level macro and risk context.",
        }
        try:
            resp = self._session.post(
                self._url("/ai/research"),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackendClientError(f"Network error calling /ai/research: {exc}") from exc

        return self._check_json(resp, "backend /ai/research")

    def ai_price(self, symbol: str) -> Dict[str, Any]:
        params = {"symbol": symbol.upper().strip()}
        try:
            resp = self._session.get(
                self._url("/ai/price"),
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackendClientError(f"Network error calling /ai/price: {exc}") from exc

        data = self._check_json(resp, "backend /ai/price")
        if "price" not in data:
            raise BackendClientError(f"/ai/price payload missing 'price': {data}")
        return data


# ======================================================================
# Teaching Layer (Coaching Mode)
# ======================================================================

def _coach_rewrite(raw_text: str, purpose: str) -> str:
    """
    Final hardened rewrite for Personality A — Ultra-Simple Teacher.
    Guarantees:
        - At most 2 simple sentences before the Why-line.
        - Exactly ONE "Why this matters:" line.
        - No repeated explanations, no long paragraphs.
        - No investment advice.
        - Very plain English for total beginners.
    """

    raw_text = (raw_text or "").strip()
    if not raw_text:
        return (
            "There isn't much useful information here.\n"
            "Why this matters: it keeps you focused on what truly affects risk."
        )

    client = _get_openai_client()
    if client is None:
        return raw_text

    system_prompt = """
You are CHAD Coach — a friendly, ultra-simple money teacher.

Rules:
- 2 short sentences maximum before the final line.
- Final line MUST start with "Why this matters: ".
- No lists. No paragraphs. No fancy terms.
- Explain things as if the user has zero financial knowledge.
- NEVER say what someone should buy or sell.
"""

    user_prompt = f"""
Rewrite the following into:
1) Two very short sentences explaining it simply.
2) One sentence starting with EXACTLY "Why this matters: ".

Keep the ENTIRE answer under 45 words before the Why-line.

Raw text:
{raw_text}
"""

    try:
        resp = client.chat.completions.create(
            model=_OPENAI_CFG.coach_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=100,
        )

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return raw_text

        # Split into lines and strip blanks
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        # Find all Why-lines and keep only the first
        why_lines = [ln for ln in lines if ln.lower().startswith("why this matters:")]
        why_line = (
            why_lines[0]
            if why_lines
            else "Why this matters: it helps you understand what is changing and stay calm."
        )

        # Remove all Why-lines from the explanation part
        expl_candidates = [
            ln for ln in lines if not ln.lower().startswith("why this matters:")
        ]

        # Merge explanation lines and split into sentences
        import re
        merged = " ".join(expl_candidates)
        merged = re.sub(r"\s+", " ", merged).strip()
        sentences = re.split(r"(?<=[.!?])\s+", merged)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Keep at most two sentences
        expl_final = sentences[:2]
        if not expl_final:
            expl_final = ["Something changed in the market."]

        final_text = "\n".join(expl_final + [why_line])
        return final_text

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Coach rewrite failed (%s). Using raw text.", exc)
        return raw_text


# ======================================================================
# Risk / Shadow helpers
# ======================================================================

def _shadow_summary_text() -> str:
    """Return the technical Mode + SCR summary string."""
    mode = get_chad_mode()
    live_enabled = is_live_mode_enabled()

    stats = load_and_compute(
        max_trades=200,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    shadow_state = evaluate_confidence(stats)
    scr_summary = format_shadow_summary(shadow_state)

    header = [
        "Mode",
        f"  CHAD_MODE   : {mode.value}",
        f"  live_enabled: {live_enabled}",
        "",
    ]
    return "\n".join(header) + scr_summary


def _reply_chunked(update: Update, text: str) -> None:
    """Send text to Telegram, chunking to respect the 4096-char limit."""
    message = update.message
    if message is None:
        return

    if len(text) <= 4000:
        message.reply_text(text)
        return

    lines = text.splitlines()
    chunk: List[str] = []
    current = 0
    for line in lines:
        if current + len(line) + 1 > 4000:
            message.reply_text("\n".join(chunk))
            chunk = []
            current = 0
        chunk.append(line)
        current += len(line) + 1
    if chunk:
        message.reply_text("\n".join(chunk))


# ======================================================================
# Auth helpers
# ======================================================================

def _authorized(update: Update, allowed_chat_id: Optional[int]) -> bool:
    if allowed_chat_id is None:
        return True
    chat = update.effective_chat
    return bool(chat and chat.id == allowed_chat_id)


def _unauth_msg(update: Update) -> str:
    chat = update.effective_chat
    return f"Unauthorized chat_id={chat.id if chat else 'unknown'}."


# ======================================================================
# Trading instruction filter
# ======================================================================

_BANNED_KEYWORDS = [
    "buy",
    "sell",
    "short",
    "go long",
    "open a position",
    "enter a position",
    "execute",
    "place order",
    "market order",
    "limit order",
    "stop loss",
]


def _looks_like_trading_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _BANNED_KEYWORDS)


# ======================================================================
# Symbol extraction
# ======================================================================

_SYMBOL_REGEX = re.compile(r"\b[A-Z]{2,6}\b")


def _extract_symbol_from_text(text: str) -> Optional[str]:
    """
    Extract a likely ticker symbol from free text.

    Rules:
        - Look for ALL-CAPS tokens 2–6 characters long.
        - Ignore common English words.
        - Return first candidate, or None.
    """
    upper = text.upper()
    candidates = _SYMBOL_REGEX.findall(upper)
    if not candidates:
        return None

    stopwords = {
        "WHAT",
        "WHATS",
        "IS",
        "ARE",
        "THE",
        "AND",
        "FOR",
        "WITH",
        "THIS",
        "THAT",
        "MODE",
        "RISK",
        "SCR",
        "WHY",
        "HOW",
        "NOW",
        "TODAY",
        "PLEASE",
        "YOU",
        "CAN",
    }

    for token in candidates:
        if token not in stopwords:
            return token
    return None


# ======================================================================
# Formatting helpers
# ======================================================================

def _format_price_raw_text(symbol: str, snap: Dict[str, Any]) -> str:
    """
    Build a short technical text from a price snapshot, ready for coaching.

    Expects:
        snap = {
          "symbol": "AAPL",
          "asset_class": "equity",
          "price": 279.28,
          "change": 2.10 | null,
          "percent_change": 0.75 | null,
          "as_of": "...",
          "source": "polygon"
        }
    """
    price = snap.get("price")
    change = snap.get("change")
    pct = snap.get("percent_change")

    parts: List[str] = []

    if isinstance(price, (int, float)):
        parts.append(f"{symbol} last price: {price:.2f}.")
    else:
        parts.append(f"No recent price available for {symbol}.")

    if isinstance(change, (int, float)) and isinstance(pct, (int, float)):
        direction = "up" if change > 0 else "down" if change < 0 else "flat"
        parts.append(
            f"It is {direction} {abs(change):.2f} ({abs(pct):.2f}%) versus yesterday."
        )

    return " ".join(parts)


def _format_research_raw_text(data: Dict[str, Any]) -> str:
    """
    Format the ResearchScenario JSON into a technical summary string.
    This is then passed through the Coaching Layer.
    """
    symbol = str(data.get("symbol", "") or "").upper()
    timeframe = str(data.get("scenario_timeframe", "") or "")
    macro_risks = data.get("macro_risks") or []
    test_ideas = data.get("test_ideas") or []
    bull = data.get("bull_case") or ""
    bear = data.get("bear_case") or ""
    base = data.get("base_case") or ""
    confidence = data.get("confidence_score")

    lines: List[str] = []

    header = f"AI Research – {symbol or 'UNKNOWN'} (tf={timeframe or 'n/a'})"
    lines.append(header)
    lines.append("")

    if isinstance(macro_risks, list) and macro_risks:
        lines.append("Macro Risks:")
        for idx, risk in enumerate(macro_risks, start=1):
            lines.append(f"  {idx}. {risk}")
        lines.append("")

    if bull:
        lines.append("Bull Case:")
        lines.append(f"  {bull}")
        lines.append("")
    if bear:
        lines.append("Bear Case:")
        lines.append(f"  {bear}")
        lines.append("")
    if base:
        lines.append("Base Case:")
        lines.append(f"  {base}")
        lines.append("")

    if isinstance(test_ideas, list) and test_ideas:
        lines.append("Test Ideas (for CHAD / research, not direct trading):")
        for idx, idea in enumerate(test_ideas, start=1):
            lines.append(f"  {idx}. {idea}")
        lines.append("")

    if isinstance(confidence, (int, float)):
        lines.append(f"Model confidence score: {confidence:.2f}")

    return "\n".join(lines)


# ======================================================================
# Command handlers
# ======================================================================

def cmd_ping(update: Update, context: CallbackContext) -> None:
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return
    update.message.reply_text("PONG — CHAD Coach is alive.")


def cmd_help(update: Update, context: CallbackContext) -> None:
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return

    lines = [
        "CHAD Coach — Commands:",
        "/ping           – Liveness check",
        "/help           – This message",
        "/shadow         – Risk & SCR summary (teaching mode)",
        "/price SYMBOL   – Live quote via CHAD backend",
        "/ai_research SYMBOL [question...] – Macro/risk research (teaching mode)",
        "",
        "You can also ask naturally:",
        "  'price of TSLA?', 'AAPL quote', 'what's NVDA doing?',",
        "  'what's the macro picture for SPY?', 'explain today's risk posture'.",
    ]
    update.message.reply_text("\n".join(lines))


def cmd_shadow(update: Update, context: CallbackContext) -> None:
    """
    Show current Mode + Shadow Confidence state, rewritten via Teaching Mode.
    """
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return

    try:
        raw = _shadow_summary_text()
        coached = _coach_rewrite(raw, purpose="risk_state")
        _reply_chunked(update, coached)
        mode = get_chad_mode()
        live_enabled = is_live_mode_enabled()
        log.info(
            "Sent /shadow teaching summary to chat_id=%s (mode=%s, live_enabled=%s)",
            update.effective_chat.id if update.effective_chat else "unknown",
            mode.value,
            live_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Error handling /shadow command: %s", exc)
        update.message.reply_text("Error computing risk state. Check logs for details.")


def cmd_price(update: Update, context: CallbackContext) -> None:
    """Explicit /price SYMBOL command."""
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return

    msg = (update.message.text or "").strip()
    parts = msg.split(maxsplit=1)
    if len(parts) < 2:
        update.message.reply_text("Usage: /price SYMBOL")
        return

    symbol = parts[1].upper().strip()
    backend = BackendClient()
    try:
        snap = backend.ai_price(symbol)
        raw_text = _format_price_raw_text(symbol, snap)
        coached = _coach_rewrite(raw_text, purpose="price")
        update.message.reply_text(coached)
        log.info(
            "Sent /price teaching response for symbol=%s to chat_id=%s",
            symbol,
            update.effective_chat.id if update.effective_chat else "unknown",
        )
    except BackendClientError as exc:
        log.warning("Backend /ai/price error for %s: %s", symbol, exc)
        update.message.reply_text(f"Could not fetch price for {symbol}.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error in /price: %s", exc)
        update.message.reply_text(
            f"Unexpected error while fetching price for {symbol}. Check logs."
        )


def cmd_ai_research(update: Update, context: CallbackContext) -> None:
    """
    Handle /ai_research SYMBOL [question...]

    Teaching Mode:
        - Fetch structured research from backend.
        - Rewrite into beginner-friendly explanation with Teaching Layer.
    """
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return

    message = update.message
    if message is None:
        return

    args = context.args or []
    if not args:
        message.reply_text(
            "Usage:\n"
            "/ai_research SYMBOL [question...]\n\n"
            "Example:\n"
            "/ai_research SPY high-level macro and risk context"
        )
        return

    symbol_raw = args[0]
    question = " ".join(args[1:]) if len(args) > 1 else ""
    symbol = symbol_raw.upper().strip()
    if not symbol:
        message.reply_text("Please provide a valid symbol, e.g. /ai_research SPY")
        return

    backend = BackendClient()
    try:
        data = backend.ai_research(symbol, question)
        raw_text = _format_research_raw_text(data)
        coached = _coach_rewrite(raw_text, purpose="research")
        message.reply_text(coached)
        log.info(
            "Sent /ai_research teaching response for symbol=%s to chat_id=%s",
            symbol,
            update.effective_chat.id if update.effective_chat else "unknown",
        )
    except BackendClientError as exc:
        log.warning("Backend /ai/research error for %s: %s", symbol, exc)
        message.reply_text(
            "I couldn’t fetch research for that symbol.\n"
            "Try again or check the CHAD backend logs."
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error in /ai_research: %s", exc)
        message.reply_text(
            "Something unexpected happened while fetching research.\n"
            "Try again or check system logs."
        )


# ======================================================================
# Free-text handler (Neural Operator + Teaching Mode)
# ======================================================================

def handle_free_text(update: Update, context: CallbackContext) -> None:
    """
    Handle non-command messages safely:

        - Block obvious trading instructions.
        - If a symbol is present:
            * If message looks like a "price now?" question → /ai/price + coach.
            * Otherwise → /ai/research + coach.
        - If no symbol but risk language → /shadow teaching summary.
        - Else → fallback teaching hint.
    """
    log = _init_logger()
    allowed_chat_id = context.bot_data.get("allowed_chat_id")
    if not _authorized(update, allowed_chat_id):
        log.warning(_unauth_msg(update))
        return

    message = update.message
    if message is None or not message.text:
        return

    text = message.text.strip()
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"

    # Safety: block direct trading instructions.
    if _looks_like_trading_instruction(text):
        message.reply_text(
            "I can't take trading or execution instructions here.\n\n"
            "CHAD Coach is advisory-only. I can help explain prices, risk, "
            "and patterns so YOU can make decisions, but I will never place trades."
        )
        log.info("Blocked apparent trading instruction from chat_id=%s", chat_id)
        return

    lower = text.lower()
    backend = BackendClient()

    # Symbol path
    symbol_candidate = _extract_symbol_from_text(text)
    if symbol_candidate:
        symbol = symbol_candidate.upper().strip()
        try:
            wants_price = any(
                token in lower
                for token in (
                    "price",
                    "quote",
                    "doing",
                    "at",
                    "today",
                    "trading",
                    "how is",
                    "what's",
                    "whats",
                )
            )

            if wants_price:
                snap = backend.ai_price(symbol)
                raw_text = _format_price_raw_text(symbol, snap)
                coached = _coach_rewrite(raw_text, purpose="price")
                message.reply_text(coached)
                log.info(
                    "Free-text routed to price: chat_id=%s symbol=%s",
                    chat_id,
                    symbol,
                )
                return

            # Otherwise: treat as research
            data = backend.ai_research(symbol, text)
            raw_text = _format_research_raw_text(data)
            coached = _coach_rewrite(raw_text, purpose="research")
            message.reply_text(coached)
            log.info(
                "Free-text routed to research: chat_id=%s symbol=%s",
                chat_id,
                symbol,
            )
            return

        except BackendClientError as exc:
            log.warning(
                "Backend error in free-text symbol handler (symbol=%s): %s",
                symbol,
                exc,
            )
            message.reply_text(
                "I ran into an issue fetching data for that symbol.\n"
                "You can also try explicitly:\n"
                f"/price {symbol}  or  /ai_research {symbol} {text}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error in free-text symbol handler: %s", exc)
            message.reply_text(
                "Something unexpected happened while fetching that symbol.\n"
                f"Try: /price {symbol}  or  /ai_research {symbol} {text}"
            )
            return

    # Risk language (no symbol)
    if any(keyword in lower for keyword in ("risk", "posture", "scr", "cautious", "confident", "warmup", "paused")):
        try:
            raw = _shadow_summary_text()
            coached = _coach_rewrite(raw, purpose="risk_state")
            _reply_chunked(update, coached)
            log.info(
                "Free-text routed to risk_state: chat_id=%s text=%r",
                chat_id,
                text,
            )
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Error in free-text risk handler: %s", exc)
            message.reply_text(
                "Error computing risk state. You can also try `/shadow` directly."
            )
            return

    # Fallback: educational hint
    fallback = (
        "I can help you understand:\n"
        "  • What a symbol (like AAPL or SPY) is doing right now.\n"
        "  • How the overall risk mood looks (cautious vs confident).\n\n"
        "Try asking:\n"
        "  \"What is the macro picture for SPY?\"\n"
        "  \"Explain today's risk posture.\"\n"
        "  \"What's AAPL price right now?\"\n\n"
        "Why this matters: having a clear mental picture of what's going on "
        "makes it easier to stay calm and avoid random decisions."
    )
    message.reply_text(fallback)


# ======================================================================
# Bot launch
# ======================================================================

def build_updater(token: str) -> Updater:
    """
    Build the Telegram Updater with all CHAD Coach handlers attached.
    """
    _init_logger()
    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher

    # Authorization scope
    allowed_raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID")
    allowed_chat_id: Optional[int] = None
    if allowed_raw:
        try:
            allowed_chat_id = int(allowed_raw)
        except ValueError:
            LOGGER.warning(
                "TELEGRAM_ALLOWED_CHAT_ID is not a valid int: %s – ignoring and allowing all chats.",
                allowed_raw,
            )
            allowed_chat_id = None

    dp.bot_data["allowed_chat_id"] = allowed_chat_id

    # Command handlers
    dp.add_handler(CommandHandler("ping", cmd_ping))
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("shadow", cmd_shadow))
    dp.add_handler(CommandHandler("price", cmd_price))
    dp.add_handler(CommandHandler("ai_research", cmd_ai_research))

    # Free-text (Neural Operator + Teaching Mode)
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_free_text))

    return updater


def run_bot() -> None:
    """
    Run the Telegram coach bot until interrupted.

    Expects:
        TELEGRAM_BOT_TOKEN (required)
        TELEGRAM_ALLOWED_CHAT_ID (optional)
        OPENAI_API_KEY (or /etc/chad/openai.env)
    """
    log = _init_logger()

    try:
        token = _env("TELEGRAM_BOT_TOKEN", required=True)
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    updater = build_updater(token)
    bot_info = updater.bot.get_me()
    log.info("Starting CHAD Coach bot as @%s (id=%s)", bot_info.username, bot_info.id)

    try:
        updater.start_polling()
        log.info("CHAD Coach bot is now polling for updates.")
        updater.idle()
    except KeyboardInterrupt:
        log.info("CHAD Coach bot interrupted by user.")
    finally:
        log.info("CHAD Coach bot stopped.")


def main() -> None:
    """CLI entrypoint: python -m chad.utils.telegram_bot"""
    run_bot()


if __name__ == "__main__":
    main()
