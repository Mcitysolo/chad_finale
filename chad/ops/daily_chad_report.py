#!/usr/bin/env python3
"""
CHAD Daily Report — Plain English, Zero Jargon

Designed for someone with zero trading knowledge.
Every financial term explained simply. Warm, encouraging tone.

Sections:
  1. Header
  2. Did We Make Money?
  3. What Did We Do?
  4. Best Move Today
  5. Worst Move Today
  6. What's Working (strategy status)
  7. Market Temperature
  8. CHAD Status
  9. Your Paper Account
  10. Sign Off

Also includes a morning brief generator for pre-market.
"""

from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/home/ubuntu/chad_finale")
DATA_TRADES = REPO_ROOT / "data" / "trades"
RUNTIME_DIR = REPO_ROOT / "runtime"
PNL_STATE_PATH = RUNTIME_DIR / "pnl_state.json"
LIVE_READINESS_PATH = RUNTIME_DIR / "live_readiness.json"
REPORTS_DIR = REPO_ROOT / "reports" / "ops"

TRADE_FILE_GLOB = "trade_history_*.ndjson"

# ---------------------------------------------------------------------------
# Instrument translations — plain English
# ---------------------------------------------------------------------------

INSTRUMENT_NAMES: Dict[str, str] = {
    "MES": "S&P 500 futures (a bet on the whole US stock market)",
    "ES": "S&P 500 futures (a bet on the whole US stock market)",
    "MNQ": "Nasdaq futures (a bet on tech stocks)",
    "NQ": "Nasdaq futures (a bet on tech stocks)",
    "MCL": "Oil futures (a bet on crude oil prices)",
    "CL": "Oil futures (a bet on crude oil prices)",
    "MGC": "Gold futures (a bet on gold prices)",
    "GC": "Gold futures (a bet on gold prices)",
    "SPY": "S&P 500 ETF (tracks the whole US stock market)",
    "QQQ": "Nasdaq ETF (tracks big tech companies)",
    "GLD": "Gold ETF (tracks gold prices)",
    "TLT": "Long-term Treasury bonds (safe government loans)",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "SVXY": "Volatility fund (profits when markets are calm)",
    "UVXY": "Volatility fund (profits when markets are panicking)",
    "SH": "Crash protection (profits if markets fall)",
    "PSQ": "Crash protection (profits if markets fall)",
}

# ---------------------------------------------------------------------------
# Strategy translations — plain English
# ---------------------------------------------------------------------------

STRATEGY_NAMES: Dict[str, str] = {
    "alpha": "Momentum trading (riding trends)",
    "beta": "Long-term holds (the slow and steady bucket)",
    "gamma": "Swing trading (medium-term moves)",
    "gamma_reversion": "Mean reversion (fading overreactions)",
    "alpha_futures": "Futures momentum (riding trends in commodities and indexes)",
    "gamma_futures": "Futures reversion (fading overreactions in commodities)",
    "omega": "Crash protection (our insurance policy)",
    "omega_macro": "Macro bets (bonds, currencies, metals)",
    "omega_vol": "Volatility plays (betting on market fear/calm)",
    "alpha_options": "Options plays (defined-risk bets on SPY)",
    "crypto": "Crypto trades (Bitcoin, Ethereum, Solana)",
    "alpha_crypto": "Crypto trades (Bitcoin, Ethereum, Solana)",
    "delta": "Execution optimizer (makes our trades smarter)",
}

# ---------------------------------------------------------------------------
# VIX translations
# ---------------------------------------------------------------------------

def vix_description(vix: float) -> str:
    """Translate VIX number into plain English."""
    if vix < 15:
        return "Very calm 😴 — markets are quiet, like a slow Tuesday"
    elif vix < 20:
        return "Normal 🌤️ — regular market day, nothing unusual"
    elif vix < 25:
        return "A bit nervous 🌥️ — some uncertainty in the air"
    elif vix < 30:
        return "Worried 🌧️ — investors are getting anxious"
    else:
        return "Fearful ⛈️ — markets are in panic mode"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today_yyyymmdd() -> str:
    return _utc_now().strftime("%Y%m%d")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _format_money(amount: float) -> str:
    """Format dollar amount in a friendly way."""
    if abs(amount) >= 1000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def translate_instrument(symbol: str) -> str:
    """Turn a ticker symbol into plain English."""
    s = symbol.strip().upper()
    return INSTRUMENT_NAMES.get(s, s)


def translate_strategy(strategy: str) -> str:
    """Turn a strategy code into plain English."""
    s = strategy.strip().lower()
    return STRATEGY_NAMES.get(s, s)


# ---------------------------------------------------------------------------
# Trade loading (reuses patterns from daily_performance_report)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeRow:
    strategy: str
    symbol: str
    side: str
    pnl: float
    ts_utc: Optional[str]


def _is_untrusted(payload: Dict[str, Any]) -> bool:
    """Check if trade row has untrusted PnL."""
    extra = payload.get("extra")
    if isinstance(extra, dict) and extra.get("pnl_untrusted") is True:
        return True
    tags = payload.get("tags") or []
    if isinstance(tags, list) and any(str(t).lower() == "pnl_untrusted" for t in tags):
        return True
    if payload.get("pnl_untrusted") is True:
        return True
    # Legacy alpha_crypto with pnl==0.0
    strat = str(payload.get("strategy", "")).strip().lower()
    if strat == "alpha_crypto":
        try:
            pnl = float(payload.get("pnl", 0.0))
        except Exception:
            pnl = 0.0
        if pnl == 0.0:
            return True
    return False


def _extract_row(obj: Dict[str, Any]) -> Optional[TradeRow]:
    """Extract a trusted trade row from an NDJSON record."""
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        payload = obj

    # Skip paper_sim
    tags = payload.get("tags", [])
    if isinstance(tags, list) and any(str(t).lower() == "paper_sim" for t in tags):
        return None

    # Skip live trades
    if payload.get("is_live") is True:
        return None

    # Skip untrusted
    if _is_untrusted(payload):
        return None

    pnl_raw = payload.get("pnl")
    if pnl_raw is None:
        return None
    try:
        pnl = float(pnl_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(pnl):
        return None

    strategy = str(payload.get("strategy", "unknown")).strip().lower()
    symbol = str(payload.get("symbol", "UNKNOWN")).strip().upper()
    side = str(payload.get("side", "unknown")).strip().lower()
    ts = obj.get("timestamp_utc") or payload.get("entry_time_utc")

    return TradeRow(strategy=strategy, symbol=symbol, side=side, pnl=pnl, ts_utc=str(ts) if ts else None)


def load_today_trades(trades_dir: Optional[Path] = None) -> List[TradeRow]:
    """Load today's trusted paper trades."""
    root = trades_dir or DATA_TRADES
    today = _today_yyyymmdd()
    ledger = root / f"trade_history_{today}.ndjson"

    if not ledger.exists():
        return []

    rows: List[TradeRow] = []
    try:
        for line in ledger.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            row = _extract_row(obj)
            if row is not None:
                rows.append(row)
    except Exception:
        pass
    return rows


def load_week_trades(trades_dir: Optional[Path] = None) -> List[TradeRow]:
    """Load this week's trusted paper trades (last 7 days)."""
    root = trades_dir or DATA_TRADES
    if not root.exists():
        return []

    cutoff = _utc_now().date() - timedelta(days=7)
    rows: List[TradeRow] = []

    for f in sorted(root.glob(TRADE_FILE_GLOB)):
        try:
            ymd = f.name.split("_", 2)[2].split(".", 1)[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            if dt < cutoff:
                continue
        except Exception:
            continue

        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                row = _extract_row(obj)
                if row is not None:
                    rows.append(row)
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------------------
# Market data helpers (best-effort, from runtime cache)
# ---------------------------------------------------------------------------

def _load_price_cache() -> Dict[str, Any]:
    """Load price cache from runtime if available."""
    for name in ("price_cache.json", "market_snapshot.json"):
        data = _read_json(RUNTIME_DIR / name)
        if data:
            return data
    return {}


def _get_vix() -> Optional[float]:
    # 1. Try price_cache.json (top-level and nested under "prices")
    cache = _load_price_cache()
    for source in (cache, cache.get("prices", {})):
        for key in ("VIX", "^VIX", "CBOE:VIX"):
            val = source.get(key)
            if val is not None:
                if isinstance(val, (int, float)):
                    return _safe_float(val)
                if isinstance(val, dict):
                    p = val.get("price") or val.get("last") or val.get("close")
                    if p is not None:
                        return _safe_float(p)

    # 2. Try ibkr_status.json
    ibkr = _read_json(RUNTIME_DIR / "ibkr_status.json")
    if ibkr:
        for key in ("VIX", "vix"):
            val = ibkr.get(key)
            if val is not None:
                return _safe_float(val)

    return None


def _get_price(symbol: str) -> Optional[float]:
    cache = _load_price_cache()
    val = cache.get(symbol)
    if val is not None:
        if isinstance(val, (int, float)):
            return _safe_float(val)
        if isinstance(val, dict):
            p = val.get("price") or val.get("last") or val.get("close")
            if p is not None:
                return _safe_float(p)
    return None


def _get_spy_change() -> Optional[float]:
    cache = _load_price_cache()
    val = cache.get("SPY")
    if isinstance(val, dict):
        chg = val.get("change_pct") or val.get("pct_change")
        if chg is not None:
            return _safe_float(chg)
    return None


# ---------------------------------------------------------------------------
# CHAD's Take — AI-generated paragraph (best-effort)
# ---------------------------------------------------------------------------

def _generate_chads_take(
    total_pnl: float,
    total_trades: int,
    wins: int,
    losses: int,
    vix: Optional[float],
    strategy_results: Dict[str, float],
) -> str:
    """Generate a 2-3 sentence plain English take using Claude Haiku."""
    try:
        from chad.intel.claude_client import ClaudeClient

        client = ClaudeClient.load()

        strat_summary = ", ".join(
            f"{translate_strategy(s)}: {'made' if pnl > 0 else 'lost'} {_format_money(abs(pnl))}"
            for s, pnl in sorted(strategy_results.items())
        )

        prompt = (
            f"Today's results: {'made' if total_pnl >= 0 else 'lost'} {_format_money(abs(total_pnl))} total. "
            f"{total_trades} trades, {wins} winners, {losses} losers. "
            f"VIX (fear gauge): {vix if vix else 'unknown'}. "
            f"Strategy results: {strat_summary}. "
            f"Write 2-3 sentences summarizing today for someone who knows nothing about trading. "
            f"Use a sports analogy or everyday comparison. Be warm and encouraging."
        )

        text, _, _ = client._call_claude(
            prompt=prompt,
            system=(
                "You are CHAD, a friendly trading assistant explaining today's performance "
                "to someone with zero trading knowledge. Use simple language, sports analogies, "
                "and everyday comparisons. Never use financial jargon. Keep it to 2-3 sentences."
            ),
            task_type="routine",
            max_tokens=200,
            temperature=0.7,
        )
        return text.strip()
    except Exception:
        # Graceful fallback
        if total_pnl >= 0:
            return (
                f"Overall a {'solid' if total_pnl > 50 else 'quiet'} day — "
                f"we ended in the green. The system is running smoothly and doing its job."
            )
        else:
            return (
                "Tough day, but that's part of the game. Even the best teams lose sometimes. "
                "The important thing is the system is working correctly and managing risk."
            )


def _generate_quiet_day_take() -> str:
    """Generate CHAD's take for weekends / no-trade days."""
    try:
        from chad.intel.claude_client import ClaudeClient

        client = ClaudeClient.load()
        prompt = (
            "CHAD had a quiet day — markets were closed. Write 2 sentences telling the owner "
            "their system is ready for Monday and what to expect. Friendly, plain English, "
            "no jargon. Sign off warmly."
        )
        text, _, _ = client._call_claude(
            prompt=prompt,
            system=(
                "You are CHAD, a friendly trading assistant. The owner knows nothing about "
                "trading. Use simple language, be warm and encouraging. Keep it to 2 sentences."
            ),
            task_type="routine",
            max_tokens=150,
            temperature=0.7,
        )
        return text.strip()
    except Exception:
        return (
            "Quiet day — markets were closed and the system took a well-deserved rest. "
            "Everything is loaded up and ready to go when Monday rolls around."
        )


# ---------------------------------------------------------------------------
# DailyCHADReport
# ---------------------------------------------------------------------------

class DailyCHADReport:
    """Generate a plain-English daily report for non-traders."""

    def __init__(self, repo_root: Optional[Path] = None, trades_dir: Optional[Path] = None):
        self._root = repo_root or REPO_ROOT
        self._trades_dir = trades_dir or (self._root / "data" / "trades")
        self._runtime_dir = self._root / "runtime"

    async def send_to_telegram(self) -> bool:
        """Generate the daily report and send it via Telegram."""
        message = self.generate()
        _send_telegram(message)
        return True

    def generate(self) -> str:
        """Generate the full daily report as a single string."""
        now = _utc_now()
        today_trades = load_today_trades(self._trades_dir)
        week_trades = load_week_trades(self._trades_dir)

        total_pnl = sum(t.pnl for t in today_trades)
        total_trades = len(today_trades)
        wins = sum(1 for t in today_trades if t.pnl > 0)
        losses = sum(1 for t in today_trades if t.pnl < 0)
        flat = total_trades - wins - losses

        # Strategy rollups
        strategy_pnl: Dict[str, float] = {}
        strategy_trades: Dict[str, int] = {}
        for t in today_trades:
            strategy_pnl[t.strategy] = strategy_pnl.get(t.strategy, 0.0) + t.pnl
            strategy_trades[t.strategy] = strategy_trades.get(t.strategy, 0) + 1

        # Best and worst trades
        best_trade = max(today_trades, key=lambda t: t.pnl, default=None)
        worst_trade = min(today_trades, key=lambda t: t.pnl, default=None)

        # Market data
        vix = _get_vix()
        spy_change = _get_spy_change()
        btc_price = _get_price("BTC-USD")

        # Account state
        pnl_state = _read_json(self._runtime_dir / "pnl_state.json") or {}
        equity = _safe_float(pnl_state.get("account_equity", 0))

        # Live readiness
        readiness = _read_json(self._runtime_dir / "live_readiness.json") or {}

        # Week PnL
        week_pnl = sum(t.pnl for t in week_trades)

        # Win rate — prefer SCR state, fall back to trade history
        overall_win_rate = 0.0
        scr_data = _read_json(self._runtime_dir / "scr_state.json")
        if scr_data:
            scr_wr = scr_data.get("win_rate")
            if scr_wr is None:
                stats = scr_data.get("stats")
                if isinstance(stats, dict):
                    scr_wr = stats.get("win_rate")
            if scr_wr is not None:
                overall_win_rate = _safe_float(scr_wr) * 100  # stored as 0-1 fraction
        if overall_win_rate == 0.0:
            all_recent = week_trades if week_trades else today_trades
            overall_win_rate = (sum(1 for t in all_recent if t.pnl > 0) / len(all_recent) * 100) if all_recent else 0.0

        sections: List[str] = []

        # 1. HEADER
        weekday = now.strftime("%A")
        date_str = now.strftime("%B %d, %Y")
        sections.append(f"📊 CHAD's End of Day — {weekday} {date_str}")
        sections.append("")

        # 2. DID WE MAKE MONEY?
        sections.append("═══ DID WE MAKE MONEY? ═══")
        if total_trades == 0:
            sections.append("Quiet day — no trades today ⚪")
            day_of_week = now.weekday()
            if day_of_week >= 5:
                sections.append("Markets are closed on weekends — we'll be back Monday!")
            else:
                sections.append("No clear signals today, so we sat on the sidelines. Sometimes the smartest move is no move at all.")
        elif total_pnl > 0:
            sections.append(f"Yes — we made {_format_money(total_pnl)} on paper today 🟢")
            sections.append(f"That's like finding {_format_money(total_pnl)} in your pocket. Not real money yet, but the approach is working.")
        elif total_pnl < 0:
            sections.append(f"We lost {_format_money(abs(total_pnl))} on paper today 🔴")
            sections.append("Some days go the wrong way — that's normal in trading. Here's what happened.")
        else:
            sections.append("Broke even today — no gain, no loss ⚪")
            sections.append("We traded but came out flat. Think of it as a tie game.")
        sections.append("")

        # 3. WHAT DID WE DO?
        if total_trades > 0:
            sections.append("═══ WHAT DID WE DO? ═══")
            sections.append(f"{total_trades} trades today. {wins} were winners, {losses} were losers.")
            if wins + losses > 0:
                record = f"{wins}-{losses}"
                quality = "a winning" if wins > losses else ("a tough" if losses > wins else "an even")
                sections.append(f"That's like going {record} — {quality} day.")
            sections.append("")

        # 4. BEST MOVE TODAY
        if best_trade and best_trade.pnl > 0:
            sections.append("═══ BEST MOVE TODAY ═══")
            inst = translate_instrument(best_trade.symbol)
            action = "Bought" if best_trade.side in ("buy", "b", "long") else "Sold"
            sections.append(f"{action} {inst}")
            sections.append(f"Result: +{_format_money(best_trade.pnl)}")
            sections.append("")

        # 5. WORST MOVE TODAY
        if worst_trade and worst_trade.pnl < 0:
            sections.append("═══ WORST MOVE TODAY ═══")
            inst = translate_instrument(worst_trade.symbol)
            action = "Bought" if worst_trade.side in ("buy", "b", "long") else "Sold"
            sections.append(f"{action} {inst}")
            sections.append(f"Result: -{_format_money(abs(worst_trade.pnl))}")
            sections.append("This is normal — even the best traders lose some trades.")
            sections.append("")

        # 6. WHAT'S WORKING
        sections.append("═══ WHAT'S WORKING ═══")
        if total_trades == 0:
            sections.append("All 12 strategies are loaded and ready for Monday.")
        else:
            all_strategies = [
                "alpha", "beta", "gamma", "gamma_reversion",
                "alpha_futures", "gamma_futures",
                "omega", "omega_macro", "omega_vol",
                "alpha_options", "crypto", "alpha_crypto", "delta",
            ]
            seen = set()
            for strat in all_strategies:
                # Dedupe crypto/alpha_crypto
                label = translate_strategy(strat)
                if label in seen:
                    continue
                seen.add(label)

                pnl = strategy_pnl.get(strat, 0.0)
                count = strategy_trades.get(strat, 0)
                if count > 0:
                    if pnl > 0:
                        status = f"✅ Made {_format_money(pnl)}"
                    elif pnl < 0:
                        status = f"🔴 Lost {_format_money(abs(pnl))}"
                    else:
                        status = "⏸ Broke even"
                else:
                    status = "😴 No trades today"
                sections.append(f"  {label}: {status}")
        sections.append("")

        # 7. MARKET TEMPERATURE — only show when we have data
        if vix is not None or spy_change is not None or btc_price is not None:
            sections.append("═══ MARKET TEMPERATURE ═══")
            sections.append("(The \"fear gauge\" tells us how nervous investors are)")
            if vix is not None:
                sections.append(f"  Fear gauge (VIX): {vix:.1f} — {vix_description(vix)}")
            if spy_change is not None:
                direction = "up" if spy_change >= 0 else "down"
                sections.append(f"  Stock market was {direction} {abs(spy_change):.1f}% today")
            if btc_price is not None:
                sections.append(f"  Bitcoin: {_format_money(btc_price)}")
            # News headlines (Yahoo Finance — no API key needed)
            try:
                from chad.market_data.yahoo_news_provider import YahooNewsProvider
                _news = YahooNewsProvider().get_market_headlines(limit=3)
                if _news:
                    sections.append("")
                    sections.append("  \U0001f4f0 In the news today:")
                    for _h in _news:
                        sections.append(f"    \u2022 {_h.headline[:100]}")
            except Exception:
                pass
            sections.append("")

        # 8. CHAD STATUS
        sections.append("═══ CHAD STATUS ═══")
        ready = readiness.get("ready_for_live", False)
        sections.append("  Mode: Practice mode (not using real money yet)")
        if overall_win_rate > 0:
            above_below = "above" if overall_win_rate >= 55 else "below"
            emoji = " 🎯" if overall_win_rate >= 55 else ""
            sections.append(f"  Win rate: {overall_win_rate:.0f}% — {above_below} our 55% target{emoji}")
        else:
            sections.append("  Win rate: not enough data yet")
        if not ready:
            sections.append("  System is in practice mode — we're testing the strategies before using real money.")
        sections.append("")

        # 9. YOUR PAPER ACCOUNT
        sections.append("═══ YOUR PAPER ACCOUNT ═══")
        sections.append("(This is practice money — think of it like a score in a video game)")
        if equity > 0:
            sections.append(f"  Value: {_format_money(equity)}")
        else:
            sections.append("  Value: data not available")
        if total_pnl != 0:
            sign = "+" if total_pnl > 0 else ""
            sections.append(f"  Change today: {sign}{_format_money(total_pnl)}")
        else:
            sections.append("  Change today: $0")
        if week_pnl != 0:
            sign = "+" if week_pnl > 0 else ""
            sections.append(f"  Change this week: {sign}{_format_money(week_pnl)}")
        else:
            sections.append("  Change this week: $0")
        sections.append("")

        # CHAD's Take (AI-generated, best-effort — always present)
        sections.append("═══ CHAD'S TAKE ═══")
        if total_trades > 0:
            take = _generate_chads_take(
                total_pnl=total_pnl,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                vix=vix,
                strategy_results=strategy_pnl,
            )
        else:
            take = _generate_quiet_day_take()
        sections.append(take)
        sections.append("")

        # 10. SIGN OFF
        sections.append("See you tomorrow. Reply anytime if you have questions — just type naturally.")
        sections.append("— CHAD 🤝")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Morning Brief
# ---------------------------------------------------------------------------

class MorningBrief:
    """Generate a short pre-market morning brief."""

    def __init__(self, repo_root: Optional[Path] = None):
        self._root = repo_root or REPO_ROOT

    async def send_to_telegram(self) -> bool:
        """Generate the morning brief and send it via Telegram."""
        message = self.generate()
        _send_telegram(message)
        return True

    def generate(self) -> str:
        vix = _get_vix()
        spy_price = _get_price("SPY")
        btc_price = _get_price("BTC-USD")

        lines: List[str] = []
        lines.append("☀️ Good morning! Markets open in 30 minutes.")
        lines.append("")
        lines.append("Here's what I'm watching today:")

        if spy_price is not None:
            spy_change = _get_spy_change()
            if spy_change is not None:
                direction = "up" if spy_change >= 0 else "down"
                lines.append(f"- Stock market (SPY): {_format_money(spy_price)} — {direction} overnight")
            else:
                lines.append(f"- Stock market (SPY): {_format_money(spy_price)}")
        else:
            lines.append("- Stock market (SPY): waiting for data")

        if vix is not None:
            lines.append(f"- Fear gauge (VIX): {vix:.1f} — {vix_description(vix)}")
        else:
            lines.append("- Fear gauge (VIX): waiting for data")

        if btc_price is not None:
            lines.append(f"- Bitcoin: {_format_money(btc_price)}")
        else:
            lines.append("- Bitcoin: waiting for data")

        lines.append("")
        lines.append("I'm ready to trade. I'll send you a full report at the end of the day.")
        lines.append("— CHAD 🤝")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _send_telegram(message: str) -> None:
    """Best-effort Telegram delivery."""
    try:
        from chad.utils.telegram_notify import notify
        # Split long messages if needed (Telegram limit is 4096 chars)
        if len(message) <= 4096:
            notify(message, severity="info")
        else:
            # Split at section breaks
            chunks: List[str] = []
            current = ""
            for line in message.split("\n"):
                if len(current) + len(line) + 1 > 4000:
                    chunks.append(current)
                    current = line
                else:
                    current = current + "\n" + line if current else line
            if current:
                chunks.append(current)
            for chunk in chunks:
                notify(chunk.strip(), severity="info")
    except Exception as exc:
        logging.getLogger("chad.daily_report").warning("Telegram send failed: %s", exc)


def run_daily_report() -> str:
    """Generate and send the daily report."""
    report = DailyCHADReport()
    message = report.generate()

    # Save to reports dir
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"DAILY_CHAD_REPORT_{ts}.txt"
    out_path.write_text(message, encoding="utf-8")

    # Send via Telegram
    _send_telegram(message)

    print(message)
    return message


def run_morning_brief() -> str:
    """Generate and send the morning brief."""
    brief = MorningBrief()
    message = brief.generate()

    # Send via Telegram
    _send_telegram(message)

    print(message)
    return message


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "morning":
        run_morning_brief()
    else:
        run_daily_report()
