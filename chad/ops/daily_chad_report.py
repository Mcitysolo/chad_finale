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
DATA_FILLS = REPO_ROOT / "data" / "fills"
RUNTIME_DIR = REPO_ROOT / "runtime"
PNL_STATE_PATH = RUNTIME_DIR / "pnl_state.json"
LIVE_READINESS_PATH = RUNTIME_DIR / "live_readiness.json"
REPORTS_DIR = REPO_ROOT / "reports" / "ops"

TRADE_FILE_GLOB = "trade_history_*.ndjson"
FILLS_FILE_TEMPLATE = "FILLS_{ymd}.ndjson"

# Live regime classifier labels (from runtime/regime_state.json).
# Keep in sync with chad/analytics/regime_classifier.py.
REGIME_LABELS: Dict[str, str] = {
    "trending_bull":  "TRENDING (bullish)",
    "trending_bear":  "TRENDING (bearish)",
    "ranging":        "RANGING (mean-reversion favored)",
    "volatile":       "VOLATILE (defensive posture)",
    "unknown":        "UNKNOWN",
    "adverse":        "ADVERSE (minimal exposure)",
}

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
    "beta_trend": "Legend-driven ETF allocation (long-term trend follower)",
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
# Intent loading from data/fills/ (open-intent records, no closed PnL yet)
#
# Live loop fills are written to data/fills/FILLS_YYYYMMDD.ndjson by
# chad/execution/paper_exec_evidence_writer.py. They represent submitted/
# executed paper-trade intents but do not contain closed PnL (positions
# have not been matched/closed). Use these to surface "what we did today"
# in the daily report; PnL still requires the closed-trade ledger or a
# future trade closer.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentRow:
    strategy: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    ts_utc: Optional[str]


def _normalize_strategy_field(value: Any) -> str:
    """Strategy may arrive as a string or a list (e.g. source_strategies)."""
    if value is None:
        return ""
    if isinstance(value, list):
        for v in value:
            if v:
                return str(v).strip().lower()
        return ""
    return str(value).strip().lower()


def _extract_intent(obj: Dict[str, Any]) -> Optional[IntentRow]:
    """
    Extract a single fill record into an IntentRow. Skips obvious non-trades
    (live records, rejected fills, missing symbol). Mirrors the defensive
    style of _extract_row.
    """
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        payload = obj

    if payload.get("is_live") is True:
        return None
    if payload.get("reject") is True:
        return None

    symbol = str(payload.get("symbol") or "").strip().upper()
    if not symbol:
        return None

    strategy = _normalize_strategy_field(payload.get("strategy"))
    if not strategy:
        strategy = _normalize_strategy_field(payload.get("source_strategies"))
    if not strategy:
        strategy = "unknown"

    side = str(payload.get("side") or "").strip().lower() or "unknown"

    try:
        qty = float(payload.get("quantity") or 0.0)
        if not math.isfinite(qty):
            qty = 0.0
    except (TypeError, ValueError):
        qty = 0.0

    try:
        price = float(payload.get("fill_price") or 0.0)
        if not math.isfinite(price):
            price = 0.0
    except (TypeError, ValueError):
        price = 0.0

    ts = obj.get("timestamp_utc") or payload.get("fill_time_utc") or payload.get("entry_time_utc")
    return IntentRow(
        strategy=strategy,
        symbol=symbol,
        side=side,
        quantity=qty,
        fill_price=price,
        ts_utc=str(ts) if ts else None,
    )


def _count_registered_strategies() -> int:
    """Count strategies dynamically; fall back to a sensible default."""
    try:
        from chad.strategies import iter_strategy_registrations
        return len(list(iter_strategy_registrations()))
    except Exception:
        return 13


def _load_open_positions(runtime_dir: Path) -> List[Dict[str, Any]]:
    """Read open paper positions from runtime/position_guard.json."""
    try:
        data = _read_json(runtime_dir / "position_guard.json")
        if not isinstance(data, dict):
            return []
        out: List[Dict[str, Any]] = []
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            if val.get("open") is True:
                out.append(val)
        return out
    except Exception:
        return []


def load_today_intents(fills_dir: Optional[Path] = None) -> List[IntentRow]:
    """
    Load today's executed paper-trade intents from
    data/fills/FILLS_YYYYMMDD.ndjson. Returns an empty list when the file
    is missing or unreadable. Never raises.
    """
    root = fills_dir or DATA_FILLS
    today = _today_yyyymmdd()
    ledger = root / FILLS_FILE_TEMPLATE.format(ymd=today)

    if not ledger.exists():
        return []

    rows: List[IntentRow] = []
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
            row = _extract_intent(obj)
            if row is not None:
                rows.append(row)
    except Exception:
        pass
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

    # 3. Fallback: most recent close from daily bar history
    bar_data = _read_json(REPO_ROOT / "data" / "bars" / "1d" / "VIX.json")
    if bar_data:
        bars = bar_data.get("bars")
        if isinstance(bars, list) and bars:
            last_bar = bars[-1]
            close = last_bar.get("close") or last_bar.get("c")
            if close is not None:
                return _safe_float(close)

    return None


def _get_price(symbol: str) -> Optional[float]:
    cache = _load_price_cache()
    for source in (cache, cache.get("prices", {})):
        val = source.get(symbol)
        if val is not None:
            if isinstance(val, (int, float)):
                return _safe_float(val)
            if isinstance(val, dict):
                p = val.get("price") or val.get("last") or val.get("close")
                if p is not None:
                    return _safe_float(p)
    return None


def _get_spy_from_bars() -> Optional[float]:
    """Fallback: read most recent close from daily bar history."""
    bar_file = REPO_ROOT / "data" / "bars" / "1d" / "SPY.json"
    data = _read_json(bar_file)
    if not data:
        return None
    bars = data.get("bars")
    if not isinstance(bars, list) or not bars:
        return None
    last_bar = bars[-1]
    close = last_bar.get("close")
    return _safe_float(close) if close is not None else None


def _get_btc_from_kraken() -> Optional[float]:
    """Read BTC-USD price from runtime/kraken_prices.json."""
    data = _read_json(RUNTIME_DIR / "kraken_prices.json")
    if not data:
        return None
    prices = data.get("prices", {})
    val = prices.get("BTC-USD")
    if val is not None:
        return _safe_float(val)
    ticks = data.get("ticks", {})
    tick = ticks.get("BTC-USD")
    if isinstance(tick, dict):
        p = tick.get("last") or tick.get("price")
        if p is not None:
            return _safe_float(p)
    return None


def _get_spy_change() -> Optional[float]:
    cache = _load_price_cache()
    for source in (cache, cache.get("prices", {})):
        val = source.get("SPY")
        if isinstance(val, dict):
            chg = val.get("change_pct") or val.get("pct_change")
            if chg is not None:
                return _safe_float(chg)
    return None


# ---------------------------------------------------------------------------
# Intelligence feed helpers (trends / reddit / short interest)
#
# Used by both the morning brief and the end-of-day report. Each feed is
# optional — a missing or stale file results in an empty highlight list, not
# an error. Freshness is determined by _is_feed_fresh(); default is 6 hours.
# ---------------------------------------------------------------------------

def _is_feed_fresh(path: Path, max_age_hours: float = 6.0) -> bool:
    """Return True if a runtime feed exists and is younger than max_age_hours."""
    try:
        if not path.is_file():
            return False
        age_s = (_utc_now() - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).total_seconds()
        return age_s < max_age_hours * 3600.0
    except Exception:
        return False


def _gather_intelligence_highlights(runtime_dir: Path) -> List[str]:
    """
    Pull 0-3 concrete highlights from trends/reddit/short-interest feeds.
    Returns [] when every feed is stale, missing, or unremarkable — never
    raises. Used by both morning brief and end-of-day take.
    """
    highlights: List[str] = []

    # Google Trends: rising/high search interest
    trends_path = runtime_dir / "trends_state.json"
    if _is_feed_fresh(trends_path):
        try:
            trends = _read_json(trends_path) or {}
            for sym, v in (trends.get("signals") or {}).items():
                if not isinstance(v, dict):
                    continue
                ratio = _safe_float(v.get("ratio"))
                sig = str(v.get("signal", "")).upper()
                if sig in ("RISING", "HIGH") or ratio >= 1.3:
                    highlights.append(
                        f"{sym} search interest surging ({ratio:.2f}x avg) — retail attention building"
                    )
                if len(highlights) >= 2:
                    break
        except Exception:
            pass

    # Reddit: heavy discussion or directional sentiment
    reddit_path = runtime_dir / "reddit_sentiment.json"
    if _is_feed_fresh(reddit_path):
        try:
            reddit = _read_json(reddit_path) or {}
            for sym, v in (reddit.get("signals") or {}).items():
                if not isinstance(v, dict):
                    continue
                mentions = int(_safe_float(v.get("mention_count")))
                score = _safe_float(v.get("sentiment_score"))
                sig = str(v.get("signal", "")).upper()
                if mentions >= 20 or sig in ("BULLISH", "BEARISH", "HYPE"):
                    tone = "bullish" if score > 0 else ("bearish" if score < 0 else "active")
                    highlights.append(
                        f"{sym} heavy Reddit discussion ({mentions} mentions, {tone}) — momentum catalyst"
                    )
                if len(highlights) >= 3:
                    break
        except Exception:
            pass

    # Short interest: squeeze candidates
    shorts_path = runtime_dir / "short_interest.json"
    if _is_feed_fresh(shorts_path):
        try:
            shorts = _read_json(shorts_path) or {}
            for sym, v in (shorts.get("signals") or {}).items():
                if not isinstance(v, dict):
                    continue
                sf = _safe_float(v.get("short_float_pct"))
                if v.get("squeeze_risk") or sf >= 0.15:
                    highlights.append(
                        f"{sym} short float at {sf*100:.0f}% — squeeze risk on upside break"
                    )
                if len(highlights) >= 3:
                    break
        except Exception:
            pass

    return highlights[:3]


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
    *,
    regime: Optional[str] = None,
    scr_state: Optional[str] = None,
    scr_sizing: Optional[float] = None,
    open_positions: Optional[int] = None,
    top_strategy: Optional[str] = None,
    intelligence_highlights: Optional[List[str]] = None,
) -> str:
    """Generate a 2-3 sentence elite-prodigy take for the end-of-day report."""
    try:
        from chad.intel.claude_client import ClaudeClient

        client = ClaudeClient.load()

        strat_summary = ", ".join(
            f"{translate_strategy(s)}: {'made' if pnl > 0 else 'lost'} {_format_money(abs(pnl))}"
            for s, pnl in sorted(strategy_results.items())
        )

        extras: List[str] = []
        if regime:
            extras.append(f"Regime: {regime}")
        if scr_state:
            sz = f" (sizing {scr_sizing:.2f}x)" if isinstance(scr_sizing, (int, float)) else ""
            extras.append(f"SCR: {scr_state}{sz}")
        if top_strategy:
            extras.append(f"Top strategy today: {top_strategy}")
        if open_positions is not None:
            extras.append(f"Open positions: {open_positions}")
        if intelligence_highlights:
            extras.append(
                "Intelligence highlights: " + "; ".join(intelligence_highlights[:3])
            )

        prompt = (
            f"Today's results: {'made' if total_pnl >= 0 else 'lost'} {_format_money(abs(total_pnl))} total. "
            f"{total_trades} trades, {wins} winners, {losses} losers, "
            f"{max(0, total_trades - wins - losses)} scratches. "
            f"VIX (fear gauge): {vix if vix else 'unknown'}. "
            f"Strategy results: {strat_summary}. "
            + (" ".join(extras) + ". " if extras else "")
            + "Write 2-3 sentences assessing today's trading. Reference the specific "
            "P&L, positions, and market conditions above. Plain English only — no "
            "jargon. Be honest about losses. Be direct."
        )

        text, _, _ = client._call_claude(
            prompt=prompt,
            system=(
                "You are CHAD — an elite autonomous trading system. You think with "
                "the quantitative precision of Jim Simons, the macro vision of Ray "
                "Dalio, and the opportunistic instincts of Stanley Druckenmiller. "
                "You are speaking to a non-technical operator who does not need "
                "jargon — use plain English only. Analyze today's trading results "
                "and give a sharp, specific 2-3 sentence assessment. Reference "
                "actual P&L, positions, and conditions. Be honest about losses. "
                "Be direct. No generic advice."
            ),
            task_type="standard",
            max_tokens=260,
            temperature=0.6,
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
            "Everything is loaded up and ready to go when markets reopen."
        )


# ---------------------------------------------------------------------------
# DailyCHADReport
# ---------------------------------------------------------------------------

class DailyCHADReport:
    """Generate a plain-English daily report for non-traders."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        trades_dir: Optional[Path] = None,
        fills_dir: Optional[Path] = None,
    ):
        self._root = repo_root or REPO_ROOT
        self._trades_dir = trades_dir or (self._root / "data" / "trades")
        self._fills_dir = fills_dir or (self._root / "data" / "fills")
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
        today_intents = load_today_intents(self._fills_dir)
        open_positions = _load_open_positions(self._runtime_dir)
        strategy_count = _count_registered_strategies()

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
        btc_price = _get_price("BTC-USD") or _get_btc_from_kraken()

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
            if today_intents:
                # Submitted trades exist but nothing has closed yet — be honest
                # about $0 closed P&L without claiming "no trades today".
                sections.append("$0 in closed profit/loss today ⚪")
                sections.append(
                    "We submitted trades today but no positions have closed yet — "
                    "open positions don't count as profit or loss until they're closed out. "
                    "See WHAT DID WE DO TODAY below."
                )
            elif open_positions:
                n_open = len(open_positions)
                pos_word = "position" if n_open == 1 else "positions"
                sections.append(f"$0 in closed profit/loss today ⚪")
                sections.append(
                    f"{n_open} {pos_word} still open — results pending when CHAD closes them. "
                    f"Open positions don't count as profit or loss until they're closed out."
                )
            else:
                sections.append("Quiet day — no trades today ⚪")
                day_of_week = now.weekday()
                if day_of_week >= 5:
                    sections.append("Markets are closed on weekends — we'll be back when they reopen!")
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

        # 3. WHAT DID WE DO?  (closed-trade P&L view; rare under live-loop today)
        if total_trades > 0:
            sections.append("═══ WHAT DID WE DO? ═══")
            scratches = max(0, total_trades - wins - losses)
            sections.append(f"{total_trades} trades today. {wins} wins, {losses} losses, {scratches} scratches (flat).")
            if wins + losses > 0:
                record = f"{wins}-{losses}"
                if total_pnl > 0:
                    quality = "a winning" if wins > losses else "a grinding"
                    sections.append(f"That's like going {record} — {quality} day.")
                elif total_pnl < 0 and wins > losses:
                    sections.append(f"Record of {record}: we had more wins than losses, but the losses were bigger — net red.")
                elif total_pnl < 0:
                    sections.append(f"Record of {record} — a tough day.")
                else:
                    sections.append(f"Record of {record} — an even day.")
            sections.append("")

        # 3b. WHAT DID WE DO TODAY  (live-loop submitted intents from data/fills/)
        if today_intents:
            n_intents = len(today_intents)
            uniq_symbols = sorted({r.symbol for r in today_intents})
            strat_counts: Dict[str, int] = {}
            for r in today_intents:
                strat_counts[r.strategy] = strat_counts.get(r.strategy, 0) + 1
            top_strategy, top_count = max(strat_counts.items(), key=lambda kv: kv[1])

            display_symbols = uniq_symbols[:8]
            sym_text = ", ".join(display_symbols)
            if len(uniq_symbols) > 8:
                sym_text += f" (+{len(uniq_symbols) - 8} more)"

            trade_word = "trade" if n_intents == 1 else "trades"
            top_word = "trade" if top_count == 1 else "trades"

            sections.append("═══ WHAT DID WE DO TODAY ═══")
            sections.append(f"We submitted {n_intents} {trade_word} across {sym_text} today.")
            sections.append(
                f"Most active: {translate_strategy(top_strategy)} with {top_count} {top_word}."
            )
            sections.append(
                "Note: These are practice trades — real profit/loss shows once positions close."
            )
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
            if open_positions:
                uniq_pos_symbols = sorted({
                    str(p.get("symbol", "")).upper() for p in open_positions if p.get("symbol")
                })
                preview = ", ".join(uniq_pos_symbols[:8])
                if len(uniq_pos_symbols) > 8:
                    preview += f" (+{len(uniq_pos_symbols) - 8} more)"
                n_open = len(open_positions)
                pos_word = "position" if n_open == 1 else "positions"
                sections.append(
                    f"CHAD has {n_open} open {pos_word} right now ({preview})."
                )
                if today_intents:
                    n_intents = len(today_intents)
                    trade_word = "trade" if n_intents == 1 else "trades"
                    sections.append(f"{n_intents} {trade_word} executed today.")
                sections.append(
                    f"{strategy_count} strategies loaded and monitoring markets."
                )
            else:
                sections.append(
                    f"All {strategy_count} strategies are loaded and ready."
                )
        else:
            all_strategies = [
                "alpha", "beta", "beta_trend", "gamma", "gamma_reversion",
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
            # Market signals (Reddit sentiment + short interest — best-effort)
            _market_signal_lines: List[str] = []
            try:
                _reddit = _read_json(RUNTIME_DIR / "reddit_sentiment.json")
                _reddit_sigs = _reddit.get("signals", {})
                _notable_reddit = [
                    (sym, s) for sym, s in _reddit_sigs.items()
                    if isinstance(s, dict) and s.get("signal") in ("HYPE", "BULLISH", "BEARISH")
                ]
                if _notable_reddit:
                    _top_reddit = sorted(_notable_reddit,
                                         key=lambda x: x[1].get("mention_count", 0),
                                         reverse=True)[:2]
                    _sent_parts = []
                    for _sym, _s in _top_reddit:
                        _label = _s.get("signal", "NEUTRAL")
                        _mc = _s.get("mention_count", 0)
                        _sent_parts.append(f"{_sym} {_label.lower()} ({_mc} mentions)")
                    if _sent_parts:
                        _market_signal_lines.append(f"  \u2022 Sentiment: {', '.join(_sent_parts)}")
            except Exception:
                pass
            try:
                _short = _read_json(RUNTIME_DIR / "short_interest.json")
                _short_sigs = _short.get("signals", {})
                _squeeze_watch = [
                    (sym, s) for sym, s in _short_sigs.items()
                    if isinstance(s, dict) and (s.get("squeeze_risk") or s.get("signal") == "EXTREME")
                ]
                if _squeeze_watch:
                    _sq_parts = []
                    for _sym, _s in _squeeze_watch[:2]:
                        _pct = _s.get("short_float_pct", 0)
                        _sq_parts.append(f"{_sym} {_pct:.0%} short float")
                    if _sq_parts:
                        _market_signal_lines.append(f"  \u2022 Short squeeze watch: {', '.join(_sq_parts)}")
            except Exception:
                pass
            if _market_signal_lines:
                sections.append("")
                sections.append("  \U0001f4e1 Market signals:")
                sections.extend(_market_signal_lines)
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
        elif open_positions:
            n_open = len(open_positions)
            pos_word = "position" if n_open == 1 else "positions"
            sections.append(
                f"  Change today: $0 closed "
                f"({n_open} {pos_word} open — results pending when CHAD closes them)"
            )
        else:
            sections.append("  Change today: $0")
        if week_pnl != 0:
            sign = "+" if week_pnl > 0 else ""
            sections.append(f"  Change this week: {sign}{_format_money(week_pnl)}")
        else:
            sections.append("  Change this week: $0")
        sections.append("")

        # 9b. YOUR REAL KRAKEN CRYPTO ACCOUNT
        # Read the snapshot the orchestrator refreshes every 5 minutes.
        kraken_snap = _read_json(RUNTIME_DIR / "kraken_balances.json") or {}
        if isinstance(kraken_snap, dict) and kraken_snap.get("ok"):
            sections.append("═══ YOUR REAL KRAKEN ACCOUNT ═══")
            sections.append("(This is real money on Kraken — read-only snapshot)")
            usd_eq = float(kraken_snap.get("usd_equivalent") or 0.0)
            if usd_eq > 0:
                sections.append(f"  Total value (USD eq): {_format_money(usd_eq)}")
            balances = kraken_snap.get("balances") or {}
            if isinstance(balances, dict):
                for asset, qty in sorted(balances.items()):
                    try:
                        sections.append(f"  {asset}: {float(qty):.6f}")
                    except (TypeError, ValueError):
                        continue
            ts = kraken_snap.get("ts_utc")
            if ts:
                sections.append(f"  Snapshot taken: {ts}")
            sections.append("")

        # CHAD's Take (AI-generated, best-effort — always present)
        sections.append("═══ CHAD'S TAKE ═══")
        if total_trades > 0:
            # Pull the richer context the elite-prodigy prompt can anchor to.
            regime_label_eod = "UNKNOWN"
            try:
                regime_eod = _read_json(self._runtime_dir / "regime_state.json") or {}
                live_regime_eod = str(regime_eod.get("regime", "unknown")).lower()
                regime_label_eod = REGIME_LABELS.get(live_regime_eod, live_regime_eod.upper())
            except Exception:
                pass
            scr_state_eod = str(scr_data.get("state", "UNKNOWN")) if scr_data else "UNKNOWN"
            scr_sizing_eod_raw = scr_data.get("sizing_factor") if scr_data else None
            scr_sizing_eod = (
                float(scr_sizing_eod_raw)
                if isinstance(scr_sizing_eod_raw, (int, float))
                else None
            )
            top_strategy_eod: Optional[str] = None
            if strategy_pnl:
                _top_sym, _top_val = max(
                    strategy_pnl.items(), key=lambda kv: kv[1]
                )
                top_strategy_eod = (
                    f"{translate_strategy(_top_sym)} "
                    f"({'+' if _top_val >= 0 else ''}{_format_money(_top_val)})"
                )

            highlights_eod = _gather_intelligence_highlights(self._runtime_dir)

            take = _generate_chads_take(
                total_pnl=total_pnl,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                vix=vix,
                strategy_results=strategy_pnl,
                regime=regime_label_eod,
                scr_state=scr_state_eod,
                scr_sizing=scr_sizing_eod,
                open_positions=len(open_positions),
                top_strategy=top_strategy_eod,
                intelligence_highlights=highlights_eod,
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

    def _load_fresh_feed(self, path: Path, max_age_hours: float = 6.0) -> Optional[Dict[str, Any]]:
        """Load a JSON feed; return None if missing or stale (> max_age_hours)."""
        try:
            if not path.exists():
                return None
            data = _read_json(path)
            if not isinstance(data, dict):
                return None
            ts_str = data.get("last_updated_utc") or data.get("ts_utc")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    age_h = (_utc_now() - ts).total_seconds() / 3600.0
                    if age_h > max_age_hours:
                        return None
                except Exception:
                    pass
            return data
        except Exception:
            return None

    def _gather_watch_items(self) -> List[str]:
        """Pull 0-3 concrete watch items from intelligence feeds."""
        return _gather_intelligence_highlights(self._root / "runtime")

    def _chads_take(self, context: Dict[str, Any]) -> Optional[str]:
        """Elite prodigy take — Simons/Dalio/Druckenmiller voice (morning brief)."""
        try:
            from chad.intel.claude_client import ClaudeClient
            client = ClaudeClient.load()
            result = client.chat_json(
                (
                    "Pre-market context:\n"
                    + json.dumps(context, default=str)[:3000]
                    + "\n\nReturn JSON {\"text\": \"2-3 sentence pre-market take\"}."
                ),
                system=(
                    "You are CHAD — an elite autonomous trading system. You think with "
                    "the quantitative precision of Jim Simons, the macro vision of Ray "
                    "Dalio, and the opportunistic instincts of Stanley Druckenmiller. "
                    "You are speaking to a non-technical operator who does not need "
                    "jargon — use plain English only. Analyze the pre-market conditions "
                    "provided and give a sharp, specific 2-3 sentence assessment. "
                    "Reference actual data. Be confident and direct. No generic "
                    "statements like 'stay disciplined' or 'markets can be volatile'. "
                    "Wrap your response in a JSON object with a single key 'text'."
                ),
                task_type="standard",
            )
            text = str(result.get("text", "")).strip()
            return text or None
        except Exception as exc:
            logging.getLogger("chad.morning_brief").debug("CHAD's take failed: %s", exc)
            return None

    def _scan_opportunities(self) -> Optional[str]:
        """Legacy opportunity scan — kept for backward compatibility."""
        try:
            runtime = self._root / "runtime"
            price_cache = _read_json(runtime / "price_cache.json") or {}
            trends_state = _read_json(runtime / "trends_state.json") or {}
            kraken_prices = _read_json(runtime / "kraken_prices.json") or {}

            reddit_sentiment = _read_json(runtime / "reddit_sentiment.json") or {}
            short_interest = _read_json(runtime / "short_interest.json") or {}

            # Build context snapshot
            context_parts = []
            if price_cache:
                context_parts.append(f"Price cache (overnight): {json.dumps(price_cache, default=str)[:2000]}")
            trends_signals = trends_state.get("signals", {})
            if trends_signals:
                context_parts.append(f"Google Trends signals: {json.dumps(trends_signals, default=str)[:1000]}")
            if kraken_prices:
                # Convert crypto volume to USD for clearer labeling
                kraken_display = {}
                kp = kraken_prices.get("prices", {})
                kt = kraken_prices.get("ticks", {})
                for sym in ("BTC-USD", "ETH-USD", "SOL-USD"):
                    price = kp.get(sym)
                    tick = kt.get(sym, {})
                    if price:
                        vol_base = tick.get("volume_24h", 0)
                        kraken_display[sym] = {
                            "price": price,
                            "volume_24h_usd": round(vol_base * price, 0) if vol_base else None,
                        }
                if kraken_display:
                    context_parts.append(f"Crypto prices (Kraken, volume is in USD): {json.dumps(kraken_display, default=str)[:1000]}")
            reddit_signals = reddit_sentiment.get("signals", {})
            if reddit_signals:
                context_parts.append(f"Reddit sentiment: {json.dumps(reddit_signals, default=str)[:1000]}")
            short_signals = short_interest.get("signals", {})
            if short_signals:
                context_parts.append(f"Short interest: {json.dumps(short_signals, default=str)[:1000]}")

            if not context_parts:
                return None

            from chad.intel.claude_client import ClaudeClient
            client = ClaudeClient.load()

            prompt = (
                "You are CHAD. Based on this overnight data, identify 2-3 instruments "
                "worth watching today and why. Be specific and direct. Plain English. "
                "No jargon. Each point max 1 sentence. "
                "For crypto (BTC, ETH, SOL), volume is in USD — label it as USD, not contracts.\n\n"
                + "\n\n".join(context_parts)
                + "\n\nReturn a JSON object: {\"text\": \"your bullet points here\"}"
            )

            result = client.chat_json(
                prompt,
                system=(
                    "You are CHAD, a trading system's intelligence layer. "
                    "Return 2-3 bullet points, each starting with the instrument symbol. "
                    "Be specific about price levels and percentages. No markdown. "
                    "Format: SYMBOL: observation. "
                    "Wrap your response in a JSON object with a single key 'text'."
                ),
                task_type="routine",
            )
            text = str(result.get("text", "") or result.get("observations", "") or "").strip()
            return text if text else None
        except Exception as exc:
            logging.getLogger("chad.morning_brief").debug("Opportunity scan failed: %s", exc)
            return None

    def generate(self) -> str:
        runtime = self._root / "runtime"
        now = _utc_now()
        vix = _get_vix()
        spy_price = _get_price("SPY") or _get_spy_from_bars()
        spy_change = _get_spy_change()
        btc_price = _get_price("BTC-USD") or _get_btc_from_kraken()

        # Crypto 24h change for context
        btc_chg_pct: Optional[float] = None
        try:
            kp = _read_json(runtime / "kraken_prices.json") or {}
            tick = (kp.get("ticks") or {}).get("BTC-USD") or {}
            btc_chg_pct = _safe_float(tick.get("change_24h_pct")) if tick.get("change_24h_pct") is not None else None
        except Exception:
            pass

        strat_intel = self._load_fresh_feed(runtime / "strategy_intelligence.json", max_age_hours=48.0) or {}
        expectancy = _read_json(runtime / "expectancy_state.json") or {}

        lines: List[str] = []
        day_date = now.strftime("%A, %B %d")
        lines.append(f"\U0001f305 CHAD Pre-Market Brief \u2014 {day_date}")
        lines.append("")

        # ═══ OVERNIGHT ═══
        lines.append("\u2550\u2550\u2550 OVERNIGHT \u2550\u2550\u2550")
        if vix is not None:
            lines.append(f"Fear gauge (VIX): {vix:.1f} \u2014 {vix_description(vix)}")
        if spy_price is not None:
            if spy_change is not None:
                direction = "up" if spy_change >= 0 else "down"
                lines.append(f"SPY: {_format_money(spy_price)} ({direction} {abs(spy_change):.2f}% overnight)")
            else:
                lines.append(f"SPY: {_format_money(spy_price)}")
        if btc_price is not None:
            if btc_chg_pct is not None:
                lines.append(f"BTC: {_format_money(btc_price)} ({btc_chg_pct:+.2f}% 24h)")
            else:
                lines.append(f"BTC: {_format_money(btc_price)}")
        lines.append("")

        # ═══ WHAT CHAD IS WATCHING ═══
        lines.append("\u2550\u2550\u2550 WHAT CHAD IS WATCHING \u2550\u2550\u2550")
        watch = self._gather_watch_items()
        if watch:
            for w in watch:
                lines.append(f"\u2022 {w}")
        else:
            lines.append("\u2022 No elevated signals across trends/reddit/shorts — baseline watch only")
        lines.append("")

        # ═══ TODAY'S STRATEGY POSTURE ═══
        lines.append("\u2550\u2550\u2550 TODAY'S STRATEGY POSTURE \u2550\u2550\u2550")
        # Regime read from the LIVE classifier (runtime/regime_state.json),
        # not the 48-hour AI cache in strategy_intelligence.json which was
        # collapsing every regime to NEUTRAL.
        try:
            regime_data = _read_json(runtime / "regime_state.json") or {}
            live_regime = str(regime_data.get("regime", "unknown")).lower()
            regime_summary = REGIME_LABELS.get(live_regime, live_regime.upper())
        except Exception:
            regime_summary = "UNKNOWN"
        lines.append(f"Regime: {regime_summary}")

        biases = strat_intel.get("confidence_bias") or []
        notable_biases: List[str] = []
        if isinstance(biases, list):
            for b in biases:
                if not isinstance(b, dict):
                    continue
                adj = _safe_float(b.get("adjustment"))
                if abs(adj) >= 0.05:
                    sym = b.get("symbol", "?")
                    strat = translate_strategy(str(b.get("strategy", "")))
                    tone = "cautious" if adj < 0 else "confident"
                    notable_biases.append(f"{strat} {tone} on {sym} ({adj:+.2f})")
                if len(notable_biases) >= 3:
                    break
        if notable_biases:
            for nb in notable_biases:
                lines.append(f"\u2022 {nb}")
        else:
            lines.append("\u2022 Confidence biases neutral across strategies")
        lines.append("")

        # ═══ CHAD'S PERFORMANCE ═══
        lines.append("\u2550\u2550\u2550 CHAD'S PERFORMANCE \u2550\u2550\u2550")
        scr = _read_json(runtime / "scr_state.json") or {}
        scr_state = scr.get("state", "UNKNOWN")
        scr_sizing = scr.get("sizing_factor")
        if isinstance(scr_sizing, (int, float)):
            lines.append(f"SCR: {scr_state} (sizing {scr_sizing:.2f}x)")
        else:
            lines.append(f"SCR: {scr_state}")

        top = expectancy.get("top_performer")
        strats = expectancy.get("strategies") or {}
        top_strategy_summary: Optional[str] = None
        if top and isinstance(strats, dict) and top in strats:
            s = strats[top]
            top_strategy_summary = (
                f"Top strategy: {translate_strategy(str(top))} \u2014 "
                f"win rate {_safe_float(s.get('win_rate'))*100:.0f}%, "
                f"expectancy {_format_money(_safe_float(s.get('expectancy')))}"
            )
            lines.append(top_strategy_summary)
        # Trade count clarity: BOTH fills-today (raw activity) AND SCR
        # effective_trades (the number that actually gates SCR progress).
        # Previously the brief only surfaced one of these and the operator
        # could not tell raw session activity from real SCR progress.
        fills_today = load_today_intents(self._root / "data" / "fills")
        fills_today_n = len(fills_today)
        scr_stats = scr.get("stats") if isinstance(scr.get("stats"), dict) else {}
        effective_trades = int(_safe_float(scr_stats.get("effective_trades", 0)))
        lines.append(
            f"Fills today: {fills_today_n} | "
            f"Clean trades toward next level: {effective_trades}/100"
        )

        # Open positions
        open_count = 0
        try:
            pg_path = runtime / "positions_snapshot.json"
            if pg_path.exists():
                snap = json.loads(pg_path.read_text(encoding="utf-8"))
                positions = snap.get("positions", [])
                if isinstance(positions, list):
                    open_count = sum(
                        1 for p in positions
                        if isinstance(p, dict) and abs(_safe_float(p.get("position", 0))) > 0
                    )
                    if open_count > 0:
                        lines.append(f"Open positions: {open_count}")
        except Exception:
            pass
        lines.append("")

        # ═══ CHAD'S TAKE ═══
        lines.append("\u2550\u2550\u2550 CHAD'S TAKE \u2550\u2550\u2550")
        take_ctx: Dict[str, Any] = {
            "vix": vix,
            "spy_price": spy_price,
            "spy_change_pct": spy_change,
            "btc_price": btc_price,
            "btc_change_24h_pct": btc_chg_pct,
            "regime": regime_summary,
            "watch_items": watch,
            "notable_biases": notable_biases,
            "top_performer": top,
            "top_strategy_summary": top_strategy_summary,
            "scr_state": scr_state,
            "scr_sizing": scr_sizing,
            "fills_today": fills_today_n,
            "effective_trades": effective_trades,
            "open_positions": open_count,
        }
        take = self._chads_take(take_ctx)
        if take:
            lines.append(take)
        else:
            lines.append(
                "Feeds are thin this morning — trading the mechanical edge, not the narrative. "
                "Position sizing stays disciplined until the tape gives us a reason to press."
            )
        lines.append("")
        lines.append("\u2014 CHAD \U0001f91d")

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


# ---------------------------------------------------------------------------
# Weekly Business Summary (Sundays 20:00 UTC)
# ---------------------------------------------------------------------------

class WeeklySummary:
    """
    Generate a Sunday-evening week-in-review for the non-technical operator.

    Pulls fills from the last 7 days directly from data/fills/, sums realized
    P&L where present, and includes SCR state, regime, beta allocation, and
    a Claude-generated elite-prodigy "take" summarizing the week.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._root = repo_root or REPO_ROOT
        self._fills_dir = self._root / "data" / "fills"
        self._runtime_dir = self._root / "runtime"

    def _load_weekly_fills(self) -> List[IntentRow]:
        if not self._fills_dir.exists():
            return []
        cutoff = _utc_now().date() - timedelta(days=7)
        rows: List[IntentRow] = []
        for f in sorted(self._fills_dir.glob("FILLS_*.ndjson")):
            try:
                ymd = f.stem.split("_", 1)[1]
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
                    row = _extract_intent(obj)
                    if row is not None:
                        rows.append(row)
            except Exception:
                continue
        return rows

    def _load_weekly_realized_pnl(self) -> float:
        """Sum realized_pnl from trade_history ledger files in the last 7 days."""
        trades = load_week_trades(self._root / "data" / "trades")
        return sum(t.pnl for t in trades)

    def _weekly_take(self, context: Dict[str, Any]) -> Optional[str]:
        """Claude Haiku elite-prodigy take summarizing the week."""
        try:
            from chad.intel.claude_client import ClaudeClient
            client = ClaudeClient.load()
            result = client.chat_json(
                (
                    "Week-in-review context:\n"
                    + json.dumps(context, default=str)[:3000]
                    + "\n\nReturn JSON {\"text\": \"2-3 sentence weekly take\"}."
                ),
                system=(
                    "You are CHAD — an elite autonomous trading system. You think with "
                    "the quantitative precision of Jim Simons, the macro vision of Ray "
                    "Dalio, and the opportunistic instincts of Stanley Druckenmiller. "
                    "You are speaking to a non-technical operator who does not need "
                    "jargon — use plain English only. Analyze this past week's trading "
                    "results and give a sharp, specific 2-3 sentence assessment. "
                    "Reference actual P&L, regime, and top-performing strategy. "
                    "Be honest about losses. Be direct. No generic advice. "
                    "Wrap your response in a JSON object with a single key 'text'."
                ),
                task_type="standard",
            )
            text = str(result.get("text", "")).strip()
            return text or None
        except Exception:
            return None

    def generate(self) -> str:
        now = _utc_now()
        weekly_fills = self._load_weekly_fills()
        weekly_fill_count = len(weekly_fills)
        weekly_realized_pnl = self._load_weekly_realized_pnl()

        scr = _read_json(self._runtime_dir / "scr_state.json") or {}
        scr_state = str(scr.get("state", "UNKNOWN"))
        scr_sizing = scr.get("sizing_factor")
        scr_stats = scr.get("stats") if isinstance(scr.get("stats"), dict) else {}
        effective_trades = int(_safe_float(scr_stats.get("effective_trades", 0)))

        # Regime (live)
        try:
            regime_data = _read_json(self._runtime_dir / "regime_state.json") or {}
            live_regime = str(regime_data.get("regime", "unknown")).lower()
            regime_label = REGIME_LABELS.get(live_regime, live_regime.upper())
        except Exception:
            regime_label = "UNKNOWN"

        # Expectancy / top strategy
        expectancy = _read_json(self._runtime_dir / "expectancy_state.json") or {}
        top = expectancy.get("top_performer")
        strats = expectancy.get("strategies") or {}
        top_line: Optional[str] = None
        if top and isinstance(strats, dict) and top in strats:
            s = strats[top]
            top_line = (
                f"{translate_strategy(str(top))} — "
                f"{_safe_float(s.get('win_rate'))*100:.0f}% win rate, "
                f"{_format_money(_safe_float(s.get('expectancy')))} expectancy"
            )

        # Beta allocation
        routing = _read_json(self._runtime_dir / "profit_routing.json") or {}
        totals = routing.get("totals") if isinstance(routing.get("totals"), dict) else {}
        beta_total = _safe_float(totals.get("beta_allocation"))
        # Beta accumulated THIS WEEK
        beta_week = 0.0
        decisions = routing.get("decisions") if isinstance(routing.get("decisions"), list) else []
        cutoff = now - timedelta(days=7)
        for d in decisions:
            if not isinstance(d, dict):
                continue
            ts = d.get("routing_timestamp") or d.get("ts_utc")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if dt >= cutoff:
                    beta_week += _safe_float(d.get("beta_allocation"))
            except Exception:
                continue

        lines: List[str] = []
        week_of = (now - timedelta(days=now.weekday() + 1)).strftime("%B %d, %Y")
        lines.append(f"\U0001f4ca CHAD Week in Review — Week of {week_of}")
        lines.append("")

        lines.append("═══ PERFORMANCE ═══")
        lines.append(f"Fills this week: {weekly_fill_count}")
        sign = "+" if weekly_realized_pnl >= 0 else ""
        lines.append(f"Realized P&L: {sign}{_format_money(weekly_realized_pnl)}")
        lines.append(f"Clean trades (SCR): {effective_trades}/100 to CAUTIOUS")
        lines.append("")

        lines.append("═══ TOP BRAIN ═══")
        if top_line:
            lines.append(top_line)
        else:
            lines.append("No strategy has logged enough trades this week to rank.")
        lines.append("")

        lines.append("═══ BETA ALLOCATION ═══")
        lines.append(f"Accumulated this week: {_format_money(beta_week)}")
        lines.append(f"Total Beta earmarked: {_format_money(beta_total)}")
        lines.append("")

        lines.append("═══ SYSTEM HEALTH ═══")
        if isinstance(scr_sizing, (int, float)):
            lines.append(f"SCR: {scr_state} (sizing {scr_sizing:.2f}x)")
        else:
            lines.append(f"SCR: {scr_state}")
        lines.append(f"Regime this week: {regime_label}")
        lines.append("")

        lines.append("═══ CHAD'S WEEKLY TAKE ═══")
        take_ctx = {
            "weekly_fills": weekly_fill_count,
            "weekly_realized_pnl": weekly_realized_pnl,
            "scr_state": scr_state,
            "regime": regime_label,
            "top_strategy": top_line,
            "beta_accumulated_week": beta_week,
            "beta_total": beta_total,
            "effective_trades": effective_trades,
        }
        take = self._weekly_take(take_ctx)
        if take:
            lines.append(take)
        else:
            lines.append(
                "Another week logged — the engine is compounding data, not ego. "
                "The SCR gate is doing its job: gating size to evidence, not opinion."
            )
        lines.append("")
        lines.append("— CHAD \U0001f91d")

        return "\n".join(lines)

    async def send_to_telegram(self) -> bool:
        message = self.generate()
        _send_telegram(message)
        return True


def run_weekly_summary() -> str:
    summary = WeeklySummary()
    message = summary.generate()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    (REPORTS_DIR / f"WEEKLY_CHAD_REPORT_{ts}.txt").write_text(message, encoding="utf-8")
    _send_telegram(message)
    print(message)
    return message


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "morning":
        run_morning_brief()
    elif len(sys.argv) > 1 and sys.argv[1] == "weekly":
        run_weekly_summary()
    else:
        run_daily_report()
