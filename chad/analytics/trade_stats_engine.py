#!/usr/bin/env python3
"""
CHAD — Trade Stats Engine (Production)

Computes aggregate performance stats used by:
- /live-gate endpoint
- SCR (Shadow Confidence Router)
- Telegram coach, status server, metrics rollups

Core Rules
----------
Trades included for performance metrics (win_rate, sharpe_like, drawdown, total_pnl) only if:
- pnl and notional are finite
- trade is NOT tagged manual
- trade is NOT tagged pnl_untrusted

This build also supports a separate enrichment ledger:
  data/trades/trade_history_enriched.ndjson

When both raw + enriched Kraken records exist for the same txid:
- enriched record wins
- raw record is excluded from effective calculations

Output fields (important)
------------------------
- total_trades: all parsed trades (including excluded)
- effective_trades: trades that count toward performance evaluation
- excluded_manual
- excluded_untrusted
- excluded_nonfinite
- live_trades / paper_trades
- win_rate, total_pnl, sharpe_like, max_drawdown

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DATA_TRADES = ROOT / "data" / "trades"
ENRICH_LEDGER = DATA_TRADES / "trade_history_enriched.ndjson"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if not math.isfinite(f):
            return default
        return f
    except Exception:
        return default


def _is_manual(tags: List[str]) -> bool:
    return any(str(t).lower() == "manual" for t in tags)


def _is_untrusted(tags: List[str], extra: Dict[str, Any]) -> bool:
    if any(str(t).lower() == "pnl_untrusted" for t in tags):
        return True
    if bool(extra.get("pnl_untrusted")):
        return True
    return False


def _is_unfilled_ibkr_paper(t: "TradeRow") -> bool:
    """Return True if this TradeRow is an IBKR PAPER record with no fill proof.

    Rationale:
      - CHAD logs paper order intent metadata for traceability.
      - SCR must ONLY count filled trades as performance sample, otherwise we
        inflate effective_trades with unfilled placeholders.
      - Filled proof is detected via:
          * 'filled' tag, OR
          * fill_price > 0, OR
          * extra.fill_price_used / extra.filled_quantity_used > 0
    """
    try:
        if str(getattr(t, "broker", "") or "").strip().lower() != "ibkr":
            return False
        if bool(getattr(t, "is_live", False)):
            return False  # only applies to paper
        tags = [str(x).strip().lower() for x in (getattr(t, "tags", []) or [])]
        if "filled" in tags:
            return False

        fp = 0.0
        try:
            fp = float(getattr(t, "fill_price", 0.0) or 0.0)
        except Exception:
            fp = 0.0
        if fp > 0.0:
            return False

        extra = getattr(t, "extra", {}) or {}
        if not isinstance(extra, dict):
            extra = {}

        fp2 = 0.0
        fq2 = 0.0
        try:
            fp2 = float(extra.get("fill_price_used") or 0.0)
        except Exception:
            fp2 = 0.0
        try:
            fq2 = float(extra.get("filled_quantity_used") or 0.0)
        except Exception:
            fq2 = 0.0

        if fp2 > 0.0 or fq2 > 0.0:
            return False

        # No fill proof => exclude from SCR sample.
        return True
    except Exception:
        # Fail-safe: do not exclude if we cannot evaluate.
        return False




def _kraken_txid(payload: Dict[str, Any]) -> Optional[str]:
    extra = payload.get("extra") or {}
    txid = extra.get("txid")
    if txid:
        return str(txid)
    return None


@dataclass(frozen=True)
class ParsedTrade:
    broker: str
    is_live: bool
    strategy: str
    symbol: str
    pnl: float
    notional: float
    tags: List[str]
    extra: Dict[str, Any]
    source: str  # "raw" or "enriched"
    txid: Optional[str] = None


def _iter_day_trade_files(days_back: int) -> List[Path]:
    out: List[Path] = []
    now = _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(max(0, int(days_back)) + 1):
        ymd = (now - timedelta(days=i)).strftime("%Y%m%d")
        p = DATA_TRADES / f"trade_history_{ymd}.ndjson"
        if p.is_file():
            out.append(p)
    if not out:
        out = sorted(DATA_TRADES.glob("trade_history_*.ndjson"))
    return out


def _parse_lines(path: Path, source: str) -> List[ParsedTrade]:
    out: List[ParsedTrade] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        payload = rec.get("payload") or {}
        broker = str(payload.get("broker") or "")
        if not broker:
            continue
        tags = list(payload.get("tags") or [])
        extra = dict(payload.get("extra") or {})
        pnl = _safe_float(payload.get("pnl"), float("nan"))
        notional = _safe_float(payload.get("notional"), float("nan"))
        is_live = bool(payload.get("is_live", False))
        pt = ParsedTrade(
            broker=broker,
            is_live=is_live,
            strategy=str(payload.get("strategy") or ""),
            symbol=str(payload.get("symbol") or ""),
            pnl=pnl,
            notional=notional,
            tags=tags,
            extra=extra,
            source=source,
            txid=_kraken_txid(payload) if broker == "kraken" else None,
        )
        out.append(pt)
    return out


def _load_all_trades(days_back: int) -> List[ParsedTrade]:
    trades: List[ParsedTrade] = []

    # Raw daily ledgers
    for f in _iter_day_trade_files(days_back):
        trades.extend(_parse_lines(f, source="raw"))

    # Enrichment ledger (optional)
    if ENRICH_LEDGER.is_file():
        trades.extend(_parse_lines(ENRICH_LEDGER, source="enriched"))

    return trades


def _dedupe_kraken(trades: List[ParsedTrade]) -> List[ParsedTrade]:
    """
    If both raw+enriched exist for same kraken txid, keep enriched only.
    """
    enriched_by_txid: Dict[str, ParsedTrade] = {}
    raw_by_txid: Dict[str, ParsedTrade] = {}

    others: List[ParsedTrade] = []
    for t in trades:
        if t.broker != "kraken" or not t.txid:
            others.append(t)
            continue
        if t.source == "enriched":
            enriched_by_txid[t.txid] = t
        else:
            raw_by_txid[t.txid] = t

    kept: List[ParsedTrade] = []
    # Add deduped kraken trades: enriched wins
    for txid, t_en in enriched_by_txid.items():
        kept.append(t_en)
    # Add raw only if no enriched exists
    for txid, t_raw in raw_by_txid.items():
        if txid not in enriched_by_txid:
            kept.append(t_raw)

    return others + kept


def _compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        dd = x - peak
        max_dd = min(max_dd, dd)
    return float(max_dd)


def _compute_sharpe_like(pnl_series: Sequence[float]) -> float:
    if not pnl_series:
        return 0.0
    mean = sum(pnl_series) / len(pnl_series)
    # population stdev
    var = sum((x - mean) ** 2 for x in pnl_series) / len(pnl_series)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    val = mean / sd
    if not math.isfinite(val):
        return 0.0
    return float(val)


def load_and_compute(
    *,
    max_trades: int = 500,
    days_back: int = 30,
    include_paper: bool = True,
    include_live: bool = True,
) -> Dict[str, Any]:
    """
    Compute trade stats from ledgers.
    """
    all_trades = _dedupe_kraken(_load_all_trades(days_back))

    # truncate for safety
    if max_trades > 0:
        all_trades = all_trades[-max_trades:]

    total_trades = len(all_trades)
    live_trades = 0
    paper_trades = 0

    excluded_manual = 0
    excluded_untrusted = 0
    excluded_nonfinite = 0

    pnls: List[float] = []

    for t in all_trades:
        if t.is_live:
            live_trades += 1
        else:
            paper_trades += 1

        # filter by include flags
        if t.is_live and not include_live:
            continue
        if (not t.is_live) and not include_paper:
            continue

        if _is_manual(t.tags):
            excluded_manual += 1
            continue
                # Exclude unfilled IBKR PAPER placeholders from SCR sample
        if _is_unfilled_ibkr_paper(t):
            excluded_untrusted += 1
            continue

        if _is_untrusted(t.tags, t.extra):
            excluded_untrusted += 1
            continue
        if not (math.isfinite(t.pnl) and math.isfinite(t.notional)):
            excluded_nonfinite += 1
            continue

        pnls.append(float(t.pnl))

    effective_trades = len(pnls)
    winners = sum(1 for x in pnls if x > 0.0)
    win_rate = float(winners / effective_trades) if effective_trades > 0 else 0.0
    total_pnl = float(sum(pnls)) if pnls else 0.0

    equity = []
    running = 0.0
    for x in pnls:
        running += float(x)
        equity.append(running)

    max_dd = _compute_max_drawdown(equity) if equity else 0.0
    sharpe_like = _compute_sharpe_like(pnls)

    return {
        "total_trades": int(total_trades),
        "effective_trades": int(effective_trades),
        "excluded_manual": int(excluded_manual),
        "excluded_untrusted": int(excluded_untrusted),
        "excluded_nonfinite": int(excluded_nonfinite),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_dd),
        "sharpe_like": float(sharpe_like),
        "live_trades": int(live_trades),
        "paper_trades": int(paper_trades),
    }


if __name__ == "__main__":
    stats = load_and_compute(max_trades=5000, days_back=60, include_paper=True, include_live=True)
    print(json.dumps(stats, indent=2, sort_keys=True))
