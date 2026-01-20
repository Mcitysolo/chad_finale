"""
CHAD Phase 10 — Weekly Investor Report (Ops + Performance, investor-style)

Purpose
-------
Generate a weekly, investor-readable report from CHAD's append-only trade ledgers:
  data/trades/trade_history_YYYYMMDD.ndjson

This report is "performance-first" (PnL, win rate, winners/losers, by-strategy),
and is safe-by-default:
- No broker calls
- No live-state mutation
- No secrets
- Tolerant of schema drift
- Skips "entry-only / untrusted" rows flagged by payload.extra.pnl_untrusted==true

Outputs
-------
reports/ops/WEEKLY_INVESTOR_REPORT_<ts>.json
reports/ops/WEEKLY_INVESTOR_REPORT_<ts>.md

Window
------
Default: last 7 UTC calendar days INCLUDING today (rolling).
Uses ledger filenames to determine date coverage.

Run
---
python -m chad.ops.weekly_investor_report
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
TRADES_DIR = REPO_ROOT / "data" / "trades"
REPORTS_DIR = REPO_ROOT / "reports" / "ops"

LEDGER_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")


# ----------------------------
# Time helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_today_date() -> date:
    return datetime.now(timezone.utc).date()


def yyyymmdd_to_date(s: str) -> date:
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def date_to_yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


# ----------------------------
# Parsing helpers
# ----------------------------

def is_finite_number(x: float) -> bool:
    return math.isfinite(x)


def to_float_maybe(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if is_finite_number(f) else None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
            return f if is_finite_number(f) else None
        except ValueError:
            return None
    return None


def to_int_maybe(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    p = obj.get("payload")
    return p if isinstance(p, dict) else obj


def is_untrusted_entry_only(payload: Dict[str, Any]) -> bool:
    extra = payload.get("extra")
    if isinstance(extra, dict):
        if extra.get("pnl_untrusted") is True:
            return True
    return False


def norm_strategy(raw: Any) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if not s:
            return "unknown"
        # normalize common labels
        if s in {"alpha", "alpha_stocks", "alpha_brain", "alpha_sniper"}:
            return "alpha"
        if s in {"beta", "beta_portfolio", "beta_brain"}:
            return "beta"
        if s in {"gamma", "gamma_swing", "gamma_brain"}:
            return "gamma"
        if s in {"omega", "omega_hedge", "omega_brain"}:
            return "omega"
        if s in {"delta", "delta_event", "delta_brain"}:
            return "delta"
        if s in {"alphacrypto", "alpha_crypto", "crypto_alpha"}:
            return "alpha_crypto"
        return s
    return "unknown"


def norm_symbol(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN"
    if isinstance(raw, str):
        s = raw.strip().upper()
        return s if s else "UNKNOWN"
    return "UNKNOWN"


def norm_side(raw: Any) -> str:
    if raw is None:
        return "unknown"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"buy", "b", "long"}:
            return "buy"
        if s in {"sell", "s", "short"}:
            return "sell"
        return s if s else "unknown"
    return "unknown"


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class TradeRow:
    day_yyyymmdd: str
    strategy: str
    symbol: str
    side: str
    pnl: float
    qty: Optional[int]
    broker: str
    ts_utc: Optional[str]


def read_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def extract_trade_row(obj: Dict[str, Any], *, day_yyyymmdd: str) -> Optional[TradeRow]:
    p = _payload(obj)

    # Skip entry-only "untrusted" rows
    if is_untrusted_entry_only(p):
        return None

    pnl_val = first_present(p, ["pnl", "pnl_usd", "realized_pnl", "profit", "profit_usd"])
    pnl = to_float_maybe(pnl_val)
    if pnl is None:
        return None

    strategy_raw = first_present(p, ["strategy", "brain", "module", "source", "brain_name", "strategy_name"])
    symbol_raw = first_present(p, ["symbol", "ticker", "instrument", "asset", "contract_symbol"])
    side_raw = first_present(p, ["side", "action", "direction", "order_side"])
    qty_raw = first_present(p, ["quantity", "qty", "shares", "size"])
    broker_raw = p.get("broker")
    broker = broker_raw.strip().lower() if isinstance(broker_raw, str) and broker_raw.strip() else "unknown"

    ts_raw = first_present(obj, ["timestamp_utc", "ts_utc", "timestamp", "time_utc", "ts"])

    return TradeRow(
        day_yyyymmdd=day_yyyymmdd,
        strategy=norm_strategy(strategy_raw),
        symbol=norm_symbol(symbol_raw),
        side=norm_side(side_raw),
        pnl=pnl,
        qty=to_int_maybe(qty_raw),
        broker=broker,
        ts_utc=str(ts_raw) if ts_raw is not None else None,
    )


# ----------------------------
# Aggregation
# ----------------------------

def win_rate(rows: List[TradeRow]) -> float:
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if r.pnl > 0)
    return wins / len(rows)


def summarize(rows: List[TradeRow]) -> Dict[str, Any]:
    total = len(rows)
    pnl_total = sum(r.pnl for r in rows)
    pnl_pos = sum(r.pnl for r in rows if r.pnl > 0)
    pnl_neg = sum(r.pnl for r in rows if r.pnl < 0)
    best = max(rows, key=lambda r: r.pnl, default=None)
    worst = min(rows, key=lambda r: r.pnl, default=None)

    return {
        "trades": total,
        "win_rate": win_rate(rows),
        "pnl_total": pnl_total,
        "pnl_positive_sum": pnl_pos,
        "pnl_negative_sum": pnl_neg,
        "best_trade": None
        if best is None
        else {
            "day": best.day_yyyymmdd,
            "strategy": best.strategy,
            "symbol": best.symbol,
            "side": best.side,
            "pnl": best.pnl,
            "qty": best.qty,
            "broker": best.broker,
        },
        "worst_trade": None
        if worst is None
        else {
            "day": worst.day_yyyymmdd,
            "strategy": worst.strategy,
            "symbol": worst.symbol,
            "side": worst.side,
            "pnl": worst.pnl,
            "qty": worst.qty,
            "broker": worst.broker,
        },
    }


def group_by(rows: List[TradeRow], key_fn) -> Dict[str, List[TradeRow]]:
    out: Dict[str, List[TradeRow]] = {}
    for r in rows:
        k = key_fn(r)
        out.setdefault(k, []).append(r)
    return out


def top_n_by_pnl(rows: List[TradeRow], n: int = 10, reverse: bool = True) -> List[Dict[str, Any]]:
    s = sorted(rows, key=lambda r: r.pnl, reverse=reverse)[:n]
    return [
        {
            "day": r.day_yyyymmdd,
            "strategy": r.strategy,
            "symbol": r.symbol,
            "side": r.side,
            "qty": r.qty,
            "pnl": r.pnl,
            "broker": r.broker,
        }
        for r in s
    ]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_md(path: Path, lines: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------
# Ledger discovery
# ----------------------------

def list_ledgers() -> List[Tuple[str, Path]]:
    """
    Returns [(yyyymmdd, path)] sorted by yyyymmdd ascending.
    """
    out: List[Tuple[str, Path]] = []
    for p in TRADES_DIR.glob("trade_history_*.ndjson"):
        m = LEDGER_RE.match(p.name)
        if not m:
            continue
        out.append((m.group(1), p))
    out.sort(key=lambda t: t[0])
    return out


def ledgers_in_window(days: int = 7) -> List[Tuple[str, Path]]:
    """
    Select ledgers whose filename date is within last N UTC days inclusive.
    """
    today = utc_today_date()
    start = today - timedelta(days=days - 1)
    ledgers = list_ledgers()
    selected: List[Tuple[str, Path]] = []
    for ymd, path in ledgers:
        d = yyyymmdd_to_date(ymd)
        if start <= d <= today:
            selected.append((ymd, path))
    return selected


# ----------------------------
# Main
# ----------------------------

def run(window_days: int = 7) -> Tuple[Path, Path]:
    safe_mkdir(REPORTS_DIR)

    chosen = ledgers_in_window(days=window_days)
    if not chosen:
        # Fallback: if no files in last N days, use newest available file(s) up to N
        all_ledgers = list_ledgers()
        if not all_ledgers:
            raise SystemExit("No trade_history_*.ndjson files found in data/trades/")
        chosen = all_ledgers[-window_days:]

    # Parse
    rows: List[TradeRow] = []
    parse_stats: Dict[str, Dict[str, int]] = {}
    for ymd, path in chosen:
        total_lines = 0
        used = 0
        skipped = 0
        for obj in read_ndjson(path):
            total_lines += 1
            tr = extract_trade_row(obj, day_yyyymmdd=ymd)
            if tr is None:
                skipped += 1
                continue
            used += 1
            rows.append(tr)
        parse_stats[ymd] = {"ledger_lines": total_lines, "rows_used": used, "rows_skipped": skipped}

    # Aggregations
    by_day = group_by(rows, lambda r: r.day_yyyymmdd)
    by_strategy = group_by(rows, lambda r: r.strategy)
    by_symbol = group_by(rows, lambda r: r.symbol)

    overall = summarize(rows)

    # Daily series (sorted)
    daily_series = []
    cumulative = 0.0
    for ymd in sorted(parse_stats.keys()):
        day_rows = by_day.get(ymd, [])
        day_pnl = sum(r.pnl for r in day_rows)
        cumulative += day_pnl
        daily_series.append(
            {
                "day": ymd,
                "trades": len(day_rows),
                "win_rate": win_rate(day_rows),
                "pnl_total": day_pnl,
                "cumulative_pnl": cumulative,
                "parse": parse_stats.get(ymd, {}),
            }
        )

    # Strategy summaries
    strat_summary = {k: summarize(v) for k, v in sorted(by_strategy.items(), key=lambda kv: kv[0])}

    # Symbol ranking
    symbol_pnl = []
    for sym, sym_rows in by_symbol.items():
        symbol_pnl.append((sym, sum(r.pnl for r in sym_rows), len(sym_rows), win_rate(sym_rows)))
    symbol_pnl.sort(key=lambda t: t[1], reverse=True)
    top_symbols = [
        {"symbol": sym, "pnl_total": pnl, "trades": cnt, "win_rate": wr}
        for sym, pnl, cnt, wr in symbol_pnl[:25]
    ]
    bottom_symbols = [
        {"symbol": sym, "pnl_total": pnl, "trades": cnt, "win_rate": wr}
        for sym, pnl, cnt, wr in sorted(symbol_pnl, key=lambda t: t[1])[:25]
    ]

    # Output
    ts = utc_now_compact()
    jpath = REPORTS_DIR / f"WEEKLY_INVESTOR_REPORT_{ts}.json"
    mpath = REPORTS_DIR / f"WEEKLY_INVESTOR_REPORT_{ts}.md"

    payload: Dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "window_days": window_days,
        "ledgers_used": [{"day": ymd, "path": str(path)} for ymd, path in chosen],
        "parse_by_day": parse_stats,
        "overall": overall,
        "daily_series": daily_series,
        "by_strategy": strat_summary,
        "top_symbols": top_symbols,
        "bottom_symbols": bottom_symbols,
        "top_trades": {
            "best_10": top_n_by_pnl(rows, n=10, reverse=True),
            "worst_10": top_n_by_pnl(rows, n=10, reverse=False),
        },
    }

    write_json(jpath, payload)

    # Markdown (investor tone, still plain)
    md: List[str] = []
    md.append("# CHAD Weekly Investor Report (Phase 10)")
    md.append(f"- Generated: `{payload['generated_utc']}`")
    md.append(f"- Host: `{payload['host']}`")
    md.append(f"- Window: last `{window_days}` UTC days")
    md.append("")

    md.append("## Executive Summary")
    md.append(f"- Trades (trusted realized): `{payload['overall']['trades']}`")
    md.append(f"- Win rate: `{payload['overall']['win_rate']}`")
    md.append(f"- Total PnL: `{payload['overall']['pnl_total']}`")
    md.append(f"- Positive PnL sum: `{payload['overall']['pnl_positive_sum']}`")
    md.append(f"- Negative PnL sum: `{payload['overall']['pnl_negative_sum']}`")
    bt = payload["overall"].get("best_trade")
    wt = payload["overall"].get("worst_trade")
    if bt:
        md.append(f"- Best trade: `{bt['day']}` `{bt['strategy']}` `{bt['symbol']}` `{bt['side']}` pnl=`{bt['pnl']}`")
    if wt:
        md.append(f"- Worst trade: `{wt['day']}` `{wt['strategy']}` `{wt['symbol']}` `{wt['side']}` pnl=`{wt['pnl']}`")
    md.append("")

    md.append("## Daily PnL (UTC)")
    for r in payload["daily_series"]:
        md.append(
            f"- `{r['day']}` trades=`{r['trades']}` win_rate=`{r['win_rate']}` "
            f"pnl=`{r['pnl_total']}` cumulative=`{r['cumulative_pnl']}`"
        )
    md.append("")

    md.append("## By Strategy")
    if not payload["by_strategy"]:
        md.append("- (none)")
    else:
        for strat, s in payload["by_strategy"].items():
            md.append(f"### {strat}")
            md.append(f"- Trades: `{s['trades']}`")
            md.append(f"- Win rate: `{s['win_rate']}`")
            md.append(f"- PnL total: `{s['pnl_total']}`")
            md.append("")

    md.append("## Top Symbols (by PnL)")
    if payload["top_symbols"]:
        for row in payload["top_symbols"][:10]:
            md.append(f"- `{row['symbol']}` pnl=`{row['pnl_total']}` trades=`{row['trades']}` win_rate=`{row['win_rate']}`")
    else:
        md.append("- (none)")
    md.append("")

    md.append("## Bottom Symbols (by PnL)")
    if payload["bottom_symbols"]:
        for row in payload["bottom_symbols"][:10]:
            md.append(f"- `{row['symbol']}` pnl=`{row['pnl_total']}` trades=`{row['trades']}` win_rate=`{row['win_rate']}`")
    else:
        md.append("- (none)")
    md.append("")

    md.append("## Top Trades (best / worst)")
    md.append("### Best 10")
    for r in payload["top_trades"]["best_10"]:
        md.append(f"- `{r['day']}` `{r['strategy']}` `{r['symbol']}` `{r['side']}` qty=`{r['qty']}` pnl=`{r['pnl']}`")
    md.append("")
    md.append("### Worst 10")
    for r in payload["top_trades"]["worst_10"]:
        md.append(f"- `{r['day']}` `{r['strategy']}` `{r['symbol']}` `{r['side']}` qty=`{r['qty']}` pnl=`{r['pnl']}`")
    md.append("")

    md.append("## Ledgers Used")
    for it in payload["ledgers_used"]:
        md.append(f"- `{it['day']}` `{it['path']}`")
    md.append("")

    md.append("## Notes / Caveats")
    md.append("- This report uses **trusted realized rows** only (entry-only rows flagged as untrusted are excluded).")
    md.append("- This is Phase 10 reporting; it does not change trading, risk, or execution behavior.")
    md.append("")

    write_md(mpath, md)

    # Telegram push (weekly investor summary)
    # Fail-safe: report generation must succeed even if Telegram is misconfigured/down.
    try:
        from chad.utils.telegram_notify import notify  # type: ignore

        snippet = "\n".join(md[:35]).strip()
        if snippet:
            notify(
                snippet,
                severity="info",
                dedupe_key="weekly_investor_report",
                raise_on_fail=False,
            )
    except Exception:
        pass

    print(str(jpath))
    print(str(mpath))
    return jpath, mpath


if __name__ == "__main__":
    run()
