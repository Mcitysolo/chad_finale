"""
CHAD Phase 10 — Daily Performance Report (Ledger v2: payload-wrapped)

Your ledger rows are NDJSON where each line is a dict with keys like:
  {
    "payload": {...trade fields...},
    "timestamp_utc": "...",
    "record_hash": "...",
    ...
  }

We must parse trade fields from obj["payload"] (source of truth).
We also exclude "untrusted" entry-only rows flagged as:
  payload.extra.pnl_untrusted == true

Outputs:
  reports/ops/DAILY_PERF_REPORT_<ts>.json
  reports/ops/DAILY_PERF_REPORT_<ts>.md
"""

from __future__ import annotations

import json
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
TRADES_DIR = REPO_ROOT / "data" / "trades"
REPORTS_DIR = REPO_ROOT / "reports" / "ops"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def today_utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


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


@dataclass(frozen=True)
class TradeRow:
    strategy: str
    symbol: str
    side: str
    pnl: float
    qty: Optional[int]
    ts_utc: Optional[str]
    is_live: Optional[bool]
    broker: str


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


def _payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    p = obj.get("payload")
    return p if isinstance(p, dict) else obj


def is_untrusted_entry_only(payload: Dict[str, Any]) -> bool:
    extra = payload.get("extra")
    if isinstance(extra, dict):
        if extra.get("pnl_untrusted") is True:
            return True
    return False


def extract_trade_row(obj: Dict[str, Any]) -> Optional[TradeRow]:
    p = _payload(obj)

    # Skip entry-only "untrusted" rows so we report real realized outcomes.
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
    ts_raw = first_present(obj, ["timestamp_utc", "ts_utc", "timestamp", "time_utc", "ts"])

    is_live_raw = p.get("is_live")
    is_live: Optional[bool]
    if isinstance(is_live_raw, bool):
        is_live = is_live_raw
    else:
        is_live = None

    broker_raw = p.get("broker")
    broker = broker_raw.strip().lower() if isinstance(broker_raw, str) and broker_raw.strip() else "unknown"

    return TradeRow(
        strategy=norm_strategy(strategy_raw),
        symbol=norm_symbol(symbol_raw),
        side=norm_side(side_raw),
        pnl=pnl,
        qty=to_int_maybe(qty_raw),
        ts_utc=str(ts_raw) if ts_raw is not None else None,
        is_live=is_live,
        broker=broker,
    )


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
        else {"symbol": best.symbol, "strategy": best.strategy, "side": best.side, "pnl": best.pnl},
        "worst_trade": None
        if worst is None
        else {"symbol": worst.symbol, "strategy": worst.strategy, "side": worst.side, "pnl": worst.pnl},
    }


def top_n_by_pnl(rows: List[TradeRow], n: int = 10, reverse: bool = True) -> List[Dict[str, Any]]:
    s = sorted(rows, key=lambda r: r.pnl, reverse=reverse)[:n]
    return [
        {"strategy": r.strategy, "symbol": r.symbol, "side": r.side, "pnl": r.pnl, "qty": r.qty, "broker": r.broker}
        for r in s
    ]


def group_by(rows: List[TradeRow], key_fn) -> Dict[str, List[TradeRow]]:
    out: Dict[str, List[TradeRow]] = {}
    for r in rows:
        k = key_fn(r)
        out.setdefault(k, []).append(r)
    return out


def run() -> Tuple[Path, Path]:
    day = today_utc_yyyymmdd()
    ledger = TRADES_DIR / f"trade_history_{day}.ndjson"

    # If today's ledger doesn't exist yet (e.g., timer runs before first trade),
    # fall back to the newest available ledger file.
    if not ledger.exists():
        candidates = sorted(TRADES_DIR.glob("trade_history_*.ndjson"))
        if not candidates:
            raise SystemExit(f"Missing ledger: {ledger} (and no trade_history_*.ndjson files found)")
        ledger = candidates[-1]

    rows: List[TradeRow] = []
    skipped = 0
    total_lines = 0

    for obj in read_ndjson(ledger):
        total_lines += 1
        tr = extract_trade_row(obj)
        if tr is None:
            skipped += 1
            continue
        rows.append(tr)

    by_strategy = group_by(rows, lambda r: r.strategy)
    by_symbol = group_by(rows, lambda r: r.symbol)

    strategies_summary = {k: summarize(v) for k, v in sorted(by_strategy.items(), key=lambda kv: kv[0])}

    # Top symbols by total pnl contribution
    symbol_pnl: List[Tuple[str, float, int, float]] = []
    for sym, sym_rows in by_symbol.items():
        symbol_pnl.append((sym, sum(r.pnl for r in sym_rows), len(sym_rows), win_rate(sym_rows)))
    symbol_pnl.sort(key=lambda t: t[1], reverse=True)

    overall = summarize(rows)

    out: Dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "ledger_path": str(ledger),
        "parse": {"ledger_lines": total_lines, "rows_used": len(rows), "rows_skipped": skipped},
        "overall": overall,
        "by_strategy": strategies_summary,
        "top_trades": {
            "best_10": top_n_by_pnl(rows, n=10, reverse=True),
            "worst_10": top_n_by_pnl(rows, n=10, reverse=False),
        },
        "top_symbols": [
            {"symbol": sym, "pnl_total": pnl, "trades": cnt, "win_rate": wr}
            for sym, pnl, cnt, wr in symbol_pnl[:25]
        ],
    }

    safe_mkdir(REPORTS_DIR)
    ts = utc_now_compact()
    jpath = REPORTS_DIR / f"DAILY_PERF_REPORT_{ts}.json"
    mpath = REPORTS_DIR / f"DAILY_PERF_REPORT_{ts}.md"

    write_json(jpath, out)

    md: List[str] = []
    md.append("# CHAD Daily Performance Report (Phase 10)")
    md.append(f"- Generated: `{out['generated_utc']}`")
    md.append(f"- Host: `{out['host']}`")
    md.append(f"- Ledger: `{out['ledger_path']}`")
    md.append("")
    md.append("## Parse")
    md.append(f"- Ledger lines: `{out['parse']['ledger_lines']}`")
    md.append(f"- Rows used (trusted realized): `{out['parse']['rows_used']}`")
    md.append(f"- Rows skipped: `{out['parse']['rows_skipped']}`")
    md.append("")
    md.append("## Overall (trusted realized rows)")
    md.append(f"- Trades: `{overall['trades']}`")
    md.append(f"- Win rate: `{overall['win_rate']}`")
    md.append(f"- PnL total: `{overall['pnl_total']}`")
    md.append(f"- PnL positive sum: `{overall['pnl_positive_sum']}`")
    md.append(f"- PnL negative sum: `{overall['pnl_negative_sum']}`")

    bt = overall.get("best_trade")
    wt = overall.get("worst_trade")
    if bt:
        md.append(f"- Best trade: `{bt['strategy']}` `{bt['symbol']}` `{bt['side']}` pnl=`{bt['pnl']}`")
    if wt:
        md.append(f"- Worst trade: `{wt['strategy']}` `{wt['symbol']}` `{wt['side']}` pnl=`{wt['pnl']}`")

    md.append("")
    md.append("## By Strategy")
    if not strategies_summary:
        md.append("- (none)")
    else:
        for strat, summ in strategies_summary.items():
            md.append(f"### {strat}")
            md.append(f"- Trades: `{summ['trades']}`")
            md.append(f"- Win rate: `{summ['win_rate']}`")
            md.append(f"- PnL total: `{summ['pnl_total']}`")
            md.append("")

    md.append("## Top Symbols (by total PnL)")
    if not out["top_symbols"]:
        md.append("- (none)")
    else:
        for row in out["top_symbols"][:15]:
            md.append(f"- `{row['symbol']}` pnl=`{row['pnl_total']}` trades=`{row['trades']}` win_rate=`{row['win_rate']}`")

    md.append("")
    md.append("## Top 10 Trades")
    md.append("### Best 10")
    for r in out["top_trades"]["best_10"]:
        md.append(f"- `{r['strategy']}` `{r['symbol']}` `{r['side']}` qty=`{r['qty']}` pnl=`{r['pnl']}` broker=`{r['broker']}`")

    md.append("")
    md.append("### Worst 10")
    for r in out["top_trades"]["worst_10"]:
        md.append(f"- `{r['strategy']}` `{r['symbol']}` `{r['side']}` qty=`{r['qty']}` pnl=`{r['pnl']}` broker=`{r['broker']}`")

    write_md(mpath, md)

    print(str(jpath))
    print(str(mpath))
    return jpath, mpath


if __name__ == "__main__":
    run()
