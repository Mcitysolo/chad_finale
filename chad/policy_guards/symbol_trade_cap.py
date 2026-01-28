"""
chad/policy/symbol_trade_cap.py

CHAD Policy Guard — Symbol Trade Cap (Deterministic)

Purpose
-------
Prevent over-trading and runaway loss streaks on a single symbol (e.g., AAPL)
using deterministic, audit-friendly rules.

This is policy-layer logic (advisory decision), not execution:
- No broker calls
- No config mutation
- No runtime mutation except optional writing of last decision (atomic)

Rules
-----
Deny *new entries* for a symbol if either:
1) trades_counted >= max_trades_per_day
2) consecutive_losses >= max_consecutive_losses  (trusted realized rows only)

Ledger schema (verified from your system)
-----------------------------------------
Each ledger line is JSON with trade fields inside obj["payload"].
Entry-only rows are marked as untrusted via:
  payload.extra.pnl_untrusted == true

Counting
--------
- trades_counted: counts ALL rows for the symbol (trusted + untrusted) to limit overtrading.
- consecutive_losses: counts the loss streak from most recent backwards using only
  trusted realized rows (not untrusted). Loss is pnl < 0.

CLI
---
python -m chad.policy_guards.symbol_trade_cap --symbol AAPL --write-runtime

Outputs
-------
- Prints a JSON decision to stdout.
- Optional: runtime/symbol_trade_cap_last.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
TRADES_DIR = REPO_ROOT / "data" / "trades"
RUNTIME_DIR = REPO_ROOT / "runtime"


# -------------------------
# Utilities
# -------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def today_utc_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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


def payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    p = obj.get("payload")
    return p if isinstance(p, dict) else obj


def is_untrusted_entry_only(p: Dict[str, Any]) -> bool:
    extra = p.get("extra")
    return isinstance(extra, dict) and extra.get("pnl_untrusted") is True


def to_float_maybe(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None


def norm_symbol(v: Any) -> str:
    s = str(v or "").strip().upper()
    return s if s else "UNKNOWN"


def get_ts(obj: Dict[str, Any]) -> str:
    ts = obj.get("timestamp_utc")
    if ts is None:
        ts = obj.get("ts_utc")
    return str(ts) if ts is not None else ""


def pick_ledger_for_today_or_latest() -> Path:
    """
    Use today's ledger if present; else fall back to the newest ledger.
    This matches the robust behavior we implemented for reporting.
    """
    today = today_utc_yyyymmdd()
    p = TRADES_DIR / f"trade_history_{today}.ndjson"
    if p.exists():
        return p
    candidates = sorted(TRADES_DIR.glob("trade_history_*.ndjson"))
    if not candidates:
        raise FileNotFoundError(f"No trade_history_*.ndjson in {TRADES_DIR}")
    return candidates[-1]


# -------------------------
# Decision model
# -------------------------

@dataclass(frozen=True)
class SymbolCapDecision:
    allowed: bool
    reason_code: str
    symbol: str
    ledger_path: str
    evaluated_utc: str

    trades_counted: int
    trusted_realized_rows: int
    consecutive_losses: int

    max_trades_per_day: int
    max_consecutive_losses: int

    notes: List[str]


# -------------------------
# Core evaluation
# -------------------------

def evaluate_symbol(
    *,
    symbol: str,
    ledger_path: Path,
    max_trades_per_day: int,
    max_consecutive_losses: int,
) -> SymbolCapDecision:
    sym = norm_symbol(symbol)
    trades_count = 0
    realized: List[Tuple[str, float]] = []  # (ts, pnl)

    for obj in read_ndjson(ledger_path):
        p = payload(obj)
        if norm_symbol(p.get("symbol")) != sym:
            continue

        trades_count += 1  # all rows count toward overtrading cap

        if is_untrusted_entry_only(p):
            continue

        pnl = to_float_maybe(p.get("pnl"))
        if pnl is None:
            continue

        realized.append((get_ts(obj), pnl))

    realized.sort(key=lambda t: t[0])

    streak = 0
    for _, pnl in reversed(realized):
        if pnl < 0:
            streak += 1
            continue
        break

    notes: List[str] = []
    if trades_count == 0:
        notes.append("No rows for symbol in selected ledger file (symbol not traded).")

    if max_trades_per_day > 0 and trades_count >= max_trades_per_day:
        return SymbolCapDecision(
            allowed=False,
            reason_code="SYMBOL_CAP_MAX_TRADES",
            symbol=sym,
            ledger_path=str(ledger_path),
            evaluated_utc=utc_now_iso(),
            trades_counted=trades_count,
            trusted_realized_rows=len(realized),
            consecutive_losses=streak,
            max_trades_per_day=max_trades_per_day,
            max_consecutive_losses=max_consecutive_losses,
            notes=notes + [f"Denied: trades_counted={trades_count} >= max_trades_per_day={max_trades_per_day}"],
        )

    if max_consecutive_losses > 0 and streak >= max_consecutive_losses:
        return SymbolCapDecision(
            allowed=False,
            reason_code="SYMBOL_CAP_CONSECUTIVE_LOSSES",
            symbol=sym,
            ledger_path=str(ledger_path),
            evaluated_utc=utc_now_iso(),
            trades_counted=trades_count,
            trusted_realized_rows=len(realized),
            consecutive_losses=streak,
            max_trades_per_day=max_trades_per_day,
            max_consecutive_losses=max_consecutive_losses,
            notes=notes + [f"Denied: consecutive_losses={streak} >= max_consecutive_losses={max_consecutive_losses}"],
        )

    return SymbolCapDecision(
        allowed=True,
        reason_code="OK",
        symbol=sym,
        ledger_path=str(ledger_path),
        evaluated_utc=utc_now_iso(),
        trades_counted=trades_count,
        trusted_realized_rows=len(realized),
        consecutive_losses=streak,
        max_trades_per_day=max_trades_per_day,
        max_consecutive_losses=max_consecutive_losses,
        notes=notes + ["Allowed: caps not exceeded."],
    )


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHAD policy guard: symbol trade cap evaluator (deterministic).")
    p.add_argument("--symbol", required=True, help="Symbol to evaluate, e.g. AAPL")
    p.add_argument("--ledger", default="", help="Optional path to trade_history_YYYYMMDD.ndjson")
    p.add_argument("--max-trades-per-day", type=int, default=200, help="Deny if trades >= this (0 disables)")
    p.add_argument("--max-consecutive-losses", type=int, default=8, help="Deny if loss streak >= this (0 disables)")
    p.add_argument("--write-runtime", action="store_true", help="Write runtime/symbol_trade_cap_last.json")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    ledger = Path(args.ledger) if args.ledger.strip() else pick_ledger_for_today_or_latest()
    if not ledger.exists():
        raise SystemExit(f"Missing ledger: {ledger}")

    decision = evaluate_symbol(
        symbol=args.symbol,
        ledger_path=ledger,
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_losses=int(args.max_consecutive_losses),
    )

    payload_out = asdict(decision)
    print(json.dumps(payload_out, indent=2, sort_keys=True))

    if args.write_runtime:
        atomic_write_json(RUNTIME_DIR / "symbol_trade_cap_last.json", payload_out)


if __name__ == "__main__":
    main()
