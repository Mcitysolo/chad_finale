from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def extract_strategy_and_pnl(obj: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """
    Supports two schemas:
    - IBKR/paper ledger: {"payload": {..., "strategy": "...", "realized_pnl": ...}}
    - Kraken trusted ledger (derived): {"payload": {..., "extra": {"pnl_trusted": bool}, "pnl": ...}}
    """
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    strategy = str(payload.get("strategy") or obj.get("strategy") or "unknown").strip() or "unknown"

    # Determine if pnl is trusted (default True for IBKR realized_pnl rows, but handle gracefully)
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}

    # Kraken trusted ledger uses pnl_trusted flag; IBKR may not.
    if isinstance(extra, dict) and "pnl_trusted" in extra:
        if not bool(extra.get("pnl_trusted", False)):
            return strategy, None  # explicitly untrusted: exclude from scoring

    # Prefer realized_pnl if present
    if "realized_pnl" in payload:
        return strategy, safe_float(payload.get("realized_pnl"), default=0.0)

    # Kraken uses pnl field
    if "pnl" in payload:
        return strategy, safe_float(payload.get("pnl"), default=0.0)

    # Fallback
    if "realized_pnl" in obj:
        return strategy, safe_float(obj.get("realized_pnl"), default=0.0)
    if "pnl" in obj:
        return strategy, safe_float(obj.get("pnl"), default=0.0)

    return strategy, None


def summarize(pnls: List[float]) -> Dict[str, Any]:
    n = len(pnls)
    wins = sum(1 for x in pnls if x > 0)
    win_rate = (wins / n) if n else 0.0
    total = sum(pnls)
    avg = (total / n) if n else 0.0
    return {
        "samples": n,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total, 6),
        "avg_pnl": round(avg, 8),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Trade Stats Engine v2 (trusted pnl only).")
    ap.add_argument("--trades-dir", default="data/trades", help="Directory containing trade_history_*.ndjson")
    ap.add_argument("--kraken-trusted", default="data/trades/trade_history_enriched_trusted.ndjson")
    ap.add_argument("--days", type=int, default=10, help="Scan last N trade_history files")
    ap.add_argument("--out", default="", help="Write JSON to this path")
    args = ap.parse_args()

    trades_dir = Path(args.trades_dir).expanduser().resolve()
    kraken_trusted = Path(args.kraken_trusted).expanduser().resolve()

    files = sorted(trades_dir.glob("trade_history_*.ndjson"))
    files = files[-int(args.days):] if files else []

    by_strat: Dict[str, List[float]] = defaultdict(list)
    meta = {
        "ts_utc": utc_now(),
        "trades_dir": str(trades_dir),
        "files_scanned": [str(p) for p in files],
        "kraken_trusted_path": str(kraken_trusted),
        "notes": [
            "Only trusted pnl is counted.",
            "Kraken BUY entry-only rows are excluded by design via pnl_trusted=false.",
        ],
    }

    # Scan IBKR/paper ledgers
    for f in files:
        for obj in iter_ndjson(f):
            s, pnl = extract_strategy_and_pnl(obj)
            if pnl is None:
                continue
            by_strat[s].append(float(pnl))

    # Scan Kraken trusted ledger (if exists)
    if kraken_trusted.exists():
        for obj in iter_ndjson(kraken_trusted):
            s, pnl = extract_strategy_and_pnl(obj)
            if pnl is None:
                continue
            by_strat[s].append(float(pnl))
    else:
        meta["notes"].append("kraken_trusted_missing: run kraken_pnl_trust_patch first")

    report = {
        **meta,
        "per_strategy": {s: summarize(pnls) for s, pnls in sorted(by_strat.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
    }

    outp = str(args.out).strip()
    if outp:
        Path(outp).expanduser().resolve().write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
