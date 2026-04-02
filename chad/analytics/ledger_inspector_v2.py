from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def iter_ndjson(path: Path, *, max_lines: int = 2_000_000) -> Iterable[Dict[str, Any]]:
    # Robust reader: skips bad lines, never throws.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return
    except Exception:
        return


@dataclass(frozen=True)
class Row:
    strategy: str
    broker: str
    pnl: Optional[float]


def extract_row(o: Dict[str, Any]) -> Row:
    # Your ledger schema has payload: {...}
    payload = o.get("payload") if isinstance(o.get("payload"), dict) else {}
    strategy = str(payload.get("strategy") or o.get("strategy") or o.get("brain") or "unknown")
    broker = str(payload.get("broker") or o.get("broker") or "unknown")

    pnl: Optional[float] = None
    # Prefer realized_pnl in payload
    if "realized_pnl" in payload:
        pnl = safe_float(payload.get("realized_pnl"), default=0.0)
    elif "pnl" in payload:
        pnl = safe_float(payload.get("pnl"), default=0.0)
    # fallback top-level
    elif "realized_pnl" in o:
        pnl = safe_float(o.get("realized_pnl"), default=0.0)
    elif "pnl" in o:
        pnl = safe_float(o.get("pnl"), default=0.0)

    return Row(strategy=strategy, broker=broker, pnl=pnl)


def find_trades_dir(root: Path) -> Path:
    # canonical expected
    a = root / "data" / "trades"
    if a.exists():
        return a
    raise FileNotFoundError(f"data/trades not found under {root}")


def newest_ledger(trades_dir: Path) -> Optional[Path]:
    files = sorted(trades_dir.glob("trade_history_*.ndjson"))
    return files[-1] if files else None


def maybe_enriched(trades_dir: Path) -> Optional[Path]:
    p = trades_dir / "trade_history_enriched.ndjson"
    return p if p.exists() else None


def summarize_ledger(path: Path, *, max_rows: int = 200_000) -> Dict[str, Any]:
    strat = Counter()
    bro = Counter()
    pnl_by_strat: Dict[str, List[float]] = defaultdict(list)
    rows = 0

    for obj in iter_ndjson(path, max_lines=max_rows):
        r = extract_row(obj)
        strat[r.strategy] += 1
        bro[r.broker] += 1
        rows += 1
        if r.pnl is not None:
            pnl_by_strat[r.strategy].append(float(r.pnl))

    top_strat = strat.most_common(15)
    top_bro = bro.most_common(10)

    # cheap stats per strat
    per = []
    for s, n in top_strat:
        pnls = pnl_by_strat.get(s, [])
        wins = sum(1 for x in pnls if x > 0)
        win_rate = (wins / len(pnls)) if pnls else None
        per.append({
            "strategy": s,
            "rows": n,
            "pnl_samples": len(pnls),
            "win_rate": None if win_rate is None else round(win_rate, 4),
            "total_pnl": round(sum(pnls), 6) if pnls else None,
        })

    return {
        "file": str(path),
        "rows_parsed": rows,
        "top_strategies": per,
        "top_brokers": top_bro,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CHAD Ledger Inspector v2 (reads payload.strategy correctly).")
    ap.add_argument("--root", default=".", help="Repo root (default: .)")
    ap.add_argument("--max-rows", type=int, default=200000, help="Max rows to scan per ledger")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    trades = find_trades_dir(root)

    out: Dict[str, Any] = {
        "ts_utc": utc_now(),
        "root": str(root),
        "trades_dir": str(trades),
        "ledger_newest": None,
        "ledger_enriched": None,
        "newest_summary": None,
        "enriched_summary": None,
        "notes": [
            "If crypto does not show in newest_summary brokers, it is likely logged elsewhere (kraken sub-ledger or pending enrichment).",
            "Use this report as your baseline before applying Beta spam governor and Kraken PnL fixes.",
        ],
    }

    newest = newest_ledger(trades)
    enriched = maybe_enriched(trades)
    out["ledger_newest"] = str(newest) if newest else None
    out["ledger_enriched"] = str(enriched) if enriched else None

    if newest:
        out["newest_summary"] = summarize_ledger(newest, max_rows=args.max_rows)
    if enriched:
        out["enriched_summary"] = summarize_ledger(enriched, max_rows=args.max_rows)

    print(jdump(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
