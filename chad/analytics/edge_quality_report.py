from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -------------------------
# Helpers
# -------------------------

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


def read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    # Defensive NDJSON reader (skips bad lines, never throws)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] not in "{[":
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except Exception:
                    continue
    except FileNotFoundError:
        return
    except Exception:
        return


def find_runtime_dir(explicit: str) -> Path:
    if explicit.strip():
        return Path(explicit).expanduser().resolve()
    env = os.getenv("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # derive repo root from this file location: chad/analytics/edge_quality_report.py -> repo root is parents[2]
    here = Path(__file__).resolve()
    repo = here.parents[2]
    return (repo / "runtime").resolve()


@dataclass(frozen=True)
class TradeRow:
    ts_utc: str
    strategy: str
    pnl: float


def extract_trade_rows(runtime: Path) -> Tuple[List[TradeRow], Dict[str, Any]]:
    """
    Best-effort extraction from whatever artifacts exist.
    Priority:
      1) trade_history_*.ndjson / decision_trace*.ndjson style logs (if present)
      2) ledger-like json files (if present)
      3) scr_state.json aggregate stats (fallback summary only)

    This is intentionally robust and never fails the report.
    """
    rows: List[TradeRow] = []

    # Heuristic: search common log patterns
    candidates: List[Path] = []
    for pat in ("*trade_history*.ndjson", "*decision_trace*.ndjson", "*trades*.ndjson"):
        candidates.extend(sorted(runtime.glob(pat))[-10:])  # last 10

    for p in candidates:
        for obj in iter_ndjson(p):
            # Try common keys
            strategy = str(obj.get("strategy") or obj.get("brain") or obj.get("module") or "").strip() or "unknown"
            ts = str(obj.get("ts_utc") or obj.get("ts") or obj.get("timestamp") or "").strip() or utc_now()

            # pnl might be direct
            pnl = None
            if "pnl" in obj:
                pnl = safe_float(obj.get("pnl"), default=0.0)
            elif "realized_pnl" in obj:
                pnl = safe_float(obj.get("realized_pnl"), default=0.0)
            elif "result" in obj and isinstance(obj["result"], dict) and "pnl" in obj["result"]:
                pnl = safe_float(obj["result"].get("pnl"), default=0.0)

            if pnl is None:
                continue
            rows.append(TradeRow(ts_utc=ts, strategy=strategy, pnl=float(pnl)))

    scr = read_json(runtime / "scr_state.json")
    return rows, scr


def sharpe_like(pnls: List[float]) -> float:
    # Simple robustness: mean / stddev with guardrails (not a “finance sharpe”; just a consistency signal)
    if len(pnls) < 5:
        return 0.0
    mu = sum(pnls) / len(pnls)
    var = sum((x - mu) ** 2 for x in pnls) / max(1, (len(pnls) - 1))
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    return float(mu / sd)


def summarize(rows: List[TradeRow], scr: Dict[str, Any]) -> Dict[str, Any]:
    by_strategy: Dict[str, List[float]] = {}
    for r in rows:
        by_strategy.setdefault(r.strategy, []).append(r.pnl)

    strat_stats: List[Dict[str, Any]] = []
    for s, pnls in sorted(by_strategy.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        wins = sum(1 for x in pnls if x > 0)
        total = len(pnls)
        wr = float(wins / total) if total else 0.0
        strat_stats.append({
            "strategy": s,
            "trades": total,
            "win_rate": round(wr, 4),
            "total_pnl": round(sum(pnls), 4),
            "avg_pnl": round((sum(pnls) / total) if total else 0.0, 6),
            "sharpe_like": round(sharpe_like(pnls), 6),
        })

    # SCR fields: keep it simple and stable
    scr_summary: Dict[str, Any] = {}
    if scr:
        scr_summary = {
            "scr_state": scr.get("state"),
            "paper_only": scr.get("paper_only"),
            "sizing_factor": scr.get("sizing_factor"),
            "reasons": scr.get("reasons"),
            "total_trades": scr.get("total_trades"),
            "effective_trades": scr.get("effective_trades"),
            "win_rate": scr.get("win_rate"),
            "sharpe_like": scr.get("sharpe_like"),
            "max_drawdown": scr.get("max_drawdown"),
            "total_pnl": scr.get("total_pnl"),
        }

    return {
        "ts_utc": utc_now(),
        "runtime_dir": str(find_runtime_dir("")),
        "observed_trade_rows": len(rows),
        "per_strategy": strat_stats,
        "scr": scr_summary,
        "notes": [
            "This report is read-only. It never places trades.",
            "If observed_trade_rows is low, your trade logs may not store per-trade pnl yet; SCR summary still reports global performance.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CHAD Edge Quality Report (paper-first, read-only).")
    ap.add_argument("--runtime-dir", default="", help="Override runtime dir (or use CHAD_RUNTIME_DIR).")
    ap.add_argument("--out", default="", help="Write JSON report to this path.")
    args = ap.parse_args()

    runtime = find_runtime_dir(args.runtime_dir)
    rows, scr = extract_trade_rows(runtime)
    rep = summarize(rows, scr)

    out = Path(args.out).expanduser().resolve() if str(args.out).strip() else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jdump(rep) + "\n", encoding="utf-8")

    print(jdump(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
