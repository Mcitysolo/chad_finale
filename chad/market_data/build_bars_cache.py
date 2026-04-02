#!/usr/bin/env python3
"""
chad/market_data/build_bars_cache.py

Build runtime/bars_cache.json from Polygon *tick* NDJSON feeds (local-only).

Input (your file format)
------------------------
NDJSON lines like:
{"timestamp_utc":"2026-02-16T03:42:14.039027+00:00","ticker":"AAPL","price":255.3005,"size":481,...}

We aggregate ticks into OHLCV bars.

Output
------
runtime/bars_cache.json:
{
  "ts_utc": "...",
  "ttl_seconds": 300,
  "source": "polygon_ticks_ndjson",
  "feed_path": "...",
  "bar_seconds": 60,
  "bars_per_symbol": 200,
  "symbols": [...],
  "bars": {
     "AAPL": [
        {"ts_utc":"...","open":..,"high":..,"low":..,"close":..,"volume":..},
        ...
     ]
  }
}

Hard guarantees
---------------
- Local-only, deterministic, bounded.
- Skips malformed rows safely.
- Bounded per symbol via bars_per_symbol.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _parse_ts_utc(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts.strip():
        return None
    s = ts.strip()
    # handle Z
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _floor_dt(dt: datetime, bar_seconds: int) -> datetime:
    # floor to bar_seconds in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    t = int(dt.timestamp())
    f = (t // bar_seconds) * bar_seconds
    return datetime.fromtimestamp(f, tz=timezone.utc)


@dataclass
class _Agg:
    bucket_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def add(self, price: float, size: float) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += float(size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.bucket_start.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }


def _iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def build_cache(*, feed: Path, out: Path, bars_per_symbol: int, bar_seconds: int, ttl_seconds: int) -> Dict[str, Any]:
    bars_per_symbol = int(max(10, min(bars_per_symbol, 5000)))
    bar_seconds = int(max(10, min(bar_seconds, 3600)))
    ttl_seconds = int(max(30, min(ttl_seconds, 24 * 3600)))

    if not feed.is_file():
        raise SystemExit(f"Feed not found: {feed}")

    # symbol -> list of bars (dict)
    bars_out: Dict[str, List[Dict[str, Any]]] = {}

    # streaming aggregation
    current: Dict[str, _Agg] = {}
    completed: Dict[str, List[_Agg]] = {}

    for obj in _iter_ndjson(feed):
        sym = _norm_sym(obj.get("ticker") or obj.get("sym") or obj.get("symbol"))
        if not sym:
            continue

        price = _safe_float(obj.get("price"))
        if price is None or price <= 0:
            continue

        size = _safe_float(obj.get("size"))
        if size is None:
            size = float(_safe_int(obj.get("raw", {}).get("size"))) if isinstance(obj.get("raw"), dict) else 0.0

        ts = _parse_ts_utc(obj.get("timestamp_utc"))
        if ts is None:
            continue

        bucket = _floor_dt(ts, bar_seconds)

        agg = current.get(sym)
        if agg is None:
            current[sym] = _Agg(bucket, price, price, price, price, float(size))
            continue

        # bucket rollover
        if bucket != agg.bucket_start:
            completed.setdefault(sym, []).append(agg)
            current[sym] = _Agg(bucket, price, price, price, price, float(size))
        else:
            agg.add(price, float(size))

    # flush last buckets
    for sym, agg in current.items():
        completed.setdefault(sym, []).append(agg)

    # sort, cap, emit
    symbols: List[str] = []
    for sym, aggs in completed.items():
        aggs_sorted = sorted(aggs, key=lambda a: a.bucket_start)
        if len(aggs_sorted) > bars_per_symbol:
            aggs_sorted = aggs_sorted[-bars_per_symbol:]
        bars_out[sym] = [a.to_dict() for a in aggs_sorted]
        symbols.append(sym)

    symbols = sorted(symbols)

    payload: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": ttl_seconds,
        "source": "polygon_ticks_ndjson",
        "feed_path": str(feed),
        "bar_seconds": bar_seconds,
        "bars_per_symbol": bars_per_symbol,
        "symbols": symbols,
        "bars": bars_out,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, payload)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build runtime/bars_cache.json from Polygon tick NDJSON (local-only).")
    ap.add_argument("--feed", required=True, help="Path to polygon_stocks_YYYYMMDD.ndjson")
    ap.add_argument("--out", default=str(REPO_ROOT / "runtime" / "bars_cache.json"), help="Output JSON path")
    ap.add_argument("--bars-per-symbol", type=int, default=200, help="Max bars stored per symbol")
    ap.add_argument("--bar-seconds", type=int, default=60, help="Bar size in seconds (default 60 = 1m)")
    ap.add_argument("--ttl-seconds", type=int, default=300, help="TTL seconds for freshness metadata")
    args = ap.parse_args(argv)

    payload = build_cache(
        feed=Path(args.feed).resolve(),
        out=Path(args.out).resolve(),
        bars_per_symbol=int(args.bars_per_symbol),
        bar_seconds=int(args.bar_seconds),
        ttl_seconds=int(args.ttl_seconds),
    )

    print(json.dumps({"ok": True, "out": str(Path(args.out).resolve()), "symbols": len(payload.get("symbols", []))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
