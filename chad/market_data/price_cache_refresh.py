#!/usr/bin/env python3
"""
CHAD — Price Cache Refresher (FINAL, PRODUCTION)

Writes runtime/price_cache.json from the latest Polygon NDJSON feed.

This is the missing producer in the system:
- Polygon streamer writes NDJSON feeds
- ContextBuilder reads runtime/price_cache.json
- BUT nothing was updating price_cache.json

This script is now the single source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


# ============================ CONSTANTS ============================

DEFAULT_TTL_SECONDS = 300
DEFAULT_TAIL_LINES = 20_000

SYMBOL_KEYS = ("symbol", "sym", "ticker", "S")
PRICE_KEYS = ("price", "p", "last_price", "last", "close", "c")
TS_KEYS = (
    "ts_utc",
    "timestamp",
    "ts",
    "t",
    "sip_timestamp",
    "participant_timestamp",
    "last_timestamp",
)


# ============================ UTILITIES ============================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v > 0.0 and v == v and v not in (float("inf"), float("-inf")):
            return v
        return None
    except Exception:
        return None


def normalize_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def extract_first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def parse_ts(val: Any) -> Optional[str]:
    if val is None:
        return None

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if s.endswith("+00:00"):
            s = s.replace("+00:00", "Z")
        return s

    if isinstance(val, (int, float)):
        n = float(val)
        if n > 1e17:
            sec = n / 1e9
        elif n > 1e14:
            sec = n / 1e6
        elif n > 1e11:
            sec = n / 1e3
        else:
            sec = n
        try:
            dt = datetime.fromtimestamp(sec, tz=timezone.utc)
            return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        except Exception:
            return None

    return None


# ============================ FEED READER ============================

def tail_lines(path: Path, max_lines: int) -> Iterable[str]:
    buf: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            buf.append(line)
            if len(buf) > max_lines:
                buf.pop(0)
    return buf


@dataclass(frozen=True)
class CacheStats:
    symbols_written: int
    bad_json_lines: int
    source_feed: str


def build_price_cache(
    feed_path: Path,
    *,
    tail_lines_n: int,
    ttl_seconds: int,
) -> Tuple[Dict[str, Any], CacheStats]:

    latest_price: Dict[str, float] = {}
    latest_ts: Dict[str, str] = {}
    bad = 0

    for line in tail_lines(feed_path, tail_lines_n):
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except Exception:
            bad += 1
            continue
        if not isinstance(rec, dict):
            continue

        sym = normalize_symbol(extract_first(rec, SYMBOL_KEYS))
        if not sym:
            continue

        px = safe_float(extract_first(rec, PRICE_KEYS))
        if px is None:
            continue

        ts = parse_ts(extract_first(rec, TS_KEYS))

        if ts:
            prev = latest_ts.get(sym)
            if prev is None or ts >= prev:
                latest_ts[sym] = ts
                latest_price[sym] = px
        else:
            latest_price[sym] = px

    payload = {
        "prices": dict(sorted(latest_price.items())),
        "ts_utc": utc_now_iso(),
        "ttl_seconds": int(ttl_seconds),
    }

    stats = CacheStats(
        symbols_written=len(latest_price),
        bad_json_lines=int(bad),
        source_feed=str(feed_path),
    )
    return payload, stats


def find_latest_feed(feed_dir: Path) -> Path:
    feeds = sorted(feed_dir.glob("polygon_stocks_*.ndjson"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not feeds:
        raise FileNotFoundError(f"No polygon_stocks_*.ndjson in {feed_dir}")
    return feeds[0]


# ============================ CLI ============================

def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh runtime/price_cache.json from Polygon NDJSON feed")
    ap.add_argument("--feed-dir", default="/home/ubuntu/CHAD FINALE/data/feeds")
    ap.add_argument("--runtime-dir", default="/home/ubuntu/CHAD FINALE/runtime")
    ap.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    ap.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    args = ap.parse_args()

    feed_dir = Path(args.feed_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    out_path = runtime_dir / "price_cache.json"

    feed = find_latest_feed(feed_dir)
    payload, stats = build_price_cache(
        feed,
        tail_lines_n=args.tail_lines,
        ttl_seconds=args.ttl_seconds,
    )

    atomic_write_json(out_path, payload)

    print(f"[price_cache_refresh] wrote {out_path}")
    print(f"  source_feed={stats.source_feed}")
    print(f"  symbols_written={stats.symbols_written}")
    print(f"  bad_json_lines={stats.bad_json_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
