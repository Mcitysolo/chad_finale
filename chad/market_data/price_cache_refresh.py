#!/usr/bin/env python3
"""
CHAD — Price Cache Refresher (FINAL, PRODUCTION)

Writes runtime/price_cache.json from the latest Polygon NDJSON feed.

Production rules:
- Trade events (ev="T"): use trade price field `p`
- Quote events (ev="Q"): use midpoint from `bp` and `ap`
- Never treat generic fields like `c` as a price
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from chad.execution.ibkr_client_ids import (
    PRICE_CACHE_REFRESH as _IBKR_PRICE_CACHE_REFRESH_CLIENT_ID,
)


DEFAULT_TTL_SECONDS = 300
DEFAULT_TAIL_LINES = 20_000

SYMBOL_KEYS = ("symbol", "sym", "ticker", "S")
TS_KEYS = (
    "ts_utc",
    "timestamp",
    "ts",
    "t",
    "sip_timestamp",
    "participant_timestamp",
    "last_timestamp",
)


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


def tail_lines(path: Path, max_lines: int) -> Iterable[str]:
    from collections import deque
    buf: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            buf.append(line)
    return buf


@dataclass(frozen=True)
class CacheStats:
    symbols_written: int
    bad_json_lines: int
    source_feed: str


def extract_polygon_price(rec: Dict[str, Any]) -> Optional[float]:
    """
    Event-aware Polygon price extraction.

    Rules:
    - ev="T": use trade price `p`
    - ev="Q": use midpoint of bid/ask if both exist and are valid
    - otherwise: ignore record
    """
    ev = str(rec.get("ev") or "").strip().upper()

    if ev == "T":
        return safe_float(rec.get("p"))

    if ev == "Q":
        bp = safe_float(rec.get("bp"))
        ap = safe_float(rec.get("ap"))
        if bp is not None and ap is not None and ap >= bp:
            return float((bp + ap) / 2.0)
        return None

    return None


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

        px = extract_polygon_price(rec)
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
    candidates = []
    modern_dir = feed_dir / "polygon_stocks"
    if modern_dir.is_dir():
        candidates += list(modern_dir.glob("*.ndjson"))
    candidates += list(feed_dir.glob("polygon_stocks_*.ndjson"))

    candidates = sorted(
        [c for c in candidates if c.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        try:
            if c.stat().st_size > 0:
                return c
        except Exception:
            continue
    raise FileNotFoundError(f"No non-empty Polygon NDJSON feeds found under {feed_dir}")


def _get_provider() -> str:
    """Read CHAD_MARKET_DATA_PROVIDER env var. Default: ibkr."""
    return os.environ.get("CHAD_MARKET_DATA_PROVIDER", "ibkr").strip().lower()


def _load_universe() -> list:
    """Load equity/ETF symbol universe from config/universe.json."""
    universe_path = Path("/home/ubuntu/chad_finale/config/universe.json")
    try:
        obj = json.loads(universe_path.read_text(encoding="utf-8"))
        syms = obj.get("symbols", [])
        return [str(s).strip().upper() for s in syms if str(s).strip()]
    except Exception:
        return ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "BAC", "GLD", "SH", "IEMG", "VWO"]


def _load_futures_universe() -> list:
    """Load futures symbols from config/universe.json 'futures' key."""
    universe_path = Path("/home/ubuntu/chad_finale/config/universe.json")
    try:
        obj = json.loads(universe_path.read_text(encoding="utf-8"))
        futures = obj.get("futures", [])
        return [str(f["symbol"]).strip().upper() for f in futures if isinstance(f, dict) and f.get("symbol")]
    except Exception:
        return []


def _refresh_ibkr(runtime_dir: Path, ttl_seconds: int) -> int:
    """Refresh price_cache.json using IBKR snapshots (equities + futures)."""
    from ib_async import IB
    from chad.market_data.ibkr_price_provider import IBKRPriceProvider

    out_path = runtime_dir / "price_cache.json"
    feed_state_path = runtime_dir / "feed_state.json"
    equity_symbols = _load_universe()
    futures_symbols = _load_futures_universe()

    ib = IB()
    try:
        ib.connect("127.0.0.1", 4002, clientId=_IBKR_PRICE_CACHE_REFRESH_CLIENT_ID, timeout=15)
        ib.reqMarketDataType(4)  # delayed-frozen for weekend/off-hours
        provider = IBKRPriceProvider(ib)

        prices: Dict[str, float] = {}

        # Equities/ETFs
        snapshots = provider.get_batch_snapshots(equity_symbols, sec_type="STK")
        for sym, snap in snapshots.items():
            px = snap.last if snap.last > 0 else snap.close
            if px > 0:
                prices[sym] = px

        # Futures
        if futures_symbols:
            fut_snapshots = provider.get_batch_snapshots(futures_symbols, sec_type="FUT")
            for sym, snap in fut_snapshots.items():
                px = snap.last if snap.last > 0 else snap.close
                if px > 0:
                    prices[sym] = px

        # --- VIX injection (CBOE Index — not fetchable via IBKR STK/FUT path) ---
        # Read the latest VIX close from the daily bar file maintained by
        # chad-ibkr-bar-provider.service. This satisfies omega_momentum_options
        # and omega_vol which require ctx.prices['VIX'].
        # NOTE: data/bars/1d/VIX.json wraps bars in {"bars": [...]}; we handle
        # both that shape and a bare list defensively.
        try:
            _vix_bar_path = Path("/home/ubuntu/chad_finale/data/bars/1d/VIX.json")
            if _vix_bar_path.exists():
                _vix_doc = json.loads(_vix_bar_path.read_text(encoding="utf-8"))
                _vix_bars = _vix_doc.get("bars", []) if isinstance(_vix_doc, dict) else _vix_doc
                if isinstance(_vix_bars, list) and _vix_bars:
                    _vix_close = float(_vix_bars[-1].get("close", 0.0))
                    if _vix_close > 0:
                        prices["VIX"] = _vix_close
        except Exception:
            pass  # VIX injection is best-effort; never block the price cache write
        # --- end VIX injection ---

        payload = {
            "prices": dict(sorted(prices.items())),
            "ts_utc": utc_now_iso(),
            "ttl_seconds": int(ttl_seconds),
        }
        atomic_write_json(out_path, payload)

        # Write feed_state.json with ibkr_stocks key
        feed_state = {
            "feeds": {
                "ibkr_stocks": {
                    "freshness_seconds": 0.0,
                    "last_update_ts_utc": utc_now_iso(),
                }
            },
            "ts_utc": utc_now_iso(),
            "ttl_seconds": 180,
        }
        atomic_write_json(feed_state_path, feed_state)

        print(f"[price_cache_refresh] IBKR wrote {out_path}")
        print(f"  symbols_written={len(prices)}")
        print(f"  spy_price={prices.get('SPY')}")
        print(f"  futures={[s for s in futures_symbols if s in prices]}")
        return 0

    except Exception as exc:
        print(f"[price_cache_refresh] IBKR error: {exc}")
        return 1
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh runtime/price_cache.json")
    ap.add_argument("--feed-dir", default="/home/ubuntu/chad_finale/data/feeds")
    ap.add_argument("--runtime-dir", default="/home/ubuntu/chad_finale/runtime")
    ap.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    ap.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    ap.add_argument("--provider", default=None, help="Force provider: ibkr or polygon")
    args = ap.parse_args()

    runtime_dir = Path(args.runtime_dir).resolve()
    provider = args.provider or _get_provider()

    if provider == "ibkr":
        return _refresh_ibkr(runtime_dir, args.ttl_seconds)

    # Legacy Polygon path
    feed_dir = Path(args.feed_dir).resolve()
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
    print(f"  spy_price={payload.get('prices', {}).get('SPY')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
