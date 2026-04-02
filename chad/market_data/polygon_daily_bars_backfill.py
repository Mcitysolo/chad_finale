#!/usr/bin/env python3
"""
chad/market_data/polygon_daily_bars_backfill.py

Polygon Daily OHLCV backfill + refresh for CHAD.

This script writes daily bars to:
  data/bars/1d/<SYMBOL>.json

Critical design guarantees
- No broker calls. No trading. Read-only Polygon market data.
- Atomic write + fsync.
- Deterministic output ordering (ascending by timestamp).
- Works with polygon-api-client return shapes:
    * list of Agg objects
    * list of dicts
    * dict with "results" list (future-proof)
- No secrets printed.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from polygon import RESTClient  # polygon-api-client

DEFAULT_DAYS_BACK = 400
DEFAULT_TTL_SECONDS = 86400
MAX_BARS_HARD_CAP = 2000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _safe_symbol(sym: str) -> str:
    return str(sym).strip().upper()


def _to_iso_utc_from_ms(epoch_ms: int) -> str:
    dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_universe(repo_root: Path) -> List[str]:
    ufile = repo_root / "config" / "universe.json"
    if ufile.is_file():
        raw = json.loads(ufile.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("symbols"), list):
            return [_safe_symbol(x) for x in raw["symbols"] if str(x).strip()]
        if isinstance(raw, list):
            return [_safe_symbol(x) for x in raw if str(x).strip()]

    from chad.utils.universe_provider import get_trade_universe
    return [_safe_symbol(x) for x in get_trade_universe()]


@dataclass(frozen=True)
class BackfillConfig:
    repo_root: Path
    out_dir: Path
    days_back: int
    ttl_seconds: int
    max_bars: int


class PolygonDailyBarsBackfill:
    def __init__(self, cfg: BackfillConfig) -> None:
        self._cfg = cfg
        api_key = _require_env("POLYGON_API_KEY")
        self._client = RESTClient(api_key)

    def _coerce_results_list(self, aggs: Any) -> List[Any]:
        # polygon-api-client sometimes returns list; future-proof dict with "results"
        if isinstance(aggs, list):
            return aggs
        if isinstance(aggs, dict):
            res = aggs.get("results")
            return res if isinstance(res, list) else []
        # iterable/generator fallback
        try:
            return list(aggs)
        except Exception:
            return []

    def _read_field(self, obj: Any, *names: str, default: Any = None) -> Any:
        # Try attribute first then dict keys
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
            if isinstance(obj, dict) and n in obj:
                v = obj.get(n)
                if v is not None:
                    return v
        return default

    def _fetch_symbol(self, symbol: str, start: date, end: date) -> List[Dict[str, Any]]:
        sym = _safe_symbol(symbol)

        aggs = self._client.get_aggs(
            ticker=sym,
            multiplier=1,
            timespan="day",
            from_=start.isoformat(),
            to=end.isoformat(),
            adjusted=True,
            sort="asc",
            limit=50000,
        )

        rows = self._coerce_results_list(aggs)
        bars: List[Dict[str, Any]] = []

        for a in rows:
            # Handle both (t,o,h,l,c,v) and (timestamp,open,high,low,close,volume)
            ts = self._read_field(a, "t", "timestamp", default=0)
            o = self._read_field(a, "o", "open", default=0.0)
            h = self._read_field(a, "h", "high", default=0.0)
            l = self._read_field(a, "l", "low", default=0.0)
            c = self._read_field(a, "c", "close", default=0.0)
            v = self._read_field(a, "v", "volume", default=0.0)

            try:
                ts_i = int(ts or 0)
                o_f = float(o or 0.0)
                h_f = float(h or 0.0)
                l_f = float(l or 0.0)
                c_f = float(c or 0.0)
                v_f = float(v or 0.0)
            except Exception:
                continue

            if ts_i <= 0 or o_f <= 0 or h_f <= 0 or l_f <= 0 or c_f <= 0:
                continue
            if h_f < max(o_f, c_f, l_f) or l_f > min(o_f, c_f, h_f):
                continue
            if v_f < 0:
                v_f = 0.0

            bars.append(
                {
                    "ts_utc": _to_iso_utc_from_ms(ts_i),
                    "open": o_f,
                    "high": h_f,
                    "low": l_f,
                    "close": c_f,
                    "volume": v_f,
                }
            )

        # Enforce deterministic sort and hard cap
        bars.sort(key=lambda x: x["ts_utc"])
        if len(bars) > self._cfg.max_bars:
            bars = bars[-self._cfg.max_bars :]

        return bars

    def run(self, symbols: Sequence[str]) -> Dict[str, Any]:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=int(self._cfg.days_back))
        end = today

        ok = 0
        fail = 0
        errors: Dict[str, str] = {}

        for sym in symbols:
            sym = _safe_symbol(sym)
            if not sym:
                continue
            try:
                bars = self._fetch_symbol(sym, start=start, end=end)
                payload: Dict[str, Any] = {
                    "symbol": sym,
                    "timeframe": "1d",
                    "source": "polygon",
                    "ts_utc": _utc_now_iso(),
                    "ttl_seconds": int(self._cfg.ttl_seconds),
                    "bars": bars,
                }
                out_path = self._cfg.out_dir / f"{sym}.json"
                _atomic_write_json(out_path, payload)
                ok += 1
            except Exception as exc:
                fail += 1
                errors[sym] = f"{type(exc).__name__}: {exc}"

        return {
            "ts_utc": _utc_now_iso(),
            "days_back": int(self._cfg.days_back),
            "symbols_requested": len(list(symbols)),
            "ok": ok,
            "fail": fail,
            "out_dir": str(self._cfg.out_dir),
            "errors": errors,
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Polygon Daily Bars Backfill (1d OHLCV).")
    parser.add_argument("--repo-root", default="", help="Repo root (default auto).")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Days of history to fetch.")
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS, help="TTL for bars files.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols override.")
    parser.add_argument("--max-bars", type=int, default=MAX_BARS_HARD_CAP, help="Hard cap bars stored per symbol.")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "bars" / "1d"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols.strip():
        symbols = [_safe_symbol(x) for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = _load_universe(root)

    cfg = BackfillConfig(
        repo_root=root,
        out_dir=out_dir,
        days_back=int(args.days_back),
        ttl_seconds=int(args.ttl_seconds),
        max_bars=int(args.max_bars),
    )

    backfill = PolygonDailyBarsBackfill(cfg)
    summary = backfill.run(symbols)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
