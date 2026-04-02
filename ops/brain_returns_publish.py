#!/usr/bin/env python3
"""
ops/brain_returns_publish.py

PHASE 11.5 — Brain Returns Publisher (Correlation + Vol + Kelly Foundation)
---------------------------------------------------------------------------
This publisher builds a *shared time-series grid* of per-brain returns so that:

- Correlation can be computed (requires overlapping timestamps).
- Volatility targeting can be computed (per-brain return std dev).
- Kelly constraints can be applied safely (requires mean/variance estimates).
- Regime-sensitive alloc can use aligned time windows.

Key idea
--------
Trade logs are *sparse* (brains trade at different minutes). If you only compute correlation
on "trade minutes", overlap can be zero. This publisher constructs:

1) ALL-MINUTES series: every minute in a chosen window exists; missing minutes = 0 return
2) ACTIVE-MINUTES series: only minutes where at least one brain has a trade/return

Outputs (atomic, SSOT-safe)
---------------------------
- runtime/intel_cache/brain_returns_state.json
- runtime/intel_cache/brain_returns_1m_all.ndjson
- runtime/intel_cache/brain_returns_1m_active.ndjson

Data source
-----------
- data/trades/trade_history_*.ndjson (paper only; is_live=false)

Return definition (scale-free)
------------------------------
return = pnl / notional   (where pnl and notional are finite; notional > 0)

Timestamps
----------
Minute bucket uses (in priority order):
- envelope: record["timestamp_utc"]
- payload:  payload["exit_time_utc"]
- payload:  payload["entry_time_utc"]

Safety guarantees
-----------------
- No broker calls. No secrets.
- Deterministic. Same inputs -> same outputs.
- Fail-closed: if parsing fails, write state with error and exit non-zero.
- Atomic writes + fsync(file) + fsync(dir) best-effort.
- Bounds: caps the minute window to avoid runaway output.

Env knobs
---------
- CHAD_ROOT (optional)                     default: /home/ubuntu/chad_finale
- CHAD_RUNTIME_DIR (optional)              default: <CHAD_ROOT>/runtime
- CHAD_TRADES_DIR (optional)               default: <CHAD_ROOT>/data/trades
- CHAD_BRAIN_RET_LOOKBACK_DAYS (int)       default: 45
- CHAD_BRAIN_RET_MAX_MINUTES (int)         default: 100000   (~69 days)
- CHAD_BRAIN_RET_EXCLUDE (csv)             default: "manual,unknown"
- CHAD_BRAIN_RET_MIN_TRADES (int)          default: 30       (strategy must have >= this many usable returns)
- CHAD_BRAIN_RET_TTL_SECONDS (int)         default: 300
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Env + Paths
# -----------------------------

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        x = int(str(v).strip())
        return x
    except Exception:
        return default


CHAD_ROOT = Path(_env_str("CHAD_ROOT", "/home/ubuntu/chad_finale")).resolve()
RUNTIME_DIR = Path(_env_str("CHAD_RUNTIME_DIR", str(CHAD_ROOT / "runtime"))).resolve()
TRADES_DIR = Path(_env_str("CHAD_TRADES_DIR", str(CHAD_ROOT / "data" / "trades"))).resolve()
INTEL_CACHE_DIR = RUNTIME_DIR / "intel_cache"

LOOKBACK_DAYS = max(0, _env_int("CHAD_BRAIN_RET_LOOKBACK_DAYS", 45))
MAX_MINUTES = max(60, _env_int("CHAD_BRAIN_RET_MAX_MINUTES", 100000))
MIN_TRADES = max(1, _env_int("CHAD_BRAIN_RET_MIN_TRADES", 30))
TTL_SECONDS = max(30, _env_int("CHAD_BRAIN_RET_TTL_SECONDS", 300))

EXCLUDE = set(
    s.strip().lower()
    for s in _env_str("CHAD_BRAIN_RET_EXCLUDE", "manual,unknown").split(",")
    if s.strip()
)

OUT_STATE = INTEL_CACHE_DIR / "brain_returns_state.json"
OUT_ALL = INTEL_CACHE_DIR / "brain_returns_1m_all.ndjson"
OUT_ACTIVE = INTEL_CACHE_DIR / "brain_returns_1m_active.ndjson"


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # fsync directory (best-effort)
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def atomic_write_ndjson(path: Path, lines: Iterable[Dict[str, Any]]) -> None:
    # Stream lines into memory buffer in a controlled way (safe for our minute caps)
    out: List[str] = []
    for obj in lines:
        out.append(json.dumps(obj, sort_keys=True))
    data = ("\n".join(out) + "\n").encode("utf-8") if out else b""
    _atomic_write_bytes(path, data)


def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _parse_iso_any(s: str) -> Optional[datetime]:
    """
    Parse ISO timestamps including:
    - 2026-01-08T20:17:56.359226+00:00
    - 2026-02-12T18:36:05Z
    """
    try:
        ss = s.strip()
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _minute_bucket(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _minute_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:00Z")

def _iter_minutes_inclusive(start: datetime, end: datetime, *, cap_minutes: int) -> List[datetime]:
    """
    Return minute buckets from start..end inclusive (UTC), capped.

    Used for paper_sim smoothing: distribute a trade's return over its holding window.
    Fail-safe: if end < start or range is huge, cap aggressively.
    """
    s = _minute_bucket(start)
    e = _minute_bucket(end)
    if e < s:
        return [s]
    span = int((e - s).total_seconds() // 60) + 1
    if span <= 1:
        return [s]
    if span > cap_minutes:
        # Cap by truncating to the most recent cap window
        s = e - timedelta(minutes=cap_minutes - 1)
        span = cap_minutes
    out: List[datetime] = []
    cur = s
    for _ in range(span):
        out.append(cur)
        cur += timedelta(minutes=1)
    return out

def _trade_files_in_window(trades_dir: Path, lookback_days: int) -> List[Path]:
    files = sorted(trades_dir.glob("trade_history_*.ndjson"))
    if not files:
        return []
    if lookback_days <= 0:
        return files
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    keep: List[Path] = []
    for f in files:
        # trade_history_YYYYMMDD.ndjson
        name = f.name
        try:
            ymd = name.split("_", 2)[2].split(".", 1)[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            if dt >= cutoff:
                keep.append(f)
        except Exception:
            # If filename doesn't parse, keep it (fail-safe).
            keep.append(f)
    return keep


@dataclass
class Accum:
    # per minute: strat -> return sum
    minute_strat_ret: Dict[datetime, Dict[str, float]]
    # per minute: whether any nonzero return exists
    minute_any: Dict[datetime, bool]
    # per strategy stats
    strat_trades: Dict[str, int]
    strat_usable: Dict[str, int]
    strat_pnl_sum: Dict[str, float]
    strat_notional_sum: Dict[str, float]


def _init_accum() -> Accum:
    return Accum(
        minute_strat_ret={},
        minute_any={},
        strat_trades={},
        strat_usable={},
        strat_pnl_sum={},
        strat_notional_sum={},
    )


def _select_timestamp(rec: Dict[str, Any], payload: Dict[str, Any]) -> Optional[datetime]:
    for key, src in (
        ("timestamp_utc", rec),
        ("exit_time_utc", payload),
        ("entry_time_utc", payload),
    ):
        v = src.get(key)
        if isinstance(v, str) and v.strip():
            dt = _parse_iso_any(v)
            if dt is not None:
                return dt
    return None


def _scan_trades(files: List[Path]) -> Tuple[Accum, Optional[str]]:
    acc = _init_accum()
    min_minute: Optional[datetime] = None
    max_minute: Optional[datetime] = None

    rows = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _safe_json_loads(line)
                    if not rec:
                        continue
                    payload = rec.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("is_live") is True:
                        continue

                    strat = str(payload.get("strategy") or "unknown").strip().lower() or "unknown"
                    if strat in EXCLUDE:
                        continue

                    acc.strat_trades[strat] = acc.strat_trades.get(strat, 0) + 1

                    dt = _select_timestamp(rec, payload)
                    if dt is None:
                        continue
                    m = _minute_bucket(dt)

                    if min_minute is None or m < min_minute:
                        min_minute = m
                    if max_minute is None or m > max_minute:
                        max_minute = m

                    pnl = _finite_float(payload.get("pnl"))
                    notional = _finite_float(payload.get("notional"))
                    if notional is None:
                        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
                        notional = _finite_float(extra.get("notional_used"))

                    if pnl is None or notional is None or notional <= 0:
                        rows += 1
                        continue

                    r = pnl / notional
                    if not math.isfinite(r):
                        rows += 1
                        continue

                    # stats
                    acc.strat_usable[strat] = acc.strat_usable.get(strat, 0) + 1
                    acc.strat_pnl_sum[strat] = acc.strat_pnl_sum.get(strat, 0.0) + pnl
                    acc.strat_notional_sum[strat] = acc.strat_notional_sum.get(strat, 0.0) + notional
                    # per-minute accumulation
                    tags = payload.get("tags", []) or []
                    is_paper_sim = (
                        isinstance(tags, list)
                        and ("paper_sim" in [str(x).strip().lower() for x in tags])
                    )

                    if is_paper_sim:
                        # Smooth the trade return across its holding window using entry/exit timestamps.
                        # Avoid single-minute spikes that inflate volatility and corrupt correlation/Kelly.
                        et = payload.get("entry_time_utc")
                        xt = payload.get("exit_time_utc")
                        dt_entry = _parse_iso_any(et) if isinstance(et, str) else None
                        dt_exit = _parse_iso_any(xt) if isinstance(xt, str) else None

                        if dt_entry is not None and dt_exit is not None:
                            minutes = _iter_minutes_inclusive(
                                dt_entry, dt_exit, cap_minutes=MAX_MINUTES
                            )
                            per_min_ret = float(r) / float(len(minutes)) if minutes else float(r)
                            for mm in minutes:
                                acc.minute_strat_ret.setdefault(mm, {})
                                acc.minute_strat_ret[mm][strat] = acc.minute_strat_ret[mm].get(strat, 0.0) + per_min_ret
                                acc.minute_any[mm] = acc.minute_any.get(mm, False) or (abs(per_min_ret) > 0)
                        else:
                            # If timestamps missing, fall back to single-minute behavior.
                            acc.minute_strat_ret.setdefault(m, {})
                            acc.minute_strat_ret[m][strat] = acc.minute_strat_ret[m].get(strat, 0.0) + float(r)
                            acc.minute_any[m] = acc.minute_any.get(m, False) or (abs(r) > 0)
                    else:
                        # Normal behavior for real paper trades: treat return as occurring at the bucket minute.
                        acc.minute_strat_ret.setdefault(m, {})
                        acc.minute_strat_ret[m][strat] = acc.minute_strat_ret[m].get(strat, 0.0) + float(r)
                        acc.minute_any[m] = acc.minute_any.get(m, False) or (abs(r) > 0)

                    rows += 1
        except Exception as exc:
            return acc, f"file_read_error:{fp}:{exc}"

    if min_minute is None or max_minute is None:
        return acc, "no_valid_minutes_found"

    # Store min/max in acc via synthetic entries (handled outside)
    acc.minute_strat_ret.setdefault(min_minute, acc.minute_strat_ret.get(min_minute, {}))
    acc.minute_strat_ret.setdefault(max_minute, acc.minute_strat_ret.get(max_minute, {}))

    return acc, None


def _strategies_for_output(acc: Accum) -> List[str]:
    # Only include strategies with enough usable return samples
    items = []
    for strat, usable in acc.strat_usable.items():
        if usable >= MIN_TRADES:
            items.append(strat)
    # Deterministic ordering
    return sorted(set(items))


def _bounded_minute_range(min_m: datetime, max_m: datetime) -> Tuple[datetime, datetime, int]:
    span = int((max_m - min_m).total_seconds() // 60) + 1
    if span <= MAX_MINUTES:
        return min_m, max_m, span
    # Cap by taking the most recent MAX_MINUTES window (keeps relevance)
    capped_min = max_m - timedelta(minutes=MAX_MINUTES - 1)
    return capped_min, max_m, MAX_MINUTES


def _emit_all_minutes(
    *,
    min_m: datetime,
    max_m: datetime,
    strategies: List[str],
    per_minute: Dict[datetime, Dict[str, float]],
) -> Iterable[Dict[str, Any]]:
    cur = min_m
    while cur <= max_m:
        row_ret = per_minute.get(cur, {})
        returns = {s: round(float(row_ret.get(s, 0.0)), 12) for s in strategies}
        yield {
            "schema_version": "brain_returns_1m.v1",
            "ts_utc": _minute_iso(cur),
            "returns": returns,
        }
        cur += timedelta(minutes=1)


def _emit_active_minutes(
    *,
    min_m: datetime,
    max_m: datetime,
    strategies: List[str],
    per_minute: Dict[datetime, Dict[str, float]],
    active_map: Dict[datetime, bool],
) -> Iterable[Dict[str, Any]]:
    cur = min_m
    while cur <= max_m:
        if active_map.get(cur, False) or (cur in per_minute and any(abs(per_minute[cur].get(s, 0.0)) > 0 for s in strategies)):
            row_ret = per_minute.get(cur, {})
            returns = {s: round(float(row_ret.get(s, 0.0)), 12) for s in strategies}
            yield {
                "schema_version": "brain_returns_1m.v1",
                "ts_utc": _minute_iso(cur),
                "returns": returns,
            }
        cur += timedelta(minutes=1)


def publish() -> int:
    INTEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not TRADES_DIR.exists():
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": f"trades_dir_missing:{TRADES_DIR}",
            },
        )
        return 2

    files = _trade_files_in_window(TRADES_DIR, LOOKBACK_DAYS)
    if not files:
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": "no_trade_history_files_found",
                "trades_dir": str(TRADES_DIR),
            },
        )
        return 2

    acc, err = _scan_trades(files)
    if err is not None:
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": err,
                "files": [str(p) for p in files],
            },
        )
        return 2

    # Determine min/max minute from keys
    minutes = sorted(acc.minute_strat_ret.keys())
    if not minutes:
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": "no_minutes_after_scan",
            },
        )
        return 2

    raw_min = minutes[0]
    raw_max = minutes[-1]
    min_m, max_m, span = _bounded_minute_range(raw_min, raw_max)

    strategies = _strategies_for_output(acc)
    if not strategies:
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": f"no_strategies_with_min_trades>={MIN_TRADES}",
                "observed_strategies": sorted(set(acc.strat_trades.keys())),
                "min_trades": MIN_TRADES,
                "usable_by_strategy": dict(sorted(acc.strat_usable.items())),
            },
        )
        return 2

    # Produce outputs
    all_lines = _emit_all_minutes(
        min_m=min_m,
        max_m=max_m,
        strategies=strategies,
        per_minute=acc.minute_strat_ret,
    )
    active_lines = _emit_active_minutes(
        min_m=min_m,
        max_m=max_m,
        strategies=strategies,
        per_minute=acc.minute_strat_ret,
        active_map=acc.minute_any,
    )

    atomic_write_ndjson(OUT_ALL, all_lines)
    atomic_write_ndjson(OUT_ACTIVE, active_lines)

    state = {
        "schema_version": "brain_returns_state.v1",
        "ts_utc": utc_now_iso(),
        "ttl_seconds": TTL_SECONDS,
        "status": "OK",
        "lookback_days": LOOKBACK_DAYS,
        "max_minutes_cap": MAX_MINUTES,
        "min_trades_required": MIN_TRADES,
        "files_used": [p.name for p in files],
        "window": {
            "raw_min_minute_utc": _minute_iso(raw_min),
            "raw_max_minute_utc": _minute_iso(raw_max),
            "published_min_minute_utc": _minute_iso(min_m),
            "published_max_minute_utc": _minute_iso(max_m),
            "published_minutes": span,
        },
        "strategies_included": strategies,
        "trade_counts": {
            "trades_by_strategy": dict(sorted(acc.strat_trades.items())),
            "usable_returns_by_strategy": dict(sorted(acc.strat_usable.items())),
        },
        "pnl_sums": {k: round(v, 6) for k, v in sorted(acc.strat_pnl_sum.items())},
        "notional_sums": {k: round(v, 6) for k, v in sorted(acc.strat_notional_sum.items())},
        "outputs": {
            "all_minutes_ndjson": str(OUT_ALL),
            "active_minutes_ndjson": str(OUT_ACTIVE),
        },
        "notes": "ALL minutes includes zeros; ACTIVE minutes only when at least one strategy has nonzero return in that minute.",
    }
    atomic_write_json(OUT_STATE, state)
    return 0


def main() -> int:
    try:
        return publish()
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            OUT_STATE,
            {
                "schema_version": "brain_returns_state.v1",
                "ts_utc": utc_now_iso(),
                "ttl_seconds": TTL_SECONDS,
                "status": "ERROR",
                "error": f"unhandled:{exc}",
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
