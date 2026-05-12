#!/usr/bin/env python3
"""
CHAD Bars Validation Report (fail-closed, market-aware)

Validates data/bars/1d/*.json for universe tickers.

Enforces schema per bar:
  ts_utc, open, high, low, close, volume

Freshness policy (Option B: long-term correct):
- We do NOT use a blind "hours since last bar" check (breaks on weekends/holidays).
- Instead, we compute the most recent expected trading *session date* for the exchange calendar
  (default: NYSE / XNYS) and require each symbol's last_ts_utc date to be >= that date.

Outputs a JSON report with:
- per-symbol: file_exists, rows, last_ts_utc, last_ts_date, expected_last_session_date, sha256, errors[]
- summary: ok, fail, total, pass, generated_ts_utc

Exit codes:
  0 = all ok
  2 = one or more symbols invalid / stale / missing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_KEYS: Tuple[str, ...] = ("ts_utc", "open", "high", "low", "close", "volume")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(ts: str) -> Optional[datetime]:
    # Accept "Z" or "+00:00"
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_bars(payload: Any) -> List[Dict[str, Any]]:
    # Support either:
    # 1) list[dict] (preferred)
    # 2) {"bars": list[dict]} (tolerate)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("bars"), list):
        return payload["bars"]
    return []


def validate_bar(bar: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in bar:
            errs.append(f"missing_key:{k}")

    # ts sanity
    if "ts_utc" in bar:
        if not isinstance(bar["ts_utc"], str) or parse_ts(bar["ts_utc"]) is None:
            errs.append("bad_ts_utc")

    # numeric sanity
    for k in ("open", "high", "low", "close"):
        if k in bar and not isinstance(bar[k], (int, float)):
            errs.append(f"bad_number:{k}")
    if "volume" in bar and not isinstance(bar.get("volume"), (int, float)):
        errs.append("bad_number:volume")

    return errs


def load_universe(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.upper())
    return out


def expected_last_session_date(exchange: str, now_utc: datetime) -> str:
    """Return the most recent *COMPLETED* trading session date (YYYY-MM-DD).

    We must not treat "today" as expected until the market has actually CLOSED.
    During market hours, the latest completed session is typically the prior session label.

    Logic:
    - Use exchange_calendars schedule.
    - Pick the last session whose `close` (or `market_close`) <= now_utc.
    """
    import exchange_calendars as ec
    import pandas as pd

    cal = ec.get_calendar(exchange)
    sched = getattr(cal, "schedule", None)
    if sched is None or len(sched) == 0:
        raise RuntimeError(f"calendar_has_no_schedule: {exchange}")

    close_col = None
    for c in ("market_close", "close"):
        if c in sched.columns:
            close_col = c
            break
    if close_col is None:
        raise RuntimeError(f"calendar_has_no_close_column: {exchange} cols={list(sched.columns)}")

    now_ts = pd.Timestamp(now_utc)  # tz-aware UTC
    eligible = sched[sched[close_col] <= now_ts]
    if len(eligible) == 0:
        raise RuntimeError(f"no_completed_sessions_before_now: {now_utc.isoformat()}")

    last_label = eligible.index[-1]
    return pd.Timestamp(last_label).date().isoformat()
def main() -> int:
    ap = argparse.ArgumentParser(description="CHAD Bars Validation Report (1d) — schema + market-aware freshness.")
    ap.add_argument("--bars-dir", required=True, help="e.g. /home/ubuntu/chad_finale/data/bars/1d")
    ap.add_argument("--universe-file", required=True, help="newline-delimited ticker file (one symbol per line, '#' comments allowed)")
    ap.add_argument("--out", required=True, help="output report JSON path")
    ap.add_argument("--exchange", default="XNYS", help="exchange calendar id (default: XNYS / NYSE)")
    args = ap.parse_args()

    bars_dir = Path(args.bars_dir)
    universe_file = Path(args.universe_file)
    out_path = Path(args.out)

    now = utc_now()

    if not bars_dir.exists():
        raise SystemExit(f"bars_dir_missing: {bars_dir}")
    if not universe_file.exists():
        raise SystemExit(f"universe_missing: {universe_file}")

    symbols = load_universe(universe_file)
    exp_date = expected_last_session_date(args.exchange, now)

    report: Dict[str, Any] = {
        "generated_ts_utc": now.isoformat().replace("+00:00", "Z"),
        "bars_dir": str(bars_dir),
        "universe_file": str(universe_file),
        "exchange": args.exchange,
        "expected_last_session_date": exp_date,
        "symbols_requested": len(symbols),
        "symbols": {},
        "summary": {},
    }

    ok = 0
    fail = 0

    for sym in symbols:
        fp = bars_dir / f"{sym}.json"
        entry: Dict[str, Any] = {
            "symbol": sym,
            "file": str(fp),
            "file_exists": fp.exists(),
            "rows": 0,
            "bad_rows": 0,
            "last_ts_utc": None,
            "last_ts_date": None,
            "expected_last_session_date": exp_date,
            "sha256": None,
            "errors": [],
        }

        if not fp.exists():
            entry["errors"].append("missing_file")
            report["symbols"][sym] = entry
            fail += 1
            continue

        try:
            payload = load_json(fp)
        except Exception as e:
            entry["errors"].append(f"json_load_failed:{type(e).__name__}")
            report["symbols"][sym] = entry
            fail += 1
            continue

        bars = extract_bars(payload)
        if not bars:
            entry["errors"].append("no_bars_found")
            report["symbols"][sym] = entry
            fail += 1
            continue

        bad_rows = 0
        last_dt: Optional[datetime] = None

        for b in bars:
            if not isinstance(b, dict):
                bad_rows += 1
                continue

            errs = validate_bar(b)
            if errs:
                bad_rows += 1
                if len(entry["errors"]) < 10:
                    entry["errors"].extend(errs)

            dt = parse_ts(b.get("ts_utc", "")) if isinstance(b.get("ts_utc"), str) else None
            if dt is not None and (last_dt is None or dt > last_dt):
                last_dt = dt

        entry["rows"] = len(bars)
        entry["bad_rows"] = bad_rows
        entry["sha256"] = sha256_file(fp)
        entry["last_ts_utc"] = last_dt.isoformat().replace("+00:00", "Z") if last_dt else None
        entry["last_ts_date"] = str(last_dt.date()) if last_dt else None

        # Fail closed if schema issues exist
        if bad_rows > 0:
            entry["errors"].append(f"schema_fail:bad_rows={bad_rows}")

        # Market-aware freshness (date-based):
        if last_dt is None:
            entry["errors"].append("no_valid_last_ts")
        else:
            if entry["last_ts_date"] is None:
                entry["errors"].append("no_valid_last_ts_date")
            else:
                # stale if last session date is behind expected trading session date
                if entry["last_ts_date"] < exp_date:
                    entry["errors"].append(f"stale:expected_session_date={exp_date}")

        report["symbols"][sym] = entry

        if entry["errors"]:
            fail += 1
        else:
            ok += 1

    report["summary"] = {
        "ok": ok,
        "fail": fail,
        "total": ok + fail,
        "pass": (fail == 0),
    }

    # SSOT parity mirrors (top-level summary_* keys)
    report["summary_ok"] = report.get("summary", {}).get("ok")
    report["summary_fail"] = report.get("summary", {}).get("fail")
    report["summary_total"] = report.get("summary", {}).get("total")
    report["summary_pass"] = report.get("summary", {}).get("pass")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
