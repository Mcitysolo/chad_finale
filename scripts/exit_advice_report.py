#!/usr/bin/env python3
"""W4B-4 (J16): exit-advice review report — the D6-rider GO corpus.

READ-ONLY. Aggregates the advice evidence (``data/exit_advice/*.ndjson``) and
the overlay's advice verdicts (``data/exit_overlay/*.ndjson``) into the review
artifact the operator gates the record->consume flip on:

  - advice volume: by day / by (strategy,symbol,side) tuple / by site;
  - every ADVICE_WOULD_CLOSE the overlay computed (the would-fire corpus);
  - every CONSUMED advice close (reason=strategy_advice WOULD_CLOSE) — expected
    empty until the flip;
  - VIOLATIONS (each must be empty for GO):
      * excluded_unflagged — an advice row on an operator-excluded symbol whose
        ``excluded`` flag is not True: the recorder's flag wall has drifted from
        the exclusion SSOT and only the aggregator's set wall still holds
        (belt-and-braces reduced to belt);
      * advice_close_on_excluded — an ADVICE_WOULD_CLOSE/consumed close on an
        excluded symbol;
      * clamp_exceeds_broker — an advice close_qty above the broker-confirmed
        qty recorded in the same verdict row.

Output: ``reports/exit_advice_review_<UTCSTAMP>.json`` (exit_advice_review.v1)
plus a human summary on stdout. Exit 0 always (a reviewer, not a gate).

Usage:
  python3 scripts/exit_advice_report.py [--days 7] [--advice-dir DIR]
      [--overlay-dir DIR] [--excluded-config PATH] [--out PATH] [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

REPORT_SCHEMA = "exit_advice_review.v1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iter_ndjson(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _dated_files(directory: Path, prefix: str, days: int, now: datetime) -> List[Path]:
    out = []
    for i in range(days):
        day = (now - timedelta(days=i)).strftime("%Y%m%d")
        p = directory / f"{prefix}_{day}.ndjson"
        if p.exists():
            out.append(p)
    return sorted(out)


def _load_excluded(config_path: Path) -> List[str]:
    """The operator-exclusion SSOT, read directly from config (no chad imports —
    this tool must run against any tree read-only)."""
    try:
        obj = json.loads(config_path.read_text(encoding="utf-8"))
        syms = set(obj.get("reconciler_non_chad_symbols") or [])
        syms |= set(obj.get("broker_preexisting_symbols") or [])
        syms |= set((obj.get("exclusion_policy") or {}).keys())
        return sorted(s.upper() for s in syms)
    except Exception:
        return ["AAPL", "MSFT"]  # the reconciler's local floor — never empty


def build_report(*, advice_dir: Path, overlay_dir: Path, excluded: List[str],
                 days: int, now: datetime) -> Dict[str, Any]:
    excluded_set = set(excluded)

    # -- advice rows ---------------------------------------------------------
    by_day: Counter = Counter()
    by_tuple: Counter = Counter()
    by_site: Counter = Counter()
    excluded_rows = 0
    excluded_unflagged: List[Dict[str, Any]] = []
    total = 0
    for f in _dated_files(advice_dir, "exit_advice", days, now):
        for row in _iter_ndjson(f):
            total += 1
            by_day[str(row.get("ts_utc", ""))[:10]] += 1
            by_tuple[f"{row.get('strategy')}|{row.get('symbol')}|{row.get('side')}"] += 1
            by_site[str(row.get("site"))] += 1
            if row.get("excluded"):
                excluded_rows += 1
            elif str(row.get("symbol", "")).upper() in excluded_set:
                # flag-wall drift: SSOT says excluded, the recorder didn't flag it
                excluded_unflagged.append({k: row.get(k) for k in (
                    "ts_utc", "site", "strategy", "symbol", "side", "excluded")})

    # -- overlay advice verdicts --------------------------------------------
    would_fire: List[Dict[str, Any]] = []
    consumed: List[Dict[str, Any]] = []
    clamp_violations: List[Dict[str, Any]] = []
    excluded_closes: List[Dict[str, Any]] = []
    for f in _dated_files(overlay_dir, "exit_overlay", days, now):
        for row in _iter_ndjson(f):
            reason = row.get("reason")
            verdict = row.get("verdict")
            if reason != "strategy_advice":
                continue
            slim = {k: row.get(k) for k in (
                "ts_utc", "verdict", "symbol", "strategy", "side",
                "open_qty", "close_qty", "broker_confirmed_qty", "price")}
            if verdict == "ADVICE_WOULD_CLOSE":
                would_fire.append(slim)
            elif verdict == "WOULD_CLOSE":
                consumed.append(slim)
            else:
                continue
            try:
                if float(row.get("close_qty") or 0) > float(row.get("broker_confirmed_qty") or 0):
                    clamp_violations.append(slim)
            except (TypeError, ValueError):
                clamp_violations.append(slim)
            if str(row.get("symbol", "")).upper() in excluded_set:
                excluded_closes.append(slim)

    violations = {
        "excluded_unflagged": excluded_unflagged,
        "advice_close_on_excluded": excluded_closes,
        "clamp_exceeds_broker": clamp_violations,
    }
    go_criteria = {
        "recorder_flag_wall_intact": not excluded_unflagged,
        "zero_advice_closes_on_excluded": not excluded_closes,
        "all_clamps_within_broker": not clamp_violations,
        "would_fire_corpus_nonempty": bool(would_fire),
        "nothing_consumed_pre_flip": not consumed,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_days": days,
        "excluded_symbols": excluded,
        "advice_rows": {
            "total": total,
            "excluded_flagged": excluded_rows,
            "by_day": dict(sorted(by_day.items())),
            # top-50 rollup; never a silent cap — the flag names what was dropped
            "by_tuple": dict(by_tuple.most_common(50)),
            "by_tuple_truncated": len(by_tuple) > 50,
            "by_tuple_distinct": len(by_tuple),
            "by_site": dict(by_site),
        },
        "advice_would_close": would_fire,
        "consumed_advice_closes": consumed,
        "violations": violations,
        "go_criteria": go_criteria,
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--days", default=7, type=int)
    ap.add_argument("--advice-dir", type=Path, default=None)
    ap.add_argument("--overlay-dir", type=Path, default=None)
    ap.add_argument("--excluded-config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root: Path = args.repo_root
    advice_dir = args.advice_dir or root / "data" / "exit_advice"
    overlay_dir = args.overlay_dir or root / "data" / "exit_overlay"
    excluded_cfg = args.excluded_config or root / "config" / "reconciliation_exclusions.json"
    now = _utcnow()

    report = build_report(
        advice_dir=advice_dir, overlay_dir=overlay_dir,
        excluded=_load_excluded(excluded_cfg), days=args.days, now=now,
    )

    out = args.out or root / "reports" / f"exit_advice_review_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    g = report["go_criteria"]
    print(f"exit_advice_review -> {out}")
    print(f"  advice rows: {report['advice_rows']['total']} "
          f"(excluded-flagged {report['advice_rows']['excluded_flagged']})")
    print(f"  top tuples: {list(report['advice_rows']['by_tuple'].items())[:5]}")
    print(f"  ADVICE_WOULD_CLOSE corpus: {len(report['advice_would_close'])}")
    print(f"  consumed (pre-flip must be 0): {len(report['consumed_advice_closes'])}")
    print("  GO criteria: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                        for k, v in g.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
