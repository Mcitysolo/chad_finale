#!/usr/bin/env python3
"""
chad/analytics/ml_veto_shadow_collector.py — W6B-4/W6B-5.

Durable ML veto shadow history, and a baseline computed from it.

WHY
---
`config/exterminator.json` asserts that "no live rate can be computed or
compared" because "veto decisions leave only log lines (ML_SHADOW / ML_VETO),
no durable artifact, no counter".

The second clause is true. **The first is not.** The ML_SHADOW lines emitted at
`chad/core/live_loop.py:2996-3010` are fully structured and carry every field a
baseline needs:

    ML_SHADOW symbol=GOOGL strategy=gamma intent_class=entry
      model_version=xgb_veto_20260510_020007 manifest_hash=sha256:74afb9e1c8f3e0fe4
      loss_prob=0.518 threshold=0.65 would_veto=False final_action=shadow_only
      reason=loss_probability_below_threshold

Parsing the current boot gave a production veto rate of **1.10% (6/545)**
against the manifest's training-time `val_veto_rate_at_0.65 = 71.2%` — a ~65x
divergence, which is exactly the veto drift EXS8 wants detectable.

WHY A COLLECTOR RATHER THAN A HOT-PATH WRITE
--------------------------------------------
This host's journald retains only the current boot, so any baseline computed
straight from the journal dies at the next restart. The obvious alternative —
writing the NDJSON row at the emit site — would put file I/O on the intent hot
path for an observability feature. So the collector runs out-of-band, parses,
and appends. The veto path is untouched; the predictor stays shadow-only.

IDEMPOTENCE
-----------
Re-running must not double-count. Each row gets a deterministic `row_key` from
(ts, symbol, strategy, loss_prob, model_version); keys already present in the
NDJSON are skipped. Safe to run on a short timer.

CLI
---
    python3 -m chad.analytics.ml_veto_shadow_collector collect [--since -6h]
    python3 -m chad.analytics.ml_veto_shadow_collector baseline [--window-hours 168]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SHADOW_PATH = ROOT / "runtime" / "ml_veto_shadow.ndjson"
BASELINE_PATH = ROOT / "runtime" / "ml_veto_baseline.json"

SHADOW_SCHEMA = "ml_veto_shadow.v1"
BASELINE_SCHEMA = "ml_veto_baseline.v1"
BASELINE_TTL_SECONDS = 26 * 3600

# A baseline computed from too few samples is worse than no baseline: it looks
# authoritative and moves on noise. Declared, not implicit.
DEFAULT_MIN_SAMPLES = 200
DEFAULT_WINDOW_HOURS = 168  # 7 days

# Sample count alone is NOT sufficiency. The first live run of this collector
# gathered 526 rows in ~12 hours of a single boot — comfortably past any n
# threshold, and still not a baseline: it is six hours of one market session,
# 94% of it one strategy. A veto rate from that window would be wired as
# "production behaviour" and would move the moment the regime changed.
#
# So sufficiency requires BOTH: enough samples AND enough elapsed coverage.
DEFAULT_MIN_SPAN_HOURS = 72  # at least 3 distinct days of behaviour

_LINE_RE = re.compile(
    r"ML_SHADOW\s+"
    r"symbol=(?P<symbol>\S+)\s+"
    r"strategy=(?P<strategy>\S+)\s+"
    r"intent_class=(?P<intent_class>\S+)\s+"
    r"model_version=(?P<model_version>\S+)\s+"
    r"manifest_hash=(?P<manifest_hash>\S+)\s+"
    r"loss_prob=(?P<loss_prob>[0-9.]+)\s+"
    r"threshold=(?P<threshold>[0-9.]+)\s+"
    r"would_veto=(?P<would_veto>\S+)\s+"
    r"final_action=(?P<final_action>\S+)\s+"
    r"reason=(?P<reason>\S+)"
)

# The logging prefix carries the authoritative timestamp:
#   "2026-07-24 01:07:26,421 [INFO] ML_SHADOW ..."
_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3})")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one ML_SHADOW journal line into a shadow row, or None."""
    m = _LINE_RE.search(line)
    if not m:
        return None
    g = m.groupdict()

    ts_utc = ""
    tm = _TS_RE.search(line)
    if tm:
        try:
            dt = datetime.strptime(tm.group("ts"), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc, microsecond=int(tm.group("ms")) * 1000
            )
            ts_utc = dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            ts_utc = ""

    try:
        loss_prob = float(g["loss_prob"])
        threshold = float(g["threshold"])
    except (TypeError, ValueError):
        return None

    row = {
        "schema_version": SHADOW_SCHEMA,
        "ts_utc": ts_utc,
        "symbol": g["symbol"],
        "strategy": g["strategy"],
        "intent_class": g["intent_class"],
        "model_version": g["model_version"],
        "manifest_hash": g["manifest_hash"],
        "loss_prob": loss_prob,
        "threshold": threshold,
        "would_veto": g["would_veto"].strip().lower() == "true",
        "final_action": g["final_action"],
        "reason": g["reason"],
    }
    row["row_key"] = _row_key(row)
    return row


def _row_key(row: Dict[str, Any]) -> str:
    raw = "|".join([
        str(row.get("ts_utc", "")), str(row.get("symbol", "")),
        str(row.get("strategy", "")), f"{row.get('loss_prob', 0):.6f}",
        str(row.get("model_version", "")),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def read_journal(unit: str = "chad-live-loop", since: str = "-24h") -> List[str]:
    """Read ML_SHADOW lines from journald. Failure yields [] — a collector that
    cannot read the journal must not crash the timer that runs it."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", unit, "--since", since, "--no-pager"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if "ML_SHADOW" in ln]


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------

def existing_keys(path: Path = SHADOW_PATH) -> set:
    keys = set()
    if not path.is_file():
        return keys
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    keys.add(json.loads(line).get("row_key"))
                except Exception:
                    continue
    except Exception:
        return keys
    keys.discard(None)
    return keys


def append_rows(rows: Iterable[Dict[str, Any]], path: Path = SHADOW_PATH) -> int:
    """Append rows not already present. Returns the count actually written."""
    rows = list(rows)
    if not rows:
        return 0
    seen = existing_keys(path)
    fresh = [r for r in rows if r.get("row_key") not in seen]
    # De-duplicate within this batch too (journalctl can repeat on overlap).
    deduped: List[Dict[str, Any]] = []
    batch_seen = set()
    for r in fresh:
        k = r.get("row_key")
        if k in batch_seen:
            continue
        batch_seen.add(k)
        deduped.append(r)
    if not deduped:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in deduped:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(deduped)


def load_rows(path: Path = SHADOW_PATH) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def compute_baseline(
    rows: List[Dict[str, Any]],
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_span_hours: int = DEFAULT_MIN_SPAN_HOURS,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Stratified veto-rate baseline over a DECLARED window.

    Stratification is not decoration. In the measured sample 510 of 545 rows
    (94%) were `strategy=gamma`, so an unstratified "portfolio veto rate" would
    really be gamma's rate wearing a portfolio label. Every stratum carries its
    own n so a thin one is visibly thin.

    `sufficient` is reported separately from the rate itself: a rate computed
    from 12 samples is still returned, but marked not-usable, so a consumer
    cannot accidentally treat it as a baseline.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)

    in_window: List[Dict[str, Any]] = []
    stamps: List[datetime] = []
    undated = 0
    for r in rows:
        ts = str(r.get("ts_utc") or "")
        if not ts:
            undated += 1
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            undated += 1
            continue
        if dt >= cutoff:
            in_window.append(r)
            stamps.append(dt)

    n = len(in_window)
    vetoes = sum(1 for r in in_window if r.get("would_veto") is True)
    rate = (vetoes / n) if n else None

    span_hours = (
        round((max(stamps) - min(stamps)).total_seconds() / 3600.0, 2)
        if len(stamps) >= 2 else 0.0
    )

    by_strategy: Dict[str, Dict[str, Any]] = {}
    by_intent: Dict[str, Dict[str, Any]] = {}
    for key, bucket in (("strategy", by_strategy), ("intent_class", by_intent)):
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for r in in_window:
            slot = counts[str(r.get(key) or "unknown")]
            slot[0] += 1
            if r.get("would_veto") is True:
                slot[1] += 1
        for name, (total, vet) in sorted(counts.items()):
            bucket[name] = {
                "n": total, "vetoes": vet,
                "veto_rate": (vet / total) if total else None,
            }

    probs = sorted(float(r.get("loss_prob") or 0.0) for r in in_window)

    def _pct(p: float) -> Optional[float]:
        if not probs:
            return None
        idx = min(len(probs) - 1, max(0, int(round(p * (len(probs) - 1)))))
        return round(probs[idx], 4)

    versions = sorted({str(r.get("model_version") or "") for r in in_window} - {""})

    return {
        "schema_version": BASELINE_SCHEMA,
        "ts_utc": now.isoformat().replace("+00:00", "Z"),
        "ttl_seconds": BASELINE_TTL_SECONDS,
        "window_hours": window_hours,
        "window_start_utc": cutoff.isoformat().replace("+00:00", "Z"),
        "min_samples_required": min_samples,
        "min_span_hours_required": min_span_hours,
        "n": n,
        "vetoes": vetoes,
        "veto_rate": rate,
        "observed_span_hours": span_hours,
        "earliest_utc": (min(stamps).isoformat().replace("+00:00", "Z") if stamps else None),
        "latest_utc": (max(stamps).isoformat().replace("+00:00", "Z") if stamps else None),
        # Both gates must pass. n alone would have called 12 hours of one boot
        # (526 rows, 94% gamma) a portfolio baseline.
        "sufficient": bool(n >= min_samples and span_hours >= min_span_hours),
        "insufficient_reasons": [
            r for r in (
                f"n={n} < min_samples={min_samples}" if n < min_samples else None,
                (f"observed_span_hours={span_hours} < min_span_hours={min_span_hours}"
                 if span_hours < min_span_hours else None),
            ) if r
        ],
        "rows_total_in_store": len(rows),
        "rows_undated_skipped": undated,
        "by_strategy": by_strategy,
        "by_intent_class": by_intent,
        "loss_prob": {
            "p50": _pct(0.50), "p90": _pct(0.90), "p99": _pct(0.99),
            "mean": round(sum(probs) / len(probs), 4) if probs else None,
        },
        "model_versions": versions,
        "single_model_version": len(versions) == 1,
        "_caveats": [
            "This is a PRODUCTION veto rate. It is NOT comparable to the "
            "manifest's val_veto_rate_at_0.65, which is a training-time "
            "validation statistic over a different distribution.",
            "Check by_strategy before quoting the headline rate: a window "
            "dominated by one strategy is that strategy's rate, not the "
            "portfolio's.",
            "sufficient=false means n or elapsed coverage is below the declared "
            "floor; the rate is reported for visibility but must NOT be wired "
            "as a baseline. See insufficient_reasons.",
        ],
    }


def write_baseline(payload: Dict[str, Any], path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ML veto shadow collector / baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="append new ML_SHADOW rows to the durable store")
    c.add_argument("--unit", default="chad-live-loop")
    c.add_argument("--since", default="-24h")
    c.add_argument("--out", default=str(SHADOW_PATH))

    b = sub.add_parser("baseline", help="compute and publish the baseline")
    b.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    b.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    b.add_argument("--min-span-hours", type=int, default=DEFAULT_MIN_SPAN_HOURS)
    b.add_argument("--store", default=str(SHADOW_PATH))
    b.add_argument("--out", default=str(BASELINE_PATH))
    b.add_argument("--dry-run", action="store_true", help="print, do not write")

    args = ap.parse_args(argv)

    if args.cmd == "collect":
        lines = read_journal(args.unit, args.since)
        rows = [r for r in (parse_line(ln) for ln in lines) if r]
        written = append_rows(rows, Path(args.out))
        print(f"lines={len(lines)} parsed={len(rows)} appended={written} store={args.out}")
        return 0

    rows = load_rows(Path(args.store))
    payload = compute_baseline(
        rows, window_hours=args.window_hours, min_samples=args.min_samples,
        min_span_hours=args.min_span_hours,
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    write_baseline(payload, Path(args.out))
    print(
        f"n={payload['n']} veto_rate={payload['veto_rate']} "
        f"sufficient={payload['sufficient']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
