#!/usr/bin/env python3
"""
Equity History Publisher.

Appends a daily equity snapshot to runtime/equity_history.ndjson so
WithdrawalManager and BusinessPhaseTracker have durable history for
high-water-mark and drawdown calculations.

Reads runtime/portfolio_snapshot.json (refreshed every 5min by the
portfolio snapshot publisher) and appends a deduplicated daily record.

Run via systemd timer: chad-equity-history.timer (daily 23:59 UTC).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone, date
from pathlib import Path

LOG = logging.getLogger("chad.ops.equity_history_publisher")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
HISTORY_PATH = RUNTIME_DIR / "equity_history.ndjson"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_utc() -> str:
    return date.today().isoformat()


def _read_existing_dates() -> set:
    if not HISTORY_PATH.is_file():
        return set()
    dates = set()
    try:
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                d = rec.get("date_utc")
                if d:
                    dates.add(d)
            except Exception:
                continue
    except Exception as exc:
        LOG.warning("history_read_failed: %s", exc)
    return dates


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not SNAPSHOT_PATH.is_file():
        LOG.error("portfolio_snapshot_missing: %s", SNAPSHOT_PATH)
        return 1

    try:
        snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.error("portfolio_snapshot_invalid: %s", exc)
        return 1

    today = _today_utc()
    existing = _read_existing_dates()

    if today in existing:
        LOG.info("equity_history_already_recorded date=%s", today)
        return 0

    # The portfolio_snapshot equity legs are broker-native CAD (ibkr_equity is
    # CAD via the v2 collector; kraken_equity is converted to CAD; coinbase is
    # CAD-base/unused). They were historically (mis)labelled as *_usd — fixed
    # here by writing them under honest *_cad keys.
    ibkr = float(snap.get("ibkr_equity", 0.0))
    kraken = float(snap.get("kraken_equity", 0.0))
    coinbase = float(snap.get("coinbase_equity", 0.0))
    total = ibkr + kraken + coinbase

    # Forward-only authoritative USD figures, sourced ONLY from the snapshot's
    # (a) fields (total_equity_usd_authoritative / ibkr_equity_usd_display).
    # FAIL-CLOSED: null when usd_ok is false (or the component is absent);
    # NEVER a CAD fallback. No FX rate is recomputed here (no duplicate rate
    # logic) — these mirror what portfolio_snapshot_publisher already validated.
    usd_ok = bool(snap.get("usd_ok", False))
    total_usd_auth = snap.get("total_equity_usd_authoritative")
    ibkr_usd_display = snap.get("ibkr_equity_usd_display")
    total_equity_usd = (
        float(total_usd_auth)
        if usd_ok and isinstance(total_usd_auth, (int, float))
        else None
    )
    ibkr_equity_usd = (
        float(ibkr_usd_display)
        if usd_ok and isinstance(ibkr_usd_display, (int, float))
        else None
    )

    record = {
        "date_utc": today,
        "ts_utc": _utc_now_iso(),
        # Honest CAD series — this is the CONTINUOUS series the drawdown and
        # withdrawal chains read (currency-agnostic ratio math). Renaming the
        # legacy total_equity_usd-as-CAD field to *_cad, and having those readers
        # prefer *_cad with a legacy total_equity_usd fallback, keeps the equity
        # series continuous (no phantom drawdown discontinuity).
        "total_equity_cad": total,
        "ibkr_equity_cad": ibkr,
        "kraken_equity_cad": kraken,
        "coinbase_equity_cad": coinbase,
        # Forward-only authoritative USD (additive, fail-closed; null when
        # usd_ok is false; never CAD). NOT read by the drawdown/withdrawal math.
        "total_equity_usd": total_equity_usd,
        "ibkr_equity_usd": ibkr_equity_usd,
        "usd_ok": usd_ok,
        "schema_version": "equity_history.v2",
    }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    LOG.info(
        "equity_history_appended date=%s total_cad=%.2f usd_ok=%s total_usd=%s",
        today, total, usd_ok,
        ("%.2f" % total_equity_usd) if total_equity_usd is not None else "None",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
