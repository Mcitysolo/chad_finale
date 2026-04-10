"""
IBKR Daily Bars Refresh — futures symbols.

Fetches 300 days of daily OHLCV bars for front-month futures
(MES, MNQ, MCL, MGC) via the IB Gateway and writes them to
data/bars/1d/{sym}.json, matching the Polygon backfill schema.

Intended to run as a systemd oneshot shortly after the
Polygon equity backfill (02:30 UTC) — see
chad-ibkr-daily-bars-refresh.timer.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import sys
import time
from typing import List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BARS_DIR = REPO_ROOT / "data" / "bars" / "1d"

FUTURES: List[Tuple[str, str]] = [
    ("MES", "CME"),
    ("MNQ", "CME"),
    ("MCL", "NYMEX"),
    ("MGC", "COMEX"),
]

IB_HOST = "127.0.0.1"
IB_PORT = 4002
IB_CLIENT_ID = 9052
DURATION = "300 D"  # IBKR caps non-year durations at 365 D


def _log(msg: str) -> None:
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def refresh() -> int:
    try:
        from ib_insync import IB, Future
    except ImportError as e:
        _log(f"FATAL: ib_insync not available: {e}")
        return 2

    BARS_DIR.mkdir(parents=True, exist_ok=True)

    ib = IB()
    failures = 0
    successes = 0

    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
        _log(f"connected to IB Gateway {IB_HOST}:{IB_PORT}")

        for sym, exchange in FUTURES:
            try:
                contract = Future(sym, exchange=exchange)
                details = ib.reqContractDetails(contract)
                if not details:
                    _log(f"{sym}: no contract details returned — SKIP")
                    failures += 1
                    continue

                front = sorted(
                    details,
                    key=lambda d: d.contract.lastTradeDateOrContractMonth,
                )[0].contract

                bars = ib.reqHistoricalData(
                    front,
                    endDateTime="",
                    durationStr=DURATION,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )

                if not bars:
                    _log(f"{sym}: no bars returned — SKIP")
                    failures += 1
                    continue

                bar_list = [
                    {
                        "ts_utc": str(b.date),
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ]

                out = BARS_DIR / f"{sym}.json"
                payload = {
                    "bars": bar_list,
                    "symbol": sym,
                    "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "source": "ibkr",
                    "contract": {
                        "localSymbol": getattr(front, "localSymbol", ""),
                        "lastTradeDateOrContractMonth": front.lastTradeDateOrContractMonth,
                        "exchange": exchange,
                    },
                }
                out.write_text(json.dumps(payload, indent=2))
                _log(f"{sym}: wrote {len(bars)} bars, latest={bars[-1].date}")
                successes += 1
                time.sleep(2)  # be polite to the pacer
            except Exception as e:
                _log(f"{sym}: ERROR {e!r}")
                failures += 1
                continue
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

    _log(f"done — success={successes} failure={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(refresh())
