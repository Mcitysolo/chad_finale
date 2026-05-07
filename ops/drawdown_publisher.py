#!/usr/bin/env python3
"""
ops/drawdown_publisher.py

GAP-016A — report-only global drawdown publisher.

Reads local equity history + the latest portfolio/PnL snapshots, computes a
60-day rolling high-water mark and the current drawdown vs. that HWM via
``chad.risk.drawdown_guard.compute_drawdown``, and writes
``runtime/drawdown_state.json`` (schema ``drawdown_state.v1``).

Contract:
* Never calls a broker.
* Never writes ``runtime/stop_bus_state.json``.
* Never suppresses signals or mutates ``live_loop.py``.
* ``enforcement_active`` is always False in this batch.
* Emits a JSON summary on stdout for orchestration scripts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk.drawdown_guard import (  # noqa: E402
    DEFAULT_LOOKBACK_DAYS,
    compute_drawdown,
    report_to_state_dict,
)

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(REPO_ROOT / "runtime"))).resolve()
DRAWDOWN_STATE_PATH = RUNTIME_DIR / "drawdown_state.json"
DEFAULT_TTL_SECONDS = 300


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def publish_drawdown(
    *,
    runtime_dir: Optional[Path] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    halt_threshold_pct: Optional[float] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """Compute drawdown report and atomically write drawdown_state.json. Returns the state dict."""
    rt = Path(runtime_dir).resolve() if runtime_dir else RUNTIME_DIR
    out_path = rt / "drawdown_state.json"

    report = compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        pnl_state_path=rt / "pnl_state.json",
        halt_threshold_pct=halt_threshold_pct,
        lookback_days=lookback_days,
    )
    state = report_to_state_dict(report, ts_utc=utc_now_iso(), ttl_seconds=ttl_seconds)
    atomic_write_json(out_path, state)
    return state


def main(argv: Optional[list] = None) -> int:
    state = publish_drawdown()
    summary = {
        "ok": True,
        "artifact": str(DRAWDOWN_STATE_PATH),
        "schema_version": state.get("schema_version"),
        "status": state.get("status"),
        "drawdown_pct": state.get("drawdown_pct"),
        "halt": state.get("halt"),
        "halt_threshold_pct": state.get("halt_threshold_pct"),
        "enforcement_active": state.get("enforcement_active", False),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
