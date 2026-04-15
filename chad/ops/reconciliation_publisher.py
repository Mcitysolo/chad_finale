#!/usr/bin/env python3
"""
CHAD Reconciliation Publisher — scheduled CLI wrapper.

Compares CHAD's position_guard.json against IBKR broker truth
(connected with clientId=83, readonly) and writes
runtime/reconciliation_state.json.

Status rules:
  GREEN  — every open CHAD position matches broker within 1 unit
  YELLOW — minor discrepancy (within 2 units) OR broker has no-guard symbols
  RED    — major discrepancy OR IBKR unavailable

Fail-soft: always writes an output file. Never raises.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger("chad.ops.reconciliation_publisher")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
GUARD_PATH = RUNTIME_DIR / "position_guard.json"
OUT_PATH = RUNTIME_DIR / "reconciliation_state.json"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 83
IBKR_TIMEOUT_SEC = 15
TTL_SECONDS = 300

# Pre-existing paper account positions not opened by CHAD. Excluded
# from the mismatch check so they do not flip status to RED.
try:
    from chad.core.position_reconciler import KNOWN_NON_CHAD_SYMBOLS as _RECONCILER_NON_CHAD  # type: ignore
except Exception:  # noqa: BLE001
    _RECONCILER_NON_CHAD = frozenset({"AAPL", "MSFT"})

# Publisher-only augmentation: symbols present at broker as pre-existing
# paper positions that CHAD never opened. Kept separate from the
# position_reconciler set so CHAD can still auto-close its own future
# positions on these symbols via thesis-flip reconciliation.
_BROKER_PREEXISTING = frozenset({"NVDA"})
KNOWN_NON_CHAD_SYMBOLS = _RECONCILER_NON_CHAD | _BROKER_PREEXISTING


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_guard_positions() -> Dict[str, float]:
    """Aggregate open CHAD positions by symbol (signed quantity)."""
    if not GUARD_PATH.exists():
        return {}
    try:
        raw = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    agg: Dict[str, float] = defaultdict(float)
    for entry in raw.values():
        if not isinstance(entry, dict) or not entry.get("open"):
            continue
        sym = entry.get("symbol")
        qty = entry.get("quantity", 0) or 0
        side = str(entry.get("side", "")).upper()
        if not sym:
            continue
        signed = -abs(float(qty)) if side in ("SELL", "SHORT") else abs(float(qty))
        agg[sym] += signed
    return dict(agg)


def _load_broker_positions() -> Dict[str, float]:
    """Connect to IBKR with clientId=83 and return signed positions by symbol."""
    from ib_insync import IB

    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   readonly=True, timeout=IBKR_TIMEOUT_SEC)
        positions = ib.positions()
        out: Dict[str, float] = defaultdict(float)
        for p in positions:
            sym = getattr(p.contract, "symbol", None)
            if not sym:
                continue
            out[sym] += float(p.position)
        return dict(out)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def _write(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload["ts_utc"] = _utc_now_iso()
    payload["ttl_seconds"] = TTL_SECONDS
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT_PATH)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    chad_side = _load_guard_positions()

    try:
        broker_side = _load_broker_positions()
    except Exception as exc:
        LOG.warning("IBKR unavailable for reconciliation: %s", exc)
        _write({
            "status": "RED",
            "broker_source": f"unavailable:{exc}",
            "chad_state_source": "position_guard.json",
            "counts": {"chad_open": len(chad_side), "broker_positions": 0},
            "mismatches": [],
            "notes": ["ibkr_unavailable"],
        })
        return 0

    mismatches: List[Dict[str, Any]] = []
    excluded: List[str] = []
    symbols = set(chad_side) | set(broker_side)
    worst = 0.0
    for sym in sorted(symbols):
        if sym in KNOWN_NON_CHAD_SYMBOLS:
            excluded.append(sym)
            continue
        c = chad_side.get(sym, 0.0)
        b = broker_side.get(sym, 0.0)
        diff = abs(c - b)
        if diff > 0:
            mismatches.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})
            worst = max(worst, diff)

    if worst <= 1.0:
        status = "GREEN"
    elif worst <= 2.0:
        status = "YELLOW"
    else:
        status = "RED"

    _write({
        "status": status,
        "broker_source": f"ibkr:clientId={IBKR_CLIENT_ID}",
        "chad_state_source": "position_guard.json",
        "counts": {"chad_open": len(chad_side), "broker_positions": len(broker_side)},
        "worst_diff": worst,
        "mismatches": mismatches,
        "excluded_symbols": excluded,
        "notes": [],
    })
    LOG.info("reconciliation status=%s worst_diff=%.2f mismatches=%d",
             status, worst, len(mismatches))
    return 0


if __name__ == "__main__":
    sys.exit(main())
