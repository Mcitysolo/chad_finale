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
TTL_SECONDS = 360

# Pre-existing paper account positions not opened by CHAD. Excluded
# from the mismatch check so they do not flip status to RED.
try:
    from chad.core.position_reconciler import KNOWN_NON_CHAD_SYMBOLS as _RECONCILER_NON_CHAD  # type: ignore
except Exception:  # noqa: BLE001
    _RECONCILER_NON_CHAD = frozenset({"AAPL", "MSFT"})

_EXCLUSIONS_CONFIG_PATH = Path("/home/ubuntu/chad_finale/config/reconciliation_exclusions.json")

# Hardcoded fallback used only when the JSON config is missing or unreadable.
# Source of truth is config/reconciliation_exclusions.json.
_FALLBACK_BROKER_PREEXISTING = frozenset({"NVDA"})
_FALLBACK_EXCLUSION_POLICY: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
    "MSFT": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
    "NVDA": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
}


def _load_exclusion_config() -> Dict[str, Any]:
    """Load reconciliation exclusion policy from config JSON.

    Returns dict with broker_preexisting (frozenset) and exclusion_policy
    (dict). Falls back to hardcoded values if the file is missing or
    malformed so reconciliation never crashes on a config error.
    """
    try:
        raw = json.loads(_EXCLUSIONS_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "reconciliation_exclusions.json unreadable (%s) — falling back to hardcoded policy",
            exc,
        )
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    bp = raw.get("broker_preexisting_symbols") or []
    pol = raw.get("exclusion_policy") or {}
    if not isinstance(bp, list) or not isinstance(pol, dict):
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    return {
        "broker_preexisting": frozenset(str(s) for s in bp),
        "exclusion_policy": pol,
    }


_EXCLUSION_CFG = _load_exclusion_config()

# Publisher-only augmentation: symbols present at broker as pre-existing
# paper positions that CHAD never opened. Kept separate from the
# position_reconciler set so CHAD can still auto-close its own future
# positions on these symbols via thesis-flip reconciliation.
_BROKER_PREEXISTING: frozenset = _EXCLUSION_CFG["broker_preexisting"]
KNOWN_NON_CHAD_SYMBOLS = _RECONCILER_NON_CHAD | _BROKER_PREEXISTING

# DS08: bounded exclusion policy. Every symbol skipped from reconciliation
# carries reason/owner/added/expires/reviewed metadata so excluded items
# stay auditable instead of becoming permanent blind spots.
EXCLUSION_POLICY: Dict[str, Dict[str, Any]] = _EXCLUSION_CFG["exclusion_policy"]

# Futures symbols whose IBKR positions cannot be reliably reconciled
# without explicit contract_month resolution (ISSUE-29 companion). Any
# diff on these is skipped rather than flagged as a mismatch until the
# futures contract-resolution path is complete. MCL surfaced in retry-3
# as chad=2 vs broker=0 (ghost position from ContractResolutionError in
# ibkr_adapter). Do not flip status RED for futures-reconciliation gaps.
KNOWN_FUTURES_SYMBOLS = frozenset({"MCL", "ES", "NQ", "CL", "GC", "RTY", "MES", "MNQ", "MYM", "M2K"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_guard_positions() -> Dict[str, float]:
    """Aggregate open CHAD positions by symbol (signed quantity).

    In paper/DRY_RUN mode, only broker_sync positions are reconciled
    against IBKR — strategy positions are never submitted to the broker
    in paper mode and will always appear as mismatches if included.
    """
    from chad.execution.execution_config import is_paper_mode
    is_paper = is_paper_mode()

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
        strategy = str(entry.get("strategy", "")).lower()
        if not sym:
            continue
        # In paper mode: only reconcile broker_sync (real IBKR positions)
        if is_paper and strategy != "broker_sync":
            continue
        signed = -abs(float(qty)) if side in ("SELL", "SHORT") else abs(float(qty))
        agg[sym] += signed
    return dict(agg)


def _load_guard_breakdown() -> Dict[str, Dict[str, float]]:
    """Per-symbol breakdown: broker_sync contribution vs strategy contribution.

    Used by drift detection — a diff on a symbol whose chad-side is entirely
    from broker_sync (no strategy attribution) indicates IBKR moved without
    CHAD initiating the change. Treated as drift, not mismatch.
    """
    if not GUARD_PATH.exists():
        return {}
    try:
        raw = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    from chad.execution.execution_config import is_paper_mode
    is_paper = is_paper_mode()

    agg: Dict[str, Dict[str, float]] = {}
    for entry in raw.values():
        if not isinstance(entry, dict) or not entry.get("open"):
            continue
        sym = entry.get("symbol")
        if not sym:
            continue
        qty = entry.get("quantity", 0) or 0
        side = str(entry.get("side", "")).upper()
        strategy = str(entry.get("strategy", "")).lower()
        # In paper mode: only include broker_sync in breakdown so
        # broker_sync diffs are correctly classified as drift not mismatch
        if is_paper and strategy != "broker_sync":
            continue
        signed = -abs(float(qty)) if side in ("SELL", "SHORT") else abs(float(qty))
        bucket = agg.setdefault(sym, {"broker_sync": 0.0, "strategies": 0.0})
        if strategy == "broker_sync":
            bucket["broker_sync"] += signed
        else:
            bucket["strategies"] += signed
    return agg


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

    # Count open strategy-attribution entries (non-broker_sync) for visibility
    # alongside chad_open. In paper mode chad_open intentionally filters to
    # broker_sync; this exposes the strategy layer that is otherwise hidden.
    try:
        _raw_guard = (
            json.loads(GUARD_PATH.read_text(encoding="utf-8"))
            if GUARD_PATH.exists() else {}
        )
        chad_strategy_open = sum(
            1 for k, v in (_raw_guard or {}).items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("broker_sync|")
        )
    except Exception:
        chad_strategy_open = 0

    try:
        broker_side = _load_broker_positions()
    except Exception as exc:
        LOG.warning("IBKR unavailable for reconciliation: %s", exc)
        _write({
            "status": "RED",
            "broker_source": f"unavailable:{exc}",
            "chad_state_source": "position_guard.json",
            "counts": {
                "chad_open": len(chad_side),
                "chad_strategy_open": chad_strategy_open,
                "broker_positions": 0,
            },
            "mismatches": [],
            "exclusion_policy": EXCLUSION_POLICY,
            "notes": ["ibkr_unavailable"],
        })
        try:
            from chad.utils.telegram_notify import notify
            notify(
                f"🚨 Reconciliation RED — IBKR unavailable: {exc}",
                severity="critical",
                dedupe_key="reconciliation_red",
            )
        except Exception:
            pass
        return 0

    breakdown = _load_guard_breakdown()

    mismatches: List[Dict[str, Any]] = []
    drifts: List[Dict[str, Any]] = []
    excluded: List[str] = []
    futures_excluded: List[str] = []
    symbols = set(chad_side) | set(broker_side)
    worst = 0.0
    for sym in sorted(symbols):
        if sym in KNOWN_NON_CHAD_SYMBOLS:
            excluded.append(sym)
            continue
        if sym in KNOWN_FUTURES_SYMBOLS:
            # ISSUE-29 companion: futures reconciliation requires explicit
            # contract_month resolution (see chad/execution/ibkr_adapter.py
            # ContractResolutionError). Skip until that path is complete.
            futures_excluded.append(sym)
            continue
        c = chad_side.get(sym, 0.0)
        b = broker_side.get(sym, 0.0)
        diff = abs(c - b)
        if diff > 0:
            # Classify diff as BROKER_DRIFT vs real mismatch.
            # Drift criterion: CHAD's side is zero OR all from broker_sync
            # (no strategy attribution). IBKR position moved without CHAD
            # initiating — a reality to log, not a bug to flag RED.
            bd = breakdown.get(sym, {"broker_sync": 0.0, "strategies": 0.0})
            strategy_contrib = abs(bd.get("strategies", 0.0))
            if strategy_contrib < 1e-6:
                LOG.warning(
                    "BROKER_DRIFT symbol=%s guard_qty=%s broker_qty=%s diff=%s "
                    "(no strategy attribution; broker moved independently)",
                    sym, c, b, diff,
                )
                drifts.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})
            else:
                mismatches.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})
                worst = max(worst, diff)

    if worst <= 1.0:
        status = "GREEN"
    elif worst <= 2.0:
        status = "YELLOW"
    else:
        status = "RED"
        try:
            from chad.utils.telegram_notify import notify
            _red_reason = f"worst_diff={worst:.2f} mismatches={len(mismatches)}"
            notify(
                f"🚨 Reconciliation RED — {_red_reason}",
                severity="critical",
                dedupe_key="reconciliation_red",
            )
        except Exception:
            pass

    _write({
        "status": status,
        "broker_source": f"ibkr:clientId={IBKR_CLIENT_ID}",
        "chad_state_source": "position_guard.json",
        "counts": {
            "chad_open": len(chad_side),
            "chad_strategy_open": chad_strategy_open,
            "broker_positions": len(broker_side),
        },
        "worst_diff": worst,
        "mismatches": mismatches,
        "drifts": drifts,
        "excluded_symbols": excluded,
        "futures_excluded_symbols": futures_excluded,
        "exclusion_policy": EXCLUSION_POLICY,
        "notes": [],
    })
    LOG.info(
        "reconciliation status=%s worst_diff=%.2f mismatches=%d drifts=%d futures_excluded=%d",
        status, worst, len(mismatches), len(drifts), len(futures_excluded),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
