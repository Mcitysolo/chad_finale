#!/usr/bin/env python3
"""
ops/micro_eod_flatten.py — Workstream 3 (CHAD v9.1)

EOD flatten gate for MICRO / STARTER tiers.

This script does NOT directly close positions. It emits *flatten intents*
to a dedicated ops-owned runtime ledger (runtime/eod_flatten_intents.json)
which downstream consumers (live_loop / paper_position_closer / operator
tooling) drain in their normal close pipeline. No broker, adapter, or
execution-router imports are pulled in here — the script reads the
canonical position_guard.json and writes one JSON status artifact.

Tier behaviour (from config/tiers.json risk_profile.flatten_before_eod):
  MICRO / STARTER  -> flatten_before_eod=true  -> emit close intents
  PRO_GROWTH / SCALE -> flatten_before_eod=false -> SKIPPED, no-op

Idempotency:
  - On every run we read the existing flatten-intent ledger and skip
    emitting a duplicate for any (strategy, symbol) that already has a
    "pending" intent within the ledger TTL.
  - A repeated invocation with no new qualifying positions writes a fresh
    status artifact but appends zero new intents.

Outputs:
  - runtime/micro_eod_flatten.json   (status artifact — atomically written)
  - runtime/eod_flatten_intents.json (intent ledger — atomically written)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── module config ────────────────────────────────────────────────────────────
LOG = logging.getLogger("chad.ops.micro_eod_flatten")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"

TIER_STATE_PATH = RUNTIME_DIR / "tier_state.json"
STATUS_OUT_PATH = RUNTIME_DIR / "micro_eod_flatten.json"
INTENT_LEDGER_PATH = RUNTIME_DIR / "eod_flatten_intents.json"

# Workstream-fixed surface.
ELIGIBLE_STRATEGY = "alpha_intraday_micro"
ELIGIBLE_SYMBOLS: frozenset = frozenset({"MES", "MNQ"})
FLATTEN_TIERS: frozenset = frozenset({"MICRO", "STARTER"})

# Status TTLs.
STATUS_TTL_SECONDS = 3600
INTENT_LEDGER_TTL_SECONDS = 7200

# Default tier-state TTL when the canonical writer omits ttl_seconds (the
# tier_manager publisher currently writes ts_utc only; per operator decision
# we default to 900s for the freshness check rather than failing closed).
DEFAULT_TIER_STATE_TTL_SECONDS = 900


# ── utilities ────────────────────────────────────────────────────────────────
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _parse_iso(ts: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; tolerate trailing 'Z' and missing tz."""
    if not isinstance(ts, str) or not ts:
        return None
    raw = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ── tier-state freshness ────────────────────────────────────────────────────
def _load_tier_state() -> Tuple[Optional[dict], Optional[str]]:
    """Return (tier_state_dict, error_reason).

    Either the dict is non-None (fresh) or error_reason is non-None.
    """
    raw = _read_json(TIER_STATE_PATH)
    if raw is None:
        return None, "EOD_FLATTEN_TIER_STATE_MISSING_OR_STALE"

    ts = _parse_iso(raw.get("ts_utc"))
    if ts is None:
        return None, "EOD_FLATTEN_TIER_STATE_MISSING_OR_STALE"

    # Operator decision (2026-05-13): when the tier_state publisher omits
    # ttl_seconds, default to 900s rather than fail-closed. tier_manager.py
    # is in the do-not-modify list for this workstream.
    ttl_raw = raw.get("ttl_seconds")
    try:
        ttl = int(ttl_raw) if ttl_raw is not None else DEFAULT_TIER_STATE_TTL_SECONDS
    except (TypeError, ValueError):
        ttl = DEFAULT_TIER_STATE_TTL_SECONDS

    age = (_utc_now() - ts).total_seconds()
    if age > ttl:
        return None, "EOD_FLATTEN_TIER_STATE_MISSING_OR_STALE"

    return raw, None


# ── position discovery (canonical SSOT: position_guard.json) ────────────────
def _load_open_positions() -> Dict[str, dict]:
    """Delegate to chad.core.position_reconciler.load_open_positions().

    That helper is the canonical reader used by live_loop, reconciliation,
    and regime_reduction — re-using it here avoids any parallel
    open-position tracking source.
    """
    try:
        from chad.core.position_reconciler import load_open_positions
        return load_open_positions()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("EOD_FLATTEN_POSITION_LOAD_FAILED err=%s", exc)
        return {}


def _is_qualifying_position(entry: dict) -> bool:
    if not isinstance(entry, dict) or not entry.get("open"):
        return False
    strategy = str(entry.get("strategy", "") or "").strip()
    if strategy != ELIGIBLE_STRATEGY:
        return False
    symbol = str(entry.get("symbol", "") or "").strip().upper()
    if symbol not in ELIGIBLE_SYMBOLS:
        return False
    try:
        qty = float(entry.get("quantity", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return qty != 0.0


# ── intent ledger ───────────────────────────────────────────────────────────
def _load_intent_ledger() -> dict:
    """Read the EOD flatten intent ledger; tolerate missing/corrupt file."""
    raw = _read_json(INTENT_LEDGER_PATH)
    if not isinstance(raw, dict):
        return {"schema_version": "eod_flatten_intents.v1", "intents": []}
    if not isinstance(raw.get("intents"), list):
        raw["intents"] = []
    raw["schema_version"] = "eod_flatten_intents.v1"
    return raw


def _pending_intent_exists(
    ledger: dict, position_key: str, now: datetime
) -> bool:
    """True iff a non-expired pending intent for position_key is present."""
    for item in ledger.get("intents", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("position_key") != position_key:
            continue
        if str(item.get("status", "")).lower() != "pending":
            continue
        created = _parse_iso(item.get("created_at_utc"))
        if created is None:
            return True  # cannot age-out a malformed entry — treat as live
        try:
            ttl = int(item.get("ttl_seconds", INTENT_LEDGER_TTL_SECONDS))
        except (TypeError, ValueError):
            ttl = INTENT_LEDGER_TTL_SECONDS
        if (now - created).total_seconds() <= ttl:
            return True
    return False


def _build_close_intent(
    position_key: str, entry: dict, tier_name: str, now_iso: str
) -> dict:
    open_side = str(entry.get("side", "")).upper()
    close_side = "SELL" if open_side == "BUY" else "BUY"
    try:
        qty = float(entry.get("quantity", 0.0) or 0.0)
    except (TypeError, ValueError):
        qty = 0.0
    return {
        "schema_version": "eod_flatten_intent.v1",
        "position_key": position_key,
        "strategy": entry.get("strategy", ELIGIBLE_STRATEGY),
        "symbol": str(entry.get("symbol", "")).upper(),
        "action": "CLOSE",
        "open_side": open_side,
        "close_side": close_side,
        "quantity": abs(qty),
        "reason": f"EOD_FLATTEN_{tier_name}",
        "tier": tier_name,
        "created_at_utc": now_iso,
        "ttl_seconds": INTENT_LEDGER_TTL_SECONDS,
        "status": "pending",
    }


def _write_intent_ledger(ledger: dict, now_iso: str) -> None:
    ledger["ts_utc"] = now_iso
    ledger["ttl_seconds"] = INTENT_LEDGER_TTL_SECONDS
    _atomic_write_json(INTENT_LEDGER_PATH, ledger)


# ── status artifact ─────────────────────────────────────────────────────────
def _write_status(
    *,
    status: str,
    tier: Optional[str],
    flatten_required: bool,
    positions_found: int,
    positions_closed: int,
    skipped_reason: Optional[str],
) -> dict:
    payload = {
        "schema_version": "micro_eod_flatten.v1",
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": STATUS_TTL_SECONDS,
        "tier": tier,
        "flatten_required": bool(flatten_required),
        "positions_found": int(positions_found),
        "positions_closed": int(positions_closed),
        "skipped_reason": skipped_reason,
        "status": status,
    }
    _atomic_write_json(STATUS_OUT_PATH, payload)
    return payload


# ── main entry point ────────────────────────────────────────────────────────
def run() -> dict:
    """Execute one EOD flatten evaluation; always writes the status artifact."""
    tier_state, error_reason = _load_tier_state()
    if tier_state is None:
        LOG.error("EOD_FLATTEN status=ERROR reason=%s", error_reason)
        return _write_status(
            status="ERROR",
            tier=None,
            flatten_required=False,
            positions_found=0,
            positions_closed=0,
            skipped_reason=error_reason,
        )

    tier_name = str(tier_state.get("tier_name", "") or "").upper()
    risk_profile = tier_state.get("risk_profile") or {}
    flatten_flag = bool(risk_profile.get("flatten_before_eod", False))

    if tier_name not in FLATTEN_TIERS or not flatten_flag:
        reason = (
            f"EOD_FLATTEN_SKIPPED tier={tier_name} "
            f"flatten_before_eod={flatten_flag}"
        )
        LOG.info(reason)
        return _write_status(
            status="SKIPPED",
            tier=tier_name or None,
            flatten_required=False,
            positions_found=0,
            positions_closed=0,
            skipped_reason=reason,
        )

    # MICRO / STARTER path with flatten_before_eod=true.
    open_positions = _load_open_positions()
    qualifying: List[Tuple[str, dict]] = [
        (pk, entry)
        for pk, entry in open_positions.items()
        if _is_qualifying_position(entry)
    ]

    now = _utc_now()
    now_iso = _utc_now_iso()
    ledger = _load_intent_ledger()
    existing_intents = ledger.get("intents", []) or []

    new_intents: List[dict] = []
    closed_count = 0
    for pk, entry in qualifying:
        if _pending_intent_exists(ledger, pk, now):
            # Idempotency: a pending intent for this position already exists.
            closed_count += 1
            LOG.info("EOD_FLATTEN_INTENT_DUP position_key=%s — skipping", pk)
            continue
        intent = _build_close_intent(pk, entry, tier_name, now_iso)
        new_intents.append(intent)
        closed_count += 1
        LOG.info(
            "EOD_FLATTEN_INTENT_EMIT position_key=%s symbol=%s qty=%s reason=%s",
            pk, intent["symbol"], intent["quantity"], intent["reason"],
        )

    ledger["intents"] = list(existing_intents) + new_intents
    _write_intent_ledger(ledger, now_iso)

    return _write_status(
        status="OK",
        tier=tier_name,
        flatten_required=True,
        positions_found=len(qualifying),
        positions_closed=closed_count,
        skipped_reason=None,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("EOD_FLATTEN_UNCAUGHT err=%s", exc)
        _write_status(
            status="ERROR",
            tier=None,
            flatten_required=False,
            positions_found=0,
            positions_closed=0,
            skipped_reason=f"UNCAUGHT_EXCEPTION:{type(exc).__name__}",
        )
    # Always exit 0 — systemd must not flag this oneshot as failed.
    return 0


if __name__ == "__main__":
    sys.exit(main())
