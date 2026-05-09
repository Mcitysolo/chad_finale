#!/usr/bin/env python3
"""
chad/core/position_guard.py

Position-state memory for CHAD.

Purpose
-------
Track currently open strategy/symbol positions and allow:
- same-side duplicate blocking
- opposite-side flip detection
- explicit open / close / replace state updates
"""

from __future__ import annotations

import logging
import os as _os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import json
import time as _time

STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/position_guard.json")

_LOG = logging.getLogger("chad.core.position_guard")

# Status values that indicate a confirmed fill (broker-accepted close).
# Anything else — PendingSubmit, error, rejected, unknown — must NOT be
# treated as a close confirmation per ISSUE-29.
_CONFIRMED_FILL_STATUSES: frozenset = frozenset({"filled", "paper_fill"})

# Status values that explicitly indicate a non-confirmed (still pending)
# state. Listed for symmetry with paper_exec_evidence_writer's pending set.
_PENDING_FILL_STATUSES: frozenset = frozenset({
    "pendingsubmit", "presubmitted", "submitted", "apipending",
    "inactive", "unknown", "",
})

# Status values that indicate an outright rejection.
_REJECTED_FILL_STATUSES: frozenset = frozenset({
    "error", "rejected", "reject", "cancelled", "canceled",
})


class PositionState(str, Enum):
    """Formal position lifecycle states (SSOT v6.4)."""
    OPEN = "OPEN"                                          # Fresh position opened
    MAINTAINED = "MAINTAINED"                              # Same-side signal on existing open — no change
    FLIPPED = "FLIPPED"                                    # Opposite-side replaced existing position
    CLOSED = "CLOSED"                                      # Position explicitly closed
    RESET_FROM_BROKER_TRUTH = "RESET_FROM_BROKER_TRUTH"    # Reconciled from broker state


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> Dict[str, dict]:
    if not STATE_PATH.is_file():
        return {}
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _validate_position_guard_schema(state: Mapping[str, Any]) -> None:
    """Validate the top-level shape of a position_guard state dict.

    The on-disk schema is a flat mapping of "<strategy>|<symbol>" → entry-dict
    (plus reserved meta keys `_version` / `_written_by`). Each entry must be
    a dict; entries with `open` set must carry strategy and symbol fields
    so downstream readers (reconciliation_publisher, net_exposure_gate)
    do not crash on a partial record. Raises ValueError on validation
    failure so callers — and the atomic writer — can refuse the write.
    """
    if not isinstance(state, dict):
        raise ValueError(
            f"position_guard state must be a dict, got {type(state).__name__}"
        )
    for key, entry in state.items():
        if not isinstance(key, str):
            raise ValueError(
                f"position_guard key must be str, got {type(key).__name__}"
            )
        if key.startswith("_"):
            # Reserved meta keys (_version / _written_by) — allowed scalars.
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"position_guard entry {key!r} must be a dict, "
                f"got {type(entry).__name__}"
            )
        # Entries that claim to be open must carry the minimum identification
        # the rest of CHAD relies on. Closed entries can be sparse.
        if entry.get("open") is True:
            for required in ("strategy", "symbol", "side"):
                if required not in entry:
                    raise ValueError(
                        f"position_guard entry {key!r} is open but missing "
                        f"required field {required!r}"
                    )


def write_position_guard(
    state: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """Single atomic writer for position_guard.json (ISSUE-75).

    Validates schema, writes atomically via temp file + fsync + os.replace,
    raises on validation failure, and never partially writes corrupted state.

    Stamps `_version` (monotonic ms) and `_written_by` so consumers can
    detect concurrent modification (CAS basis — CB08/DS03).

    All callers — internal position_guard mutators, live_loop reconcilers,
    position_reconciler.apply_close_intents — must route writes through
    this function. Direct json.dump or write_text against position_guard.json
    is forbidden and asserted in chad/tests/test_position_guard_atomic_writer.py.
    """
    target = Path(path) if path is not None else STATE_PATH
    _validate_position_guard_schema(state)
    state["_version"] = int(_time.time() * 1000)
    state["_written_by"] = "position_guard"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(state, indent=2, default=str)
    # Open with write+fsync to harden against truncation on crash.
    fd = _os.open(
        str(tmp),
        _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC,
        0o644,
    )
    try:
        _os.write(fd, payload.encode("utf-8"))
        try:
            _os.fsync(fd)
        except OSError:
            # fsync may fail on some filesystems (tmpfs); the atomic
            # rename below still guarantees no partial write is observed.
            pass
    finally:
        _os.close(fd)
    _os.replace(str(tmp), str(target))


def save_state(state: Dict[str, dict]) -> None:
    """Backward-compatible alias for write_position_guard (ISSUE-75).

    Existing callers (live_loop, position_reconciler) import save_state.
    Keeping the alias avoids a churn-PR while routing all writes through
    the single atomic writer.
    """
    write_position_guard(state)


def _intent_strategy(intent) -> str:
    return str(getattr(intent, "strategy", "") or "")


def _intent_symbol(intent) -> str:
    return str(getattr(intent, "symbol", "") or "")


def _intent_side(intent) -> str:
    return str(getattr(intent, "side", "") or "")


def _position_key(strategy: str, symbol: str) -> str:
    return f"{strategy}|{symbol}"


def get_open_position(intent) -> Optional[dict]:
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    record = state.get(key)
    if record and record.get("open") is True:
        return record
    return None


def has_open_position(intent) -> bool:
    return get_open_position(intent) is not None


def is_same_side_open(intent) -> bool:
    record = get_open_position(intent)
    if not record:
        return False
    record_side = str(record.get("side", "") or "").upper().strip()
    intent_side = _intent_side(intent).upper().strip()
    match = record_side != "" and record_side == intent_side
    if match:
        # Record MAINTAINED state — position unchanged by this signal.
        # Skip the write when last_state is already MAINTAINED to avoid
        # write amplification on repeated same-side signals.
        if str(record.get("last_state", "") or "") != PositionState.MAINTAINED.value:
            state = _load_state()
            key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
            if key in state:
                state[key]["last_state"] = PositionState.MAINTAINED.value
                save_state(state)
    return match


def is_flip_signal(intent) -> bool:
    record = get_open_position(intent)
    if not record:
        return False
    open_side = str(record.get("side", "") or "").upper().strip()
    new_side = _intent_side(intent).upper().strip()
    return open_side != "" and new_side != "" and open_side != new_side


def _reduce_or_close_broker_sync(
    state: Dict[str, dict],
    broker_sync_key: str,
    strategy: str,
    side: str,
    quantity: float,
) -> None:
    """ISSUE-56 v2: reduce-not-close for partial broker_sync attribution.

    Same side, residual > 0  → reduce quantity, keep open=True.
    Same side, residual <= 0 → soft-close (full attribution).
    Opposite side            → untouched (flip intent; reconciliation surfaces drift).
    """
    if strategy == "broker_sync":
        return
    bs_entry = state.get(broker_sync_key)
    if not bs_entry or not bs_entry.get("open"):
        return
    bs_side = str(bs_entry.get("side", "") or "").upper()
    incoming_side = str(side or "").upper()
    if bs_side != incoming_side:
        return
    bs_qty = abs(float(bs_entry.get("quantity", 0) or 0))
    residual = bs_qty - abs(float(quantity or 0))
    bs_entry["updated_at_utc"] = _utc_now_iso()
    if residual <= 0:
        bs_entry["open"] = False
        bs_entry["closed_by"] = "strategy_ownership_assumed"
    else:
        bs_entry["quantity"] = residual
        bs_entry["open"] = True
        bs_entry["closed_by"] = "partial_attribution_residual"


def mark_position_open(intent) -> None:
    state = _load_state()
    strategy = _intent_strategy(intent)
    symbol = _intent_symbol(intent)
    side = _intent_side(intent)
    quantity = float(getattr(intent, "quantity", 0.0) or 0.0)
    key = _position_key(strategy, symbol)
    broker_sync_key = _position_key("broker_sync", symbol)
    _reduce_or_close_broker_sync(state, broker_sync_key, strategy, side, quantity)
    now_iso = _utc_now_iso()
    prior = state.get(key) or {}
    prior_opened = None
    if prior.get("open") is True and str(prior.get("side")) == str(side):
        prior_opened = prior.get("opened_at")
    state[key] = {
        "open": True,
        "opened_at": prior_opened or now_iso,
        "updated_at_utc": now_iso,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "last_state": PositionState.OPEN.value,
        "_entry_version": int(_time.time() * 1000),
    }
    save_state(state)


def is_fill_confirmed(fill_evidence: Optional[Mapping[str, Any]]) -> bool:
    """Return True iff `fill_evidence` represents a confirmed broker/paper fill.

    Per ISSUE-29 a position_guard.json entry may only be marked closed
    when ALL of the following hold:
      - `fill_id` exists (non-empty),
      - status is `filled` or `paper_fill`,
      - status is NOT `PendingSubmit` (or any other pending value),
      - the fill is NOT flagged `pnl_untrusted`, rejected, or otherwise
        untrusted (via `pnl_untrusted` bool or `tags` containing the marker).

    Any other shape — missing dict, missing fill_id, pending status,
    untrusted flag — must return False so callers leave guard state
    unchanged and log a warning instead of mutating to a phantom close.
    """
    if not isinstance(fill_evidence, Mapping):
        return False
    fill_id = fill_evidence.get("fill_id")
    if not fill_id or not str(fill_id).strip():
        return False
    status = str(fill_evidence.get("status", "") or "").strip().lower()
    if status in _PENDING_FILL_STATUSES:
        return False
    if status in _REJECTED_FILL_STATUSES:
        return False
    if status not in _CONFIRMED_FILL_STATUSES:
        return False
    if bool(fill_evidence.get("pnl_untrusted")):
        return False
    if bool(fill_evidence.get("reject")):
        return False
    tags = fill_evidence.get("tags")
    if isinstance(tags, (list, tuple)) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags
    ):
        return False
    extra = fill_evidence.get("extra")
    if isinstance(extra, Mapping) and bool(extra.get("pnl_untrusted")):
        return False
    return True


def mark_position_closed(
    intent,
    fill_evidence: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Close the guard entry for `intent` only when fill is confirmed.

    ISSUE-29: previously this function unconditionally flipped the entry
    to `open=False`, which produced phantom-closed positions when called
    before the broker (or paper executor) actually confirmed the fill.

    Now requires `fill_evidence` to satisfy `is_fill_confirmed`. When
    confirmation is absent the function logs a warning and returns False
    without mutating state; the caller is expected to retry on the next
    cycle once a confirmed fill exists.

    Returns True iff the guard entry was mutated and persisted.
    """
    if not is_fill_confirmed(fill_evidence):
        _LOG.warning(
            "ISSUE29_GUARD_SKIP mark_position_closed: fill not confirmed for "
            "strategy=%s symbol=%s — guard left unchanged (evidence=%s)",
            _intent_strategy(intent),
            _intent_symbol(intent),
            (fill_evidence or {}).get("status") if isinstance(fill_evidence, Mapping)
            else None,
        )
        return False
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    if key in state:
        state[key]["open"] = False
        state[key]["updated_at_utc"] = _utc_now_iso()
        state[key]["last_state"] = PositionState.CLOSED.value
        state[key]["closed_by"] = "fill_confirmed"
        state[key]["closed_fill_id"] = str(fill_evidence.get("fill_id"))
        state[key]["_entry_version"] = int(_time.time() * 1000)
        write_position_guard(state)
        return True
    return False


def replace_position(intent) -> None:
    """
    Used for a flip:
    close whatever side was open for strategy+symbol,
    then open the new side.
    """
    state = _load_state()
    strategy = _intent_strategy(intent)
    symbol = _intent_symbol(intent)
    side = _intent_side(intent)
    quantity = float(getattr(intent, "quantity", 0.0) or 0.0)
    key = _position_key(strategy, symbol)
    broker_sync_key = _position_key("broker_sync", symbol)
    _reduce_or_close_broker_sync(state, broker_sync_key, strategy, side, quantity)
    now_iso = _utc_now_iso()
    state[key] = {
        "open": True,
        "opened_at": now_iso,
        "updated_at_utc": now_iso,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "last_state": PositionState.FLIPPED.value,
        "_entry_version": int(_time.time() * 1000),
    }
    save_state(state)


def reset_from_broker(strategy: str, symbol: str) -> None:
    """Reconcile a position from broker truth — marks entry as reset."""
    state = _load_state()
    key = _position_key(strategy, symbol)
    if key in state:
        state[key]["open"] = False
        state[key]["updated_at_utc"] = _utc_now_iso()
        state[key]["last_state"] = PositionState.RESET_FROM_BROKER_TRUTH.value
    else:
        state[key] = {
            "open": False,
            "updated_at_utc": _utc_now_iso(),
            "strategy": strategy,
            "symbol": symbol,
            "side": "",
            "quantity": 0.0,
            "last_state": PositionState.RESET_FROM_BROKER_TRUTH.value,
        }
    save_state(state)


def reset_all_positions() -> None:
    save_state({})


def detect_guard_vs_broker_truth_drift(
    state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Pure read-only detector — flag strategy positions that disagree with broker_sync truth.

    Scans `state` for non-`broker_sync` open entries, then compares each one to
    the matching `broker_sync|<symbol>` entry. Returns a list of drift records:

      - drift_kind="broker_truth_missing": strategy says open but broker_sync
        has no open record for the same symbol.
      - drift_kind="side_mismatch": strategy and broker_sync disagree on side.

    No disk reads, no writes — caller passes the state dict in. Symbol/side
    comparisons mirror the same `.upper().strip()` normalization used by
    is_same_side_open / is_flip_signal so the detector cannot be fooled by
    case or whitespace drift between writers.
    """
    if not isinstance(state, Mapping):
        return []

    broker_by_symbol: Dict[str, Dict[str, Any]] = {}
    for key, entry in state.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if not key.startswith("broker_sync|"):
            continue
        if entry.get("open") is not True:
            continue
        symbol = str(entry.get("symbol", "") or "").strip().upper()
        if not symbol:
            continue
        broker_by_symbol[symbol] = entry

    drift: List[Dict[str, Any]] = []
    for key, entry in state.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if key.startswith("_") or key.startswith("broker_sync|"):
            continue
        if entry.get("open") is not True:
            continue
        strategy = str(entry.get("strategy", "") or "").strip()
        symbol = str(entry.get("symbol", "") or "").strip().upper()
        guard_side = str(entry.get("side", "") or "").strip().upper()
        if not symbol:
            continue
        broker_entry = broker_by_symbol.get(symbol)
        if broker_entry is None:
            drift.append({
                "key": key,
                "strategy": strategy,
                "symbol": symbol,
                "guard_side": guard_side,
                "broker_side": None,
                "broker_present": False,
                "drift_kind": "broker_truth_missing",
            })
            continue
        broker_side = str(broker_entry.get("side", "") or "").strip().upper()
        if guard_side and broker_side and guard_side != broker_side:
            drift.append({
                "key": key,
                "strategy": strategy,
                "symbol": symbol,
                "guard_side": guard_side,
                "broker_side": broker_side,
                "broker_present": True,
                "drift_kind": "side_mismatch",
            })
    return drift


def close_stale_position_from_broker_truth(
    strategy: str,
    symbol: str,
    reason: str,
    evidence: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Close a single stale guard entry reconciled against broker truth.

    Use only after operator-level confirmation that broker truth shows no
    matching position for (strategy, symbol). No fill is fabricated and no
    `closed_fill_id` is written — auditability is preserved via the
    `closed_by`, `closed_reason`, and `closed_evidence` fields.

    Mutates exactly one key (`<strategy>|<symbol>`); other entries are
    untouched. Returns True iff the entry existed and was mutated;
    returns False if the key is absent so callers can detect a no-op.
    """
    state = _load_state()
    key = _position_key(strategy, symbol)
    entry = state.get(key)
    if not isinstance(entry, dict):
        return False
    now_iso = _utc_now_iso()
    entry["open"] = False
    entry["updated_at_utc"] = now_iso
    entry["last_state"] = PositionState.CLOSED.value
    entry["closed_by"] = str(reason)
    entry["closed_reason"] = "stale_guard_entry"
    entry["closed_evidence"] = dict(evidence) if isinstance(evidence, Mapping) else {}
    entry["_entry_version"] = int(_time.time() * 1000)
    write_position_guard(state)
    return True
