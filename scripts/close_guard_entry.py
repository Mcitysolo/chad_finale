#!/usr/bin/env python3
"""
scripts/close_guard_entry.py

GAP-028 §5.2 — operator CLI to surgically close a stale per-strategy
position_guard entry against broker truth.

Honours the operational invariant from GAP-028 §4: closing the guard
entry alone is a no-op because `_rebuild_guard_from_paper_ledger` will
re-open it from the trade_closer FIFO within ~60 seconds. This CLI
clears the matching `trade_closer_state.queues[strategy|symbol]` entry
in the same invocation so the next live-loop cycle cannot rebuild it.

Fail-closed: refuses to run when SCR is unsafe, when exec_mode is not
paper/dry_run, or when LiveGate operator intent is ALLOW_LIVE. Refuses
on broker_sync|* keys (those are owned by `_rebuild_guard_from_broker`
and reflect IBKR truth).

Usage:
    python3 scripts/close_guard_entry.py \
        --strategy delta --symbol AAPL \
        --reason "broker_truth_short_2_exclusion_AAPL" \
        --by operator --confirm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.core import position_guard  # noqa: E402

LOG = logging.getLogger("scripts.close_guard_entry")

RUNTIME_DIR = REPO_ROOT / "runtime"
TRADE_CLOSER_STATE_PATH = RUNTIME_DIR / "trade_closer_state.json"
SCR_STATE_PATH = RUNTIME_DIR / "scr_state.json"
OPERATOR_INTENT_PATH = RUNTIME_DIR / "operator_intent.json"
OPERATOR_ACTIONS_DIR = REPO_ROOT / "data" / "operator_actions"

_SAFE_SCR_STATES = frozenset({"CONFIDENT", "CAUTIOUS"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _check_exec_mode_paper() -> Tuple[bool, str]:
    """Refuse unless CHAD_EXECUTION_MODE resolves to paper or dry_run."""
    try:
        from chad.execution.execution_config import is_paper_mode, get_execution_mode
        mode = get_execution_mode().value
        if not is_paper_mode():
            return False, f"exec_mode={mode} (not paper/dry_run)"
        return True, mode
    except Exception as exc:  # noqa: BLE001
        return False, f"exec_mode check raised {type(exc).__name__}: {exc}"


def _check_scr_safe() -> Tuple[bool, str]:
    """Refuse unless SCR state is CONFIDENT or CAUTIOUS."""
    if not SCR_STATE_PATH.is_file():
        return False, "scr_state.json missing"
    try:
        scr = json.loads(SCR_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"scr_state.json unreadable: {exc}"
    state = str(scr.get("state", "") or "").upper().strip()
    if state not in _SAFE_SCR_STATES:
        return False, f"scr_state={state or 'UNKNOWN'} (not CONFIDENT/CAUTIOUS)"
    return True, state


def _check_livegate_not_allow_live() -> Tuple[bool, str]:
    """Refuse if LiveGate operator intent is ALLOW_LIVE.

    Reads operator_intent.json directly to avoid pulling the full evaluate()
    path which would also touch market state. Fail-closed: missing/unreadable
    file is treated as 'not ALLOW_LIVE' (the conservative direction here is
    to ALLOW the close because we want maintenance available in non-live
    postures), so we only refuse when we positively read ALLOW_LIVE.
    """
    if not OPERATOR_INTENT_PATH.is_file():
        return True, "operator_intent.json missing (not ALLOW_LIVE)"
    try:
        oi = json.loads(OPERATOR_INTENT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return True, f"operator_intent.json unreadable ({exc}); treated as not ALLOW_LIVE"
    mode = str(oi.get("operator_mode") or oi.get("mode") or "").upper().strip()
    if mode == "ALLOW_LIVE" or mode == "ALLOW":
        return False, f"operator_intent.operator_mode={mode}"
    return True, mode or "UNSET"


def _gate_check() -> Tuple[bool, Dict[str, str]]:
    """Run all fail-closed gates. Returns (allowed, reasons)."""
    reasons: Dict[str, str] = {}
    ok_exec, exec_msg = _check_exec_mode_paper()
    reasons["exec_mode"] = exec_msg
    ok_scr, scr_msg = _check_scr_safe()
    reasons["scr"] = scr_msg
    ok_livegate, lg_msg = _check_livegate_not_allow_live()
    reasons["livegate"] = lg_msg
    return (ok_exec and ok_scr and ok_livegate), reasons


def _read_position_guard() -> Dict[str, Any]:
    if not position_guard.STATE_PATH.is_file():
        return {}
    try:
        return json.loads(position_guard.STATE_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def clear_trade_closer_queue_entry(
    strategy: str,
    symbol: str,
    state_path: Optional[Path] = None,
) -> bool:
    """Atomically remove the (strategy, symbol) FIFO entry from trade_closer_state.

    On-disk shape: `{"queues": [{"strategy": ..., "symbol": ..., "lots": [...]}],
    "processed_fill_ids": [...]}`. Symbols are normalized to upper-case for the
    match (mirrors `_rebuild_guard_from_paper_ledger` at chad/core/live_loop.py).

    Returns True iff at least one entry was removed and persisted.
    """
    target = Path(state_path) if state_path is not None else TRADE_CLOSER_STATE_PATH
    if not target.is_file():
        return False
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    queues = data.get("queues")
    if not isinstance(queues, list):
        return False
    needle_strategy = str(strategy).strip()
    needle_symbol = str(symbol).strip().upper()
    new_queues = []
    removed = 0
    for entry in queues:
        if not isinstance(entry, dict):
            new_queues.append(entry)
            continue
        e_strategy = str(entry.get("strategy", "") or "").strip()
        e_symbol = str(entry.get("symbol", "") or "").strip().upper()
        if e_strategy == needle_strategy and e_symbol == needle_symbol:
            removed += 1
            continue
        new_queues.append(entry)
    if removed == 0:
        return False
    data["queues"] = new_queues
    data["last_operator_clear_utc"] = _utc_now_iso()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(target)
    return True


def _append_audit_record(record: Dict[str, Any]) -> Path:
    OPERATOR_ACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = OPERATOR_ACTIONS_DIR / f"{_today_utc_date()}.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="GAP-028 §5.2 — close a stale per-strategy guard entry against broker truth.",
    )
    ap.add_argument("--strategy", required=True, help="Strategy name (left side of guard key).")
    ap.add_argument("--symbol", required=True, help="Symbol (right side of guard key).")
    ap.add_argument("--reason", required=True, help="Free-text reason recorded in closed_by + audit.")
    ap.add_argument("--by", required=True, help="Operator/agent identifier recorded in audit.")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Required affirmative confirmation. No auto-yes default.",
    )
    return ap


def run(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_argparser().parse_args(argv)

    strategy = str(args.strategy).strip()
    symbol = str(args.symbol).strip().upper()
    reason = str(args.reason).strip()
    by = str(args.by).strip()

    if not args.confirm:
        LOG.error("REFUSE: --confirm flag is required (no auto-yes default).")
        return 2

    if strategy.startswith("broker_sync") or strategy == "broker_sync":
        LOG.error(
            "REFUSE: broker_sync|* entries are owned by _rebuild_guard_from_broker; "
            "use broker reconciliation tooling, not this CLI (strategy=%s symbol=%s).",
            strategy, symbol,
        )
        return 3

    allowed, gate_reasons = _gate_check()
    if not allowed:
        LOG.error(
            "REFUSE: fail-closed gate(s) tripped — exec_mode=%s scr=%s livegate=%s",
            gate_reasons["exec_mode"], gate_reasons["scr"], gate_reasons["livegate"],
        )
        return 4

    key = f"{strategy}|{symbol}"
    state = _read_position_guard()
    previous_entry = state.get(key)

    # Idempotent path: missing entry OR already closed → no-op WARNING, exit 0.
    if not isinstance(previous_entry, dict):
        LOG.warning(
            "IDEMPOTENT_NOOP: position_guard has no entry for key=%s — nothing to close.",
            key,
        )
        return 0
    if previous_entry.get("open") is not True:
        LOG.warning(
            "IDEMPOTENT_NOOP: position_guard entry key=%s already closed "
            "(open=%s, last_state=%s, closed_by=%s).",
            key,
            previous_entry.get("open"),
            previous_entry.get("last_state"),
            previous_entry.get("closed_by"),
        )
        return 0

    # ---- atomic close + FIFO clear (GAP-028 §4 invariant) -----------------
    closed_ok = position_guard.close_stale_position_from_broker_truth(
        strategy=strategy,
        symbol=symbol,
        reason=reason,
        evidence={
            "operator": by,
            "operator_action_utc": _utc_now_iso(),
            "source": "scripts/close_guard_entry.py",
            "previous_entry_open": True,
        },
    )
    if not closed_ok:
        LOG.error(
            "INTERNAL: close_stale_position_from_broker_truth returned False for key=%s "
            "after presence check (race?). Aborting before FIFO clear.",
            key,
        )
        return 5

    fifo_cleared = clear_trade_closer_queue_entry(strategy, symbol)

    audit_record = {
        "schema_version": "operator_action.close_guard_entry.v1",
        "ts_utc": _utc_now_iso(),
        "action": "close_guard_entry",
        "strategy": strategy,
        "symbol": symbol,
        "key": key,
        "reason": reason,
        "by": by,
        "previous_entry": previous_entry,
        "trade_closer_fifo_cleared": fifo_cleared,
        "exec_mode": gate_reasons["exec_mode"],
        "scr_state": gate_reasons["scr"],
        "livegate_intent": gate_reasons["livegate"],
    }
    audit_path = _append_audit_record(audit_record)

    LOG.info(
        "CLOSED key=%s reason=%s by=%s fifo_cleared=%s audit=%s",
        key, reason, by, fifo_cleared, audit_path,
    )
    return 0


def main() -> int:
    try:
        return run()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("UNCAUGHT: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
