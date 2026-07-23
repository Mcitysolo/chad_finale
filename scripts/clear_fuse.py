#!/usr/bin/env python3
"""
scripts/clear_fuse.py

Operator recovery tool — manually clear a tripped fuse (LC2/LC3) BEFORE the
session roll. Fuses self-clear at the next session window automatically
(runtime/fuse_box_state.json re-derives from the trusted ledger every cycle);
this tool is for the case where an operator has reviewed a trip and wants the
bucket back on the menu early.

Mechanism: writes an entry to runtime/fuse_manual_clears.json keyed by the
CURRENT session window. The fuse evaluator honours it — the bucket reads
untripped for the rest of this session even though the ledger still shows the
streak (recorded honestly as manually_cleared=True, with the previous_* counters
preserved, GAP-018 idiom). The override auto-expires at the session roll; no
cleanup needed.

Fail-closed (mirrors scripts/close_guard_entry.py): refuses unless exec_mode
resolves to paper/dry_run and SCR is CONFIDENT/CAUTIOUS. This tool never
touches an order path and never closes a position — it only un-blocks future
ENTRIES for one bucket for one session.

Usage:
    python3 scripts/clear_fuse.py --fuse-id family:gamma
    python3 scripts/clear_fuse.py --fuse-id symbol:TLT --by reviewer --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNTIME_DIR = REPO_ROOT / "runtime"
STATE_PATH = RUNTIME_DIR / "fuse_box_state.json"
CLEARS_PATH = RUNTIME_DIR / "fuse_manual_clears.json"
SCR_STATE_PATH = RUNTIME_DIR / "scr_state.json"
_SAFE_SCR_STATES = frozenset({"CONFIDENT", "CAUTIOUS"})


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _check_exec_mode_paper():
    try:
        from chad.execution.execution_config import get_execution_mode, is_paper_mode

        if not is_paper_mode():
            return False, f"exec_mode={get_execution_mode()} (not paper/dry_run)"
        return True, f"exec_mode={get_execution_mode()}"
    except Exception as exc:  # noqa: BLE001
        return False, f"exec_mode check raised {type(exc).__name__}: {exc}"


def _check_scr_safe():
    if not SCR_STATE_PATH.is_file():
        return False, "scr_state.json missing"
    try:
        scr = json.loads(SCR_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"scr_state unreadable: {exc}"
    state = str(scr.get("state", "")).upper()
    if state not in _SAFE_SCR_STATES:
        return False, f"scr_state={state or 'UNKNOWN'} (not CONFIDENT/CAUTIOUS)"
    return True, f"scr_state={state}"


def _gate_check():
    reasons = {}
    ok_exec, reasons["exec_mode"] = _check_exec_mode_paper()
    ok_scr, reasons["scr"] = _check_scr_safe()
    return (ok_exec and ok_scr), reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manually clear a tripped fuse for the current session."
    )
    parser.add_argument("--fuse-id", required=True,
                        help="Fuse id to clear, e.g. family:gamma / symbol:TLT.")
    parser.add_argument("--by", default="operator", help="Who is clearing (cleared_by).")
    parser.add_argument("--reason", default="manual_operator_clear",
                        help="Reason (stored in the override + evidence).")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt.")
    args = parser.parse_args()

    allowed, reasons = _gate_check()
    print(json.dumps({"gate_reasons": reasons}, indent=2))
    if not allowed:
        print("REFUSED: fail-closed gate(s) not satisfied.", file=sys.stderr)
        return 2

    state = {}
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            state = {}
    fuses = {r.get("fuse_id"): r for r in (state.get("fuses") or [])
             if isinstance(r, dict)}
    row = fuses.get(args.fuse_id)
    if row is None:
        print(f"REFUSED: fuse_id {args.fuse_id!r} not present in "
              f"{STATE_PATH.name}. Known: {sorted(fuses)}", file=sys.stderr)
        return 2
    if not row.get("tripped"):
        print(f"REFUSED: fuse {args.fuse_id!r} is not tripped "
              f"(nothing to clear).", file=sys.stderr)
        return 2

    window = state.get("session_window_start_utc")
    if not window:
        print("REFUSED: fuse_box_state has no session_window_start_utc "
              "(evaluator has not run this session).", file=sys.stderr)
        return 2

    if not args.yes:
        resp = input(
            f"Clear tripped fuse {args.fuse_id} (streak="
            f"{row.get('consecutive_losers')}, net={row.get('session_net_pnl')}) "
            f"for session {window}? [y/N] "
        )
        if resp.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    existing = {}
    if CLEARS_PATH.is_file():
        try:
            existing = json.loads(CLEARS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    cleared = existing.get("cleared") if isinstance(existing.get("cleared"), dict) else {}
    cleared[args.fuse_id] = {
        "session_window_start": window,
        "by": args.by,
        "reason": args.reason,
        "cleared_at_utc": _iso_now(),
        # GAP-018 idiom: preserve the counters at clear time.
        "previous_tripped_at_utc": row.get("tripped_at_utc"),
        "previous_consecutive_losers": row.get("consecutive_losers"),
        "previous_session_net_pnl": row.get("session_net_pnl"),
    }
    out = dict(existing)
    out["cleared"] = cleared
    out["schema_version"] = "fuse_manual_clears.v1"
    out["updated_at_utc"] = _iso_now()

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(CLEARS_PATH) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.replace(CLEARS_PATH)

    print(json.dumps({
        "cleared_fuse_id": args.fuse_id,
        "session_window_start": window,
        "previous_consecutive_losers": row.get("consecutive_losers"),
        "note": "Effective next evaluator cycle; auto-expires at session roll.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
