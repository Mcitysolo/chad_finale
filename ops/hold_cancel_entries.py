#!/usr/bin/env python3
"""
ops/hold_cancel_entries.py — W6B-13 (INCIDENT-0723 follow-up).

Cancel ENTRY orders already working at the broker when an operator hold is
applied. Flag-gated, default OFF, dry-run by default, paper-only, Channel-1.

THE GAP THIS CLOSES
-------------------
Both INCIDENT-0723 D4 defects are fixed: a hold now PERSISTS
(`operator_intent_refresher._unexpired_hold`) and is CONSULTED
(`live_gate._load_operator_intent`, failing closed to DENY_ALL). Neither
touches orders **already working at the broker**. `live_gate`/`operator_intent`
are admission control on NEW intents; an order submitted seconds before the
hold keeps living at IBKR and can still fill. The incident record's own
language — "the hard brake remains operator-side: systemctl stop
chad-live-loop" — is an admission that no in-band brake reaches working orders.

TWO TRAPS THIS DESIGN EXISTS TO AVOID
-------------------------------------
`chad/core/paper_position_closer.py:359-367` already implements order
cancellation, and reusing it as-is would be actively dangerous:

1. **It cancels indiscriminately.** Its loop cancels EVERY open order. In a
   flatten that is intended; applied to a hold it would cancel protective and
   exit orders, leaving positions naked — strictly worse than the problem being
   solved.
2. **It enumerates with `ib.openOrders()`, which is clientId-scoped.** Per the
   standing IBKR probe methodology, that call silently returns nothing for
   other clients' orders — it does not error. A hold-cancel built on it would
   report "cancelled 0, all clear" while orders placed under a different
   clientId keep working. That is the trap most likely to produce a confident,
   wrong success message, so it is treated here as the primary correctness
   requirement rather than a footnote.

   Enumeration therefore uses `reqAllOpenOrders*` and **aborts** if unavailable
   — it never falls back to the scoped call. (Same contract as
   `scripts/flatten_all.py:262-271`. That file was not read while writing this:
   the chad-order-guard hook blocked access, which is the guard working
   correctly, so the pattern is reimplemented from the documented contract.)

CLASSIFICATION IS FAIL-CLOSED, AND HAS NO CHAD TAG TO LEAN ON
--------------------------------------------------------------
CHAD does not set `orderRef` on broker orders, so an order enumerated from the
broker carries **no CHAD intent_class**. Entry-vs-exit cannot be looked up; it
must be inferred from broker-visible attributes plus the position book.

An order is cancellable ONLY if positively classified ENTRY. Every other
outcome — protective, reduce-only, unknown, unparseable — is LEFT ALONE.
Ambiguity resolves to doing nothing.

Usage (operator, in the terminal — Channel 1):

    python3 -m ops.hold_cancel_entries --plan            # dry run (default)
    CHAD_HOLD_CANCEL_ENTRIES=1 python3 -m ops.hold_cancel_entries --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runtime" / "hold_cancel_report.json"
REPORT_SCHEMA = "hold_cancel_report.v1"

ENABLE_FLAG = "CHAD_HOLD_CANCEL_ENTRIES"
PAPER_MODES = ("paper", "dry_run")

# Order types that are protective by construction. A stop/trail is never an
# entry in CHAD's book, and cancelling one leaves a position unprotected.
PROTECTIVE_ORDER_TYPES = frozenset({
    "STP", "STP LMT", "STPLMT", "TRAIL", "TRAIL LIMIT", "TRAILLIMIT",
    "MIT", "LIT", "MTL", "STP PRT",
})

# Only these can ever be an entry. Anything else is unknown -> left alone.
ENTRY_CAPABLE_ORDER_TYPES = frozenset({"LMT", "MKT", "MIDPRICE", "REL"})


class HoldCancelAbort(RuntimeError):
    """Fail-closed abort. Nothing was cancelled."""


@dataclass
class Classification:
    verdict: str          # entry | protective | reduce_only | unknown
    reason: str
    cancellable: bool = False


@dataclass
class OrderRow:
    order_id: int
    client_id: int
    perm_id: int
    symbol: str
    sec_type: str
    action: str
    total_qty: float
    order_type: str
    parent_id: int
    oca_group: str
    status: str
    classification: Optional[Classification] = None
    action_taken: str = "none"
    result: str = ""

    def to_dict(self) -> Dict[str, Any]:
        c = self.classification
        return {
            "order_id": self.order_id,
            "client_id": self.client_id,
            "perm_id": self.perm_id,
            "symbol": self.symbol,
            "sec_type": self.sec_type,
            "action": self.action,
            "total_qty": self.total_qty,
            "order_type": self.order_type,
            "parent_id": self.parent_id,
            "oca_group": self.oca_group,
            "status": self.status,
            "verdict": c.verdict if c else "unknown",
            "verdict_reason": c.reason if c else "unclassified",
            "cancellable": bool(c.cancellable) if c else False,
            "action_taken": self.action_taken,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_gates(env: Optional[Mapping[str, str]] = None, *, execute: bool) -> None:
    """Fail closed before touching the broker.

    Mirrors scripts/close_guard_entry.py's posture: paper-only, and the flag is
    required for execution but NOT for planning — an operator must always be
    able to see what WOULD be cancelled without arming anything.
    """
    env = env if env is not None else os.environ

    mode = str(env.get("CHAD_EXECUTION_MODE", "") or "").strip().lower()
    if mode not in PAPER_MODES:
        raise HoldCancelAbort(
            f"refusing: CHAD_EXECUTION_MODE={mode!r} is not one of {PAPER_MODES}. "
            "This tool cancels working broker orders and is paper-only."
        )

    if execute:
        flag = str(env.get(ENABLE_FLAG, "") or "").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            raise HoldCancelAbort(
                f"refusing: --execute requires {ENABLE_FLAG}=1. Default is OFF, and "
                "planning does not require it."
            )


# ---------------------------------------------------------------------------
# Cross-client enumeration
# ---------------------------------------------------------------------------

def enumerate_all_open_orders(ib: Any) -> List[Any]:
    """Cross-client open orders.

    MUST use reqAllOpenOrders. `ib.openOrders()` is clientId-scoped and returns
    an EMPTY list for other clients' orders without erroring, which would make
    this tool report a confident, wrong "nothing to cancel".

    Aborts rather than degrading — a fallback to the scoped call is exactly the
    failure this function exists to prevent.
    """
    for name in ("reqAllOpenOrdersAsync", "reqAllOpenOrders", "reqAllOpenOrdersSync"):
        fn = getattr(ib, name, None)
        if fn is None:
            continue
        try:
            result = fn()
            if name.endswith("Async"):
                result = ib.run(result)
            return list(result or [])
        except Exception as exc:  # noqa: BLE001
            raise HoldCancelAbort(
                f"cross-client enumeration via {name} failed: {exc!r}. Refusing to "
                "fall back to clientId-scoped openOrders(), which would silently "
                "hide other clients' working orders."
            ) from exc
    raise HoldCancelAbort(
        "cross-client open-order enumeration unavailable (no reqAllOpenOrders* on "
        "this IB client). Refusing to proceed with a clientId-scoped call."
    )


def _position_map(ib: Any) -> Dict[str, float]:
    """symbol -> signed position. Empty dict means UNKNOWN, not flat."""
    out: Dict[str, float] = {}
    try:
        for p in ib.positions() or []:
            sym = str(getattr(getattr(p, "contract", None), "symbol", "") or "").upper()
            if not sym:
                continue
            out[sym] = out.get(sym, 0.0) + float(getattr(p, "position", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return {}
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_order(
    row: OrderRow,
    positions: Mapping[str, float],
    *,
    positions_known: bool,
) -> Classification:
    """Positively identify an ENTRY, or refuse to touch the order.

    Order of checks matters: every protective signal is tested BEFORE any
    entry signal, so an order that looks like both is protective.
    """
    otype = str(row.order_type or "").strip().upper()

    if row.parent_id:
        return Classification(
            "protective",
            f"parentId={row.parent_id} — child leg of a bracket; cancelling it "
            "would unprotect the parent position",
        )

    if str(row.oca_group or "").strip():
        return Classification(
            "protective",
            f"ocaGroup={row.oca_group!r} — OCA leg; cancelling one leg silently "
            "changes the behaviour of the others",
        )

    if otype in PROTECTIVE_ORDER_TYPES:
        return Classification(
            "protective", f"orderType={otype} is a stop/trail — protective by construction"
        )

    if otype not in ENTRY_CAPABLE_ORDER_TYPES:
        return Classification(
            "unknown",
            f"orderType={otype!r} is not on the entry-capable list "
            f"{sorted(ENTRY_CAPABLE_ORDER_TYPES)} — cannot positively classify, so left alone",
        )

    if not positions_known:
        return Classification(
            "unknown",
            "broker positions unavailable — cannot tell an entry from a "
            "reduce-only order, so left alone",
        )

    pos = float(positions.get(str(row.symbol or "").upper(), 0.0) or 0.0)
    side = str(row.action or "").strip().upper()
    if side not in ("BUY", "SELL"):
        return Classification("unknown", f"unrecognised action={row.action!r}")

    # Reduce-only: the order opposes an existing position. Treat ANY opposing
    # order as an exit, including one larger than the position (a flip) — a
    # flip's closing half is still an exit, and cancelling it is not this
    # tool's business.
    if pos > 0 and side == "SELL":
        return Classification(
            "reduce_only",
            f"SELL against a long position of {pos:g} — reduces or flips existing "
            "exposure, so it is an exit, not an entry",
        )
    if pos < 0 and side == "BUY":
        return Classification(
            "reduce_only",
            f"BUY against a short position of {pos:g} — reduces or flips existing "
            "exposure, so it is an exit, not an entry",
        )

    return Classification(
        "entry",
        f"orderType={otype}, no parentId, no OCA group, and {side} "
        f"{'opens' if pos == 0 else 'increases'} exposure "
        f"(current position {pos:g})",
        cancellable=True,
    )


def _row_from_broker(o: Any) -> Optional[OrderRow]:
    """Adapt an ib_async OpenOrder/Trade into an OrderRow. Returns None if the
    shape is unrecognisable — which the caller records as unknown."""
    order = getattr(o, "order", o)
    contract = getattr(o, "contract", None)
    status_obj = getattr(o, "orderStatus", None)

    # A bare object() would otherwise sail through every getattr default and
    # become an all-zero OrderRow — parseable-looking garbage. It would still
    # be left alone (order_type "" is unclassifiable), but it would vanish from
    # the unparseable list, and an operator reading the report would never know
    # the broker returned something this tool did not understand.
    if not hasattr(order, "orderId") or not hasattr(order, "action"):
        return None

    try:
        return OrderRow(
            order_id=int(getattr(order, "orderId", 0) or 0),
            client_id=int(getattr(order, "clientId", 0) or 0),
            perm_id=int(getattr(order, "permId", 0) or 0),
            symbol=str(getattr(contract, "symbol", "") or "").upper(),
            sec_type=str(getattr(contract, "secType", "") or ""),
            action=str(getattr(order, "action", "") or ""),
            total_qty=float(getattr(order, "totalQuantity", 0.0) or 0.0),
            order_type=str(getattr(order, "orderType", "") or ""),
            parent_id=int(getattr(order, "parentId", 0) or 0),
            oca_group=str(getattr(order, "ocaGroup", "") or ""),
            status=str(getattr(status_obj, "status", "") or ""),
        )
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Plan / execute
# ---------------------------------------------------------------------------

def build_plan(
    open_orders: Sequence[Any],
    positions: Mapping[str, float],
    *,
    positions_known: bool,
) -> Tuple[List[OrderRow], List[Dict[str, Any]]]:
    rows: List[OrderRow] = []
    unparseable: List[Dict[str, Any]] = []
    for o in open_orders:
        row = _row_from_broker(o)
        if row is None:
            unparseable.append({"repr": repr(o)[:200], "note": "left alone"})
            continue
        row.classification = classify_order(row, positions, positions_known=positions_known)
        rows.append(row)
    return rows, unparseable


def write_report(payload: Dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_report(
    rows: Sequence[OrderRow],
    unparseable: Sequence[Dict[str, Any]],
    *,
    executed: bool,
    positions_known: bool,
    enumeration: str,
) -> Dict[str, Any]:
    """A partial or zero-cancel outcome must be LOUD, never a clean success."""
    cancellable = [r for r in rows if r.classification and r.classification.cancellable]
    cancelled = [r for r in cancellable if r.action_taken == "cancelled"]
    failed = [r for r in cancellable if r.action_taken == "cancel_failed"]
    not_cancelled = [
        {
            "order_id": r.order_id, "client_id": r.client_id, "symbol": r.symbol,
            "verdict": r.classification.verdict if r.classification else "unknown",
            "reason": r.classification.reason if r.classification else "unclassified",
        }
        for r in rows
        if not (r.classification and r.classification.cancellable)
    ]

    # `complete` requires that something was actually cancelled. Without the
    # `cancelled` term, an execute run that found nothing cancellable scores
    # 0 == 0 and reports "complete" — precisely the confident-wrong-success this
    # tool is built to avoid. Zero cancels is `nothing_cancelled`, always.
    complete = executed and not failed and bool(cancelled) and len(cancelled) == len(cancellable)
    return {
        "schema_version": REPORT_SCHEMA,
        "ts_utc": _utc_now_iso(),
        "mode": "execute" if executed else "plan",
        "enumeration": enumeration,
        "positions_known": positions_known,
        "orders_seen": len(rows),
        "orders_unparseable": list(unparseable),
        "client_ids_seen": sorted({r.client_id for r in rows}),
        "counts": {
            "cancellable": len(cancellable),
            "cancelled": len(cancelled),
            "cancel_failed": len(failed),
            "left_alone": len(not_cancelled),
        },
        "orders": [r.to_dict() for r in rows],
        "not_cancelled": not_cancelled,
        "outcome": (
            "complete" if complete
            else "planned" if not executed
            else "partial" if cancelled
            else "nothing_cancelled"
        ),
        "loud_notes": [
            note for note in (
                ("NO cross-client enumeration confirmation — treat counts as "
                 "unreliable") if enumeration != "reqAllOpenOrders" else None,
                ("broker positions were UNAVAILABLE; every order was left alone "
                 "because an entry cannot be distinguished from a reduce-only "
                 "order without them") if not positions_known else None,
                (f"{len(failed)} cancel(s) FAILED — orders may still be working")
                if failed else None,
                ("zero orders cancelled; this is NOT proof the book is clear, "
                 "only that nothing was positively classified as an entry")
                if executed and not cancelled else None,
            ) if note
        ],
    }


def run(
    ib: Any,
    *,
    execute: bool,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    check_gates(env, execute=execute)

    open_orders = enumerate_all_open_orders(ib)
    positions = _position_map(ib)
    positions_known = bool(positions) or _positions_probe_ok(ib)

    rows, unparseable = build_plan(open_orders, positions, positions_known=positions_known)

    if execute:
        by_id = {}
        for o in open_orders:
            r = _row_from_broker(o)
            if r is not None:
                by_id[r.order_id] = o
        for row in rows:
            if not (row.classification and row.classification.cancellable):
                continue
            target = by_id.get(row.order_id)
            if target is None:
                row.action_taken = "cancel_failed"
                row.result = "order object not found for cancellation"
                continue
            try:
                ib.cancelOrder(getattr(target, "order", target))
                row.action_taken = "cancelled"
                row.result = "cancelOrder submitted"
            except Exception as exc:  # noqa: BLE001
                row.action_taken = "cancel_failed"
                row.result = repr(exc)[:200]

    return build_report(
        rows, unparseable, executed=execute,
        positions_known=positions_known, enumeration="reqAllOpenOrders",
    )


def _positions_probe_ok(ib: Any) -> bool:
    """Distinguish "flat" from "could not read positions".

    An empty position map is ambiguous, and the ambiguity is dangerous: read as
    "flat", every SELL looks like an entry. Only an explicit, successful call
    returning an empty sequence counts as a known-flat book.
    """
    try:
        return ib.positions() is not None
    except Exception:  # noqa: BLE001
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cancel ENTRY orders working at the broker (hold support). "
                    "Dry-run by default; Channel-1 operator tool."
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", default=True,
                   help="show what WOULD be cancelled (default)")
    g.add_argument("--execute", action="store_true",
                   help=f"actually cancel. Requires {ENABLE_FLAG}=1.")
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002)
    ap.add_argument("--client-id", type=int, default=97)
    args = ap.parse_args(argv)

    execute = bool(args.execute)
    try:
        check_gates(execute=execute)
    except HoldCancelAbort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2

    try:
        from ib_async import IB
    except Exception as exc:  # noqa: BLE001
        print(f"ABORT: ib_async unavailable: {exc}", file=sys.stderr)
        return 2

    ib = IB()
    try:
        try:
            ib.connect(args.host, args.port, clientId=args.client_id, timeout=20)
        except Exception as exc:  # noqa: BLE001
            print(f"ABORT: IBKR connect failed: {exc}", file=sys.stderr)
            return 2
        report = run(ib, execute=execute)
    except HoldCancelAbort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:  # noqa: BLE001
            pass

    write_report(report, Path(args.report))
    print(json.dumps(report["counts"], indent=2))
    print(f"outcome={report['outcome']} report={args.report}")
    for note in report["loud_notes"]:
        print(f"!! {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
