#!/usr/bin/env python3
"""LC6 EMERGENCY FLATTEN-ALL (W4B-5/6) — one command, both lanes.

Cancel all open orders + close all positions: reduce-only, clamped to LIVE
broker truth at submit time (INCIDENT-0713 defense — quantities NEVER come
from the ledger), broker-confirmed with a residual re-probe and per-order
ack/fill SLA measurement, idempotent, typed-confirm gated, coach-narrated.

DRILL (dry-run) IS THE DEFAULT INVOCATION: without ``--execute`` the full
chain runs read-only probes, builds the complete cancel + close inventories,
pushes every IBKR close through the adapter's ``dry_run`` short-circuit
(proving the chain reaches the placeOrder boundary without executing), and
writes the drill-proof artifact. Execution requires BOTH
``--execute --confirm FLATTEN-ALL`` and an interactive re-typed token; a
wrong token is a HARD ERROR (exit 2) — never a silent degrade (anti-pattern:
epoch_reset). Gates are deliberately MINIMAL (D4): paper/dry_run execution
mode fail-closed + the tokens. NOT SCR/reconciliation-gated — emergencies
are exactly when those are broken.

Scope (D1): ``--scope chad`` (default) closes CHAD-attributed strategy legs,
each clamped to the broker net; operator-excluded symbols are untouchable
and every one left standing is NAMED in the report (D1 rider).
``--scope broker-all`` additionally closes operator/excluded positions and
requires the second token ``INCLUDE-EXCLUDED``.

Cancel (D3): ``reqGlobalCancel`` — emergency semantics: every working order
across ALL clientIds dies, including manual ones; the report enumerates the
non-CHAD collateral by name (D3 rider), never silently.

Operator-terminal-only by standing policy (chad-order-guard hook).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

REPORT_SCHEMA = "flatten_all_report.v1"
DRILL_SCHEMA = "flatten_drill_proof.v1"
CONFIRM_TOKEN = "FLATTEN-ALL"
INCLUDE_EXCLUDED_TOKEN = "INCLUDE-EXCLUDED"

# Bounded waits (seconds)
CANCEL_VERIFY_TIMEOUT_S = 30.0
CANCEL_VERIFY_POLL_S = 1.0
ORDER_TERMINAL_TIMEOUT_S = 60.0
ORDER_POLL_S = 0.5

_TERMINAL_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
# Adapter pseudo-statuses that end an order's lifecycle without a broker fill.
_ADAPTER_TERMINAL = {"dry_run", "duplicate_blocked", "margin_blocked",
                     "market_closed", "futures_execution_disabled",
                     "duplicate_open_order", "suppressed_open_orders_cap"}


class FlattenAbort(RuntimeError):
    """Refuse-and-preserve: raised when truth is UNKNOWN (never guessed flat)
    or a gate refuses. The tool exits loudly; nothing has been mutated beyond
    whatever phase already completed (each phase is idempotent)."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Gates (D4: minimal + typed; wrong token = hard error)
# --------------------------------------------------------------------------- #

def check_gates(env: Mapping[str, str],
                live_readiness_path: Optional[Path] = None) -> Dict[str, Any]:
    """Fail-closed mode gate + live_readiness posture cross-check (plan §2.1).

    Only CHAD_EXECUTION_MODE paper|dry_run may pass (the ONLY modes that exist
    besides live — never accept invented variants); missing/unknown REFUSES: an
    emergency tool must still never touch a live posture without its own
    separate authorization. Cross-check: a readable live_readiness publisher
    state with ``ready_for_live: true`` REFUSES too (mid-transition mismatch —
    a live-mode flatten is a separate authorization, out of W4B scope). An
    unreadable file only WARNS: the env mode gate is the hard wall, and an
    emergency in paper must not be blocked by a dead advisory publisher."""
    raw = str(env.get("CHAD_EXECUTION_MODE", "") or "").strip().lower()
    mode_ok = raw in ("paper", "dry_run")
    ready_for_live: Optional[bool] = None
    readiness_warn: Optional[str] = None
    if live_readiness_path is not None:
        try:
            doc = json.loads(live_readiness_path.read_text(encoding="utf-8"))
            ready_for_live = bool(doc.get("ready_for_live"))
        except Exception as exc:
            readiness_warn = f"live_readiness unreadable (advisory): {exc}"
    return {
        "execution_mode_raw": raw or None,
        "mode_gate_ok": mode_ok,
        "ready_for_live": ready_for_live,
        "readiness_warn": readiness_warn,
        "ok": mode_ok and ready_for_live is not True,
    }


def verify_tokens(*, execute: bool, confirm: Optional[str], scope: str,
                  scope_confirm: Optional[str]) -> None:
    """Token walls. Raises FlattenAbort (exit 2 at the CLI) on ANY mismatch —
    a bad token must never degrade to a quieter mode."""
    if execute and confirm != CONFIRM_TOKEN:
        raise FlattenAbort(
            f"--execute requires --confirm {CONFIRM_TOKEN} (got {confirm!r})")
    if scope == "broker-all" and scope_confirm != INCLUDE_EXCLUDED_TOKEN:
        raise FlattenAbort(
            "--scope broker-all touches operator-excluded positions and "
            f"requires --scope-confirm {INCLUDE_EXCLUDED_TOKEN}")


# --------------------------------------------------------------------------- #
# Phase 0 — PROBE (read-only, fail-closed; unknown is never flat)
# --------------------------------------------------------------------------- #

@dataclass
class BrokerProbe:
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # SYM -> {qty(signed), sec_type, con}
    all_open_orders: List[Dict[str, Any]] = field(default_factory=list)
    own_open_orders_count: int = 0
    probed_at_utc: str = ""


def probe_broker(ib: Any) -> BrokerProbe:
    """Positions + cross-client open orders from a CONNECTED ib. Fail-closed:
    an unanswerable connection raises FlattenAbort (XOV2: a dead socket's
    reset cache is byte-identical to flat — unknown must never read as flat)."""
    if not _api_connected(ib):
        raise FlattenAbort("BROKER_TRUTH_UNAVAILABLE: connection cannot answer "
                           "— refusing (unknown is never flat)")
    probe = BrokerProbe(probed_at_utc=_iso(_utcnow()))
    for pos in (ib.positions() or []):
        try:
            contract = getattr(pos, "contract", None)
            symbol = str(getattr(contract, "symbol", "") or "").upper()
            qty = float(getattr(pos, "position", 0.0) or 0.0)
            if not symbol or qty == 0.0:
                continue
            probe.positions[symbol] = {
                "qty": qty,
                "sec_type": str(getattr(contract, "secType", "") or ""),
                "local_symbol": str(getattr(contract, "localSymbol", "") or ""),
                "expiry": str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""),
                "avg_cost": float(getattr(pos, "avgCost", 0.0) or 0.0),
            }
        except Exception:
            continue
    # Cross-client enumeration — reqAllOpenOrders, NEVER the clientId-scoped
    # variant (2026-05-27 lore: a fresh probe id sees 0/0). The scoped count
    # is also captured for the drill's visibility-split proof.
    trades = _req_all_open_trades(ib)
    for t in trades:
        try:
            order = getattr(t, "order", None)
            contract = getattr(t, "contract", None)
            status = getattr(getattr(t, "orderStatus", None), "status", "") or ""
            probe.all_open_orders.append({
                "order_id": int(getattr(order, "orderId", 0) or 0),
                "perm_id": int(getattr(order, "permId", 0) or 0),
                "client_id": int(getattr(order, "clientId", -1)
                                 if getattr(order, "clientId", None) is not None else -1),
                "symbol": str(getattr(contract, "symbol", "") or "").upper(),
                "action": str(getattr(order, "action", "") or ""),
                "total_qty": float(getattr(order, "totalQuantity", 0.0) or 0.0),
                "order_type": str(getattr(order, "orderType", "") or ""),
                "status": str(status),
            })
        except Exception:
            continue
    try:
        probe.own_open_orders_count = len(ib.openOrders() or [])
    except Exception:
        probe.own_open_orders_count = -1
    return probe


def _api_connected(ib: Any) -> bool:
    """An unanswerable probe is not truth (broker_position_sync contract)."""
    try:
        if not bool(ib.isConnected()):
            return False
        return bool(getattr(getattr(ib, "client", None), "isConnected", lambda: True)())
    except Exception:
        return False


def _req_all_open_trades(ib: Any) -> List[Any]:
    """Cross-client open orders. Prefers reqAllOpenOrders (sync wrapper);
    a facade/fake may expose either name."""
    for name in ("reqAllOpenOrders", "reqAllOpenOrdersSync"):
        fn = getattr(ib, name, None)
        if callable(fn):
            try:
                return list(fn() or [])
            except Exception:
                continue
    raise FlattenAbort("open-order enumeration unavailable (reqAllOpenOrders)")


def crosscheck_snapshot(probe: BrokerProbe, snapshot_path: Path) -> List[Dict[str, Any]]:
    """Two-source confirm (advisory): live probe vs positions_snapshot.json.
    Divergences are WARNED and listed; the live probe always wins."""
    diffs: List[Dict[str, Any]] = []
    try:
        doc = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snap: Dict[str, float] = {}
        for row in doc.get("positions") or []:
            sym = str(row.get("symbol", "") or "").upper()
            if sym:
                snap[sym] = snap.get(sym, 0.0) + float(row.get("position") or row.get("qty") or 0.0)
        for sym in sorted(set(snap) | set(probe.positions)):
            live = float(probe.positions.get(sym, {}).get("qty", 0.0))
            s = snap.get(sym, 0.0)
            if abs(live - s) > 1e-9:
                diffs.append({"symbol": sym, "live_qty": live, "snapshot_qty": s})
    except Exception as exc:
        diffs.append({"error": f"snapshot unreadable: {exc}"})
    return diffs


# --------------------------------------------------------------------------- #
# Scope resolution (D1) — targets + NAMED untouched remainder
# --------------------------------------------------------------------------- #

def load_excluded_symbols(config_path: Path) -> List[str]:
    """Operator-exclusion SSOT read directly from config (no chad imports;
    unreadable config falls back to the reconciler's {AAPL, MSFT} floor —
    never to an empty set)."""
    try:
        obj = json.loads(config_path.read_text(encoding="utf-8"))
        syms = set(obj.get("reconciler_non_chad_symbols") or [])
        syms |= set(obj.get("broker_preexisting_symbols") or [])
        syms |= set((obj.get("exclusion_policy") or {}).keys())
        out = sorted(s.upper() for s in syms)
        return out or ["AAPL", "MSFT"]
    except Exception:
        return ["AAPL", "MSFT"]


def _clamp_to_broker(strategy_qty: float, broker_qty: float) -> float:
    """context_positions clamp: same sign, min magnitude; 0 on mismatch/flat."""
    if broker_qty == 0.0 or (strategy_qty > 0.0) != (broker_qty > 0.0):
        return 0.0
    mag = min(abs(strategy_qty), abs(broker_qty))
    return mag if broker_qty > 0.0 else -mag


def resolve_targets(
    *,
    probe: BrokerProbe,
    guard_state: Mapping[str, Any],
    scope: str,
    excluded: List[str],
) -> Dict[str, Any]:
    """Build per-leg close targets clamped to the broker book, plus the NAMED
    untouched remainder (D1 rider: the operator must SEE what stays standing).

    chad scope: per-symbol strategy-leg net (guard strategy legs only — the
    broker_sync mirror book is truth, never summed with legs), clamped to the
    broker signed qty; excluded symbols untouchable. Legs consume a per-symbol
    broker budget so multi-leg symbols stay reduce-only in aggregate.
    broker-all: chad targets PLUS the remaining broker qty per symbol
    (excluded included) as operator_flatten legs."""
    excluded_set = {s.upper() for s in excluded}
    broker_signed = {s: float(v["qty"]) for s, v in probe.positions.items()}

    # strategy-leg net per symbol from the guard (open, non-mirror entries)
    strat_net: Dict[str, float] = {}
    legs_by_symbol: Dict[str, List[Tuple[str, float]]] = {}
    for key, entry in (guard_state or {}).items():
        if not isinstance(key, str) or key.startswith("_") or key.startswith("broker_sync|"):
            continue
        if not isinstance(entry, dict) or not entry.get("open"):
            continue
        sym = str(entry.get("symbol", "") or "").upper()
        side = str(entry.get("side", "") or "").upper()
        qty = abs(float(entry.get("quantity", 0.0) or 0.0))
        if not sym or qty <= 0.0 or side not in ("BUY", "SELL"):
            continue
        signed = qty if side == "BUY" else -qty
        strat_net[sym] = strat_net.get(sym, 0.0) + signed
        legs_by_symbol.setdefault(sym, []).append((key, signed))

    targets: List[Dict[str, Any]] = []
    untouched: List[Dict[str, Any]] = []

    for sym in sorted(set(broker_signed) | set(strat_net)):
        b = broker_signed.get(sym, 0.0)
        info = probe.positions.get(sym, {})
        if sym in excluded_set and scope != "broker-all":
            if b != 0.0:
                untouched.append({"symbol": sym, "broker_qty": b,
                                  "reason": "operator_excluded"})
            continue
        if b == 0.0:
            # broker flat -> nothing to close, whatever the ledger believes
            # (INCIDENT-0713: no order is ever minted from ledger belief).
            # NAMED, never silent: a ledger-only leg is a split-brain worth seeing.
            untouched.append({"symbol": sym, "broker_qty": 0.0,
                              "unclosed_qty": 0.0,
                              "ledger_qty": strat_net.get(sym, 0.0),
                              "reason": "broker_flat_ledger_only"})
            continue

        chad_net = _clamp_to_broker(strat_net.get(sym, 0.0), b)
        budget = abs(chad_net)
        # chad legs (skipped entirely for excluded symbols in chad scope above)
        if budget > 0.0 and sym not in excluded_set:
            for key, signed in sorted(legs_by_symbol.get(sym, [])):
                if budget <= 0.0:
                    break
                if (signed > 0.0) != (b > 0.0):
                    continue  # leg opposes broker sign -> not closable reduce-only
                leg_qty = min(abs(signed), budget)
                if leg_qty <= 0.0:
                    continue
                budget -= leg_qty
                targets.append({
                    "symbol": sym,
                    "position_key": key,
                    "strategy": key.split("|", 1)[0],
                    "open_side": "BUY" if b > 0.0 else "SELL",
                    "close_side": "SELL" if b > 0.0 else "BUY",
                    "quantity": leg_qty,
                    "sec_type": info.get("sec_type") or "STK",
                    "origin": "chad",
                })
        remainder = abs(b) - sum(t["quantity"] for t in targets if t["symbol"] == sym)
        if remainder > 1e-9:
            if scope == "broker-all":
                targets.append({
                    "symbol": sym,
                    "position_key": f"operator|{sym}",
                    "strategy": "operator_flatten",
                    "open_side": "BUY" if b > 0.0 else "SELL",
                    "close_side": "SELL" if b > 0.0 else "BUY",
                    "quantity": remainder,
                    "sec_type": info.get("sec_type") or "STK",
                    "origin": "operator" if sym in excluded_set else "unattributed",
                })
            else:
                untouched.append({
                    "symbol": sym, "broker_qty": b if b > 0 else b,
                    "unclosed_qty": remainder,
                    "reason": ("operator_excluded" if sym in excluded_set
                               else "not_chad_attributed"),
                })

    return {"targets": targets, "untouched": untouched, "scope": scope,
            "excluded_symbols": sorted(excluded_set)}


# --------------------------------------------------------------------------- #
# Phase 1 — CANCEL (D3: global, verified, collateral NAMED)
# --------------------------------------------------------------------------- #

def cancel_phase(ib: Any, *, chad_client_ids: List[int], execute: bool,
                 sleep: Callable[[float], None] = time.sleep,
                 timeout_s: float = CANCEL_VERIFY_TIMEOUT_S) -> Dict[str, Any]:
    """Enumerate (cross-client) -> reqGlobalCancel -> poll to zero.
    D3 rider: every order whose clientId is NOT a registered CHAD id goes in
    ``collateral_non_chad`` BY NAME — emergency semantics accepted, collateral
    listed, never silent. Drill mode enumerates and classifies but cancels
    nothing."""
    before = probe_broker(ib).all_open_orders
    known = set(chad_client_ids)
    collateral = [o for o in before if o.get("client_id") not in known]
    result: Dict[str, Any] = {
        "orders_before": before,
        "orders_before_count": len(before),
        "collateral_non_chad": collateral,       # D3 rider — named, never silent
        "executed": bool(execute),
        "verified_zero": None,
        "survivors": [],
    }
    if not execute:
        return result
    ib.reqGlobalCancel()
    deadline = time.monotonic() + timeout_s
    while True:  # always >=1 filtered re-probe, even at timeout_s=0
        remaining = [o for o in probe_broker(ib).all_open_orders
                     if o.get("status") not in _TERMINAL_STATUSES]
        if not remaining or time.monotonic() >= deadline:
            break
        sleep(CANCEL_VERIFY_POLL_S)
    result["verified_zero"] = not remaining
    result["survivors"] = remaining
    return result


# --------------------------------------------------------------------------- #
# Phase 2+3 — CLOSE + CONFIRM (submit, SLA, residual re-probe)
# --------------------------------------------------------------------------- #

def close_dicts_from_targets(
    targets: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reconciler-shaped close dicts (apply_close_intents contract) for the
    chad-origin legs. Operator/unattributed legs are returned separately —
    they must NOT go through apply_close_intents (its GAP-001 exclusion
    backstop is correct for every caller except this explicitly
    double-token-authorized override, which uses the direct adapter path)."""
    chad, operator = [], []
    for t in targets:
        d = {
            "symbol": t["symbol"],
            "action": "CLOSE",
            "open_side": t["open_side"],
            "close_side": t["close_side"],
            "quantity": float(t["quantity"]),
            "reason": f"flatten_all_{t['origin']}",
            "position_key": t["position_key"],
            "strategy": t["strategy"],
        }
        (chad if t["origin"] == "chad" else operator).append(d)
    return chad, operator


def measure_orders(submitted: List[Any], *, now_fn=_utcnow,
                   poll: Callable[[], None] = lambda: time.sleep(ORDER_POLL_S),
                   timeout_s: float = ORDER_TERMINAL_TIMEOUT_S) -> Dict[str, Any]:
    """Per-order ack/fill SLA from SubmittedOrder + (when present) the live
    ib_async Trade in ``raw['trade']`` (Trade.log rows are event-timestamped).
    Adapter pseudo-terminal statuses (dry_run, duplicate_blocked, ...) are
    terminal-with-no-broker-leg; duplicate_blocked is classified BENIGN
    (idempotent re-run)."""
    rows: List[Dict[str, Any]] = []
    for so in submitted or []:
        status = str(getattr(so, "status", "") or "")
        row: Dict[str, Any] = {
            "symbol": getattr(so, "symbol", None),
            "side": getattr(so, "side", None),
            "quantity": getattr(so, "quantity", None),
            "status": status,
            "idempotency_key": getattr(so, "idempotency_key", None),
            "ib_order_id": getattr(so, "ib_order_id", None),
            "benign_duplicate": status == "duplicate_blocked",
            "ack_ms": None,
            "fill_ms": None,
        }
        trade = None
        raw = getattr(so, "raw", None)
        if isinstance(raw, Mapping):
            trade = raw.get("trade")
        submitted_at = getattr(so, "submitted_at", None)
        if trade is not None and submitted_at is not None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                st = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
                if st in _TERMINAL_STATUSES:
                    break
                poll()
            try:
                log = list(getattr(trade, "log", []) or [])
                acks = [e for e in log
                        if str(getattr(e, "status", "")) not in ("PendingSubmit", "")]
                fills = [e for e in log if str(getattr(e, "status", "")) == "Filled"]
                if acks:
                    row["ack_ms"] = max(0.0, (acks[0].time - submitted_at).total_seconds() * 1000.0)
                if fills:
                    row["fill_ms"] = max(0.0, (fills[0].time - submitted_at).total_seconds() * 1000.0)
                row["status"] = str(getattr(getattr(trade, "orderStatus", None), "status", row["status"]))
            except Exception:
                pass
        rows.append(row)

    def _pct(vals: List[float], q: float) -> Optional[float]:
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return vals[idx]

    acks = [r["ack_ms"] for r in rows if r["ack_ms"] is not None]
    fills = [r["fill_ms"] for r in rows if r["fill_ms"] is not None]
    return {
        "orders": rows,
        "sla": {
            "ack_ms_p50": _pct(acks, 0.5), "ack_ms_p95": _pct(acks, 0.95),
            "fill_ms_p50": _pct(fills, 0.5), "fill_ms_p95": _pct(fills, 0.95),
            "measured_acks": len(acks), "measured_fills": len(fills),
        },
    }


def residual_check(ib: Any, resolution: Dict[str, Any]) -> Dict[str, Any]:
    """Re-probe (fail-closed) and verdict per targeted symbol: FLAT | RESIDUAL;
    untouched excluded/unattributed positions restated BY NAME (D1 rider)."""
    probe = probe_broker(ib)
    targeted = sorted({t["symbol"] for t in resolution["targets"]})
    per_symbol = []
    overall_flat = True
    for sym in targeted:
        left = float(probe.positions.get(sym, {}).get("qty", 0.0))
        # chad scope: "flat" means the CHAD-attributed share is gone; any
        # remaining broker qty must be exactly the untouched remainder.
        expected_left = sum(
            u.get("unclosed_qty", u.get("broker_qty", 0.0)) or 0.0
            for u in resolution["untouched"] if u.get("symbol") == sym
        )
        residual = abs(left) - abs(expected_left)
        flat = residual <= 1e-9
        overall_flat = overall_flat and flat
        per_symbol.append({"symbol": sym, "broker_qty_after": left,
                           "expected_untouched": expected_left,
                           "verdict": "FLAT" if flat else "RESIDUAL",
                           "residual_qty": max(0.0, residual)})
    untouched_named = [
        {**u, "verdict": "EXCLUDED_UNTOUCHED" if u.get("reason") == "operator_excluded"
                         else "UNTOUCHED"}
        for u in resolution["untouched"]
    ]
    return {
        "reprobed_at_utc": probe.probed_at_utc,
        "per_symbol": per_symbol,
        "untouched_named": untouched_named,   # D1 rider — always NAMED
        "overall": "FLAT" if overall_flat else "INCOMPLETE",
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_report(out_dir: Path, payload: Dict[str, Any], *, drill: bool,
                 now: Optional[datetime] = None) -> Path:
    now = now or _utcnow()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    if drill:
        out = out_dir / "ratification" / f"PROOF_FLATTEN_DRILL_{now.strftime('%Y%m%d')}.json"
        payload = {"schema_version": DRILL_SCHEMA, **payload}
    else:
        out = out_dir / f"flatten_all_{stamp}.json"
        payload = {"schema_version": REPORT_SCHEMA, **payload}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# CLI (W4B-6 wires the act phases + Kraken lane + narration + hold)
# --------------------------------------------------------------------------- #

def main(argv: List[str]) -> int:  # pragma: no cover - integration shell
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true",
                    help="actually cancel+close (default: DRILL dry-run)")
    ap.add_argument("--confirm", default=None,
                    help=f"required with --execute: {CONFIRM_TOKEN}")
    ap.add_argument("--scope", choices=("chad", "broker-all"), default="chad")
    ap.add_argument("--scope-confirm", default=None,
                    help=f"required with --scope broker-all: {INCLUDE_EXCLUDED_TOKEN}")
    ap.add_argument("--repo-root", type=Path, default=Path("/home/ubuntu/chad_finale"))
    ap.add_argument("--no-hold", action="store_true",
                    help="skip the post-flatten operator_intent EXIT_ONLY hold")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args(argv)

    import os
    try:
        verify_tokens(execute=args.execute, confirm=args.confirm,
                      scope=args.scope, scope_confirm=args.scope_confirm)
    except FlattenAbort as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    gates = check_gates(os.environ,
                        args.repo_root / "runtime" / "live_readiness.json")
    if gates["readiness_warn"]:
        print(f"WARN: {gates['readiness_warn']}", file=sys.stderr)
    if not gates["ok"]:
        print(f"REFUSED: execution-mode/readiness gate failed: {gates}",
              file=sys.stderr)
        return 2
    if args.execute:
        resp = input(f"Type {CONFIRM_TOKEN} to proceed with a REAL flatten: ")
        if resp.strip() != CONFIRM_TOKEN:
            print("REFUSED: interactive token mismatch", file=sys.stderr)
            return 2
    # The act-phase orchestration (connect, probe, cancel, close, confirm,
    # kraken lane, narration, hold, report) lands in W4B-6.
    print("flatten_all core loaded; act-phase orchestration arrives in W4B-6",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
