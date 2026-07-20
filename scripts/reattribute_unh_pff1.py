#!/usr/bin/env python3
"""scripts/reattribute_unh_pff1.py — trusted re-attribution of the real +228 UNH long to gamma.

Wave-2A item 2 (see PLAN_W2A.md + docs/W2A_item2_guard_dedup_proof.md). PFF1-Q1's double-book
left the real broker-held +228 UNH long UNTRACKED by any strategy: the gamma UNH FIFO is empty,
so the position surfaces only as the `broker_sync|UNH` broker mirror — gamma and the exit
overlay cannot see or manage it. This script re-attributes it to `gamma` with a REAL, TRUSTED
cost basis so the overlay finally manages it reduce-only.

Why a NEW narrow script (not scripts/reconcile_ledger_to_broker.py) — D3: the general
reconcile tool would land the lot under `epoch3_adopted` at a FABRICATED basis marked
`pnl_untrusted` + `UNATTRIBUTED_EPOCH3_ACCUMULATION` (correct for the general case where the
broker basis is unknown). For UNH we have the exact real re-buy fills, so we can do a *trusted*
re-attribution: gamma-attributed, real broker-truth VWAP, NOT pnl_untrusted.

Cost basis — D4: broker-truth VWAP 424.97 (the account's re-buys: 5@425.57 + 40@424.96 +
80@424.96 + 80@424.96 + 23@424.95 over the held 228). Corroborated by the broker's own reported
avgCost (≈424.98 in runtime/positions_snapshot.json). The overlay's anchor should equal what
the broker paid, so the position we re-track IS the broker's 228.

Guard-dedup (D7 — PROVEN in docs/W2A_item2_guard_dedup_proof.md + tests): writing an OPEN
`gamma|UNH`=228 alongside the ever-present `broker_sync|UNH`=228 mirror creates NO over-count.
The guard dual-books; every reader compares legs like-with-like, never sums (drift_v2 goes from
`broker_untracked_position` to matched; v4 legs all agree at 228; the overlay skips broker_sync
and caps a close at broker truth 228). So this script writes ONLY the gamma leg and NEVER
touches `broker_sync|UNH` (it is the truth anchor, re-created every cycle regardless).

Durable target — runtime/trade_closer_state.json (the FIFO book): the guard is rebuilt from it
every cycle (_rebuild_guard_from_paper_ledger). position_guard.json is ALSO written in the exact
rebuild shape for immediate consistency, but the FIFO edit is what makes it stick.

Safety rails (identical class to scripts/ghost_scrub_pff1.py / reconcile_ledger_to_broker.py):
DRY-RUN by default; --execute needs --confirm REATTRIBUTE-UNH-PFF1; fail-closed gates
(exec_mode ∈ {paper,dry_run}; broker truth UNH==228 confirmed by BOTH the guard broker_sync
mirror AND a FRESH positions_snapshot, refusing on any disagreement/staleness; the gamma UNH
FIFO must be empty and no existing gamma|UNH lot — idempotency); .bak of every mutated file;
idempotent NOOP re-run; signed report under reports/. NO broker I/O, NO order path.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Make `chad` importable no matter the CWD (mirrors reconcile_ledger_to_broker F1).
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

LOG = logging.getLogger("chad.reattribute_unh_pff1")

CONFIRM_TOKEN = "REATTRIBUTE-UNH-PFF1"
REPORT_SCHEMA = "reattribute_unh_pff1_report.v1"

TARGET_STRATEGY = "gamma"
TARGET_SYMBOL = "UNH"
TARGET_QTY = 228.0
TARGET_SIDE = "BUY"
BROKER_TRUTH_VWAP = 424.97          # D4 (broker-truth VWAP of the real re-buys)
FILL_ID_PREFIX = "PFF1_REATTR_UNH_"
LOT_SOURCE = "pff1_reattribution"
LOT_BASIS = "broker_truth_vwap"
LOT_PROVENANCE = "PFF1_REATTRIBUTION_TRUSTED"

_QTY_TOL = 1e-6
# Snapshot freshness: refuse if older than ttl*_SNAP_TTL_MULT (mirrors the v4 detector's
# ttl*3 tolerance in chad/core/position_guard.py).
_SNAP_TTL_MULT = 3.0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso_z(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stamp(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _backup(path: Path, stamp: str, backups: List[Dict[str, str]]) -> None:
    if not path.is_file():
        return
    bak = path.with_name(path.name + f".bak_reattr_unh_{stamp}")
    shutil.copy2(path, bak)
    backups.append({"original": str(path), "backup": str(bak), "sha256": _sha256_file(bak)})


def _parse_iso(ts: Any) -> Optional[_dt.datetime]:
    if not isinstance(ts, str) or not ts.strip():
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


# --------------------------------------------------------------------------- #
# gate (fail-closed; mirrors ghost_scrub_pff1._gate_exec_mode_paper)
# --------------------------------------------------------------------------- #
def _gate_exec_mode_paper() -> Tuple[bool, str]:
    try:
        from chad.execution.execution_config import get_execution_mode, is_paper_mode
        mode = get_execution_mode().value
        if not is_paper_mode():
            return False, f"exec_mode={mode} (not paper/dry_run)"
        return True, f"exec_mode={mode}"
    except Exception as exc:  # noqa: BLE001 — fail-closed
        return False, f"exec_mode check raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# broker-truth verification (two independent sources must agree on UNH==228)
# --------------------------------------------------------------------------- #
def verify_broker_truth(runtime_dir: Path, now: _dt.datetime) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Confirm broker holds UNH==228 via BOTH the guard broker_sync mirror AND a fresh
    positions_snapshot. Returns (ok, problems, evidence). Fail-closed on any disagreement."""
    problems: List[str] = []
    evidence: Dict[str, Any] = {}

    # Source 1 — position_guard.json broker_sync|UNH (the guard's broker mirror).
    guard = _load_json(runtime_dir / "position_guard.json")
    bs = None
    if isinstance(guard, Mapping):
        bs = guard.get(f"broker_sync|{TARGET_SYMBOL}")
    if not isinstance(bs, Mapping):
        problems.append(f"broker_sync|{TARGET_SYMBOL} not present in position_guard.json")
    else:
        bs_qty = abs(_f(bs.get("quantity")))
        bs_side = str(bs.get("side", "") or "").upper()
        evidence["guard_broker_sync"] = {"open": bool(bs.get("open")), "side": bs_side, "quantity": bs_qty}
        if not bs.get("open"):
            problems.append(f"broker_sync|{TARGET_SYMBOL} is not open")
        if bs_side != TARGET_SIDE:
            problems.append(f"broker_sync|{TARGET_SYMBOL} side={bs_side}, expected {TARGET_SIDE}")
        if abs(bs_qty - TARGET_QTY) > _QTY_TOL:
            problems.append(f"broker_sync|{TARGET_SYMBOL} qty={bs_qty}, expected {TARGET_QTY}")

    # Source 2 — positions_snapshot.json (independent clientId=99 leg), must be FRESH.
    snap = _load_json(runtime_dir / "positions_snapshot.json")
    if not isinstance(snap, Mapping):
        problems.append("positions_snapshot.json missing/unreadable (independent leg required)")
    else:
        snap_ts = _parse_iso(snap.get("ts_utc"))
        ttl = _f(snap.get("ttl_seconds"), 300.0)
        age = (now - snap_ts).total_seconds() if snap_ts else None
        snap_qty = None
        for row in (snap.get("positions") or []):
            if isinstance(row, Mapping) and str(row.get("symbol", "")).upper() == TARGET_SYMBOL:
                snap_qty = _f(row.get("position"))
                evidence["snapshot"] = {
                    "position": snap_qty, "avgCost": _f(row.get("avgCost")),
                    "ts_utc": snap.get("ts_utc"), "age_seconds": age,
                }
                break
        if snap_ts is None:
            problems.append("positions_snapshot.json has no parseable ts_utc")
        elif age is not None and age > ttl * _SNAP_TTL_MULT:
            problems.append(f"positions_snapshot.json STALE (age {age:.0f}s > ttl*{_SNAP_TTL_MULT:g}={ttl*_SNAP_TTL_MULT:.0f}s)")
        if snap_qty is None:
            problems.append(f"{TARGET_SYMBOL} not present in positions_snapshot.json positions")
        elif abs(snap_qty - TARGET_QTY) > _QTY_TOL:
            problems.append(f"snapshot {TARGET_SYMBOL} position={snap_qty}, expected {TARGET_QTY}")

    return (not problems), problems, evidence


# --------------------------------------------------------------------------- #
# FIFO / idempotency
# --------------------------------------------------------------------------- #
def _gamma_unh_row_index(queues: List[Any]) -> Optional[int]:
    for i, row in enumerate(queues):
        if (isinstance(row, Mapping)
                and str(row.get("strategy", "")).strip() == TARGET_STRATEGY
                and str(row.get("symbol", "")).strip().upper() == TARGET_SYMBOL):
            return i
    return None


def _already_reattributed(row: Mapping[str, Any]) -> bool:
    """True iff a gamma UNH queue row already holds exactly our re-attribution lot (228 BUY)."""
    lots = [l for l in (row.get("lots") or []) if isinstance(l, Mapping)]
    if len(lots) != 1:
        return False
    lot = lots[0]
    fid = str(lot.get("fill_id", "") or "")
    return (
        fid.startswith(FILL_ID_PREFIX)
        and str(lot.get("side", "")).upper() == TARGET_SIDE
        and abs(_f(lot.get("quantity")) - TARGET_QTY) <= _QTY_TOL
    )


def _last_rebuy_ts(fifo_state: Mapping[str, Any]) -> Optional[str]:
    """The most-recent broker_sync UNH re-buy lot ts_utc (the real establishment time)."""
    best: Optional[_dt.datetime] = None
    best_raw: Optional[str] = None
    for row in (fifo_state.get("queues") or []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("strategy", "")) != "broker_sync" or str(row.get("symbol", "")).upper() != TARGET_SYMBOL:
            continue
        for lot in (row.get("lots") or []):
            if not isinstance(lot, Mapping):
                continue
            dt = _parse_iso(lot.get("ts_utc"))
            if dt is not None and (best is None or dt > best):
                best, best_raw = dt, str(lot.get("ts_utc"))
    return best_raw


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_lot(fill_id: str, ts_utc: str) -> Dict[str, Any]:
    return {
        "fill_id": fill_id,
        "side": TARGET_SIDE,
        "quantity": TARGET_QTY,
        "fill_price": BROKER_TRUTH_VWAP,
        "ts_utc": ts_utc,
        "multiplier": 1.0,
        "meta": {
            "source": LOT_SOURCE,
            "basis": LOT_BASIS,
            "provenance": LOT_PROVENANCE,
            "reattributed": True,
            # DELIBERATELY NOT pnl_untrusted / scoring_excluded: this is a real broker
            # position with a real (VWAP) cost basis, so its eventual close IS scoreable.
            "note": "PFF1 re-attribution of the real +228 UNH long; broker-truth VWAP 424.97 (D4).",
        },
    }


def build_guard_entry(ts_utc: str, now_iso: str) -> Dict[str, Any]:
    """position_guard.json entry in the exact _rebuild_guard_from_paper_ledger shape."""
    return {
        "open": True,
        "opened_at": ts_utc,
        "updated_at_utc": now_iso,
        "strategy": TARGET_STRATEGY,
        "symbol": TARGET_SYMBOL,
        "side": TARGET_SIDE,
        "quantity": TARGET_QTY,
        "last_state": "OPEN",
        "source": LOT_SOURCE,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="Trusted re-attribution of the real +228 UNH long to gamma (dry-run default)."
    )
    ap.add_argument("--execute", action="store_true", help="Apply (mutates runtime/). Default is dry-run.")
    ap.add_argument("--confirm", default="", help=f"Required with --execute: the exact token {CONFIRM_TOKEN}.")
    ap.add_argument("--repo-root", default=None, help="Override repo root (tests).")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_FOR_IMPORT
    runtime_dir = repo_root / "runtime"
    reports_dir = repo_root / "reports"
    fifo_path = runtime_dir / "trade_closer_state.json"
    guard_path = runtime_dir / "position_guard.json"
    now = _utcnow()
    now_iso = _iso_z(now)
    stamp = _stamp(now)

    print("=" * 78)
    print("RE-ATTRIBUTE UNH (PFF1) — trusted +228 gamma|UNH lot @ broker-truth VWAP 424.97")
    print("=" * 78)
    print(f"repo_root : {repo_root}")
    print(f"fifo      : {fifo_path}")
    print(f"guard     : {guard_path}")
    print(f"mode      : {'EXECUTE' if args.execute else 'DRY-RUN (no mutation)'}")

    # 1. Broker truth: UNH == 228 via BOTH sources, snapshot fresh.
    bt_ok, bt_problems, bt_evidence = verify_broker_truth(runtime_dir, now)
    print("\n-- broker-truth verification (both sources must agree on UNH==228) --")
    print(f"  guard broker_sync : {bt_evidence.get('guard_broker_sync')}")
    print(f"  independent snap  : {bt_evidence.get('snapshot')}")
    for p in bt_problems:
        print(f"   !!  {p}")
    if not bt_ok:
        print("\nREFUSING: broker truth did not verify (UNH must be a fresh, agreed 228). Nothing written.")
        return 2

    # 2. Idempotency + FIFO precondition.
    fifo_state = _load_json(fifo_path)
    if not isinstance(fifo_state, Mapping):
        print(f"\nREFUSING: {fifo_path} missing/unreadable.")
        return 2
    queues = fifo_state.get("queues")
    if not isinstance(queues, list):
        print(f"\nREFUSING: trade_closer_state.json has no queues list.")
        return 2
    idx = _gamma_unh_row_index(queues)
    if idx is not None:
        row = queues[idx]
        if _already_reattributed(row):
            print("\nIDEMPOTENT NOOP: a gamma|UNH re-attribution lot (228 BUY) already exists. Nothing to do.")
            return 0
        print(f"\nREFUSING: a gamma UNH FIFO queue already exists with unexpected content "
              f"({len(row.get('lots') or [])} lots). Not overwriting — resolve manually.")
        return 2

    ts_utc = _last_rebuy_ts(fifo_state) or (
        (bt_evidence.get("snapshot") or {}).get("ts_utc")) or now_iso
    fill_id = f"{FILL_ID_PREFIX}{stamp}"
    lot = build_lot(fill_id, ts_utc)
    new_fifo_row = {"strategy": TARGET_STRATEGY, "symbol": TARGET_SYMBOL, "lots": [lot]}
    guard_entry = build_guard_entry(ts_utc, now_iso)

    print(f"\n-- planned write --")
    print(f"  FIFO row  : gamma|UNH  1 lot  {json.dumps(lot)}")
    print(f"  guard     : gamma|UNH  {json.dumps(guard_entry)}")
    print(f"  broker_sync|UNH is NOT touched (D7: truth anchor, re-created every cycle).")

    # 3. Dry-run stops here.
    if not args.execute:
        print("\nDRY-RUN complete. Re-run with --execute --confirm " + CONFIRM_TOKEN + " to apply.")
        return 0

    # 4. Execute — gated.
    if args.confirm != CONFIRM_TOKEN:
        print(f"\nREFUSING: --execute requires --confirm {CONFIRM_TOKEN} (got {args.confirm!r}).")
        return 2
    gate_ok, gate_reason = _gate_exec_mode_paper()
    print(f"\n-- gate -- exec_mode: {gate_reason}")
    if not gate_ok:
        print("REFUSING: exec_mode gate failed (must be paper/dry_run). Nothing written.")
        return 2

    backups: List[Dict[str, str]] = []
    _backup(fifo_path, stamp, backups)
    _backup(guard_path, stamp, backups)

    # 4a. FIFO: append the gamma|UNH lot; retain processed_fill_ids; mark our fill processed.
    new_fifo = copy.deepcopy(fifo_state)
    new_fifo.setdefault("queues", []).append(new_fifo_row)
    pfi = new_fifo.get("processed_fill_ids")
    if isinstance(pfi, list) and fill_id not in pfi:
        pfi.append(fill_id)
    new_fifo["saved_at_utc"] = now_iso
    _atomic_write_json(fifo_path, new_fifo)
    print(f"WROTE {fifo_path}")

    # 4b. Guard: write gamma|UNH in the rebuild shape; broker_sync|UNH untouched.
    guard = _load_json(guard_path)
    if not isinstance(guard, dict):
        guard = {}
    guard[f"{TARGET_STRATEGY}|{TARGET_SYMBOL}"] = guard_entry
    _atomic_write_json(guard_path, guard)
    print(f"WROTE {guard_path}")

    # 5. Signed report.
    report = {
        "schema_version": REPORT_SCHEMA,
        "applied_at_utc": now_iso,
        "confirm_token": CONFIRM_TOKEN,
        "lot": lot,
        "guard_entry": guard_entry,
        "broker_truth_evidence": bt_evidence,
        "cost_basis": {"vwap": BROKER_TRUTH_VWAP, "source": "D4 broker-truth VWAP",
                       "broker_avgcost_corroboration": (bt_evidence.get("snapshot") or {}).get("avgCost")},
        "broker_sync_untouched": True,
        "backups": backups,
        "gate": {"exec_mode": gate_reason},
    }
    report_path = reports_dir / f"reattribute_unh_pff1_{stamp}.json"
    _atomic_write_json(report_path, report)
    print(f"WROTE {report_path}")
    print("\nRE-ATTRIBUTION applied. gamma|UNH=228 is now tracked at a trusted basis; the exit "
          "overlay will manage it reduce-only (broker_sync|UNH remains the truth anchor).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
