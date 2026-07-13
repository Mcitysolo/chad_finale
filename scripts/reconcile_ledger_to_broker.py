#!/usr/bin/env python3
"""scripts/reconcile_ledger_to_broker.py — governed, reversible ledger↔broker reconciliation.

Implements the disposition in ops/pending_actions/DRIFT_RECON_ledger_broker_2026-07-13.md:
bring CHAD's strategy ledger into agreement with authoritative IBKR (paper) broker truth
WITHOUT destroying evidence, via a HYBRID rebaseline + adopt.

Root cause (EXIT-AUDIT + DRIFT-RECON D1): post-Epoch-3 the CHAD paper-SIM ledger over-counted the
real IBKR paper account (entries confirm, exits never fire → one-sided accumulation; SELL exits
rejected as phantom). The trade_closer FIFO book holds e.g. gamma|TLT=1220 lots while the broker
holds TLT=420; other symbols hold nothing at the broker at all (SPY/BAC phantom).

Durable-source fact (verified): the position guard is REBUILT every cycle from the trade_closer
FIFO queues by chad/core/live_loop._rebuild_guard_from_paper_ledger (faithful sum, no broker
clamp), and broker_sync|<sym> is rewritten every cycle from live IBKR truth by
_rebuild_guard_from_broker. Therefore:
  * the ONLY durable mutation target is runtime/trade_closer_state.json (the FIFO book);
  * broker_sync|<sym> is READ-ONLY broker truth and is NEVER touched here;
  * we also write position_guard.json in the exact rebuild shape for immediate consistency, but
    the FIFO edit is what makes the reconciliation stick across the next cycle's rebuild.

Disposition (HYBRID), per symbol:
  * CHAD symbol (NOT operator-excluded) with broker_qty>0  → ADOPT: replace the over-counted FIFO
    lots with ONE reconciliation seed lot at broker truth qty (dominant FIFO strategy), marked
    provenance=UNATTRIBUTED_EPOCH3_ACCUMULATION + pnl_untrusted, and open the matching guard entry.
  * CHAD symbol with broker_qty==0 (phantom)             → REBASELINE: remove the FIFO lots.
  * operator-EXCLUDED symbol (exclusion_policy / _EFFECTIVE_NON_CHAD_SYMBOLS) → REBASELINE: remove
    the stray CHAD FIFO lots; NEVER adopt (the broker position is the operator's).

Evidence preservation: every mutated file is copied to a timestamped .bak first; nothing is
hard-deleted; processed_fill_ids is RETAINED (never re-open old fills); removed lots + adoptions
are written to a reconciliation ledger (runtime/ledger_reconciliation_<stamp>.ndjson) and a signed
report (reports/ledger_recon_<stamp>.json). Nothing synthetic is EVER written into data/fills, so
no reconciliation row can enter effective_trades; belt-and-suspenders, every synthetic record
carries extra.pnl_untrusted=true (the canonical Stage-2 trust-filter exclusion,
chad/analytics/trade_stats_engine._is_untrusted).

Safety: DRY-RUN by default (prints the full plan, mutates NOTHING). --execute requires the typed
token --confirm RECONCILE-LEDGER-TO-BROKER and passes fail-closed gates (exec mode paper/dry_run,
SCR ∈ {CONFIDENT,CAUTIOUS}, reconciliation status not RED). Idempotent: a symbol already carrying a
reconciliation seed lot at broker truth is a NOOP, so re-running is safe.

This script performs NO broker I/O and places NO orders.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger("chad.reconcile_ledger_to_broker")

CONFIRM_TOKEN = "RECONCILE-LEDGER-TO-BROKER"
MARKER = "LEDGER_RECON_APPLIED"

PROV_ADOPT = "UNATTRIBUTED_EPOCH3_ACCUMULATION"
PROV_FLATTEN = "OPERATOR_FLATTEN"
RECON_SOURCE = "ledger_broker_reconciliation"
LEDGER_SCHEMA = "ledger_reconciliation.v1"
REPORT_SCHEMA = "ledger_recon_report.v1"

DISP_ADOPT = "ADOPT"
DISP_REBASELINE_EXCLUDED = "REBASELINE_EXCLUDED"
DISP_REBASELINE_PHANTOM = "REBASELINE_PHANTOM"

_SAFE_SCR_STATES = frozenset({"CONFIDENT", "CAUTIOUS"})


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso_z(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stamp(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v else default  # NaN guard


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# broker-truth aggregation (mirrors position_guard._signed_qty; broker_sync is truth)
# --------------------------------------------------------------------------- #
def broker_signed_by_symbol(guard_state: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, entry in guard_state.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if not key.startswith("broker_sync|"):
            continue
        sym = str(entry.get("symbol", "") or "").strip().upper()
        if not sym:
            continue
        qty = abs(_f(entry.get("quantity")))
        side = str(entry.get("side", "") or "").strip().upper()
        out[sym] = out.get(sym, 0.0) + (-qty if side == "SELL" else qty)
    return out


def _lot_qty(lot: Mapping[str, Any]) -> float:
    for k in ("remaining", "quantity", "qty"):
        if k in lot:
            return abs(_f(lot.get(k)))
    return 0.0


def fifo_by_symbol(queues: List[Mapping[str, Any]]) -> Dict[str, List[Tuple[int, dict]]]:
    """symbol -> list of (index_in_queues, row)."""
    out: Dict[str, List[Tuple[int, dict]]] = {}
    for i, row in enumerate(queues):
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "") or "").strip().upper()
        if not sym:
            continue
        out.setdefault(sym, []).append((i, row))
    return out


def _row_lots(row: Mapping[str, Any]) -> List[dict]:
    lots = row.get("lots")
    if isinstance(lots, list):
        return [l for l in lots if isinstance(l, dict)]
    # tolerate alternate key names
    for k in ("queue", "fills", "open_lots", "entries"):
        v = row.get(k)
        if isinstance(v, list):
            return [l for l in v if isinstance(l, dict)]
    return []


def _row_qty(row: Mapping[str, Any]) -> float:
    return sum(_lot_qty(l) for l in _row_lots(row))


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class PlanRow:
    symbol: str
    disposition: str
    excluded: bool
    broker_qty: float
    broker_side: str
    fifo_qty_before: float
    target_strategy: Optional[str]           # ADOPT only
    mark: Optional[float]                     # ADOPT seed price
    seed_qty: float                           # ADOPT only
    removed_strategies: List[str] = field(default_factory=list)
    removed_lot_count: int = 0
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "disposition": self.disposition,
            "excluded": self.excluded,
            "broker_qty": self.broker_qty,
            "broker_side": self.broker_side,
            "fifo_qty_before": self.fifo_qty_before,
            "target_strategy": self.target_strategy,
            "mark": self.mark,
            "seed_qty": self.seed_qty,
            "removed_strategies": self.removed_strategies,
            "removed_lot_count": self.removed_lot_count,
            "provenance": self.provenance,
        }

    def marker_line(self) -> str:
        return (
            f"{MARKER} symbol={self.symbol} disposition={self.disposition} "
            f"excluded={str(self.excluded).lower()} broker_qty={self.broker_qty:g} "
            f"fifo_qty_before={self.fifo_qty_before:g} seed_qty={self.seed_qty:g} "
            f"target_strategy={self.target_strategy or '-'} "
            f"removed_lots={self.removed_lot_count} mark={self.mark if self.mark is not None else 'null'}"
        )


def _is_already_reconciled(rows: List[Tuple[int, dict]], broker_qty: float) -> bool:
    """A symbol is already reconciled iff it has exactly ONE strategy row whose single lot is a
    reconciliation seed lot (meta.reconciled) at the current broker qty."""
    if len(rows) != 1:
        return False
    lots = _row_lots(rows[0][1])
    if len(lots) != 1:
        return False
    meta = lots[0].get("meta") if isinstance(lots[0].get("meta"), dict) else {}
    return bool(meta.get("reconciled")) and abs(_lot_qty(lots[0]) - abs(broker_qty)) < 1e-6


def compute_plan(
    *,
    queues: List[Mapping[str, Any]],
    broker_signed: Mapping[str, float],
    exclusions: set,
    marks: Mapping[str, float],
) -> List[PlanRow]:
    """Pure planner. Decides a per-symbol disposition from the FIFO book + broker truth +
    exclusions. Returns only symbols that require a change (idempotent NOOPs are omitted)."""
    fifo = fifo_by_symbol(list(queues))
    symbols = set(fifo.keys()) | {s for s, q in broker_signed.items() if abs(q) > 1e-9}
    plan: List[PlanRow] = []

    for symbol in sorted(symbols):
        rows = fifo.get(symbol, [])
        fifo_qty = sum(_row_qty(r) for _, r in rows)
        broker_qty = _f(broker_signed.get(symbol, 0.0))
        excluded = symbol in exclusions
        removed_strats = sorted({str(r.get("strategy", "") or "") for _, r in rows})
        removed_lots = sum(len(_row_lots(r)) for _, r in rows)

        if excluded:
            # Operator-owned / non-CHAD symbol: purge any stray CHAD FIFO lots, never adopt.
            if rows:
                plan.append(PlanRow(
                    symbol=symbol, disposition=DISP_REBASELINE_EXCLUDED, excluded=True,
                    broker_qty=broker_qty, broker_side=_side(broker_qty),
                    fifo_qty_before=fifo_qty, target_strategy=None, mark=None, seed_qty=0.0,
                    removed_strategies=removed_strats, removed_lot_count=removed_lots,
                    provenance=PROV_FLATTEN,
                ))
            continue

        if broker_qty > 0:
            if _is_already_reconciled(rows, broker_qty):
                continue  # idempotent no-op
            target = _dominant_strategy(rows) or "epoch3_adopted"
            plan.append(PlanRow(
                symbol=symbol, disposition=DISP_ADOPT, excluded=False,
                broker_qty=broker_qty, broker_side=_side(broker_qty),
                fifo_qty_before=fifo_qty, target_strategy=target,
                mark=(_f(marks.get(symbol)) or None), seed_qty=abs(broker_qty),
                removed_strategies=removed_strats, removed_lot_count=removed_lots,
                provenance=PROV_ADOPT,
            ))
        elif broker_qty == 0 and rows:
            plan.append(PlanRow(
                symbol=symbol, disposition=DISP_REBASELINE_PHANTOM, excluded=False,
                broker_qty=0.0, broker_side="", fifo_qty_before=fifo_qty,
                target_strategy=None, mark=None, seed_qty=0.0,
                removed_strategies=removed_strats, removed_lot_count=removed_lots,
                provenance=PROV_FLATTEN,
            ))
        # broker_qty < 0 (short) with no CHAD context is left untouched (out of scope v1).

    return plan


def _side(qty: float) -> str:
    if qty > 0:
        return "BUY"
    if qty < 0:
        return "SELL"
    return ""


def _dominant_strategy(rows: List[Tuple[int, dict]]) -> Optional[str]:
    best: Optional[str] = None
    best_q = -1.0
    for _, row in rows:
        s = str(row.get("strategy", "") or "")
        q = _row_qty(row)
        if s and q > best_q:
            best_q, best = q, s
    return best


# --------------------------------------------------------------------------- #
# apply (pure transform: returns new structures + audit; no file I/O here)
# --------------------------------------------------------------------------- #
@dataclass
class ReconResult:
    new_guard: Dict[str, Any]
    new_fifo: Dict[str, Any]
    ledger_records: List[Dict[str, Any]]
    markers: List[str]
    plan_dicts: List[Dict[str, Any]]
    lots_before: int
    lots_after: int
    ledger_rows_out: int


def apply_plan(
    *,
    plan: List[PlanRow],
    guard_state: Mapping[str, Any],
    fifo_state: Mapping[str, Any],
    now_iso: str,
    stamp: str,
) -> ReconResult:
    """Pure transform. Given the plan + current guard/FIFO state, return the reconciled guard and
    FIFO structures plus the audit ledger records + markers. Preserves every removed lot in the
    ledger records (relabelled, not destroyed). Never mutates its inputs."""
    guard = copy.deepcopy(dict(guard_state))
    fifo = copy.deepcopy(dict(fifo_state))
    queues: List[dict] = [r for r in (fifo.get("queues") or []) if isinstance(r, dict)]
    processed = list(fifo.get("processed_fill_ids") or [])
    by_symbol = {p.symbol: p for p in plan}

    lots_before = sum(len(_row_lots(r)) for r in queues)
    ledger_records: List[Dict[str, Any]] = []
    markers: List[str] = []
    new_queues: List[dict] = []
    seen_adopt: set = set()

    for row in queues:
        sym = str(row.get("symbol", "") or "").strip().upper()
        strat = str(row.get("strategy", "") or "")
        p = by_symbol.get(sym)
        if p is None:
            new_queues.append(row)  # untouched symbol
            continue

        removed_lots = _row_lots(row)
        if p.disposition == DISP_ADOPT and strat == p.target_strategy and sym not in seen_adopt:
            # Replace this (dominant) strategy row's lots with a single reconciliation seed lot.
            seed_fid = f"RECON_ADOPT_{sym}_{stamp}"
            seed_lot = {
                "fill_id": seed_fid,
                "side": p.broker_side,
                "quantity": p.seed_qty,
                "fill_price": p.mark if p.mark is not None else 0.0,
                "ts_utc": now_iso,
                "multiplier": 1.0,
                "meta": {
                    "reconciled": True,
                    "provenance": PROV_ADOPT,
                    "pnl_untrusted": True,
                    "scoring_excluded": True,
                    "source": RECON_SOURCE,
                    "seeded_from": "broker_truth",
                },
            }
            new_row = dict(row)
            new_row["lots"] = [seed_lot]
            new_queues.append(new_row)
            seen_adopt.add(sym)
            if seed_fid not in processed:
                processed.append(seed_fid)
            ledger_records.append(_ledger_record(
                now_iso, sym, strat, "ADOPT", PROV_ADOPT,
                removed_lots=removed_lots, seed_qty=p.seed_qty, broker_qty=p.broker_qty,
                fifo_qty_before=p.fifo_qty_before, mark=p.mark,
                note="rebaselined SIM over-count and seeded strategy book at broker truth",
            ))
        else:
            # Rebaseline: drop this row entirely (excluded/phantom, or a non-dominant strat row
            # of an ADOPT symbol). Its lots are preserved in the ledger record below.
            ledger_records.append(_ledger_record(
                now_iso, sym, strat, "REBASELINE", p.provenance,
                removed_lots=removed_lots, seed_qty=0.0, broker_qty=p.broker_qty,
                fifo_qty_before=_row_qty(row), mark=None,
                note=("operator-excluded: purge stray CHAD lots" if p.excluded
                      else "phantom (broker holds nothing): purge CHAD lots"),
            ))

    # ADOPT symbols that had NO existing FIFO row (broker-only): create a fresh seeded row.
    for p in plan:
        if p.disposition == DISP_ADOPT and p.symbol not in seen_adopt:
            seed_fid = f"RECON_ADOPT_{p.symbol}_{stamp}"
            new_queues.append({
                "strategy": p.target_strategy,
                "symbol": p.symbol,
                "lots": [{
                    "fill_id": seed_fid, "side": p.broker_side, "quantity": p.seed_qty,
                    "fill_price": p.mark if p.mark is not None else 0.0, "ts_utc": now_iso,
                    "multiplier": 1.0,
                    "meta": {"reconciled": True, "provenance": PROV_ADOPT,
                             "pnl_untrusted": True, "scoring_excluded": True,
                             "source": RECON_SOURCE, "seeded_from": "broker_truth"},
                }],
            })
            seen_adopt.add(p.symbol)
            if seed_fid not in processed:
                processed.append(seed_fid)
            ledger_records.append(_ledger_record(
                now_iso, p.symbol, p.target_strategy or "epoch3_adopted", "ADOPT", PROV_ADOPT,
                removed_lots=[], seed_qty=p.seed_qty, broker_qty=p.broker_qty,
                fifo_qty_before=0.0, mark=p.mark,
                note="broker-held with no CHAD FIFO book: seeded at broker truth",
            ))

    fifo["queues"] = new_queues
    fifo["processed_fill_ids"] = processed
    fifo["saved_at_utc"] = now_iso

    # Guard mutations — exact rebuild shape for immediate consistency (broker_sync UNTOUCHED).
    for p in plan:
        _apply_guard(guard, p, now_iso)

    for p in plan:
        markers.append(p.marker_line())

    lots_after = sum(len(_row_lots(r)) for r in new_queues)
    return ReconResult(
        new_guard=guard, new_fifo=fifo, ledger_records=ledger_records, markers=markers,
        plan_dicts=[p.to_dict() for p in plan],
        lots_before=lots_before, lots_after=lots_after, ledger_rows_out=len(ledger_records),
    )


def _ledger_record(now_iso, symbol, strategy, action, provenance, *, removed_lots, seed_qty,
                   broker_qty, fifo_qty_before, mark, note) -> Dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA,
        "ts_utc": now_iso,
        "symbol": symbol,
        "strategy": strategy,
        "action": action,
        "provenance": provenance,
        "broker_qty": broker_qty,
        "fifo_qty_before": fifo_qty_before,
        "seed_qty": seed_qty,
        "mark": mark,
        "removed_lot_count": len(removed_lots),
        "removed_fill_ids": [str(l.get("fill_id")) for l in removed_lots if l.get("fill_id")],
        "removed_lots": removed_lots,   # full lot bodies preserved (evidence, not destroyed)
        "pnl_untrusted": True,          # canonical Stage-2 exclusion (trade_stats_engine._is_untrusted)
        "tags": ["pnl_untrusted", "ledger_reconciliation"],
        "note": note,
    }


def _apply_guard(guard: Dict[str, Any], p: PlanRow, now_iso: str) -> None:
    if p.disposition == DISP_ADOPT:
        key = f"{p.target_strategy}|{p.symbol}"
        existing = guard.get(key) if isinstance(guard.get(key), dict) else {}
        guard[key] = {
            "open": True,
            "opened_at": existing.get("opened_at") or now_iso,
            "updated_at_utc": now_iso,
            "strategy": p.target_strategy,
            "symbol": p.symbol,
            "side": p.broker_side,
            "quantity": p.seed_qty,
            "last_state": "OPEN",
            "source": RECON_SOURCE,
            "provenance": PROV_ADOPT,
        }
    else:
        # Force-close any OPEN CHAD strategy entries for this symbol (broker_sync untouched).
        for key, entry in list(guard.items()):
            if not isinstance(key, str) or key.startswith("_") or key.startswith("broker_sync|"):
                continue
            if not isinstance(entry, dict):
                continue
            if str(entry.get("symbol", "") or "").strip().upper() != p.symbol:
                continue
            if entry.get("open"):
                entry["open"] = False
                entry["last_state"] = "CLOSED"
                entry["closed_by"] = RECON_SOURCE
                entry["updated_at_utc"] = now_iso


# --------------------------------------------------------------------------- #
# gates (fail-closed; mirror scripts/close_guard_entry.py)
# --------------------------------------------------------------------------- #
def _gate_exec_mode_paper() -> Tuple[bool, str]:
    try:
        from chad.execution.execution_config import get_execution_mode, is_paper_mode
        mode = get_execution_mode().value
        if not is_paper_mode():
            return False, f"exec_mode={mode} (not paper/dry_run)"
        return True, f"exec_mode={mode}"
    except Exception as exc:  # noqa: BLE001
        return False, f"exec_mode check raised {type(exc).__name__}: {exc}"


def _gate_scr(runtime_dir: Path) -> Tuple[bool, str]:
    path = runtime_dir / "scr_state.json"
    if not path.is_file():
        return False, "scr_state.json missing"
    try:
        state = str(json.loads(path.read_text(encoding="utf-8")).get("state") or "").upper()
    except Exception as exc:  # noqa: BLE001
        return False, f"scr_state.json unreadable: {exc}"
    if state not in _SAFE_SCR_STATES:
        return False, f"scr_state={state or 'UNKNOWN'} (not CONFIDENT/CAUTIOUS)"
    return True, f"scr_state={state}"


def _gate_reconciliation_not_red(runtime_dir: Path) -> Tuple[bool, str]:
    path = runtime_dir / "reconciliation_state.json"
    if not path.is_file():
        return False, "reconciliation_state.json missing"
    try:
        status = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "").upper()
    except Exception as exc:  # noqa: BLE001
        return False, f"reconciliation_state.json unreadable: {exc}"
    if status == "RED":
        return False, "reconciliation status=RED"
    return True, f"reconciliation status={status or 'UNKNOWN'}"


def run_gates(runtime_dir: Path) -> Tuple[bool, Dict[str, str]]:
    checks = {
        "exec_mode": _gate_exec_mode_paper(),
        "scr": _gate_scr(runtime_dir),
        "reconciliation": _gate_reconciliation_not_red(runtime_dir),
    }
    reasons = {k: v[1] for k, v in checks.items()}
    return all(v[0] for v in checks.values()), reasons


# --------------------------------------------------------------------------- #
# exclusions
# --------------------------------------------------------------------------- #
def load_exclusions(runtime_dir: Path) -> set:
    ex: set = set()
    try:
        rec = json.loads((runtime_dir / "reconciliation_state.json").read_text(encoding="utf-8"))
        for s in (rec.get("exclusion_policy") or {}):
            ex.add(str(s).strip().upper())
        for s in (rec.get("excluded_symbols") or []):
            ex.add(str(s).strip().upper())
    except Exception:  # noqa: BLE001
        pass
    try:
        from chad.core.position_reconciler import _EFFECTIVE_NON_CHAD_SYMBOLS
        ex |= {str(s).strip().upper() for s in _EFFECTIVE_NON_CHAD_SYMBOLS}
    except Exception:  # noqa: BLE001
        pass
    return ex


def load_marks(runtime_dir: Path) -> Dict[str, float]:
    try:
        data = json.loads((runtime_dir / "price_cache.json").read_text(encoding="utf-8"))
        prices = data.get("prices") if isinstance(data, dict) else None
        if isinstance(prices, dict):
            return {str(k).upper(): _f(v) for k, v in prices.items() if _f(v) > 0}
    except Exception:  # noqa: BLE001
        pass
    return {}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _print_plan(plan: List[PlanRow]) -> None:
    if not plan:
        print("  (no drift requiring reconciliation — already converged / idempotent no-op)")
        return
    print(f"  {'symbol':6s} {'disposition':22s} {'excl':5s} {'broker':>8s} {'fifo_before':>11s} "
          f"{'seed':>6s} {'strat':10s} {'lots':>5s} {'mark':>9s}")
    for p in plan:
        print(f"  {p.symbol:6s} {p.disposition:22s} {str(p.excluded):5s} {p.broker_qty:8g} "
              f"{p.fifo_qty_before:11g} {p.seed_qty:6g} {str(p.target_strategy or '-'):10s} "
              f"{p.removed_lot_count:5d} {('%.2f'%p.mark) if p.mark is not None else 'null':>9s}")


def _backup(path: Path, stamp: str, backups: List[Dict[str, str]]) -> None:
    if not path.is_file():
        return
    bak = path.with_name(path.name + f".bak_ledger_recon_{stamp}")
    shutil.copy2(path, bak)
    backups.append({"original": str(path), "backup": str(bak), "sha256": _sha256_file(bak)})


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Governed ledger↔broker reconciliation (dry-run default).")
    ap.add_argument("--execute", action="store_true",
                    help="Apply the plan (mutates files). Default is dry-run.")
    ap.add_argument("--confirm", default="",
                    help=f"Required with --execute: the exact token {CONFIRM_TOKEN}.")
    ap.add_argument("--repo-root", default=None, help="Override repo root (tests).")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    runtime_dir = repo_root / "runtime"
    now = _utcnow()
    now_iso = _iso_z(now)
    stamp = _stamp(now)

    guard_path = runtime_dir / "position_guard.json"
    fifo_path = runtime_dir / "trade_closer_state.json"
    try:
        guard_state = json.loads(guard_path.read_text(encoding="utf-8"))
        fifo_state = json.loads(fifo_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.error("REFUSE: cannot read guard/FIFO state: %s", exc)
        return 2

    exclusions = load_exclusions(runtime_dir)
    marks = load_marks(runtime_dir)
    broker_signed = broker_signed_by_symbol(guard_state)
    queues = [r for r in (fifo_state.get("queues") or []) if isinstance(r, dict)]

    plan = compute_plan(queues=queues, broker_signed=broker_signed,
                        exclusions=exclusions, marks=marks)

    print(f"\n=== LEDGER↔BROKER RECONCILIATION PLAN ({now_iso}) ===")
    print(f"broker truth (broker_sync): "
          f"{ {k: broker_signed[k] for k in sorted(broker_signed)} }")
    print(f"operator-excluded: {sorted(exclusions)}")
    _print_plan(plan)

    if not plan:
        print("\nNothing to do. Ledger already agrees with broker truth (idempotent).")
        return 0

    if not args.execute:
        print("\nDRY-RUN — no files mutated. To apply:")
        print(f"  python3 scripts/reconcile_ledger_to_broker.py --execute --confirm {CONFIRM_TOKEN}")
        return 0

    # ---- execute path ----
    if args.confirm != CONFIRM_TOKEN:
        LOG.error("REFUSE: --execute requires --confirm %s (got %r).", CONFIRM_TOKEN, args.confirm)
        return 2
    gates_ok, gate_reasons = run_gates(runtime_dir)
    if not gates_ok:
        LOG.error("REFUSE: fail-closed gate(s) tripped — %s", gate_reasons)
        return 2

    result = apply_plan(plan=plan, guard_state=guard_state, fifo_state=fifo_state,
                        now_iso=now_iso, stamp=stamp)

    # Backups first (evidence preservation), then atomic writes.
    backups: List[Dict[str, str]] = []
    _backup(guard_path, stamp, backups)
    _backup(fifo_path, stamp, backups)

    _atomic_write_json(fifo_path, result.new_fifo)
    _atomic_write_json(guard_path, result.new_guard)

    ledger_path = runtime_dir / f"ledger_reconciliation_{stamp}.ndjson"
    with ledger_path.open("w", encoding="utf-8") as fh:
        for rec in result.ledger_records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    for m in result.markers:
        LOG.info(m)

    report = {
        "schema_version": REPORT_SCHEMA,
        "ts_utc": now_iso,
        "git_head": _git_head(repo_root),
        "confirm_token_used": CONFIRM_TOKEN,
        "gates": gate_reasons,
        "broker_truth": {k: broker_signed[k] for k in sorted(broker_signed)},
        "exclusions": sorted(exclusions),
        "plan": result.plan_dicts,
        "lots_before": result.lots_before,
        "lots_after": result.lots_after,
        "ledger_rows": result.ledger_rows_out,
        "backups": backups,
        "files_written": [str(fifo_path), str(guard_path), str(ledger_path)],
        "marker_count": len(result.markers),
    }
    reports_dir = repo_root / "reports"
    report_path = reports_dir / f"ledger_recon_{stamp}.json"
    _atomic_write_json(report_path, report)

    print(f"\n=== APPLIED. lots {result.lots_before} -> {result.lots_after}; "
          f"ledger rows {result.ledger_rows_out}; report {report_path} ===")
    print("Backups:")
    for b in backups:
        print(f"  {b['original']} -> {b['backup']}")
    print("Rollback: restore the .bak_ledger_recon_* files over the originals.")
    print("Next live-loop cycle rebuilds the guard from the reconciled FIFO -> drift converges.")
    return 0


def _git_head(repo_root: Path) -> Optional[str]:
    try:
        head = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            return (repo_root / ".git" / ref).read_text(encoding="utf-8").strip()[:12]
        return head[:12]
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
