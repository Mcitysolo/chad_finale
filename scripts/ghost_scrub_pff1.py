#!/usr/bin/env python3
"""scripts/ghost_scrub_pff1.py — non-destructive scrub of the 6 PFF1 phantom UNH closes.

Wave-2A item 1 (see PLAN_W2A.md). PFF1-Q1 fixed the trade_closer double-book at the SOURCE,
but the 6 phantom ``gamma UNH`` closes it already emitted on 2026-07-20 are hash-chained into
``data/trades/trade_history_20260720.ndjson`` and both scorekeepers still count them:

  * SCR (chad.analytics.trade_stats_engine) counts them in ``effective_trades`` (they carry no
    ``pnl_untrusted`` marker — they are the re-buys closing the phantom shorts, and look like
    genuine round-trips);
  * Stage-2 (chad.validation.trade_log_adapter) would ADMIT them for the same reason.

The ledger is APPEND-ONLY and hash-chained — rewriting it in place would break every
``prev_hash`` link downstream. So this scrub is NON-DESTRUCTIVE: it writes an operator
quarantine manifest (``runtime/quarantine_manifest_pff1_ghost_scrub.json``) pinning the 6
``record_hash`` values. Both scorekeepers already honour that manifest (SCR natively via
``chad.utils.quarantine.get_exclusion_sets``; Stage-2 via W2A-1) and drop exactly those 6 rows
BEFORE parse, counted as ``excluded_quarantined`` — the original evidence file is untouched.

The 6 records (verified against the ledger before pinning): the ``gamma UNH`` closes with NO
``pnl_untrusted`` marker, sequence 2..7 of trade_history_20260720.ndjson. Sum 233 sh / −145.79.
Sequence 1 (the +625.17 untrusted seed close) is NOT scrubbed — it is already
``excluded_untrusted`` and its total_pnl leak is PFF1-Q2's job, not this one.

Expected effect once applied against the live runtime (Phase-3, operator GO):
  * SCR ``effective_trades``: 73 → 67 (−6).
  * SCR ``total_pnl`` reaches the −375.60 target ONLY once the SCR shadow server (:9618) has
    reloaded PFF1-Q2 (see D2) — scrub alone (Q2 not yet active) yields +103.78 + 145.79 =
    +249.57. This script asserts nothing about the live SCR number; the acceptance test asserts
    the deterministic DELTA (effective −6, trusted-pnl += 145.79) on a fixture.

Safety rails (identical class to scripts/reconcile_ledger_to_broker.py):
  * DRY-RUN by default — prints the full plan, mutates NOTHING.
  * ``--execute`` requires the typed token ``--confirm GHOST-SCRUB-PFF1``.
  * Fail-closed gates: exec_mode ∈ {paper, dry_run}; every one of the 6 record_hashes MUST be
    found in the ledger AND match its expected (gamma/UNH, qty, pnl, no pnl_untrusted) shape;
    refuses if the manifest already exists unless ``--idempotent-ok``.
  * Idempotent: an already-correct manifest is a NOOP; a to-be-overwritten file is ``.bak``'d first.
  * Writes a signed audit report to ``reports/ghost_scrub_pff1_<stamp>.json``.
  * NO broker I/O, NO order path, NO ledger rewrite.
"""

from __future__ import annotations

import argparse
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

LOG = logging.getLogger("chad.ghost_scrub_pff1")

CONFIRM_TOKEN = "GHOST-SCRUB-PFF1"
MANIFEST_NAME = "quarantine_manifest_pff1_ghost_scrub.json"
MANIFEST_SCHEMA = "quarantine_manifest.v1"
REPORT_SCHEMA = "ghost_scrub_pff1_report.v1"
QUARANTINE_REASON = "pff1_phantom_double_book"

# The exact 6 phantom records (trade_history_20260720.ndjson seq 2..7). Each is verified
# against the live ledger before it is pinned: strategy=gamma, symbol=UNH, matching qty/pnl,
# and NOT pnl_untrusted (defense against pinning the wrong row on a hash typo).
EXPECTED_RECORDS: Tuple[Dict[str, Any], ...] = (
    {"record_hash": "0c70922e1dc265aaf171f4fc8a7c17bf5f8d282221483fb94113b3ca1f087dd6", "side": "BUY",  "qty": 5.0,  "pnl": 12.05},
    {"record_hash": "7bd3a872261ff9a4965c222d8fa39dcae059a7236f768f8cee817eca577e635e", "side": "SELL", "qty": 5.0,  "pnl": -3.55},
    {"record_hash": "e6f51f9923947089e7f5ed4e9b4dabc44f65ab29bb06b1adad6ca0fb663f10b9", "side": "SELL", "qty": 32.0, "pnl": -22.72},
    {"record_hash": "687133278be20e0d87854ce71f80efb0c39e8bfa7bf8000fe54b7fbbd47a29fc", "side": "SELL", "qty": 40.0, "pnl": -28.00},
    {"record_hash": "ead6c9f71dc97e01ea7d6ab8a5b468494d56c9bb2100ad86e6b310bd35a79747", "side": "SELL", "qty": 80.0, "pnl": -56.00},
    {"record_hash": "3d0627208074e0dd3817853a07f114f6e2eeafb582a4b3cbec9b339c652a429d", "side": "SELL", "qty": 71.0, "pnl": -47.57},
)
_TARGET_STRATEGY = "gamma"
_TARGET_SYMBOL = "UNH"
_PNL_TOL = 0.005
_QTY_TOL = 1e-6


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


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _backup(path: Path, stamp: str, backups: List[Dict[str, str]]) -> None:
    if not path.is_file():
        return
    bak = path.with_name(path.name + f".bak_ghost_scrub_{stamp}")
    shutil.copy2(path, bak)
    backups.append({"original": str(path), "backup": str(bak), "sha256": _sha256_file(bak)})


# --------------------------------------------------------------------------- #
# gate (fail-closed; mirrors scripts/reconcile_ledger_to_broker._gate_exec_mode_paper)
# --------------------------------------------------------------------------- #
def _gate_exec_mode_paper() -> Tuple[bool, str]:
    try:
        from chad.execution.execution_config import get_execution_mode, is_paper_mode
        mode = get_execution_mode().value
        if not is_paper_mode():
            return False, f"exec_mode={mode} (not paper/dry_run)"
        return True, f"exec_mode={mode}"
    except Exception as exc:  # noqa: BLE001 — any failure resolving the mode is fail-closed
        return False, f"exec_mode check raised {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# ledger scan — locate & verify the 6 records before pinning them
# --------------------------------------------------------------------------- #
def _iter_ledger_files(trades_dir: Path) -> List[Path]:
    if not trades_dir.is_dir():
        return []
    import re
    rx = re.compile(r"^trade_history_\d{8}\.ndjson$")
    return sorted(p for p in trades_dir.iterdir() if p.is_file() and rx.match(p.name))


def _find_records(trades_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``record_hash -> {found row summary}`` for the 6 targets found in the ledgers."""
    wanted = {r["record_hash"] for r in EXPECTED_RECORDS}
    found: Dict[str, Dict[str, Any]] = {}
    for fp in _iter_ledger_files(trades_dir):
        with fp.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, Mapping):
                    continue
                rh = rec.get("record_hash")
                if not (isinstance(rh, str) and rh in wanted):
                    continue
                payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else rec
                extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
                tags = [str(t).lower() for t in (payload.get("tags") or [])]
                found[rh] = {
                    "source_file": fp.name,
                    "strategy": str(payload.get("strategy") or ""),
                    "symbol": str(payload.get("symbol") or "").upper(),
                    "side": str(payload.get("side") or "").upper(),
                    "qty": _f(payload.get("quantity")),
                    "pnl": _f(payload.get("pnl")),
                    "pnl_untrusted": bool(
                        payload.get("pnl_untrusted") is True
                        or extra.get("pnl_untrusted") is True
                        or "pnl_untrusted" in tags
                    ),
                }
    return found


def verify_targets(trades_dir: Path) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """Fail-closed verification of the 6 targets. Returns (ok, problems, verified_rows)."""
    found = _find_records(trades_dir)
    problems: List[str] = []
    verified: List[Dict[str, Any]] = []
    for spec in EXPECTED_RECORDS:
        rh = spec["record_hash"]
        row = found.get(rh)
        if row is None:
            problems.append(f"record_hash {rh[:16]}… NOT FOUND in ledgers")
            continue
        if row["strategy"] != _TARGET_STRATEGY or row["symbol"] != _TARGET_SYMBOL:
            problems.append(
                f"{rh[:16]}… is {row['strategy']}|{row['symbol']}, expected "
                f"{_TARGET_STRATEGY}|{_TARGET_SYMBOL}"
            )
            continue
        if row["pnl_untrusted"]:
            problems.append(f"{rh[:16]}… carries pnl_untrusted — refusing (would be Q2's row)")
            continue
        if abs(row["qty"] - spec["qty"]) > _QTY_TOL or abs(row["pnl"] - spec["pnl"]) > _PNL_TOL:
            problems.append(
                f"{rh[:16]}… qty/pnl {row['qty']}/{row['pnl']} != expected "
                f"{spec['qty']}/{spec['pnl']}"
            )
            continue
        verified.append({"record_hash": rh, **row})
    return (not problems), problems, verified


# --------------------------------------------------------------------------- #
# manifest build + idempotency
# --------------------------------------------------------------------------- #
def build_manifest(now_iso: str, verified: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "quarantined_at_utc": now_iso,
        "reason": QUARANTINE_REASON,
        "source": "scripts/ghost_scrub_pff1.py",
        "note": (
            "PFF1-Q1 double-book phantoms (gamma UNH re-buys closing phantom shorts). "
            "Non-destructive scrub — the hash-chained ledger is untouched; both scorekeepers "
            "drop these before parse. Sum 233 sh / -145.79. seq1 (+625.17 seed close) NOT here."
        ),
        "invalid_trades": [
            {
                "record_hash": r["record_hash"],
                "reason": QUARANTINE_REASON,
                "strategy": r.get("strategy"),
                "symbol": r.get("symbol"),
                "side": r.get("side"),
                "quantity": r.get("qty"),
                "pnl": r.get("pnl"),
                "source_file": r.get("source_file"),
            }
            for r in verified
        ],
        "invalid_fills": [],
    }


def manifest_is_current(path: Path) -> bool:
    """True iff an existing manifest already pins exactly the 6 target hashes (idempotent NOOP)."""
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(doc, Mapping):
        return False
    pinned = {
        e.get("record_hash")
        for e in (doc.get("invalid_trades") or [])
        if isinstance(e, Mapping)
    }
    return pinned == {r["record_hash"] for r in EXPECTED_RECORDS}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description="Non-destructive ghost-scrub of the 6 PFF1 phantom UNH closes (dry-run default)."
    )
    ap.add_argument("--execute", action="store_true", help="Write the manifest (mutates runtime/). Default is dry-run.")
    ap.add_argument("--confirm", default="", help=f"Required with --execute: the exact token {CONFIRM_TOKEN}.")
    ap.add_argument("--idempotent-ok", action="store_true",
                    help="Permit overwriting an existing manifest of the same name (.bak'd first).")
    ap.add_argument("--repo-root", default=None, help="Override repo root (tests).")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_FOR_IMPORT
    runtime_dir = repo_root / "runtime"
    trades_dir = repo_root / "data" / "trades"
    reports_dir = repo_root / "reports"
    manifest_path = runtime_dir / MANIFEST_NAME
    now = _utcnow()
    now_iso = _iso_z(now)
    stamp = _stamp(now)

    print("=" * 78)
    print("GHOST-SCRUB PFF1 — 6 phantom gamma UNH closes (non-destructive quarantine pin)")
    print("=" * 78)
    print(f"repo_root   : {repo_root}")
    print(f"trades_dir  : {trades_dir}")
    print(f"manifest    : {manifest_path}")
    print(f"mode        : {'EXECUTE' if args.execute else 'DRY-RUN (no mutation)'}")

    # 1. Verify the 6 targets exist AND match their expected shape (fail-closed).
    ok, problems, verified = verify_targets(trades_dir)
    print("\n-- target verification --")
    for r in verified:
        print(f"  OK  {r['record_hash'][:16]}…  {r['strategy']}|{r['symbol']} "
              f"{r['side']:4s} qty={r['qty']:g} pnl={r['pnl']:+.2f}  [{r['source_file']}]")
    for p in problems:
        print(f"   !!  {p}")
    if not ok:
        print("\nREFUSING: one or more target records did not verify. Nothing written.")
        return 2
    sum_qty = sum(r["qty"] for r in verified)
    sum_pnl = sum(r["pnl"] for r in verified)
    print(f"  => {len(verified)} records verified; sum_qty={sum_qty:g}  sum_pnl={sum_pnl:+.2f}")

    # 2. Idempotency: an already-correct manifest is a NOOP.
    already = manifest_is_current(manifest_path)
    if already:
        print("\nIDEMPOTENT NOOP: manifest already pins exactly these 6 records. Nothing to do.")
        return 0
    if manifest_path.is_file() and not args.idempotent_ok:
        print(f"\nREFUSING: {MANIFEST_NAME} already exists (different content). "
              f"Re-run with --idempotent-ok to overwrite (a .bak is taken first).")
        return 2

    manifest = build_manifest(now_iso, verified)

    # 3. Dry-run stops here.
    if not args.execute:
        print("\n-- planned manifest (DRY-RUN, not written) --")
        print(json.dumps(manifest, indent=2))
        print("\nExpected effect once applied to LIVE runtime (Phase-3, operator GO):")
        print("  SCR effective_trades 73 -> 67 (-6); trusted total_pnl += 145.79.")
        print("  total_pnl reaches -375.60 ONLY after the SCR shadow server (:9618) reloads Q2 (D2);")
        print("  scrub-alone (Q2 not yet active) => +103.78 + 145.79 = +249.57.")
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
    _backup(manifest_path, stamp, backups)  # only if overwriting (--idempotent-ok)
    _atomic_write_json(manifest_path, manifest)
    print(f"WROTE {manifest_path}")

    # 5. Signed audit report.
    report = {
        "schema_version": REPORT_SCHEMA,
        "applied_at_utc": now_iso,
        "confirm_token": CONFIRM_TOKEN,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "records_pinned": verified,
        "sum_qty": sum_qty,
        "sum_pnl": round(sum_pnl, 2),
        "backups": backups,
        "expected_effect": {
            "scr_effective_delta": -6,
            "trusted_total_pnl_delta": round(-sum_pnl, 2),
            "scr_total_pnl_target_note": (
                "-375.60 requires the SCR shadow server (:9618) to have reloaded PFF1-Q2 (D2); "
                "scrub-alone yields +249.57"
            ),
        },
        "gate": {"exec_mode": gate_reason},
    }
    report_path = reports_dir / f"ghost_scrub_pff1_{stamp}.json"
    _atomic_write_json(report_path, report)
    print(f"WROTE {report_path}")
    print("\nGHOST-SCRUB applied. Both scorekeepers now drop these 6 records (excluded_quarantined).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
