"""Quarantine tool for historical placeholder fills (HISTORICAL-PLACEHOLDER-1).

Read-only by default. ``--apply`` requires ``--operator-approve "<reason>"``
with a non-empty reason; it writes a timestamped backup of the input file
and a sidecar quarantine manifest. The original NDJSON is never mutated —
even in ``--apply`` mode the existing append-only ledger is preserved per
the audit's §20 prohibited-actions list.

Two output artefacts in ``--apply`` mode:
  1. ``<input>.backup.<UTC-ISO>.ndjson`` — byte-identical copy of the input.
  2. ``<dir>/quarantine_placeholder_<base>.json`` — sidecar manifest in the
     shape consumed by ``chad/analytics/quarantine.py`` (fill_id list +
     reason metadata).

In ``--check`` mode the tool prints the evidence report only and exits 0
(read-only). It never writes anything to disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chad.execution.fill_validation import classify_placeholder, DEFAULT_SIGNALS

EXIT_OK = 0
EXIT_USER_ERROR = 1


def _utc_iso_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _scan(input_path: Path) -> list[dict[str, Any]]:
    """Yield rows tagged with classifier output. Pure read — no side effects."""
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                rows.append({
                    "line": line_no,
                    "parse_error": str(exc),
                    "is_placeholder": False,
                    "signals_fired": [],
                })
                continue
            result = classify_placeholder(row)
            payload = row.get("payload", {}) if isinstance(row, dict) else {}
            rows.append({
                "line": line_no,
                "is_placeholder": result.is_placeholder,
                "signals_fired": result.signals_fired,
                "fill_id": payload.get("fill_id"),
                "sequence_id": row.get("sequence_id") if isinstance(row, dict) else None,
                "record_hash": row.get("record_hash") if isinstance(row, dict) else None,
                "fill_price": payload.get("fill_price"),
                "symbol": payload.get("symbol"),
                "strategy": payload.get("strategy"),
            })
    return rows


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_errors = sum(1 for r in rows if r.get("parse_error"))
    placeholders = [r for r in rows if r.get("is_placeholder")]
    return {
        "scanned_rows": total,
        "parse_errors": parse_errors,
        "placeholder_rows": len(placeholders),
        "placeholder_lines": [r["line"] for r in placeholders],
        "placeholder_fill_ids": [r["fill_id"] for r in placeholders if r.get("fill_id")],
        "signals_chain": list(DEFAULT_SIGNALS),
    }


def _apply(input_path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    stamp = _utc_iso_compact()
    backup_path = input_path.with_name(f"{input_path.name}.backup.{stamp}.ndjson")
    shutil.copy2(input_path, backup_path)
    base = input_path.stem
    sidecar_path = input_path.with_name(f"quarantine_placeholder_{base}.json")
    sidecar = {
        "schema_version": "fills_quarantine_sidecar.v1",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path.name),
        "operator_reason": reason,
        "scanned_rows": summary["scanned_rows"],
        "placeholder_rows": summary["placeholder_rows"],
        "fill_ids": summary["placeholder_fill_ids"],
        "lines": summary["placeholder_lines"],
        "signals_chain": summary["signals_chain"],
        "evidence": rows,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return {
        "backup_path": str(backup_path),
        "sidecar_path": str(sidecar_path),
        "wrote_files": [str(backup_path), str(sidecar_path)],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quarantine_placeholder_fills", description=__doc__)
    p.add_argument("--input", required=True, help="Path to FILLS_*.ndjson to scan.")
    p.add_argument("--check", action="store_true", help="Read-only mode (default).")
    p.add_argument("--apply", action="store_true", help="Write a backup + sidecar manifest. Requires --operator-approve.")
    p.add_argument("--operator-approve", default=None, help="Non-empty reason justifying --apply.")
    args = p.parse_args(argv)

    if not args.check and not args.apply:
        args.check = True

    in_path = Path(args.input)
    if not in_path.is_file():
        print(json.dumps({"error": "input_not_found", "input": str(in_path)}), file=sys.stderr)
        return EXIT_USER_ERROR

    rows = _scan(in_path)
    summary = _summarise(rows)
    out = {
        "tool": "quarantine_placeholder_fills.v1",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "check",
        "input": str(in_path),
        "summary": summary,
        "rows": rows,
    }

    if args.apply:
        reason = (args.operator_approve or "").strip()
        if not reason:
            print(
                json.dumps({"error": "operator_approve_required", "hint": "--apply requires --operator-approve '<reason>'"}),
                file=sys.stderr,
            )
            return EXIT_USER_ERROR
        applied = _apply(in_path, summary, rows, reason=reason)
        out["applied"] = applied

    print(json.dumps(out, indent=2, default=str))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
