"""Position-authority validator (R1 / TRUTH-RECONCILE-1).

Read-only cross-surface validator for the four paper-position truth files.

Compares:
  - runtime/positions_snapshot.json     (symbol-keyed list)
  - runtime/ibkr_paper_ledger_state.json (hash-keyed dict)
  - runtime/position_guard.json          (strategy|symbol keyed)
  - runtime/reconciliation_state.json    (classifier output)

Mismatch categories: missing_symbol, extra_symbol, qty_mismatch, side_mismatch,
stale_ts, key_shape_mismatch.

The validator never writes to runtime/. It does not designate a canonical
writer; that decision is operator-domain (see
ops/pending_actions/R1_canonical_position_authority_gap_2026-05-27.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RUNTIME = Path("runtime")
DEFAULT_STALE_SECONDS = 1800  # 30 min
QTY_TOLERANCE = 1e-6
EXIT_OK = 0
EXIT_MISMATCH = 2
EXIT_READ_ERROR = 3


@dataclass
class SurfaceLoad:
    path: str
    ok: bool
    error: str | None = None
    ts_utc: str | None = None
    age_seconds: float | None = None
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> signed qty
    raw_count: int = 0
    key_shape: str = "unknown"


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def _age_seconds(ts: str | None, now: datetime) -> float | None:
    parsed = _parse_ts(ts)
    if parsed is None:
        return None
    return (now - parsed).total_seconds()


def load_snapshot(path: Path, now: datetime) -> SurfaceLoad:
    out = SurfaceLoad(path=str(path), ok=False, key_shape="symbol_list")
    if not path.exists():
        out.error = "missing"
        return out
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        out.error = f"json_error: {exc}"
        return out
    rows = data.get("positions") or []
    for r in rows:
        sym = r.get("symbol")
        qty = r.get("position")
        if sym is None or qty is None:
            continue
        out.positions[sym] = out.positions.get(sym, 0.0) + float(qty)
    out.raw_count = len(rows)
    out.ts_utc = data.get("ts_utc")
    out.age_seconds = _age_seconds(out.ts_utc, now)
    out.ok = True
    return out


def load_ledger(path: Path, now: datetime) -> SurfaceLoad:
    out = SurfaceLoad(path=str(path), ok=False, key_shape="hash_dict")
    if not path.exists():
        out.error = "missing"
        return out
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        out.error = f"json_error: {exc}"
        return out
    if not isinstance(data, dict):
        out.error = "ledger_not_dict"
        return out
    missing_symbol_keys: list[str] = []
    for hash_key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        sym = entry.get("symbol")
        qty = entry.get("qty")
        if sym is None:
            missing_symbol_keys.append(hash_key)
            continue
        if qty is None:
            continue
        out.positions[sym] = out.positions.get(sym, 0.0) + float(qty)
    out.raw_count = len(data)
    # Ledger has no top-level ts_utc; use file mtime as a proxy.
    try:
        mtime = path.stat().st_mtime
        out.ts_utc = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        out.age_seconds = (now - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds()
    except Exception:
        pass
    out.ok = True
    if missing_symbol_keys:
        out.error = f"ledger_entries_missing_symbol_count={len(missing_symbol_keys)}"
    return out


def load_guard(path: Path, now: datetime) -> SurfaceLoad:
    out = SurfaceLoad(path=str(path), ok=False, key_shape="strategy_symbol")
    if not path.exists():
        out.error = "missing"
        return out
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        out.error = f"json_error: {exc}"
        return out
    if not isinstance(data, dict):
        out.error = "guard_not_dict"
        return out
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("open"):
            continue
        sym = entry.get("symbol")
        qty = entry.get("quantity")
        side = (entry.get("side") or "").upper()
        if sym is None or qty is None:
            continue
        signed = float(qty) * (-1.0 if side == "SELL" else 1.0)
        out.positions[sym] = out.positions.get(sym, 0.0) + signed
    out.raw_count = len(data)
    try:
        mtime = path.stat().st_mtime
        out.ts_utc = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        out.age_seconds = (now - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds()
    except Exception:
        pass
    out.ok = True
    return out


def load_reconciliation(path: Path, now: datetime) -> SurfaceLoad:
    out = SurfaceLoad(path=str(path), ok=False, key_shape="classifier_state")
    if not path.exists():
        out.error = "missing"
        return out
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        out.error = f"json_error: {exc}"
        return out
    out.ts_utc = data.get("ts_utc")
    out.age_seconds = _age_seconds(out.ts_utc, now)
    out.raw_count = int((data.get("counts") or {}).get("chad", 0) or 0)
    out.ok = True
    # Reconciliation does not expose per-symbol qty here; it is a classifier
    # output. We keep positions={} but surface status as metadata.
    out.error = data.get("status")  # e.g. "GREEN" / "RED"
    return out


def compare_surfaces(
    snapshot: SurfaceLoad,
    ledger: SurfaceLoad,
    guard: SurfaceLoad,
    stale_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "missing_symbol": [],
        "extra_symbol": [],
        "qty_mismatch": [],
        "side_mismatch": [],
        "stale_ts": [],
        "key_shape_mismatch": [],
        "worst_qty_diff": 0.0,
    }

    if snapshot.key_shape != "symbol_list" or ledger.key_shape != "hash_dict":
        report["key_shape_mismatch"].append(
            {"snapshot": snapshot.key_shape, "ledger": ledger.key_shape}
        )

    for surf in (snapshot, ledger, guard):
        if surf.age_seconds is not None and surf.age_seconds > stale_seconds:
            report["stale_ts"].append(
                {
                    "path": surf.path,
                    "age_seconds": surf.age_seconds,
                    "threshold": stale_seconds,
                }
            )

    # Snapshot vs ledger
    snap_syms = set(snapshot.positions)
    led_syms = set(ledger.positions)
    for sym in sorted(snap_syms - led_syms):
        report["extra_symbol"].append({"symbol": sym, "in": "snapshot", "missing_from": "ledger"})
    for sym in sorted(led_syms - snap_syms):
        report["missing_symbol"].append({"symbol": sym, "in": "ledger", "missing_from": "snapshot"})

    for sym in sorted(snap_syms & led_syms):
        s_qty = snapshot.positions[sym]
        l_qty = ledger.positions[sym]
        diff = abs(s_qty - l_qty)
        if diff > QTY_TOLERANCE:
            entry = {
                "symbol": sym,
                "snapshot_qty": s_qty,
                "ledger_qty": l_qty,
                "diff": diff,
            }
            report["qty_mismatch"].append(entry)
            report["worst_qty_diff"] = max(report["worst_qty_diff"], diff)
        if (s_qty > 0) != (l_qty > 0) and (s_qty != 0 and l_qty != 0):
            report["side_mismatch"].append(
                {"symbol": sym, "snapshot_qty": s_qty, "ledger_qty": l_qty}
            )

    return report


def build_report(runtime_dir: Path, stale_seconds: float = DEFAULT_STALE_SECONDS) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc)
    snapshot = load_snapshot(runtime_dir / "positions_snapshot.json", now)
    ledger = load_ledger(runtime_dir / "ibkr_paper_ledger_state.json", now)
    guard = load_guard(runtime_dir / "position_guard.json", now)
    recon = load_reconciliation(runtime_dir / "reconciliation_state.json", now)

    mismatch = compare_surfaces(snapshot, ledger, guard, stale_seconds)
    surfaces = {
        "snapshot": asdict(snapshot),
        "ledger": asdict(ledger),
        "guard": asdict(guard),
        "reconciliation": asdict(recon),
    }
    any_load_error = not (snapshot.ok and ledger.ok and guard.ok)
    has_mismatch = (
        bool(mismatch["missing_symbol"])
        or bool(mismatch["extra_symbol"])
        or bool(mismatch["qty_mismatch"])
        or bool(mismatch["side_mismatch"])
        or bool(mismatch["stale_ts"])
        or bool(mismatch["key_shape_mismatch"])
    )
    return {
        "validator": "position_authority.v1",
        "ts_utc": now.isoformat(),
        "runtime_dir": str(runtime_dir),
        "stale_seconds_threshold": stale_seconds,
        "surfaces": surfaces,
        "mismatch": mismatch,
        "verdict": "MISMATCH" if has_mismatch else ("LOAD_ERROR" if any_load_error else "OK"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="position_authority", description=__doc__)
    p.add_argument("--check", action="store_true", help="Run validator (default action).")
    p.add_argument(
        "--runtime",
        default=str(DEFAULT_RUNTIME),
        help="Runtime directory containing the truth files (default: ./runtime).",
    )
    p.add_argument(
        "--stale-seconds",
        type=float,
        default=DEFAULT_STALE_SECONDS,
        help="Stale-timestamp threshold in seconds (default 1800).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to also write the structured report to (not under runtime/).",
    )
    args = p.parse_args(argv)

    runtime_dir = Path(args.runtime)
    report = build_report(runtime_dir, stale_seconds=args.stale_seconds)
    print(json.dumps(report, indent=2, default=str))

    if args.output:
        out = Path(args.output)
        resolved = out.resolve()
        if "runtime" in resolved.parts and DEFAULT_RUNTIME.resolve() in resolved.parents:
            print(
                f"position_authority: refusing to write inside runtime/: {resolved}",
                file=sys.stderr,
            )
            return EXIT_READ_ERROR
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))

    verdict = report["verdict"]
    if verdict == "OK":
        return EXIT_OK
    if verdict == "LOAD_ERROR":
        return EXIT_READ_ERROR
    return EXIT_MISMATCH


if __name__ == "__main__":
    raise SystemExit(main())
