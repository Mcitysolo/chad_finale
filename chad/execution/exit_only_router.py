#!/usr/bin/env python3
"""
chad/execution/exit_only_router.py

Phase 9.1 (Part 2/4): Exit-Only Router Artifact (NO broker calls).

This module converts an ExitPlan (produced by chad.execution.exit_only_executor)
into a deterministic, auditable runtime artifact that downstream systems can consume
later (Phase 9.1 Part 3/4).

Hard guarantees:
- No broker calls (IBKR/Kraken/etc). Pure data transformation + atomic file write.
- Deterministic output ordering for stable diffing and reproducibility.
- Fail-closed: refuses to write unless LiveGate indicates exits-only AND entry lanes are disabled.
- Strict validation (types, quantities, supported asset classes).
- Crash-safe atomic write: tmp -> fsync -> replace -> best-effort dir fsync.

Runtime artifact (default):
- /home/ubuntu/CHAD FINALE/runtime/exit_only_plan.json

Input sources (recommended):
- LiveGate: http://127.0.0.1:9618/live-gate
- Exit plan builder: chad.execution.exit_only_executor.build_exit_only_plan()

Note:
- lane_id is accepted and preserved as metadata only. No lane behavior in Phase 9.1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from chad.execution.exit_only_executor import (
    AssetClass,
    ExitIntent,
    ExitOnlyError,
    ExitPlan,
    LiveGateDecision,
    Side,
)

DEFAULT_RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
DEFAULT_OUT_PATH = DEFAULT_RUNTIME_DIR / "exit_only_plan.json"


@dataclass(frozen=True)
class RouterResult:
    """Result of routing + writing an exit-only plan artifact."""
    out_path: str
    wrote: bool
    exits_count: int
    ts_utc: str
    sha256: str


def _utc_now_iso() -> str:
    # ISO-8601 Zulu, seconds precision (stable for logs)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Best-effort directory fsync for crash safety
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def _require_exits_only(gate: LiveGateDecision) -> None:
    if not bool(gate.allow_exits_only):
        raise ExitOnlyError("router_denied: allow_exits_only is false")

    # In exits-only mode, entries must be blocked at both paper and live levels.
    if bool(gate.allow_ibkr_paper) or bool(gate.allow_ibkr_live):
        raise ExitOnlyError("router_denied: exits-only requires paper/live entry disallowed")


def _validate_exit(exit_: ExitIntent) -> None:
    if not exit_.symbol or not isinstance(exit_.symbol, str):
        raise ExitOnlyError("invalid_exit: missing symbol")
    if float(exit_.qty) <= 0.0:
        raise ExitOnlyError(f"invalid_exit: qty must be > 0 for {exit_.symbol}")

    # Phase 9.1: conservative support set (expand later).
    if exit_.asset_class not in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.UNKNOWN):
        raise ExitOnlyError(f"invalid_exit: unsupported asset_class={exit_.asset_class}")

    if exit_.side not in (Side.SELL, Side.BUY):
        raise ExitOnlyError(f"invalid_exit: unsupported side={exit_.side}")

    if not exit_.currency:
        raise ExitOnlyError(f"invalid_exit: missing currency for {exit_.symbol}")


def _deterministic_sort(exits: Iterable[ExitIntent]) -> List[ExitIntent]:
    # Stable ordering: asset_class, symbol, side, qty
    def key(e: ExitIntent) -> Tuple[str, str, str, float]:
        return (str(e.asset_class.value), e.symbol.upper(), str(e.side.value), float(e.qty))

    return sorted(list(exits), key=key)


def build_router_artifact(
    *,
    plan: ExitPlan,
    live_gate: LiveGateDecision,
    out_path: Path = DEFAULT_OUT_PATH,
) -> RouterResult:
    """
    Build and write the deterministic exit-only artifact.

    Raises ExitOnlyError on any validation / gating failure.
    """
    _require_exits_only(live_gate)

    exits = _deterministic_sort(plan.exits)
    for e in exits:
        _validate_exit(e)

    payload: Dict[str, Any] = {
        "ts_utc": _utc_now_iso(),
        "lane_id": plan.lane_id,
        "exits_count": len(exits),
        "live_gate": asdict(live_gate),
        "exits": [asdict(e) for e in exits],
        "notes": list(plan.notes),
        "schema_version": "phase9.1.exit_only_plan.v1",
    }

    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    sha = _sha256_bytes(data)
    _atomic_write_bytes(out_path, data)

    return RouterResult(
        out_path=str(out_path),
        wrote=True,
        exits_count=len(exits),
        ts_utc=payload["ts_utc"],
        sha256=sha,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_live_gate_decision(obj: Dict[str, Any]) -> LiveGateDecision:
    return LiveGateDecision(
        allow_exits_only=bool(obj.get("allow_exits_only", False)),
        allow_ibkr_paper=bool(obj.get("allow_ibkr_paper", False)),
        allow_ibkr_live=bool(obj.get("allow_ibkr_live", False)),
        operator_mode=str(obj.get("operator_mode") or ""),
        reasons=tuple(obj.get("reasons") or ()),
    )


def cli(argv: Optional[List[str]] = None) -> int:
    """
    CLI helper to route an already-built plan to a runtime artifact.

    Inputs:
      - plan JSON on stdin OR via --plan-json
      - live-gate JSON via --live-gate-json (required)
      - output path via --out

    This CLI does NOT call brokers. It may call the backend only if you feed it backend JSON files.
    """
    ap = argparse.ArgumentParser(description="Phase 9.1 Exit-Only Router (writes runtime artifact; no broker calls).")
    ap.add_argument("--plan-json", default="", help="Path to plan JSON (as produced by ops/phase9_exit_only_plan.py). If empty, read from stdin.")
    ap.add_argument("--live-gate-json", default="", help="Path to LiveGate JSON (required).")
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH), help=f"Output artifact path (default: {DEFAULT_OUT_PATH})")
    args = ap.parse_args(argv)

    if not args.live_gate_json:
        raise SystemExit("--live-gate-json is required (router is fail-closed without LiveGate proof).")

    if args.plan_json:
        plan_obj = _load_json(Path(args.plan_json))
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("No plan JSON provided (stdin empty and --plan-json not set).")
        plan_obj = json.loads(raw)

    lg_obj = _load_json(Path(args.live_gate_json))
    gate = _to_live_gate_decision(lg_obj)

    exits_in = plan_obj.get("exits") or []
    notes_in = plan_obj.get("notes") or []
    lane_id = plan_obj.get("lane_id")
    ts_utc = str(plan_obj.get("ts_utc") or _utc_now_iso())

    exits: List[ExitIntent] = []
    for e in exits_in:
        ac_raw = str(e.get("asset_class") or "unknown").lower()
        side_raw = str(e.get("side") or "").upper()
        qty = float(e.get("qty") or 0.0)

        # Parse enums fail-closed
        try:
            ac = AssetClass(ac_raw)
        except Exception:
            ac = AssetClass.UNKNOWN

        try:
            side = Side(side_raw)
        except Exception as ex:
            raise ExitOnlyError(f"invalid_exit: unsupported side={side_raw}") from ex

        exits.append(
            ExitIntent(
                symbol=str(e.get("symbol") or ""),
                asset_class=ac,
                side=side,
                qty=abs(qty),
                currency=str(e.get("currency") or "USD"),
                reason=str(e.get("reason") or "exit_only"),
                lane_id=e.get("lane_id"),
            )
        )

    plan = ExitPlan(
        ts_utc=ts_utc,
        lane_id=lane_id,
        exits=tuple(exits),
        notes=tuple(str(x) for x in notes_in),
    )

    res = build_router_artifact(plan=plan, live_gate=gate, out_path=Path(args.out))
    print(json.dumps(asdict(res), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
