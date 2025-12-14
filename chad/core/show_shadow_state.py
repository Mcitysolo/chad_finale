"""
Show Shadow State – CLI (Phase 5 / 6 Bridge)

This small CLI reads the latest Shadow Confidence snapshot from:

    data/shadow/shadow_state.json

and prints a human-readable summary to stdout.

Usage:
    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_shadow_state
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


SHADOW_STATE_PATH = Path("data") / "shadow" / "shadow_state.json"


def _load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"shadow state snapshot not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    try:
        snapshot = _load_snapshot(SHADOW_STATE_PATH)
    except FileNotFoundError as exc:
        print(f"[show_shadow_state] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[show_shadow_state] ERROR reading snapshot: {exc}", file=sys.stderr)
        sys.exit(1)

    state = snapshot.get("state", "UNKNOWN")
    sizing_factor = snapshot.get("sizing_factor", 0.0)
    paper_only = snapshot.get("paper_only", True)
    reasons = snapshot.get("reasons", [])
    stats = snapshot.get("stats", {}) or {}

    total_trades = int(stats.get("total_trades", 0))
    win_rate = float(stats.get("win_rate", 0.0))
    total_pnl = float(stats.get("total_pnl", 0.0))
    live_trades = int(stats.get("live_trades", 0))
    paper_trades = int(stats.get("paper_trades", 0))

    print("=== CHAD Shadow State ===")
    print(f"state        : {state}")
    print(f"sizing_factor: {sizing_factor:.3f}")
    print(f"paper_only   : {paper_only}")
    print()
    print("=== Performance Snapshot ===")
    print(f"total_trades : {total_trades}")
    print(f"win_rate     : {win_rate:.3f}")
    print(f"total_pnl    : {total_pnl:.2f}")
    print(f"live_trades  : {live_trades}")
    print(f"paper_trades : {paper_trades}")
    print()
    print("=== Reasons ===")
    if not reasons:
        print("  (no reasons recorded)")
    else:
        for idx, reason in enumerate(reasons, start=1):
            print(f"  {idx}. {reason}")

    sys.exit(0)


if __name__ == "__main__":
    main()
