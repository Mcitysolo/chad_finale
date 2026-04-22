#!/usr/bin/env python3
"""
scripts/clear_edge_decay.py

Operator recovery tool — clear the F4 edge-decay halt for a strategy.

Usage:
    python3 scripts/clear_edge_decay.py --strategy alpha
    python3 scripts/clear_edge_decay.py --strategy alpha --by reviewer

Prints the previous allocation entry and the cleared state so the
audit log in the terminal reflects exactly what changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk.edge_decay_monitor import (  # noqa: E402
    clear_strategy_halt,
    read_allocations,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear an F4 edge-decay halt for a strategy."
    )
    parser.add_argument("--strategy", required=True, help="Strategy identifier to un-halt.")
    parser.add_argument("--by", default="operator", help="Who is clearing the halt (cleared_by).")
    args = parser.parse_args()

    before = read_allocations()
    previous = before.get("allocations", {}).get(args.strategy)
    print("previous_entry:")
    print(json.dumps(previous or {}, indent=2, sort_keys=True))

    after = clear_strategy_halt(strategy=args.strategy, cleared_by=args.by)
    updated = after.get("allocations", {}).get(args.strategy)
    print("\ncleared_entry:")
    print(json.dumps(updated or {}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
