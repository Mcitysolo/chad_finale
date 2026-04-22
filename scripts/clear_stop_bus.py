#!/usr/bin/env python3
"""
scripts/clear_stop_bus.py

Operator recovery tool — clear the STOP bus after investigating the halt.

Usage:
    python3 scripts/clear_stop_bus.py [--by NAME]

Prints the previous state (for the audit trail) and then clears.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk.stop_bus_state import clear_stop_bus, read_stop_bus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear the CHAD STOP bus.")
    parser.add_argument(
        "--by",
        default="operator",
        help="Who is clearing the bus (written to cleared_by).",
    )
    args = parser.parse_args()

    before = read_stop_bus()
    print("previous_state:")
    print(json.dumps(before, indent=2, sort_keys=True))

    after = clear_stop_bus(cleared_by=args.by)
    print("\ncleared_state:")
    print(json.dumps(after, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
