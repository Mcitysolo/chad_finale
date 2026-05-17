#!/usr/bin/env python3
"""
scripts/preview_bag_intent.py

Dry-run preview of a BAG (combo) options intent from CLI arguments.

Phase D Item 2 Tier 1 — paper-only hardening. This script lets an operator
construct an OptionsSpreadSpec, project it to legacy meta, and print the
combined payload as JSON without ever connecting to IBKR.

Hard safety contract
--------------------
* Never imports ``ib_async`` or any IBKR connection class.
* Never calls ``placeOrder`` (the IBKR adapter is not imported).
* Refuses to run unless ``CHAD_EXECUTION_MODE`` is unset / "paper" /
  "dry_run" / "preview".

Usage
-----
::

    CHAD_EXECUTION_MODE=paper python3 scripts/preview_bag_intent.py \\
        --symbol SPY \\
        --expiry 20260618 \\
        --long-strike 737 \\
        --short-strike 744 \\
        --long-right C \\
        --short-right C \\
        --contracts 1 \\
        --spread-type BULL_CALL \\
        --net-debit-estimate 350 \\
        --max-loss-per-contract 700
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the chad package is importable when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chad.options.spread_spec import OptionsSpreadSpec  # noqa: E402


_ALLOWED_MODES = frozenset({"", "paper", "dry_run", "preview"})


def _check_execution_mode() -> None:
    """Refuse to run in any live mode. Returns silently on success, calls
    ``sys.exit(2)`` with a JSON error otherwise."""
    mode = os.environ.get("CHAD_EXECUTION_MODE", "").strip().lower()
    if mode not in _ALLOWED_MODES:
        payload = {
            "ok": False,
            "error": "preview_refused_non_paper_mode",
            "CHAD_EXECUTION_MODE": mode,
            "allowed_modes": sorted(m for m in _ALLOWED_MODES if m),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        sys.exit(2)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dry-run BAG combo intent preview (paper-only).",
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--expiry", required=True, help="YYYYMMDD")
    p.add_argument("--long-strike", required=True, type=float)
    p.add_argument("--short-strike", required=True, type=float)
    p.add_argument("--long-right", required=True, choices=["C", "P", "c", "p"])
    p.add_argument("--short-right", required=True, choices=["C", "P", "c", "p"])
    p.add_argument("--contracts", required=True, type=int)
    p.add_argument("--spread-type", default="CUSTOM")
    p.add_argument("--net-debit-estimate", required=False, type=float, default=None)
    p.add_argument("--max-loss-per-contract", required=False, type=float, default=None)
    p.add_argument("--ratio-long", type=int, default=1)
    p.add_argument("--ratio-short", type=int, default=1)
    p.add_argument("--exchange", default="SMART")
    p.add_argument("--currency", default="USD")
    p.add_argument("--spread-id", default=None)
    p.add_argument("--dte", type=int, default=None)
    return p


def build_preview(args: argparse.Namespace) -> dict:
    """Build the JSON payload for a single preview request. Pure function —
    no I/O. Validation errors propagate as ValueError so the caller can
    serialize them deterministically."""
    spec = OptionsSpreadSpec(
        symbol=args.symbol,
        expiry=args.expiry,
        long_strike=float(args.long_strike),
        short_strike=float(args.short_strike),
        long_right=args.long_right,
        short_right=args.short_right,
        ratio_long=int(args.ratio_long),
        ratio_short=int(args.ratio_short),
        exchange=args.exchange,
        currency=args.currency,
        spread_type=args.spread_type,
        max_loss_per_contract=args.max_loss_per_contract,
        net_debit_estimate=args.net_debit_estimate,
        spread_id=args.spread_id,
        dte=args.dte,
    )
    if int(args.contracts) < 1:
        raise ValueError("contracts must be >= 1")

    return {
        "ok": True,
        "symbol": spec.symbol,
        "sec_type": "BAG",
        "contracts": int(args.contracts),
        "spread_spec": spec.as_dict(),
        "legacy_meta": spec.to_legacy_meta(),
        "bag_legs": spec.bag_leg_dicts(),
    }


def main(argv: list[str] | None = None) -> int:
    _check_execution_mode()
    parser = _build_argparser()
    args = parser.parse_args(argv)
    try:
        payload = build_preview(args)
    except ValueError as exc:
        payload = {
            "ok": False,
            "error": "spread_spec_validation_failed",
            "detail": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
