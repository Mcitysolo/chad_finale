#!/usr/bin/env python3
"""
chad/portfolio/ibkr_portfolio_collector.py

IBKR Portfolio Collector for CHAD
=================================

This module is responsible for **writing IBKR equity into a shared portfolio
snapshot JSON file** (runtime/portfolio_snapshot.json by default).

It does NOT fetch balances directly from IBKR. Instead, it expects an equity
value to be injected via:

    - CLI argument:     --ibkr-equity
    - or env var:       CHAD_IBKR_EQUITY_USD
    - or fallback env:  CHAD_IBKR_EQUITY_FALLBACK

This keeps the collector:

    - Honest (no fake API calls)
    - Composable (can be driven by any upstream IBKR client)
    - Deterministic and testable

Schema
------

The snapshot file is a single JSON object. This collector ensures:

    {
      "ibkr_equity_usd": <float>,
      "coinbase_equity_usd": <float, optional>,
      "total_equity_usd": <float, optional, convenience>,
      ... (other fields may be added by other collectors)
    }

When IBKR equity is written:

- "ibkr_equity_usd" is set/overwritten.
- "total_equity_usd" is recomputed if coinbase_equity_usd exists.

Typical sequence
----------------

1. Some upstream job fetches IBKR NetLiquidation and calls:

       python -m chad.portfolio.ibkr_portfolio_collector \\
           --ibkr-equity 123456.78

2. Coinbase collector merges its equity into the same snapshot.
3. The dynamic risk allocator reads the snapshot and computes caps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


# --------------------------------------------------------------------------- #
# Data structures and helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IbkrEquitySource:
    """
    Source of IBKR equity in USD.

    Attributes
    ----------
    cli_value:
        Value passed via CLI, if any.
    env_primary:
        Value from CHAD_IBKR_EQUITY_USD, if set.
    env_fallback:
        Value from CHAD_IBKR_EQUITY_FALLBACK, if set.
    """

    cli_value: Optional[float]
    env_primary: Optional[float]
    env_fallback: Optional[float]

    def resolve(self) -> float:
        """
        Resolve the effective IBKR equity in USD, with precedence:

            1) CLI value (if provided and valid)
            2) CHAD_IBKR_EQUITY_USD (if valid)
            3) CHAD_IBKR_EQUITY_FALLBACK (if valid)

        Returns
        -------
        float
            The resolved equity value.

        Raises
        ------
        ValueError
            If no positive numeric value can be resolved.
        """
        if self.cli_value is not None and self.cli_value > 0:
            return self.cli_value

        if self.env_primary is not None and self.env_primary > 0:
            return self.env_primary

        if self.env_fallback is not None and self.env_fallback > 0:
            return self.env_fallback

        raise ValueError(
            "Unable to resolve IBKR equity: provide --ibkr-equity or set "
            "CHAD_IBKR_EQUITY_USD / CHAD_IBKR_EQUITY_FALLBACK to a positive number."
        )


def _read_float_env(name: str) -> Optional[float]:
    """Read a float from env; return None if missing or invalid."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value


def _load_snapshot(path: Path) -> Dict[str, Any]:
    """
    Load an existing portfolio snapshot if present, else return an empty dict.

    Malformed JSON results in an empty dict to avoid hard crashes in the
    collector (logging should be handled by the caller or surrounding system).
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _write_snapshot(path: Path, snapshot: Dict[str, Any]) -> None:
    """
    Persist the snapshot to disk using deterministic JSON formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI and main
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibkr_portfolio_collector.py",
        description=(
            "Merge IBKR equity (USD) into runtime/portfolio_snapshot.json as "
            "ibkr_equity_usd and recompute total_equity_usd if possible."
        ),
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="runtime/portfolio_snapshot.json",
        help="Optional override for portfolio_snapshot.json path.",
    )
    parser.add_argument(
        "--ibkr-equity",
        type=float,
        default=None,
        help=(
            "IBKR equity in USD (overrides CHAD_IBKR_EQUITY_USD and "
            "CHAD_IBKR_EQUITY_FALLBACK if provided)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("chad.portfolio.ibkr")

    snapshot_path = Path(args.snapshot_path)

    # Resolve IBKR equity from CLI + env
    source = IbkrEquitySource(
        cli_value=args.ibkr_equity,
        env_primary=_read_float_env("CHAD_IBKR_EQUITY_USD"),
        env_fallback=_read_float_env("CHAD_IBKR_EQUITY_FALLBACK"),
    )

    try:
        ibkr_equity = source.resolve()
    except ValueError as exc:
        logger.error("Failed to resolve IBKR equity: %s", exc)
        return 1

    logger.info("Resolved IBKR equity: %.2f USD", ibkr_equity)

    snapshot = _load_snapshot(snapshot_path)
    snapshot["ibkr_equity_usd"] = ibkr_equity

    # Recompute total_equity_usd if we also have coinbase_equity_usd
    coinbase_equity = snapshot.get("coinbase_equity_usd")
    if isinstance(coinbase_equity, (int, float)):
        total_equity = float(ibkr_equity) + float(coinbase_equity)
        snapshot["total_equity_usd"] = total_equity
        logger.info(
            "Updated total_equity_usd using IBKR + Coinbase: %.2f USD", total_equity
        )
    else:
        # Leave total_equity_usd untouched if we don't know the other side yet.
        logger.info(
            "Coinbase equity not present; wrote only ibkr_equity_usd into snapshot."
        )

    _write_snapshot(snapshot_path, snapshot)
    logger.info("Portfolio snapshot updated at %s", snapshot_path)
    print(f"[IBKR Portfolio Collector] Updated {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
