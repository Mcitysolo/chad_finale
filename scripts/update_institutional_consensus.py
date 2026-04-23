#!/usr/bin/env python3
"""
scripts/update_institutional_consensus.py

Weekly refresh for runtime/institutional_consensus.json.

Pulls the latest 13F-HR filings for the tracked institutional filers
(chad.market_data.sec_13f_fetcher.TARGET_FUNDS), aggregates them via
chad.analytics.institutional_consensus, and writes the top-N ranked
consensus plus normalized weights to runtime/institutional_consensus.json.

This is intentionally dumb: it runs once and exits. Wire it to a cron
or a weekly systemd timer — 13F data moves quarterly, so daily is
overkill.

Usage:
    python3 scripts/update_institutional_consensus.py
    python3 scripts/update_institutional_consensus.py --top-n 25
    python3 scripts/update_institutional_consensus.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.analytics.institutional_consensus import (  # noqa: E402
    CONSENSUS_PATH,
    InstitutionalConsensus,
)
from chad.market_data.sec_13f_fetcher import SEC13FFetcher, TARGET_FUNDS  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh institutional_consensus.json")
    p.add_argument("--top-n", type=int, default=25, help="Top N consensus holdings (default: 25)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print but do not write the consensus file",
    )
    p.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Keep holdings where the issuer name can't be mapped to a ticker",
    )
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Override output path (default: {CONSENSUS_PATH})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _setup_logging(args.verbose)
    log = logging.getLogger("update_institutional_consensus")

    fetcher = SEC13FFetcher()
    log.info("fetching 13F filings for %d tracked funds", len(TARGET_FUNDS))
    fund_holdings = fetcher.get_all_fund_holdings()

    # Sanity: how many funds came back non-empty?
    filled = {
        name: payload for name, payload in fund_holdings.items()
        if (payload.get("holdings") or [])
    }
    log.info("funds with non-empty holdings: %d / %d", len(filled), len(fund_holdings))

    consensus_obj = InstitutionalConsensus()
    entries = consensus_obj.compute_consensus(
        fund_holdings,
        top_n=args.top_n,
        include_unresolved=args.include_unresolved,
    )
    weights = consensus_obj.get_consensus_weights(entries)

    log.info(
        "consensus computed: top_n=%d entries=%d weighted_symbols=%d",
        args.top_n, len(entries), len(weights),
    )

    if not entries:
        log.warning(
            "no consensus entries produced; output will be empty. "
            "Check network access / SEC throttling / fund CIKs.",
        )

    funds_included = sorted(filled.keys())

    if args.dry_run:
        payload_preview = {
            "funds_included": funds_included,
            "top_holdings": [e.to_dict() for e in entries[:10]],
            "weights": weights,
        }
        print(json.dumps(payload_preview, indent=2))
        return 0

    out_path = consensus_obj.write_cache(
        entries,
        weights,
        funds_included=funds_included,
        path=args.output,
    )
    log.info("wrote %s (%d holdings, %d weights)", out_path, len(entries), len(weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
