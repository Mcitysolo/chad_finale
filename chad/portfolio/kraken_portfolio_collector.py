from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig


@dataclass(frozen=True)
class KrakenPortfolioSnapshot:
    """
    Simple snapshot of Kraken spot account balances.

    Fields:
        balances: Mapping from Kraken asset code to balance (float).
                  Only non-zero balances are included.
    """

    balances: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"balances": self.balances}


class KrakenPortfolioCollector:
    """
    Reads balances from Kraken via KrakenClient and optionally persists them
    to a JSON file for CHAD to consume.

    This collector is deliberately read-only:
        * No trading.
        * No withdrawals.
        * No funding.
    """

    def __init__(self, client: KrakenClient) -> None:
        self._client = client

    def collect(self) -> KrakenPortfolioSnapshot:
        """
        Fetch balances from Kraken and return a snapshot.

        Returns:
            KrakenPortfolioSnapshot with non-zero balances only.
        """
        raw_balances = self._client.get_balances()
        non_zero: Dict[str, float] = {
            asset: amount
            for asset, amount in raw_balances.items()
            if amount != 0.0
        }
        return KrakenPortfolioSnapshot(balances=non_zero)

    @staticmethod
    def write_snapshot(snapshot: KrakenPortfolioSnapshot, path: Path) -> None:
        """
        Write the snapshot to a JSON file atomically.

        Args:
            snapshot: Snapshot to persist.
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(snapshot.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def default_output_path() -> Path:
    """
    Default location for Kraken balances snapshot, under repo_root/runtime/.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "runtime" / "kraken_balances.json"


def _build_client_from_env() -> KrakenClient:
    cfg = KrakenClientConfig.from_env()
    return KrakenClient(cfg)


def _cmd_collect(output: Path) -> int:
    client = _build_client_from_env()
    collector = KrakenPortfolioCollector(client)

    try:
        snapshot = collector.collect()
    except Exception as exc:  # noqa: BLE001
        print(f"[KRAKEN PORTFOLIO] ERROR during collection: {exc}")
        return 1

    KrakenPortfolioCollector.write_snapshot(snapshot, output)
    print(f"[KRAKEN PORTFOLIO] Snapshot written to: {output}")
    return 0


def _cmd_print() -> int:
    client = _build_client_from_env()
    collector = KrakenPortfolioCollector(client)

    try:
        snapshot = collector.collect()
    except Exception as exc:  # noqa: BLE001
        print(f"[KRAKEN PORTFOLIO] ERROR during collection: {exc}")
        return 1

    if not snapshot.balances:
        print("[KRAKEN PORTFOLIO] No non-zero balances.")
        return 0

    print("[KRAKEN PORTFOLIO] Balances:")
    for asset, amount in sorted(snapshot.balances.items()):
        print(f"  {asset}: {amount}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Kraken portfolio collector.\n"
            "Uses KRAKEN_API_KEY and KRAKEN_API_SECRET from the environment to "
            "read balances via the Kraken private API and optionally persist "
            "them to runtime/kraken_balances.json."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect balances and write them to a JSON file.",
    )
    collect_parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional path to write the balances snapshot. "
            "Default: repo_root/runtime/kraken_balances.json"
        ),
    )

    subparsers.add_parser(
        "print",
        help="Print balances to stdout (no file write).",
    )

    args = parser.parse_args(argv)

    if args.command == "collect":
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
        else:
            out_path = default_output_path()
        return _cmd_collect(out_path)

    if args.command == "print":
        return _cmd_print()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
