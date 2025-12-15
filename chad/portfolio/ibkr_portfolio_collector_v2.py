from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IBKRConnectionConfig:
    """
    Connection configuration for IBKR Gateway / TWS via ib_insync.

    Environment variables:

      * IBKR_HOST              (default: 127.0.0.1)
      * IBKR_PORT              (default: 4002 – typical paper gateway)
      * IBKR_CLIENT_ID         (required)
      * IBKR_ACCOUNT_ID        (optional, filter specific account; else sum)
    """

    host: str
    port: int
    client_id: int
    account_id: Optional[str]

    @classmethod
    def from_env(cls) -> IBKRConnectionConfig:
        host = os.getenv("IBKR_HOST", "127.0.0.1")
        port_str = os.getenv("IBKR_PORT", "4002")
        client_id_str = os.getenv("IBKR_CLIENT_ID")
        account_id = os.getenv("IBKR_ACCOUNT_ID")

        missing: list[str] = []
        if not client_id_str:
            missing.append("IBKR_CLIENT_ID")

        if missing:
            raise RuntimeError(
                f"Missing required IBKR env vars: {', '.join(missing)}"
            )

        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(f"Invalid IBKR_PORT value: {port_str!r}") from exc

        try:
            client_id = int(client_id_str)
        except ValueError as exc:
            raise ValueError(f"Invalid IBKR_CLIENT_ID value: {client_id_str!r}") from exc

        return cls(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id,
        )


# ---------------------------------------------------------------------------
# Portfolio collector
# ---------------------------------------------------------------------------


class IBKRPortfolioCollector:
    """
    Reads NetLiquidation from IBKR via ib_insync and writes it into
    runtime/portfolio_snapshot.json as ibkr_equity, preserving coinbase_equity and kraken_equity.

    This collector is read-only: it does NOT place orders or move funds.
    """

    def __init__(self, config: IBKRConnectionConfig) -> None:
        self._cfg = config

    def _connect_ib(self):
        try:
            from ib_insync import IB  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ib_insync is not installed. Install it with 'pip install ib-insync' "
                "inside the CHAD venv."
            ) from exc

        ib = IB()
        ib.connect(
            host=self._cfg.host,
            port=self._cfg.port,
            clientId=self._cfg.client_id,
            timeout=10.0,
        )
        return ib

    def get_net_liquidation(self) -> float:
        """
        Fetch NetLiquidation in account currency for the configured account.

        If IBKR_ACCOUNT_ID is set, use that account. Otherwise, sum across all.
        """
        from ib_insync import IB  # type: ignore[import]  # noqa: F401

        ib = self._connect_ib()
        try:
            summary = ib.accountSummary()

            if not summary:
                # Give IBKR a moment to populate summary
                ib.sleep(2.0)
                summary = ib.accountSummary()

            if not summary:
                raise RuntimeError("No account summary data returned from IBKR")

            values_by_account: Dict[str, float] = {}

            for item in summary:
                # item has attributes: tag, account, value, currency, etc.
                tag = str(getattr(item, "tag", ""))
                currency = str(getattr(item, "currency", ""))
                value_str = str(getattr(item, "value", ""))
                account = str(getattr(item, "account", ""))

                if tag != "NetLiquidation":
                    continue

                # We accept any currency here; CHAD treats this as "base equity"
                try:
                    val = float(value_str)
                except ValueError:
                    continue

                values_by_account[account] = val

            if not values_by_account:
                raise RuntimeError("No NetLiquidation values found in account summary")

            if self._cfg.account_id:
                acct = self._cfg.account_id
                if acct not in values_by_account:
                    raise RuntimeError(
                        f"IBKR_ACCOUNT_ID={acct!r} not found in NetLiquidation summary"
                    )
                total = values_by_account[acct]
            else:
                total = sum(values_by_account.values())

            return total
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # snapshot integration                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_snapshot_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "runtime" / "portfolio_snapshot.json"

    def update_snapshot(self, snapshot_path: Optional[Path] = None) -> Path:
        """
        Read existing portfolio_snapshot.json (if any), overwrite ibkr_equity
        with the current NetLiquidation, preserve coinbase_equity, and
        write the updated snapshot back atomically.
        """
        path = snapshot_path or self._default_snapshot_path()

        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        ibkr_equity = self.get_net_liquidation()

        coinbase_raw = data.get("coinbase_equity", 0.0)
        try:
            coinbase_equity = float(coinbase_raw)
        except (TypeError, ValueError):
            coinbase_equity = 0.0

        kraken_raw = data.get("kraken_equity", 0.0)
        try:
            kraken_equity = float(kraken_raw)
        except (TypeError, ValueError):
            kraken_equity = 0.0

        new_payload: Dict[str, Any] = {
            "ibkr_equity": float(ibkr_equity),
            "coinbase_equity": coinbase_equity,
            "kraken_equity": kraken_equity,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(new_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)

        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "IBKR Portfolio Collector v2\n"
            "Reads NetLiquidation via ib_insync and updates "
            "runtime/portfolio_snapshot.json with ibkr_equity, preserving "
            "coinbase_equity."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "print",
        help="Print NetLiquidation to stdout (no file writes).",
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="Update portfolio_snapshot.json with ibkr_equity.",
    )
    collect_parser.add_argument(
        "--snapshot-path",
        type=str,
        default="",
        help="Optional override for portfolio_snapshot.json path.",
    )

    args = parser.parse_args(argv)

    try:
        cfg = IBKRConnectionConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        print(f"[IBKR PORTFOLIO] Config error: {exc}")
        return 1

    collector = IBKRPortfolioCollector(cfg)

    if args.command == "print":
        try:
            netliq = collector.get_net_liquidation()
        except Exception as exc:  # noqa: BLE001
            print(f"[IBKR PORTFOLIO] ERROR fetching NetLiquidation: {exc}")
            return 1
        print(f"[IBKR PORTFOLIO] NetLiquidation: {netliq:.2f}")
        return 0

    if args.command == "collect":
        snapshot_path: Optional[Path] = None
        if args.snapshot_path:
            snapshot_path = Path(args.snapshot_path).expanduser().resolve()

        try:
            out_path = collector.update_snapshot(snapshot_path=snapshot_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[IBKR PORTFOLIO] ERROR updating snapshot: {exc}")
            return 1

        print(f"[IBKR PORTFOLIO] Updated snapshot at {out_path}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
