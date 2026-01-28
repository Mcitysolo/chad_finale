from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Runtime file freshness for portfolio snapshot (short on purpose)
PORTFOLIO_SNAPSHOT_TTL_SECONDS = 300  # 5 minutes


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class IBKRConnectionConfig:
    """
    IBKR connection settings for paper gateway by default.
    Reads from environment variables used across this repo.
    """
    host: str
    port: int
    client_id: int
    account_id: str

    @staticmethod
    def from_env() -> "IBKRConnectionConfig":
        host = os.getenv("IBKR_HOST", "127.0.0.1")
        port = int(os.getenv("IBKR_PORT", "4002"))
        client_id = int(os.getenv("IBKR_CLIENT_ID", "99"))
        account_id = os.getenv("IBKR_ACCOUNT_ID", "").strip()
        return IBKRConnectionConfig(host=host, port=port, client_id=client_id, account_id=account_id)


class IBKRPortfolioCollector:
    """
    Collector that reads NetLiquidation via ib_insync and writes runtime/portfolio_snapshot.json.

    Contract:
      - Never writes secrets
      - Atomic + fsync
      - Adds ts_utc + ttl_seconds for audit freshness
      - Preserves other venue equities already present in the snapshot
    """

    def __init__(self, cfg: IBKRConnectionConfig) -> None:
        self._cfg = cfg

    def get_net_liquidation(self) -> float:
        """
        Returns NetLiquidation (USD) for the configured account, or sum across accounts if not set.
        """
        from ib_insync import IB  # type: ignore[import]

        ib = IB()
        try:
            ib.connect(self._cfg.host, self._cfg.port, clientId=self._cfg.client_id, readonly=True, timeout=6.0)

            # accountSummary returns a list of AccountValue(tag, value, currency, account)
            rows = ib.accountSummary()
            values_by_account: Dict[str, float] = {}
            for r in rows:
                try:
                    if str(getattr(r, "tag", "")) != "NetLiquidation":
                        continue
                    acct = str(getattr(r, "account", "")).strip()
                    val = float(getattr(r, "value", "0") or 0.0)
                    values_by_account[acct] = val
                except Exception:
                    continue

            if not values_by_account:
                raise RuntimeError("No NetLiquidation values found in account summary")

            if self._cfg.account_id:
                acct = self._cfg.account_id
                if acct not in values_by_account:
                    raise RuntimeError(f"IBKR_ACCOUNT_ID={acct!r} not found in NetLiquidation summary")
                total = values_by_account[acct]
            else:
                total = sum(values_by_account.values())

            return float(total)
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
        with current NetLiquidation, preserve coinbase_equity and kraken_equity,
        write updated snapshot back atomically, TTL-stamped.
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

        # Preserve other venues if present
        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        coinbase_equity = _to_float(data.get("coinbase_equity", 0.0))
        kraken_equity = _to_float(data.get("kraken_equity", 0.0))

        new_payload: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": int(PORTFOLIO_SNAPSHOT_TTL_SECONDS),
            "ibkr_equity": float(ibkr_equity),
            "coinbase_equity": float(coinbase_equity),
            "kraken_equity": float(kraken_equity),
        }

        _atomic_write_json(path, new_payload)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "IBKR Portfolio Collector v2\n"
            "Reads NetLiquidation via ib_insync and updates runtime/portfolio_snapshot.json "
            "with ibkr_equity, preserving coinbase_equity and kraken_equity."
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

    if args.command not in ("print", "collect"):
        parser.print_help()
        return 2

    cfg = IBKRConnectionConfig.from_env()
    collector = IBKRPortfolioCollector(cfg)

    if args.command == "print":
        try:
            val = collector.get_net_liquidation()
        except Exception as exc:
            print(f"[IBKR PORTFOLIO] Error: {exc}")
            return 1
        print(val)
        return 0

    # collect
    try:
        snapshot_path = Path(args.snapshot_path).expanduser().resolve() if args.snapshot_path else None
        out_path = collector.update_snapshot(snapshot_path=snapshot_path)
        print(f"[IBKR PORTFOLIO] Updated snapshot: {out_path}")
        return 0
    except Exception as exc:
        print(f"[IBKR PORTFOLIO] Error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
