from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Representation of CHAD's portfolio snapshot JSON on disk.

    Schema (current, kraken-aware)
    ------------------------------
    We now treat IBKR, Coinbase, and Kraken as separate first-class sources:

        {
          "ibkr_equity": 1011718.62,
          "coinbase_equity": 0.0,
          "kraken_equity": 50.0
        }

    Notes
    -----
    * Older snapshots that only have ibkr_equity / coinbase_equity are still
      supported. In that case, kraken_equity is treated as 0.0.
    * This module does NOT compute total_equity; that is derived elsewhere
      (DynamicRiskAllocator) as:

          total_equity = ibkr_equity + coinbase_equity + kraken_equity

    * This module does NOT merge Kraken into ibkr_equity anymore. It writes a
      dedicated kraken_equity field so that IBKR remains IBKR-only.
    """

    ibkr_equity: float
    coinbase_equity: float
    kraken_equity: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ibkr_equity": float(self.ibkr_equity),
            "coinbase_equity": float(self.coinbase_equity),
            "kraken_equity": float(self.kraken_equity),
        }

    @classmethod
    def from_file(cls, path: Path) -> "PortfolioSnapshot":
        """
        Load snapshot from JSON, handling missing/corrupt files gracefully.
        """
        if not path.is_file():
            # Default snapshot: no equity yet.
            return cls(ibkr_equity=0.0, coinbase_equity=0.0, kraken_equity=0.0)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # If corrupted, fall back to zero; orchestrator will log separately.
            return cls(ibkr_equity=0.0, coinbase_equity=0.0, kraken_equity=0.0)

        ibkr_raw = data.get("ibkr_equity", 0.0)
        coinbase_raw = data.get("coinbase_equity", 0.0)
        kraken_raw = data.get("kraken_equity", 0.0)

        try:
            ibkr_val = float(ibkr_raw)
        except (TypeError, ValueError):
            ibkr_val = 0.0

        try:
            coinbase_val = float(coinbase_raw)
        except (TypeError, ValueError):
            coinbase_val = 0.0

        try:
            kraken_val = float(kraken_raw)
        except (TypeError, ValueError):
            kraken_val = 0.0

        return cls(
            ibkr_equity=ibkr_val,
            coinbase_equity=coinbase_val,
            kraken_equity=kraken_val,
        )

    def write(self, path: Path) -> None:
        """
        Atomically write the snapshot JSON back to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)


@dataclass(frozen=True)
class KrakenBalances:
    """
    Wrapper for balances from runtime/kraken_balances.json.

    Expected structure:
      {
        "balances": {
          "ZCAD": 50.0,
          "XXBT": 0.01,
          ...
        }
      }

    For now, we only treat ZCAD as fiat (1:1 CAD). Other assets are logged
    but ignored for equity aggregation, which is exact for your current state
    (ZCAD only) and conservative if you later start holding crypto.
    """

    balances: Dict[str, float]

    @classmethod
    def from_file(cls, path: Path) -> "KrakenBalances":
        if not path.is_file():
            return cls(balances={})

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(balances={})

        raw = data.get("balances", {})
        if not isinstance(raw, dict):
            return cls(balances={})

        cleaned: Dict[str, float] = {}
        for asset, amount in raw.items():
            try:
                cleaned[str(asset)] = float(amount)
            except (TypeError, ValueError):
                continue

        return cls(balances=cleaned)

    def fiat_equity_cad(self) -> float:
        """
        Compute total fiat equity in CAD using Kraken fiat asset codes.

        Currently supported:
          * ZCAD -> treated as CAD 1:1

        Other fiat or crypto assets are ignored for now (conservative).
        """
        total = 0.0

        # ZCAD is Kraken's CAD fiat code.
        zcad = self.balances.get("ZCAD", 0.0)
        total += zcad

        # NOTE: if you later start holding other fiat (ZUSD, ZEUR, etc.) or
        # crypto, we will extend this function to value them using Kraken's
        # public ticker. For now, this is exact for your current balance
        # (ZCAD only) and conservative otherwise.
        return total


def default_paths() -> Tuple[Path, Path]:
    """
    Return (portfolio_snapshot_path, kraken_balances_path) under repo_root/runtime.
    """
    root = Path(__file__).resolve().parents[2]
    snapshot_path = root / "runtime" / "portfolio_snapshot.json"
    kraken_path = root / "runtime" / "kraken_balances.json"
    return snapshot_path, kraken_path


def merge_kraken_into_snapshot(
    *,
    snapshot_path: Path,
    kraken_balances_path: Path,
) -> PortfolioSnapshot:
    """
    Read existing portfolio_snapshot.json and kraken_balances.json, compute
    Kraken fiat equity (ZCAD only, for now), and write an updated snapshot
    where:

        ibkr_equity   := ibkr_equity        (unchanged)
        coinbase_eq   := coinbase_equity    (unchanged)
        kraken_equity := kraken_fiat_cad    (overwritten with latest Kraken)

    This keeps IBKR equity IBKR-only and tracks Kraken as a separate field,
    while allowing the orchestrator + DynamicRiskAllocator to treat all three
    sources as part of total_equity.
    """
    snapshot = PortfolioSnapshot.from_file(snapshot_path)
    kraken_balances = KrakenBalances.from_file(kraken_balances_path)

    kraken_fiat_cad = kraken_balances.fiat_equity_cad()

    updated_snapshot = PortfolioSnapshot(
        ibkr_equity=snapshot.ibkr_equity,
        coinbase_equity=snapshot.coinbase_equity,
        kraken_equity=kraken_fiat_cad,
    )

    updated_snapshot.write(snapshot_path)
    return updated_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Kraken fiat equity into CHAD portfolio_snapshot.json.\n"
            "Reads runtime/kraken_balances.json and runtime/portfolio_snapshot.json, "
            "sets kraken_equity based on ZCAD balances, and writes an updated snapshot."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="",
        help="Optional override for portfolio_snapshot.json path.",
    )
    parser.add_argument(
        "--kraken-balances-path",
        type=str,
        default="",
        help="Optional override for kraken_balances.json path.",
    )

    args = parser.parse_args(argv)

    default_snapshot_path, default_kraken_path = default_paths()

    snapshot_path = (
        Path(args.snapshot_path).expanduser().resolve()
        if args.snapshot_path
        else default_snapshot_path
    )
    kraken_balances_path = (
        Path(args.kraken_balances_path).expanduser().resolve()
        if args.kraken_balances_path
        else default_kraken_path
    )

    updated = merge_kraken_into_snapshot(
        snapshot_path=snapshot_path,
        kraken_balances_path=kraken_balances_path,
    )

    print(
        f"Updated snapshot at {snapshot_path} "
        f"(ibkr_equity={updated.ibkr_equity:.2f}, "
        f"coinbase_equity={updated.coinbase_equity:.2f}, "
        f"kraken_equity={updated.kraken_equity:.2f})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

