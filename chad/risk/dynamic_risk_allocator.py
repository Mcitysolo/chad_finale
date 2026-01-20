from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# These are the weights that previously summed to 1.03 and triggered the error.
# We now treat them as RELATIVE WEIGHTS and normalize them internally.
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "alpha": 0.35,
    "beta": 0.30,
    "gamma": 0.15,
    "omega": 0.10,
    "delta": 0.05,
    "crypto": 0.03,
    "forex": 0.03,
}


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyAllocation:
    """
    Holds raw strategy weights and provides normalized weights.

    Raw weights may sum to ANY value (> 1.0 is allowed).
    Normalization is done when computing caps.
    """

    weights: Dict[str, float]

    @classmethod
    def from_env_or_default(cls) -> "StrategyAllocation":
        """
        CHAD_STRATEGY_WEIGHTS env format:

            CHAD_STRATEGY_WEIGHTS="alpha=0.35,beta=0.30,gamma=0.15,omega=0.10,delta=0.05,crypto=0.03,forex=0.03"
        """
        env_val = os.getenv("CHAD_STRATEGY_WEIGHTS")
        if env_val:
            weights: Dict[str, float] = {}
            for chunk in env_val.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "=" not in chunk:
                    raise ValueError(
                        f"Invalid CHAD_STRATEGY_WEIGHTS chunk {chunk!r} "
                        "(expected name=value)",
                    )
                name, value = chunk.split("=", 1)
                name = name.strip().lower()
                weights[name] = float(value)
            return cls(weights=weights)

        # Fallback to the canonical map we’ve been using.
        return cls(weights=dict(DEFAULT_STRATEGY_WEIGHTS))

    def normalized(self) -> Dict[str, float]:
        """
        Return weights normalized to sum to 1.0.

        If they sum to > 1.0 (e.g. 1.03) we KEEP proportions but rescale.
        """
        total = sum(self.weights.values())
        if total <= 0.0:
            raise ValueError("StrategyAllocation: total weight must be > 0")

        if abs(total - 1.0) > 1e-6:
            logger.info(
                "StrategyAllocation: normalizing weights sum=%.6f to 1.0 "
                "(relative proportions unchanged)",
                total,
            )

        return {name: weight / total for name, weight in self.weights.items()}

    def raw_sum(self) -> float:
        return float(sum(self.weights.values()))


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Simple snapshot of current portfolio equity in base currency.

    Fields
    ------
    ibkr_equity:
        Equity held at IBKR (paper or live), in base units (e.g. USD or CAD).

    coinbase_equity:
        Equity held at Coinbase (or other crypto venue aggregated as Coinbase),
        in base units.

    kraken_equity:
        Equity held at Kraken that we want to participate in the global risk
        budget. This is a first-class field rather than being folded into
        ibkr_equity, so that upstream collectors can remain explicit and
        downstream logic can always compute:

            total_equity = ibkr + coinbase + kraken

    Backwards compatibility
    -----------------------
    Older callers (including existing orchestrator tests) construct
    PortfolioSnapshot with only (ibkr_equity, coinbase_equity). The
    kraken_equity field has a default of 0.0 so those call sites remain valid.
    """

    ibkr_equity: float
    coinbase_equity: float
    kraken_equity: float = 0.0

    @property
    def total_equity(self) -> float:
        """
        Return total equity used for risk budgeting.

        All components are clamped at >= 0.0 to avoid negative-equity artefacts
        from upstream data glitches.
        """
        ibkr = max(0.0, float(self.ibkr_equity))
        coinbase = max(0.0, float(self.coinbase_equity))
        kraken = max(0.0, float(self.kraken_equity))
        return ibkr + coinbase + kraken


@dataclass(frozen=True)
class DynamicRiskAllocator:
    """
    Given:

    - StrategyAllocation (weights per strategy)
    - daily_risk_fraction (0.0 .. 1.0)

    And a PortfolioSnapshot, compute per-strategy dollar caps and
    build a dynamic_caps payload suitable for the daily throttle.

    The allocator is deliberately stateless: all configuration is passed in via
    StrategyAllocation and the daily_risk_fraction, and all results are
    returned as a pure dict payload.
    """

    strategy_allocation: StrategyAllocation
    daily_risk_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.daily_risk_fraction <= 1.0):
            raise ValueError("daily_risk_fraction must be between 0.0 and 1.0")

    # ------------------------------------------------------------------ #
    # Core math                                                          #
    # ------------------------------------------------------------------ #

    def compute_caps(self, *, snapshot: PortfolioSnapshot) -> Dict[str, float]:
        """
        Compute dollar caps per strategy based on current equity.
        """
        total_equity = snapshot.total_equity
        if total_equity < 0.0:
            # This should never happen given clamping in PortfolioSnapshot,
            # but we keep the guard as a hard assertion.
            raise ValueError("total_equity must be >= 0")

        norm = self.strategy_allocation.normalized()
        portfolio_risk_cap = total_equity * self.daily_risk_fraction

        caps: Dict[str, float] = {}
        for name, frac in norm.items():
            caps[name] = portfolio_risk_cap * frac

        logger.info(
            "DynamicRiskAllocator: total_equity=%.2f daily_risk_fraction=%.3f "
            "portfolio_risk_cap=%.2f",
            total_equity,
            self.daily_risk_fraction,
            portfolio_risk_cap,
        )
        return caps

    def build_payload(self, *, snapshot: PortfolioSnapshot) -> Dict[str, object]:
        """
        Build the JSON payload we’ll write to dynamic_caps.json.
        """
        total_equity = snapshot.total_equity
        raw = dict(self.strategy_allocation.weights)
        norm = self.strategy_allocation.normalized()
        caps = self.compute_caps(snapshot=snapshot)

        portfolio_risk_cap = total_equity * self.daily_risk_fraction
        sum_raw = self.strategy_allocation.raw_sum()
        sum_norm = sum(norm.values())

        per_strategy: Dict[str, Dict[str, float]] = {}
        for name in sorted(norm.keys()):
            per_strategy[name] = {
                "raw_weight": float(raw[name]),
                "normalized_weight": float(norm[name]),
                "fraction_of_total_equity": float(norm[name] * self.daily_risk_fraction),
                "dollar_cap": float(caps[name]),
            }

        return {
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": int(DYNAMIC_CAPS_TTL_SECONDS),
            "total_equity": float(total_equity),
            "daily_risk_fraction": float(self.daily_risk_fraction),
            "portfolio_risk_cap": float(portfolio_risk_cap),
            "sum_raw_weights": float(sum_raw),
            "sum_normalized_weights": float(sum_norm),
            "raw_weights": raw,
            "normalized_weights": norm,
            "strategy_caps": caps,
            "strategies": per_strategy,
        }


# ---------------------------------------------------------------------------
# Paths / helpers
# ---------------------------------------------------------------------------

DYNAMIC_CAPS_TTL_SECONDS = 300  # 5 minutes (caps should be refreshed frequently)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")



def default_output_path() -> Path:
    """
    Default to repo_root/runtime/dynamic_caps.json
    (matches your earlier manual example).
    """
    root = Path(__file__).resolve().parents[2]
    return root / "runtime" / "dynamic_caps.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_allocator_from_args(
    args: argparse.Namespace,
) -> tuple[DynamicRiskAllocator, PortfolioSnapshot]:
    """
    Helper for the CLI entrypoint.

    Supports both the original two-field interface and an optional kraken
    equity flag. This keeps existing tests / scripts valid while allowing you
    to experiment with three-source equity from the command line.
    """
    ibkr_equity = float(args.ibkr_equity)
    coinbase_equity = float(args.coinbase_equity)
    kraken_equity = float(getattr(args, "kraken_equity", 0.0))

    snapshot = PortfolioSnapshot(
        ibkr_equity=ibkr_equity,
        coinbase_equity=coinbase_equity,
        kraken_equity=kraken_equity,
    )
    daily_risk_fraction = float(args.daily_risk_pct) / 100.0

    allocation = StrategyAllocation.from_env_or_default()
    allocator = DynamicRiskAllocator(
        strategy_allocation=allocation,
        daily_risk_fraction=daily_risk_fraction,
    )
    return allocator, snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dynamic per-strategy dollar caps from portfolio equity, "
            "daily risk percentage, and strategy weights. "
            "Writes a dynamic_caps.json file compatible with the daily throttle."
        )
    )
    parser.add_argument(
        "--ibkr-equity",
        type=float,
        required=True,
        help="Current IBKR equity in base currency.",
    )
    parser.add_argument(
        "--coinbase-equity",
        type=float,
        required=True,
        help="Current Coinbase equity in base currency.",
    )
    parser.add_argument(
        "--kraken-equity",
        type=float,
        default=0.0,
        help="Optional Kraken equity in base currency (default: 0.0).",
    )
    parser.add_argument(
        "--daily-risk-pct",
        type=float,
        required=True,
        help="Percent of total equity to expose today (e.g. 10 = 10%%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Output path for dynamic_caps.json. "
            "Defaults to repo_root/runtime/dynamic_caps.json."
        ),
    )

    args = parser.parse_args(argv)

    allocator, snapshot = _build_allocator_from_args(args)
    payload = allocator.build_payload(snapshot=snapshot)

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(output_path)
    print(f"dynamic_caps written to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

