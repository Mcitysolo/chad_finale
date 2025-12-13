"""
CHAD Orchestrator v2
====================

Core responsibilities
---------------------
* Read current portfolio snapshot (IBKR + Coinbase + Kraken)
* Apply global daily risk fraction to compute a portfolio risk budget
* Use DynamicRiskAllocator to convert strategy weights → per-strategy dollar caps
* Write runtime/dynamic_caps.json for the Daily Throttle + execution layer
* Run either a single cycle (--once) or a continuous loop with sleep

Design principles
-----------------
* Pure orchestration: no direct broker or strategy logic here.
* Everything broker/platform-specific happens *outside* and feeds a single
  runtime/portfolio_snapshot.json produced by whichever collector you use.
* Fully typed (PEP 484), dataclasses for config, explicit error handling.
* Async-ready loop (even though the allocator work is CPU-light and sync).

Environment
-----------
This module uses the following environment variables:

* CHAD_DAILY_RISK_PCT             (float, default 5.0)   # e.g. 5 = risk 5% of equity
* CHAD_ORCH_INTERVAL_SECONDS      (float, default 60.0)  # loop sleep
* CHAD_ORCH_RUN_FOREVER           ("1"/"0", default "0") # if "1", ignore --once
* CHAD_ORCH_LOG_LEVEL             (str, default "INFO")  # logging level name
* CHAD_IBKR_EQUITY_FALLBACK       (float, default 0.0)   # fallback if snapshot missing
* CHAD_COINBASE_EQUITY_FALLBACK   (float, default 0.0)   # fallback if snapshot missing
* CHAD_KRAKEN_EQUITY_FALLBACK     (float, default 0.0)   # fallback if snapshot missing

Files
-----
* runtime/portfolio_snapshot.json   # input, optional

  Canonical kraken-aware schema:

    {
      "ibkr_equity": 1011718.62,
      "coinbase_equity": 0.0,
      "kraken_equity": 50.0
    }

  Backwards compatibility: we also support the older USD-style schema:

    {
      "ibkr_equity_usd": 123456.78,
      "coinbase_equity_usd": 20000.0,
      "total_equity_usd": 143456.78
    }

  In that case, kraken_equity defaults to 0.0 and is derived only if an
  explicit kraken_equity / kraken_equity_usd field is present.

* runtime/dynamic_caps.json         # output, produced every cycle
    (exact structure defined by DynamicRiskAllocator)

CLI
---
Run once:

    python -m chad.core.orchestrator --once --log-level INFO

Run as a loop (env-driven):

    export CHAD_ORCH_RUN_FOREVER=1
    python -m chad.core.orchestrator --log-level INFO
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chad.risk.dynamic_risk_allocator import (
    DynamicRiskAllocator,
    PortfolioSnapshot,
    StrategyAllocation,
    default_output_path as default_dynamic_caps_path,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RUNTIME_DIR: Path = REPO_ROOT / "runtime"
PORTFOLIO_SNAPSHOT_PATH: Path = RUNTIME_DIR / "portfolio_snapshot.json"

LOGGER_NAME = "chad.orchestrator"
logger = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorSettings:
    """
    Immutable configuration for the orchestrator.

    Populated from environment variables, with a small number of CLI overrides.
    """

    daily_risk_pct: float
    loop_interval_seconds: float
    run_forever: bool
    log_level: str
    portfolio_snapshot_path: Path
    dynamic_caps_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        portfolio_snapshot_path: Optional[Path] = None,
        dynamic_caps_path: Optional[Path] = None,
    ) -> "OrchestratorSettings":
        """
        Build settings from environment variables.

        This avoids assumptions about upstream brokers and lets you control
        risk and loop behaviour via env.
        """
        daily_risk_pct = float(os.getenv("CHAD_DAILY_RISK_PCT", "5.0"))
        loop_interval_seconds = float(os.getenv("CHAD_ORCH_INTERVAL_SECONDS", "60.0"))
        run_forever_env = os.getenv("CHAD_ORCH_RUN_FOREVER", "0")
        run_forever = run_forever_env == "1"
        log_level = os.getenv("CHAD_ORCH_LOG_LEVEL", "INFO").upper()

        snapshot_path = portfolio_snapshot_path or PORTFOLIO_SNAPSHOT_PATH
        caps_path = dynamic_caps_path or default_dynamic_caps_path()

        return cls(
            daily_risk_pct=daily_risk_pct,
            loop_interval_seconds=loop_interval_seconds,
            run_forever=run_forever,
            log_level=log_level,
            portfolio_snapshot_path=snapshot_path,
            dynamic_caps_path=caps_path,
        )


@dataclass(frozen=True)
class OrchestratorCycleResult:
    """
    Summary of a single orchestrator cycle.
    """

    total_equity: float
    daily_risk_fraction: float
    portfolio_risk_cap: float
    dynamic_caps_path: Path
    used_fallback_snapshot: bool


# ---------------------------------------------------------------------------
# Orchestrator core
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Core orchestration entrypoint for CHAD.

    This class is deliberately focused on *risk orchestration*:
    it computes and persists per-strategy dollar caps based on live or
    snapshot portfolio equity, leaving brokers & strategies to other modules.
    """

    def __init__(self, settings: OrchestratorSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> OrchestratorSettings:
        return self._settings

    # ------------------------------------------------------------------ #
    # Portfolio snapshot                                                 #
    # ------------------------------------------------------------------ #

    def _load_portfolio_snapshot(self) -> Tuple[PortfolioSnapshot, bool]:
        """
        Load portfolio snapshot from JSON, with environment-based fallbacks.

        Returns:
            (snapshot, used_fallback)

        Supports both schema styles:
          * ibkr_equity / coinbase_equity / kraken_equity
          * ibkr_equity_usd / coinbase_equity_usd / total_equity_usd

        If the JSON file is missing or invalid, falls back to env vars:
            CHAD_IBKR_EQUITY_FALLBACK
            CHAD_COINBASE_EQUITY_FALLBACK
            CHAD_KRAKEN_EQUITY_FALLBACK
        """
        path = self._settings.portfolio_snapshot_path

        if path.is_file():
            try:
                raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

                # Prefer *_usd keys if present, else base keys, else 0.0.
                ibkr_raw = raw.get(
                    "ibkr_equity_usd",
                    raw.get("ibkr_equity", 0.0),
                )
                coinbase_raw = raw.get(
                    "coinbase_equity_usd",
                    raw.get("coinbase_equity", 0.0),
                )
                # Kraken historically had no *_usd variant; we still allow one
                # for symmetry in case you introduce it later.
                kraken_raw = raw.get(
                    "kraken_equity",
                    raw.get("kraken_equity_usd", 0.0),
                )

                try:
                    ibkr_equity = float(ibkr_raw)
                except (TypeError, ValueError):
                    ibkr_equity = 0.0

                try:
                    coinbase_equity = float(coinbase_raw)
                except (TypeError, ValueError):
                    coinbase_equity = 0.0

                try:
                    kraken_equity = float(kraken_raw)
                except (TypeError, ValueError):
                    kraken_equity = 0.0

                snapshot = PortfolioSnapshot(
                    ibkr_equity=ibkr_equity,
                    coinbase_equity=coinbase_equity,
                    kraken_equity=kraken_equity,
                )

                logger.info(
                    "orchestrator.portfolio_snapshot_loaded",
                    extra={
                        "path": str(path),
                        "ibkr_equity": ibkr_equity,
                        "coinbase_equity": coinbase_equity,
                        "kraken_equity": kraken_equity,
                        "total_equity": snapshot.total_equity,
                    },
                )
                return snapshot, False
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "orchestrator.portfolio_snapshot_invalid",
                    extra={"path": str(path), "error": str(exc)},
                )

        # Fallback path (no file or invalid JSON)
        ibkr_equity_fallback = float(os.getenv("CHAD_IBKR_EQUITY_FALLBACK", "0.0"))
        coinbase_equity_fallback = float(
            os.getenv("CHAD_COINBASE_EQUITY_FALLBACK", "0.0")
        )
        kraken_equity_fallback = float(
            os.getenv("CHAD_KRAKEN_EQUITY_FALLBACK", "0.0")
        )

        snapshot = PortfolioSnapshot(
            ibkr_equity=ibkr_equity_fallback,
            coinbase_equity=coinbase_equity_fallback,
            kraken_equity=kraken_equity_fallback,
        )

        logger.info(
            "orchestrator.portfolio_snapshot_fallback",
            extra={
                "ibkr_equity": ibkr_equity_fallback,
                "coinbase_equity": coinbase_equity_fallback,
                "kraken_equity": kraken_equity_fallback,
                "total_equity": snapshot.total_equity,
            },
        )
        return snapshot, True

    # ------------------------------------------------------------------ #
    # Dynamic caps                                                       #
    # ------------------------------------------------------------------ #

    def _build_allocator(self) -> DynamicRiskAllocator:
        """
        Construct a DynamicRiskAllocator using StrategyAllocation derived from env.

        We use StrategyAllocation.from_env_or_default() so the behaviour is
        consistent with your standalone allocator CLI.
        """
        allocation = StrategyAllocation.from_env_or_default()
        daily_fraction = self._settings.daily_risk_pct / 100.0
        return DynamicRiskAllocator(
            strategy_allocation=allocation,
            daily_risk_fraction=daily_fraction,
        )

    def refresh_dynamic_caps(self) -> OrchestratorCycleResult:
        """
        Refresh runtime/dynamic_caps.json based on the current portfolio snapshot.

        This is the *core* action of the orchestrator. It is safe to call
        multiple times; each call overwrites the caps file atomically.
        """
        logger.info(
            "orchestrator.cycle_start",
            extra={
                "daily_risk_pct": self._settings.daily_risk_pct,
                "dynamic_caps_path": str(self._settings.dynamic_caps_path),
            },
        )

        snapshot, used_fallback = self._load_portfolio_snapshot()
        allocator = self._build_allocator()
        payload = allocator.build_payload(snapshot=snapshot)

        out_path = self._settings.dynamic_caps_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")

        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        tmp.replace(out_path)

        total_equity = float(payload["total_equity"])
        daily_risk_fraction = float(payload["daily_risk_fraction"])
        portfolio_risk_cap = float(payload["portfolio_risk_cap"])

        logger.info(
            "orchestrator.execution_config",
            extra={
                "total_equity": total_equity,
                "daily_risk_fraction": daily_risk_fraction,
                "portfolio_risk_cap": portfolio_risk_cap,
                "dynamic_caps_path": str(out_path),
            },
        )

        logger.info(
            "orchestrator.cycle_summary",
            extra={
                "total_equity": total_equity,
                "portfolio_risk_cap": portfolio_risk_cap,
                "used_fallback_snapshot": used_fallback,
            },
        )

        return OrchestratorCycleResult(
            total_equity=total_equity,
            daily_risk_fraction=daily_risk_fraction,
            portfolio_risk_cap=portfolio_risk_cap,
            dynamic_caps_path=out_path,
            used_fallback_snapshot=used_fallback,
        )

    # ------------------------------------------------------------------ #
    # Run loop                                                           #
    # ------------------------------------------------------------------ #

    async def run_once(self) -> OrchestratorCycleResult:
        """
        Run a single orchestrator cycle.

        This is split out for testability and for your `--once` CLI flag.
        """
        loop = asyncio.get_running_loop()
        # Offload the sync work to default executor so future async work can coexist.
        result = await loop.run_in_executor(None, self.refresh_dynamic_caps)
        return result

    async def run_forever(self) -> None:
        """
        Run the orchestrator in an infinite loop with configured sleep.

        Safe for systemd / Docker; exits only if an unhandled exception bubbles
        out of refresh_dynamic_caps (which is rare, and logged).
        """
        interval = self._settings.loop_interval_seconds
        logger.info(
            "orchestrator.loop_start",
            extra={
                "interval_seconds": interval,
                "daily_risk_pct": self._settings.daily_risk_pct,
                "portfolio_snapshot_path": str(self._settings.portfolio_snapshot_path),
                "dynamic_caps_path": str(self._settings.dynamic_caps_path),
            },
        )

        while True:
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "orchestrator.cycle_error", extra={"error": str(exc)}
                )
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CHAD Orchestrator — compute dynamic per-strategy dollar caps "
            "from portfolio snapshot and daily risk percentage."
        )
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single orchestrator cycle and exit.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="",
        help=(
            "Optional log level override (e.g. DEBUG, INFO). "
            "Defaults to CHAD_ORCH_LOG_LEVEL or INFO if unset."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    settings = OrchestratorSettings.from_env()

    # Configure logging based on env/CLI.
    effective_level = (
        args.log_level.upper() if args.log_level else settings.log_level
    )
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(getattr(logging, effective_level, logging.INFO))

    orch = Orchestrator(settings=settings)

    if args.once or not settings.run_forever:
        asyncio.run(orch.run_once())
    else:
        asyncio.run(orch.run_forever())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
