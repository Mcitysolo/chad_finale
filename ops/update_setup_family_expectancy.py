#!/usr/bin/env python3
"""ops/update_setup_family_expectancy.py — Gap-2 (v9.1 audit).

Thin runner around
:class:`chad.analytics.setup_family_expectancy_updater.SetupFamilyExpectancyUpdater`.

The script:
- resolves trades_dir / output_path / lookback_days from environment
  variables (falling back to canonical defaults),
- invokes ``SetupFamilyExpectancyUpdater.run()`` once,
- logs start/end timestamps and elapsed runtime,
- exits 0 on success and 1 only on an unhandled fatal error.

No broker, adapter, or strategy imports are pulled in.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Standalone invocation (`python3 ops/update_setup_family_expectancy.py`)
# requires the repo root on sys.path so the `chad` package resolves. The
# systemd unit sets PYTHONPATH=/home/ubuntu/chad_finale and this insert
# is a no-op there.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.analytics.setup_family_expectancy_updater import (  # noqa: E402
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRADES_DIR,
    SetupFamilyExpectancyUpdater,
)


LOG = logging.getLogger("ops.update_setup_family_expectancy")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _resolve_path(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return default
    return Path(raw)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    trades_dir = _resolve_path("CHAD_TRADES_DIR", DEFAULT_TRADES_DIR)
    output_path = _resolve_path(
        "CHAD_SETUP_FAMILY_EXPECTANCY_OUTPUT", DEFAULT_OUTPUT_PATH
    )
    lookback_days = _resolve_int(
        "CHAD_SETUP_FAMILY_EXPECTANCY_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS
    )

    start_iso = _iso_now()
    LOG.info(
        "setup_family_expectancy_updater_start ts=%s trades_dir=%s "
        "output=%s lookback_days=%d",
        start_iso,
        trades_dir,
        output_path,
        lookback_days,
    )

    started = time.monotonic()
    try:
        updater = SetupFamilyExpectancyUpdater(
            trades_dir=trades_dir,
            output_path=output_path,
            lookback_days=lookback_days,
        )
        payload = updater.run()
    except Exception as exc:  # noqa: BLE001
        LOG.exception(
            "setup_family_expectancy_updater_fatal err=%s", exc
        )
        return 1

    elapsed = time.monotonic() - started
    end_iso = _iso_now()
    LOG.info(
        "setup_family_expectancy_updater_end ts=%s elapsed_seconds=%.3f "
        "families_processed=%d trades_processed=%d trades_skipped_corrupt=%d "
        "last_trade_ts=%s",
        end_iso,
        elapsed,
        payload["summary"]["families_processed"],
        payload["summary"]["trades_processed"],
        payload["summary"]["trades_skipped_corrupt"],
        payload["summary"]["last_trade_ts_utc"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
