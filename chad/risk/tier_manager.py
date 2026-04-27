#!/usr/bin/env python3
"""
TierManager — equity-tier strategy enable/disable.

Reads current portfolio equity and config/tiers.json, determines which
tier the account is in, and writes runtime/tier_state.json with the
list of enabled strategies for the current tier.

Hysteresis: a 5% buffer prevents flapping between tiers when equity
hovers near a threshold. Once promoted, equity must drop below
threshold * (1 - hysteresis) to demote.

INPUTS:
  - runtime/portfolio_snapshot.json
  - config/tiers.json
  - runtime/tier_state.json (previous state, for hysteresis)

OUTPUT:
  - runtime/tier_state.json
    {
      "tier_name": "MID",
      "current_equity_usd": float,
      "tier_min_equity": float,
      "tier_max_equity": float | null,
      "enabled_strategies": [...],
      "previous_tier": "SMALL",
      "promoted_at_utc": ISO-8601,
      "ts_utc": ISO-8601
    }
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

LOG = logging.getLogger("chad.risk.tier_manager")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_DIR = REPO_ROOT / "config"

SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
TIERS_CONFIG_PATH = CONFIG_DIR / "tiers.json"
OUT_PATH = RUNTIME_DIR / "tier_state.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("read_failed path=%s err=%s", path, exc)
        return {}


def _select_tier(
    equity: float, tiers: List[Dict[str, Any]], previous_tier: Optional[str], hysteresis_pct: float
) -> Dict[str, Any]:
    """
    Find the matching tier for the equity. Apply hysteresis to prevent
    flapping: a tier's lower bound is effectively raised by hysteresis
    when checking demotion from a higher tier.
    """
    sorted_tiers = sorted(tiers, key=lambda t: float(t.get("min_equity_usd", 0)))

    naive_match: Optional[Dict[str, Any]] = None
    for tier in sorted_tiers:
        lo = float(tier.get("min_equity_usd", 0))
        hi_raw = tier.get("max_equity_usd")
        hi = float(hi_raw) if hi_raw is not None else float("inf")
        if lo <= equity < hi:
            naive_match = tier
            break

    if naive_match is None:
        # Equity above all tiers — pick the top tier
        naive_match = sorted_tiers[-1]

    # Hysteresis: if previous tier is higher than naive match, only demote
    # when equity falls below previous tier's min by hysteresis margin.
    if previous_tier and previous_tier != naive_match["name"]:
        prev = next((t for t in sorted_tiers if t["name"] == previous_tier), None)
        if prev:
            prev_min = float(prev.get("min_equity_usd", 0))
            prev_idx = sorted_tiers.index(prev)
            naive_idx = sorted_tiers.index(naive_match)
            # Only resist demotion (prev higher than naive)
            if prev_idx > naive_idx:
                hysteresis_floor = prev_min * (1.0 - hysteresis_pct / 100.0)
                if equity >= hysteresis_floor:
                    LOG.info(
                        "tier_hysteresis_held previous=%s naive=%s equity=%.2f floor=%.2f",
                        previous_tier, naive_match["name"], equity, hysteresis_floor,
                    )
                    return prev

    return naive_match


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    snap = _read_json(SNAPSHOT_PATH)
    if not snap:
        LOG.error("portfolio_snapshot_missing")
        return 1
    equity = (
        float(snap.get("ibkr_equity", 0.0))
        + float(snap.get("kraken_equity", 0.0))
        + float(snap.get("coinbase_equity", 0.0))
    )

    config = _read_json(TIERS_CONFIG_PATH)
    if not config or "tiers" not in config:
        LOG.error("tiers_config_missing_or_invalid")
        return 1
    tiers = config["tiers"]
    hysteresis = float(config.get("hysteresis_pct", 5.0))

    previous_state = _read_json(OUT_PATH)
    previous_tier = previous_state.get("tier_name")

    selected = _select_tier(equity, tiers, previous_tier, hysteresis)

    promoted_at = previous_state.get("promoted_at_utc")
    if previous_tier != selected["name"]:
        promoted_at = _utc_now_iso()
        LOG.info(
            "tier_change previous=%s new=%s equity=%.2f",
            previous_tier, selected["name"], equity,
        )

    payload = {
        "schema_version": "tier_state.v1",
        "tier_name": selected["name"],
        "tier_description": selected.get("description", ""),
        "current_equity_usd": equity,
        "tier_min_equity": float(selected.get("min_equity_usd", 0)),
        "tier_max_equity": (
            float(selected["max_equity_usd"])
            if selected.get("max_equity_usd") is not None
            else None
        ),
        "enabled_strategies": list(selected.get("enabled_strategies", [])),
        "previous_tier": previous_tier,
        "promoted_at_utc": promoted_at,
        "ts_utc": _utc_now_iso(),
    }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "tier_state_published tier=%s equity=$%.2f strategies=%d",
        selected["name"], equity, len(payload["enabled_strategies"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
