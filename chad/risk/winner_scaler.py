#!/usr/bin/env python3
"""
WinnerScaler — per-strategy performance multipliers from expectancy.

Reads runtime/expectancy_state.json and computes a small bounded
multiplier for each strategy. Strong recent expectancy gets a boost,
negative expectancy gets a haircut. Bounded so even the best strategy
can't dominate the book — and broker_sync (which is reconciliation
plumbing, not a strategy) is excluded entirely.

INPUTS:
  - runtime/expectancy_state.json
  - config/winner_scaling_policy.json

OUTPUT:
  - runtime/winner_scaling.json
    {
      "schema_version": "winner_scaling.v1",
      "multipliers": {<strategy>: <float>, ...},
      "max_multiplier": 1.50,
      "min_multiplier": 0.50,
      "median_expectancy": float,
      "n_strategies_scaled": int,
      "n_strategies_neutral": int,
      "ts_utc": ISO-8601
    }

LOGIC:
  - Take strategies with >= min_trades_for_scaling trades (default 5).
  - Compute median expectancy across that scoring pool (excluded
    strategies are never in the pool).
  - For each strategy: multiplier = clamp(expectancy / |median|, min, max).
  - Strategies with < min_trades get multiplier = 1.0 (neutral).
  - Excluded strategies always get 1.0.
  - When |median_expectancy| ~ 0 we degrade to all-1.0 multipliers
    (cannot rank yet).

PHASE 11C per SSOT v8.2 roadmap.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

LOG = logging.getLogger("chad.risk.winner_scaler")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_DIR = REPO_ROOT / "config"

EXPECTANCY_PATH = RUNTIME_DIR / "expectancy_state.json"
POLICY_PATH = CONFIG_DIR / "winner_scaling_policy.json"
OUT_PATH = RUNTIME_DIR / "winner_scaling.json"

DEFAULT_POLICY: Dict[str, Any] = {
    "schema_version": "winner_scaling_policy.v1",
    "min_trades_for_scaling": 5,
    "max_multiplier": 1.50,
    "min_multiplier": 0.50,
    "exclude_strategies": ["broker_sync", "manual", "paper_exec", "unknown"],
}


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


def _load_policy() -> Dict[str, Any]:
    on_disk = _read_json(POLICY_PATH)
    if not on_disk:
        return dict(DEFAULT_POLICY)
    merged = dict(DEFAULT_POLICY)
    merged.update(on_disk)
    return merged


def _is_reconciliation_artifact(name: str) -> bool:
    """True for reconciliation/closeout-strategy labels that must never
    appear in winner_scaling.json (they are bookkeeping rows, not real
    trading strategies). Keeps the writer side authoritative so the
    health monitor doesn't have to keep clearing them downstream.
    """
    nlow = str(name or "").strip().lower()
    if not nlow:
        return False
    if nlow.startswith("reconciled_phase"):
        return True
    if nlow.startswith("reconciled_"):
        return True
    return False


def compute_multipliers(
    expectancy_doc: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Pure function: produce the winner_scaling payload."""
    min_trades = int(policy["min_trades_for_scaling"])
    max_mult = float(policy["max_multiplier"])
    min_mult = float(policy["min_multiplier"])
    excluded = {s.lower() for s in policy.get("exclude_strategies", [])}

    strategies = expectancy_doc.get("strategies", {}) or {}

    # Build scoring pool: enough trades and not excluded.
    pool: List[tuple[str, float]] = []
    for name, info in strategies.items():
        if not isinstance(info, dict):
            continue
        nlow = name.lower()
        if nlow in excluded:
            continue
        if _is_reconciliation_artifact(name):
            # Reconciliation artifacts (RECONCILED_PHASE2_* etc.) are
            # closeout bookkeeping, not strategies — they must not
            # influence the median-expectancy denominator.
            continue
        try:
            trades = int(info.get("total_trades", 0))
            exp = float(info.get("expectancy", 0.0))
        except (TypeError, ValueError):
            continue
        if trades >= min_trades:
            pool.append((name, exp))

    multipliers: Dict[str, float] = {}
    median_abs = 0.0
    if pool:
        expectancies = [e for _, e in pool]
        median_abs = abs(statistics.median(expectancies))

    use_ratio = median_abs > 1e-6

    n_scaled = 0
    n_neutral = 0
    for name, info in strategies.items():
        # Drop reconciliation artifacts from the published map entirely so
        # downstream consumers (allocator, health monitor) never see them.
        if _is_reconciliation_artifact(name):
            continue
        nlow = name.lower()
        if nlow in excluded:
            multipliers[name] = 1.0
            n_neutral += 1
            continue
        if not isinstance(info, dict):
            multipliers[name] = 1.0
            n_neutral += 1
            continue
        try:
            trades = int(info.get("total_trades", 0))
            exp = float(info.get("expectancy", 0.0))
        except (TypeError, ValueError):
            multipliers[name] = 1.0
            n_neutral += 1
            continue
        if trades < min_trades or not use_ratio:
            multipliers[name] = 1.0
            n_neutral += 1
            continue
        ratio = exp / median_abs
        # clamp
        scaled = max(min_mult, min(max_mult, ratio))
        multipliers[name] = round(scaled, 3)
        n_scaled += 1

    # Ensure all 16 canonical strategies appear explicitly in the output.
    # Strategies not in the scoring pool (too few trades or excluded) get
    # an explicit 1.0 rather than being absent — makes the contract between
    # winner_scaler and dynamic_risk_allocator explicit not implicit.
    _CANONICAL_STRATEGIES = {
        "alpha", "alpha_intraday", "alpha_crypto", "alpha_options",
        "alpha_futures", "beta", "beta_trend", "gamma", "gamma_futures",
        "gamma_reversion", "delta", "delta_pairs", "omega", "omega_vol",
        "omega_macro", "omega_momentum_options",
    }
    for _s in _CANONICAL_STRATEGIES:
        if _s not in multipliers:
            multipliers[_s] = 1.0

    return {
        "schema_version": "winner_scaling.v1",
        "multipliers": multipliers,
        "max_multiplier": max_mult,
        "min_multiplier": min_mult,
        "median_expectancy": round(median_abs, 4),
        "n_strategies_scaled": n_scaled,
        "n_strategies_neutral": n_neutral,
        "min_trades_for_scaling": min_trades,
        "excluded_strategies": sorted(excluded),
        "ts_utc": _utc_now_iso(),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    expectancy = _read_json(EXPECTANCY_PATH)
    if not expectancy:
        LOG.warning("expectancy_state_missing — writing neutral multipliers")
        expectancy = {"strategies": {}}

    policy = _load_policy()

    result = compute_multipliers(expectancy, policy)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "winner_scaling_published scaled=%d neutral=%d median=%.4f sample=%s",
        result["n_strategies_scaled"], result["n_strategies_neutral"],
        result["median_expectancy"],
        {k: v for k, v in list(result["multipliers"].items())[:5]},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
