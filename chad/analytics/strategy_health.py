"""
chad/analytics/strategy_health.py

Phase-8 Session 6 (F3): composite strategy health score.

A single float in [0, 1] per strategy, written to
``runtime/strategy_health.json`` each cycle. The score fuses four
measurement surfaces that each capture a different failure mode:

    health_score = 0.40 * normalized_sharpe     # profitability
                 + 0.30 * win_rate              # consistency
                 + 0.20 * (1 - slippage_ratio)  # execution quality
                 + 0.10 * regime_alignment      # regime discipline

Clamped to [0, 1] so downstream consumers can feed it to multiplicative
allocators without a sign flip.

Inputs
------

``expectancy_tracker`` is passed as either the module, a precomputed
state dict, or an object exposing a ``.compute()`` / ``.strategies`` /
``.state`` accessor. The scorer tolerates all three so live-loop and
tests can pass whichever shape is most convenient.

``slippage_tracker`` exposes ``get_rolling_stats(strategy=..., ...)``.
Only the ``mean`` field is consulted — a non-zero positive mean means
fills are coming in worse than expected and docks the score.

``regime_state`` is the current classifier label from regime_classifier
(trending_bull / trending_bear / ranging / volatile / unknown). The
per-strategy favorable-regime table in this module converts the label
into a 0/1 alignment contribution. When no mapping exists for a
strategy we default to 0.5 (neutral) so unknown strategies are neither
rewarded nor punished.

Cold-start
----------

If expectancy has no data for a strategy the constraint in the task
says to return 0.5 — explicitly neutral, not penalised. The scorer
honours that contract via the ``DEFAULT_NEUTRAL_SCORE`` constant.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
HEALTH_PATH = ROOT / "runtime" / "strategy_health.json"

SCHEMA_VERSION = "strategy_health.v1"

# Composition weights. Sum must equal 1.0 so clamp([0,1]) is tight.
WEIGHT_SHARPE: float = 0.40
WEIGHT_WIN_RATE: float = 0.30
WEIGHT_SLIPPAGE: float = 0.20
WEIGHT_REGIME: float = 0.10

DEFAULT_NEUTRAL_SCORE: float = 0.5

# Per-strategy favorable-regime table. The keys overlap with
# execution_pipeline._STRATEGY_SIGNAL_FAMILY so the same names used for
# vote aggregation are reused here. A strategy missing from the table
# contributes DEFAULT_NEUTRAL_SCORE for its regime_alignment term.
#
# Rough taxonomy:
#   * trend-following strategies prefer trending_bull / trending_bear
#   * mean-reversion prefers ranging
#   * volatility strategies prefer volatile
#   * macro / options are less regime-bound; treat as neutral
_FAVORABLE_REGIMES: Dict[str, frozenset] = {
    "alpha": frozenset({"trending_bull", "trending_bear"}),
    "alpha_crypto": frozenset({"trending_bull", "trending_bear"}),
    "alpha_forex": frozenset({"trending_bull", "trending_bear"}),
    "alpha_futures": frozenset({"trending_bull", "trending_bear"}),
    "alpha_intraday": frozenset({"trending_bull", "trending_bear"}),
    "alpha_options": frozenset({"trending_bull", "trending_bear", "volatile"}),
    "beta": frozenset({"trending_bull", "trending_bear"}),
    "delta": frozenset({"trending_bull", "trending_bear"}),
    "delta_pairs": frozenset({"ranging"}),
    "gamma": frozenset({"volatile"}),
    "gamma_futures": frozenset({"trending_bull", "trending_bear"}),
    "gamma_reversion": frozenset({"ranging"}),
    "omega": frozenset({"volatile"}),
    "omega_macro": frozenset({"trending_bull", "trending_bear", "volatile", "ranging"}),
    "omega_momentum_options": frozenset({"trending_bull", "trending_bear"}),
    "omega_vol": frozenset({"volatile"}),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _extract_strategy_dict(expectancy_tracker: Any) -> Dict[str, Any]:
    """Normalise several expectancy-tracker shapes to a plain dict."""
    if expectancy_tracker is None:
        return {}
    # Module exposing compute()
    compute = getattr(expectancy_tracker, "compute", None)
    if callable(compute):
        try:
            state = compute()
            if isinstance(state, dict):
                strategies = state.get("strategies")
                if isinstance(strategies, dict):
                    return strategies
        except Exception:  # noqa: BLE001
            pass
    # A precomputed dict returned by compute()
    if isinstance(expectancy_tracker, dict):
        if "strategies" in expectancy_tracker and isinstance(expectancy_tracker["strategies"], dict):
            return expectancy_tracker["strategies"]
        return expectancy_tracker
    # Attribute-shaped: tracker.strategies
    strategies = getattr(expectancy_tracker, "strategies", None)
    if isinstance(strategies, dict):
        return strategies
    return {}


def _get_slippage_stats(
    slippage_tracker: Any, strategy: str
) -> Dict[str, Any]:
    if slippage_tracker is None:
        return {"n": 0, "mean": None}
    getter = getattr(slippage_tracker, "get_rolling_stats", None)
    if not callable(getter):
        return {"n": 0, "mean": None}
    try:
        stats = getter(strategy=strategy)
    except Exception:  # noqa: BLE001
        return {"n": 0, "mean": None}
    if not isinstance(stats, dict):
        return {"n": 0, "mean": None}
    return stats


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class StrategyHealthScorer:
    """Composite health score per strategy with atomic JSON persistence."""

    def __init__(self, output_path: Path = HEALTH_PATH) -> None:
        self._output_path = Path(output_path)

    # ------------------------------------------------------------------
    # individual sub-scores
    # ------------------------------------------------------------------

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Map sharpe to [0, 1].

        sharpe =  2.0 → 1.0
        sharpe =  0.0 → 0.5
        sharpe = -2.0 → 0.0

        Values outside ±2.0 clamp at the ends. This matches the task spec.
        """
        return _clamp01((sharpe + 2.0) / 4.0)

    def _slippage_ratio(
        self, mean_slippage: Optional[float], expected_pnl: Optional[float]
    ) -> float:
        """Fraction of edge consumed by slippage, clamped to [0, 1].

        mean_slippage is per-share in the slippage-tracker's side-adjusted
        convention (positive = adverse). If expected_pnl is zero or
        negative we cannot form a ratio, so we fall back to a neutral
        0.5 so the scorer does not penalise strategies for missing data.
        """
        slip = _safe_float(mean_slippage, 0.0)
        edge = _safe_float(expected_pnl, 0.0)
        if edge <= 0.0:
            return 0.5
        if slip <= 0.0:
            return 0.0  # negative slippage = favourable, clamp to 0 consumption
        return _clamp01(slip / edge)

    def _regime_alignment(self, strategy: str, regime_state: str) -> float:
        """Return 1.0 if the strategy favours the current regime, else 0.0.

        Unknown strategy or missing regime → DEFAULT_NEUTRAL_SCORE so the
        composite score is not skewed by table coverage gaps.
        """
        regime = str(regime_state or "unknown").lower()
        favorable = _FAVORABLE_REGIMES.get(str(strategy).lower())
        if favorable is None or regime in ("", "unknown"):
            return DEFAULT_NEUTRAL_SCORE
        return 1.0 if regime in favorable else 0.0

    # ------------------------------------------------------------------
    # single-strategy composite
    # ------------------------------------------------------------------

    def compute(
        self,
        strategy: str,
        expectancy_tracker: Any,
        slippage_tracker: Any,
        regime_state: str = "unknown",
    ) -> Dict[str, Any]:
        """Return the health record for one strategy."""
        strategies = _extract_strategy_dict(expectancy_tracker)
        entry = strategies.get(strategy) if isinstance(strategies, dict) else None

        # Contract: no data → neutral 0.5, not penalised.
        if not isinstance(entry, dict) or int(entry.get("total_trades") or 0) <= 0:
            return {
                "strategy": str(strategy),
                "health_score": DEFAULT_NEUTRAL_SCORE,
                "normalized_sharpe": DEFAULT_NEUTRAL_SCORE,
                "win_rate": 0.0,
                "slippage_ratio": 0.5,
                "regime_alignment": self._regime_alignment(strategy, regime_state),
                "sample_count": 0,
                "reason": "no_expectancy_data",
            }

        win_rate = _clamp01(_safe_float(entry.get("win_rate"), 0.0))
        expectancy = _safe_float(entry.get("expectancy"), 0.0)

        # Sharpe proxy: expectancy-tracker does not surface rolling_sharpe
        # directly, so we use a simple transformation of expectancy per trade
        # scaled by win_rate as a resilient stand-in. Real Sharpe from a
        # rolling-window engine can be substituted later without changing
        # this contract.
        sharpe_proxy = _safe_float(entry.get("rolling_sharpe"), None)
        if sharpe_proxy is None:
            # Best-effort derivation when rolling_sharpe is not present:
            # treat expectancy * 10 as a sharpe-like ratio. Bounded by the
            # normalization map (sharpe=2.0 → 1.0).
            sharpe_proxy = max(-2.0, min(2.0, expectancy * 10.0))
        normalized_sharpe = self._normalize_sharpe(sharpe_proxy)

        slippage_stats = _get_slippage_stats(slippage_tracker, strategy)
        slip_mean = slippage_stats.get("mean") if isinstance(slippage_stats, dict) else None
        avg_win = _safe_float(entry.get("avg_win"), 0.0)
        slippage_ratio = self._slippage_ratio(slip_mean, avg_win)
        execution_quality = _clamp01(1.0 - slippage_ratio)

        regime_alignment = self._regime_alignment(strategy, regime_state)

        score = (
            WEIGHT_SHARPE * normalized_sharpe
            + WEIGHT_WIN_RATE * win_rate
            + WEIGHT_SLIPPAGE * execution_quality
            + WEIGHT_REGIME * regime_alignment
        )
        health_score = _clamp01(score)

        return {
            "strategy": str(strategy),
            "health_score": round(health_score, 4),
            "normalized_sharpe": round(normalized_sharpe, 4),
            "win_rate": round(win_rate, 4),
            "slippage_ratio": round(slippage_ratio, 4),
            "regime_alignment": round(regime_alignment, 4),
            "sample_count": int(entry.get("total_trades") or 0),
            "reason": "computed",
        }

    # ------------------------------------------------------------------
    # batch + persistence
    # ------------------------------------------------------------------

    def compute_all(
        self,
        strategy_names: Iterable[str],
        expectancy_tracker: Any,
        slippage_tracker: Any,
        regime_state: str = "unknown",
    ) -> Dict[str, Dict[str, Any]]:
        """Compute every named strategy and write the JSON snapshot.

        Returns the ``{strategy: record}`` dict that was written.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name in strategy_names:
            results[str(name)] = self.compute(
                strategy=str(name),
                expectancy_tracker=expectancy_tracker,
                slippage_tracker=slippage_tracker,
                regime_state=regime_state,
            )
        self._write_json(results, regime_state=regime_state)
        return results

    def _write_json(
        self,
        results: Mapping[str, Mapping[str, Any]],
        regime_state: str = "unknown",
    ) -> None:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ts_utc": _utc_now_iso(),
            "regime_state": str(regime_state),
            "weights": {
                "sharpe": WEIGHT_SHARPE,
                "win_rate": WEIGHT_WIN_RATE,
                "slippage": WEIGHT_SLIPPAGE,
                "regime": WEIGHT_REGIME,
            },
            "strategies": {k: dict(v) for k, v in results.items()},
        }
        _write_atomic(self._output_path, payload)


def read_health(path: Path = HEALTH_PATH) -> Dict[str, Any]:
    """Return the on-disk health snapshot or a safe default if missing."""
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "ts_utc": "", "strategies": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "ts_utc": "", "strategies": {}}
    if not isinstance(data, dict):
        return {"schema_version": SCHEMA_VERSION, "ts_utc": "", "strategies": {}}
    return data


__all__ = [
    "DEFAULT_NEUTRAL_SCORE",
    "HEALTH_PATH",
    "SCHEMA_VERSION",
    "WEIGHT_REGIME",
    "WEIGHT_SHARPE",
    "WEIGHT_SLIPPAGE",
    "WEIGHT_WIN_RATE",
    "StrategyHealthScorer",
    "read_health",
]
