from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# ── Baseline Fallback Mode Definition (SSOT v6.4) ─────────────────────
# "Conservative mode" / "baseline fallback" means:
#   - DEFAULT_STRATEGY_WEIGHTS applied as-is (normalized to 1.0)
#   - No savage overlay, no v3 overlay, no regime tilt, no Kelly constraint
#   - Pure proportional allocation based on the base chassis below
#   - daily_risk_fraction unchanged (not reduced)
# This mode activates when ANY of:
#   - Savage allocator is disabled or returns all-zero adjusted weights
#   - Allocator v3 is disabled or has insufficient return data
#   - Any overlay raises an exception (orchestrator fail-closed to base)
# The base weights are the absolute floor — overlays may adjust upward
# or downward, but on failure the system reverts to exactly these weights.
# Zero-weight strategies are never resurrected by any overlay.
# ──────────────────────────────────────────────────────────────────────
DEFAULT_STRATEGY_WEIGHTS: Dict[str, float] = {
    "alpha": 0.16,
    "beta_trend": 0.25,
    "gamma": 0.07,
    "gamma_reversion": 0.04,
    "alpha_futures": 0.09,
    "gamma_futures": 0.05,
    "alpha_options": 0.04,
    "omega_momentum_options": 0.03,
    "omega": 0.05,
    "omega_macro": 0.03,
    "omega_vol": 0.05,
    "delta": 0.02,
    "delta_pairs": 0.05,
    "alpha_crypto": 0.04,
    "alpha_intraday": 0.03,
}

# Governed config file path — written atomically by ActionApplier
STRATEGY_WEIGHTS_CONFIG = "config/strategy_weights.json"
STRATEGY_WEIGHTS_SCHEMA_VERSION = "strategy_weights.v1"


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
    source: str = "hardcoded_default"

    @classmethod
    def _load_from_config_file(cls, repo_root: Path) -> "StrategyAllocation | None":
        """
        Attempt to load weights from config/strategy_weights.json.

        Returns None on any failure (missing file, bad JSON, schema mismatch,
        invalid weights). Callers fall through to the hardcoded default.
        This is fail-open-to-next-tier, not fail-open-to-trading.
        """
        config_path = repo_root / STRATEGY_WEIGHTS_CONFIG
        if not config_path.is_file():
            return None

        try:
            raw = config_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "StrategyAllocation: config file unreadable, skipping: %s",
                exc,
            )
            return None

        if not isinstance(doc, dict):
            logger.warning("StrategyAllocation: config file is not a JSON object")
            return None

        schema = str(doc.get("schema_version") or "").strip()
        if schema != STRATEGY_WEIGHTS_SCHEMA_VERSION:
            logger.warning(
                "StrategyAllocation: config file schema %r != expected %r, skipping",
                schema,
                STRATEGY_WEIGHTS_SCHEMA_VERSION,
            )
            return None

        weights_raw = doc.get("weights")
        if not isinstance(weights_raw, dict) or not weights_raw:
            logger.warning("StrategyAllocation: config file ‘weights’ missing or empty")
            return None

        # Validate every entry is a non-negative float
        weights: Dict[str, float] = {}
        for key, val in weights_raw.items():
            if not isinstance(key, str) or not key.strip():
                logger.warning(
                    "StrategyAllocation: config file has non-string or empty key"
                )
                return None
            try:
                fval = float(val)
            except (TypeError, ValueError):
                logger.warning(
                    "StrategyAllocation: config file weight %r is not a number", key
                )
                return None
            if fval < 0.0:
                logger.warning(
                    "StrategyAllocation: config file weight %r is negative", key
                )
                return None
            weights[key.strip().lower()] = fval

        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            logger.warning("StrategyAllocation: config file weights sum to zero")
            return None

        logger.info(
            "StrategyAllocation: loaded %d weights from %s (sum=%.4f)",
            len(weights),
            config_path,
            weight_sum,
        )
        return cls(weights=weights, source=f"config_file:{config_path}")

    @classmethod
    def from_env_or_default(
        cls, repo_root: Path | None = None,
    ) -> "StrategyAllocation":
        """
        Resolution order (first wins):

        1. CHAD_STRATEGY_WEIGHTS env var   — operator override
        2. config/strategy_weights.json    — governed source (ActionApplier)
        3. DEFAULT_STRATEGY_WEIGHTS dict   — emergency fallback

        CHAD_STRATEGY_WEIGHTS env format:

            CHAD_STRATEGY_WEIGHTS="alpha=0.35,beta_trend=0.30,gamma=0.15,..."
        """
        # --- Tier 1: environment variable override ---
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
            return cls(weights=weights, source="env:CHAD_STRATEGY_WEIGHTS")

        # --- Tier 2: governed config file ---
        if repo_root is None:
            # Infer repo root from this file’s location:
            # chad/risk/dynamic_risk_allocator.py -> ../../
            repo_root = Path(__file__).resolve().parent.parent.parent

        config_alloc = cls._load_from_config_file(repo_root)
        if config_alloc is not None:
            return config_alloc

        # --- Tier 3: hardcoded emergency fallback ---
        logger.info(
            "StrategyAllocation: using hardcoded DEFAULT_STRATEGY_WEIGHTS fallback"
        )
        return cls(weights=dict(DEFAULT_STRATEGY_WEIGHTS), source="hardcoded_default")

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


# W6B-9 (P2-3): HALT_BOOST_SUPPRESSED log de-duplication.
#
# The halt clamp itself is not in question — a halted strategy whose
# winner_factor exceeds 1.0 is clamped to 1.0 and the fact is recorded
# structurally in applied_overlays["halt_clamp_applied"], which is what
# health_monitor_rules.py:591 and the exterminator contradiction check
# actually consume. The log line is pure narration, and it fired at WARNING
# once per halted strategy PER CYCLE — with several strategies halted for
# days, that is the dominant source of WARNING noise in the live journal and
# it trains operators to skim past warnings that matter.
#
# Dedupe on TRANSITION, not on first-sight: a clamp engaging and a clamp
# releasing are both worth one line. Today there is no release signal at all,
# so this is strictly more informative than the version it replaces while
# emitting ~2 lines per halt episode instead of ~thousands.
#
# Deliberately module-level, not instance state: DynamicRiskAllocator is a
# frozen dataclass documented as "deliberately stateless", and log dedupe is
# a logging concern rather than allocation state. Per-process, so a restart
# re-announces the current clamp set once — which is the desired behaviour.
_HALT_CLAMP_LOGGED: dict[str, bool] = {}


def _log_halt_clamp_transition(strategy: str, clamped: bool) -> None:
    """Emit one INFO line when a strategy's halt clamp engages or releases."""
    if _HALT_CLAMP_LOGGED.get(strategy) == clamped:
        return
    was_known = strategy in _HALT_CLAMP_LOGGED
    _HALT_CLAMP_LOGGED[strategy] = clamped
    if clamped:
        logger.info(
            "HALT_BOOST_SUPPRESSED strategy=%s winner_factor_clamped_to=1.0 "
            "transition=engaged (structural record: applied_overlays"
            "[%s][halt_clamp_applied]=true)",
            strategy,
            strategy,
        )
    elif was_known:
        logger.info(
            "HALT_BOOST_SUPPRESSED strategy=%s transition=released "
            "(strategy no longer halted, or winner_factor no longer >1.0)",
            strategy,
        )


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

    Execution-first promotion gate (SSOT v6.4):
    The allocator has no runtime gate checking P0 execution contract health
    (10s timeout, single IB session, reqContractDetails removal, attribution).
    Overlay aggressiveness (savage/v3) is controlled by env var enablement and
    data availability, not by execution subsystem health.  The P0-before-P1
    ordering is enforced by operational governance (CLAUDE.md work order), not
    by code.  This is a known architectural decision: the allocator is stateless
    and deliberately decoupled from execution health.
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
        Compute dollar caps per strategy. The cap chain is:

          base_weight (from config/strategy_weights.json)
            -> correlation_overlay (applied upstream by orchestrator)
            -> chassis_enforcement (50/30/20, applied upstream)
            -> tier_filter      (zero out strategies not in current tier)
            -> winner_scaling   (per-strategy performance multiplier)
            -> regime_booster   (global multiplier on top)
            -> SCR sizing_factor (applied separately at execution)

        The tier/winner/regime overlays are read from runtime/. If any
        file is missing or stale (>10 minutes) the corresponding overlay
        falls back to neutral.
        """
        details = self._compute_caps_with_overlays(snapshot=snapshot)
        return details["caps"]

    def _compute_caps_with_overlays(
        self, *, snapshot: PortfolioSnapshot,
    ) -> Dict[str, object]:
        """
        Internal — returns caps plus the overlay context so build_payload
        can surface the multipliers without recomputing.
        """
        total_equity = snapshot.total_equity
        if total_equity < 0.0:
            raise ValueError("total_equity must be >= 0")

        norm = self.strategy_allocation.normalized()
        portfolio_risk_cap = total_equity * self.daily_risk_fraction

        # --- PROFIT LOCK (system-level sizing kill switch) ---
        try:
            profit_lock_path = (
                Path(__file__).resolve().parents[2]
                / "runtime"
                / "profit_lock_state.json"
            )
            if profit_lock_path.is_file():
                with profit_lock_path.open("r", encoding="utf-8") as f:
                    pl = json.load(f)
                sizing_factor = float(pl.get("sizing_factor", 1.0))
                stop_new_entries = bool(pl.get("stop_new_entries", False))
                portfolio_risk_cap *= max(0.0, min(1.0, sizing_factor))
                if stop_new_entries:
                    portfolio_risk_cap = 0.0
        except Exception:
            pass  # Fail-safe: never break allocator

        # --- BUSINESS OVERLAYS (tier / winner / regime) ---
        tier_set = load_tier_filter()
        winner_mults, winner_stale = load_winner_multipliers_or_stale()
        # CB05: when winner scaling is stale, fall back to a conservative
        # 0.5x multiplier per strategy rather than silently leaving sizing
        # at 1.0x — stale-fail-open was the audit finding.
        if winner_stale:
            logger.warning(
                "winner_scaling_stale — using conservative 0.5x multipliers"
            )
            winner_mults = {name.lower(): 0.5 for name in norm.keys()}
        regime_mult = load_regime_booster_multiplier()
        # Clamp regime multiplier to the documented 1.0..1.5 band.
        regime_mult = max(1.0, min(1.5, float(regime_mult)))

        # Halt-aware boost suppression: a strategy halted by edge decay
        # monitor must never receive an aggressive winner_factor > 1.0,
        # even if winner_scaling.json still publishes a stale boost. The
        # halt itself is preserved (suppression happens here on the read
        # path only — winner_scaling.json and strategy_allocations.json
        # are not mutated).
        halted_lookup = _load_halted_strategy_set()

        caps: Dict[str, float] = {}
        applied_overlays: Dict[str, Dict[str, float]] = {}
        for name, frac in norm.items():
            base_cap = portfolio_risk_cap * frac
            tier_factor = 1.0
            if tier_set is not None and name.lower() not in tier_set:
                tier_factor = 0.0
            # CB05: when stale, every strategy gets 0.5x (default in lookup).
            winner_default = 0.5 if winner_stale else 1.0
            winner_factor = float(winner_mults.get(name.lower(), winner_default))
            halt_clamp_applied = False
            if name.lower() in halted_lookup and winner_factor > 1.0:
                winner_factor = 1.0
                halt_clamp_applied = True
            cap = base_cap * tier_factor * winner_factor * regime_mult
            caps[name] = cap
            applied_overlays[name] = {
                "base_cap": base_cap,
                "tier_factor": tier_factor,
                "winner_factor": winner_factor,
                "regime_factor": regime_mult,
                "final_cap": cap,
                "halt_clamp_applied": halt_clamp_applied,
            }
            _log_halt_clamp_transition(name.lower(), halt_clamp_applied)

        logger.info(
            "DynamicRiskAllocator: total_equity=%.2f daily_risk_fraction=%.3f "
            "portfolio_risk_cap=%.2f tier_filter=%s winners=%d regime_mult=%.3f",
            total_equity,
            self.daily_risk_fraction,
            portfolio_risk_cap,
            "active" if tier_set is not None else "neutral",
            len(winner_mults),
            regime_mult,
        )
        return {
            "caps": caps,
            "portfolio_risk_cap_after_profit_lock": portfolio_risk_cap,
            "tier_filter_active": tier_set is not None,
            "tier_enabled_strategies": sorted(tier_set) if tier_set else [],
            "winner_multipliers_applied": winner_mults,
            "regime_booster_multiplier": regime_mult,
            "per_strategy_overlay": applied_overlays,
        }

    def build_payload(self, *, snapshot: PortfolioSnapshot) -> Dict[str, object]:
        """
        Build the JSON payload we’ll write to dynamic_caps.json.
        """
        total_equity = snapshot.total_equity
        raw = dict(self.strategy_allocation.weights)
        norm = self.strategy_allocation.normalized()
        details = self._compute_caps_with_overlays(snapshot=snapshot)
        caps = details["caps"]
        per_overlay = details["per_strategy_overlay"]

        portfolio_risk_cap = total_equity * self.daily_risk_fraction
        sum_raw = self.strategy_allocation.raw_sum()
        sum_norm = sum(norm.values())

        per_strategy: Dict[str, Dict[str, float]] = {}
        for name in sorted(norm.keys()):
            ov = per_overlay.get(name, {})
            per_strategy[name] = {
                "raw_weight": float(raw[name]),
                "normalized_weight": float(norm[name]),
                "fraction_of_total_equity": float(norm[name] * self.daily_risk_fraction),
                "dollar_cap": float(caps[name]),
                "tier_factor": float(ov.get("tier_factor", 1.0)),
                "winner_factor": float(ov.get("winner_factor", 1.0)),
                "regime_factor": float(ov.get("regime_factor", 1.0)),
                "base_cap_pre_overlay": float(ov.get("base_cap", 0.0)),
                "halt_clamp_applied": bool(ov.get("halt_clamp_applied", False)),
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
            "business_overlays": {
                "tier_filter_active": bool(details["tier_filter_active"]),
                "tier_enabled_strategies": list(details["tier_enabled_strategies"]),
                "winner_multipliers": dict(details["winner_multipliers_applied"]),
                "regime_booster_multiplier": float(details["regime_booster_multiplier"]),
            },
        }


# ---------------------------------------------------------------------------
# 50/30/20 Chassis Enforcement
# ---------------------------------------------------------------------------

ALPHA_STRATEGIES = frozenset({
    "alpha", "alpha_futures", "alpha_intraday", "alpha_options",
    "omega_momentum_options", "gamma", "gamma_futures", "gamma_reversion",
})
BETA_STRATEGIES = frozenset({"beta", "beta_trend"})
ADAPTIVE_STRATEGIES = frozenset({
    "omega", "omega_macro", "omega_vol", "delta", "delta_pairs", "alpha_crypto",
})

ALPHA_TARGET = 0.50
BETA_TARGET = 0.30
ADAPTIVE_TARGET = 0.20
CHASSIS_TOLERANCE = 0.05  # allow 5% drift before hard clamp


def _chassis_enabled() -> bool:
    raw = os.environ.get("CHAD_CHASSIS_ENFORCEMENT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def enforce_chassis(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Enforce the 50/30/20 sleeve allocation:
      - ALPHA sleeve (50%): alpha, alpha_futures, alpha_options, gamma, gamma_futures, gamma_reversion
      - BETA sleeve (30%): beta_trend — structural savings sleeve, protected
      - ADAPTIVE sleeve (20%): omega, omega_macro, omega_vol, delta, crypto

    Only fires when sleeve drift exceeds CHASSIS_TOLERANCE.
    Preserves intra-sleeve proportions. Returns weights summing to ~1.0.
    Disabled via CHAD_CHASSIS_ENFORCEMENT=0.
    """
    if not _chassis_enabled():
        return dict(weights)

    result = dict(weights)
    total = sum(result.values())
    if total <= 0.0:
        return result

    # Normalize to 1.0 for sleeve math
    for k in result:
        result[k] /= total

    alpha_total = sum(result.get(s, 0.0) for s in ALPHA_STRATEGIES)
    beta_total = sum(result.get(s, 0.0) for s in BETA_STRATEGIES)
    adaptive_total = sum(result.get(s, 0.0) for s in ADAPTIVE_STRATEGIES)

    needs_enforcement = (
        abs(alpha_total - ALPHA_TARGET) > CHASSIS_TOLERANCE
        or abs(beta_total - BETA_TARGET) > CHASSIS_TOLERANCE
        or abs(adaptive_total - ADAPTIVE_TARGET) > CHASSIS_TOLERANCE
    )

    if not needs_enforcement:
        return result

    logger.info(
        "chassis_enforcement: BEFORE alpha=%.3f beta=%.3f adaptive=%.3f",
        alpha_total, beta_total, adaptive_total,
    )

    # Scale each sleeve to its target, preserving intra-sleeve proportions
    def _scale_sleeve(strategies: frozenset, target: float, current: float) -> None:
        if current > 0.0:
            scale = target / current
            for s in strategies:
                if s in result:
                    result[s] *= scale
        elif target > 0.0:
            # Sleeve has zero weight but should have some — distribute equally
            members = [s for s in strategies if s in result]
            if members:
                each = target / len(members)
                for s in members:
                    result[s] = each

    _scale_sleeve(ALPHA_STRATEGIES, ALPHA_TARGET, alpha_total)
    _scale_sleeve(BETA_STRATEGIES, BETA_TARGET, beta_total)
    _scale_sleeve(ADAPTIVE_STRATEGIES, ADAPTIVE_TARGET, adaptive_total)

    # Any unknown strategies (not in any sleeve) get zeroed
    known = ALPHA_STRATEGIES | BETA_STRATEGIES | ADAPTIVE_STRATEGIES
    for k in list(result.keys()):
        if k not in known:
            result[k] = 0.0

    logger.info(
        "chassis_enforcement: AFTER alpha=%.3f beta=%.3f adaptive=%.3f",
        sum(result.get(s, 0.0) for s in ALPHA_STRATEGIES),
        sum(result.get(s, 0.0) for s in BETA_STRATEGIES),
        sum(result.get(s, 0.0) for s in ADAPTIVE_STRATEGIES),
    )

    return result


# ---------------------------------------------------------------------------
# Paths / helpers
# ---------------------------------------------------------------------------

DYNAMIC_CAPS_TTL_SECONDS = 300  # 5 minutes (caps should be refreshed frequently)

# Business-framework runtime files consumed by the cap chain. If any file
# is missing or older than BUSINESS_OVERLAY_STALE_SECONDS, that overlay
# falls back to neutral (no filtering, multiplier 1.0).
BUSINESS_OVERLAY_STALE_SECONDS = 600  # 10 minutes


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read_business_runtime(path: Path, stale_seconds: int = 600) -> Dict[str, object] | None:
    """
    Read a business-overlay runtime JSON file. Returns None if the file
    is missing, unreadable, or its ts_utc is older than the stale window.
    """
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("business_overlay_unreadable path=%s err=%s", path, exc)
        return None
    if not isinstance(doc, dict):
        return None
    ts_raw = str(doc.get("ts_utc") or "").strip()
    if not ts_raw:
        # No timestamp: accept the doc but log.
        logger.info("business_overlay_no_ts path=%s — accepting", path)
        return doc
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except Exception:
        return doc
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    if age > stale_seconds:
        logger.warning(
            "business_overlay_stale path=%s age_s=%.0f threshold=%d",
            path, age, stale_seconds,
        )
        return None
    return doc


def _runtime_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime"


def load_tier_filter() -> set[str] | None:
    """
    Return the set of enabled strategy names per the current tier, or
    None to disable filtering (file missing or stale).
    """
    doc = _read_business_runtime(_runtime_dir() / "tier_state.json", stale_seconds=600)
    if not doc:
        return None
    enabled = doc.get("enabled_strategies")
    if not isinstance(enabled, list) or not enabled:
        return None
    return {str(s).strip().lower() for s in enabled if isinstance(s, str)}


def load_winner_multipliers() -> Dict[str, float]:
    """Per-strategy multipliers from runtime/winner_scaling.json (empty if stale)."""
    doc = _read_business_runtime(_runtime_dir() / "winner_scaling.json", stale_seconds=1800)
    if not doc:
        return {}
    raw = doc.get("multipliers")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        try:
            out[k.strip().lower()] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_winner_multipliers_or_stale() -> Tuple[Dict[str, float], bool]:
    """
    CB05: returns (multipliers, stale).

    Differs from load_winner_multipliers() only by exposing whether the
    underlying file was missing/stale, so callers can fall back to a
    conservative default (e.g. 0.5x) instead of silently no-op'ing to 1.0.
    """
    doc = _read_business_runtime(_runtime_dir() / "winner_scaling.json", stale_seconds=1800)
    if not doc:
        return {}, True
    raw = doc.get("multipliers")
    if not isinstance(raw, dict):
        return {}, True
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        try:
            out[k.strip().lower()] = float(v)
        except (TypeError, ValueError):
            continue
    return out, False


def load_regime_booster_multiplier() -> float:
    """Global regime boost. Returns 1.0 when missing/stale."""
    doc = _read_business_runtime(_runtime_dir() / "regime_booster.json", stale_seconds=120)
    if not doc:
        # CB05: stale booster already fails conservative (1.0x = no boost),
        # but log so operators see the missing input.
        logger.warning("regime_booster_stale — no boost applied (1.0x)")
        return 1.0
    try:
        return float(doc.get("multiplier", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _load_halted_strategy_set() -> set[str]:
    """Read runtime/strategy_allocations.json and return the lower-case
    set of strategies currently flagged halted. Fail-safe: any error
    returns an empty set so the allocator continues to compute caps —
    halt enforcement is layered (live_loop also drops signals from
    halted strategies), so a transient read failure must not break
    risk allocation.
    """
    halted: set[str] = set()
    try:
        path = _runtime_dir() / "strategy_allocations.json"
        if not path.is_file():
            return halted
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return halted
        allocations = doc.get("allocations", {})
        if not isinstance(allocations, dict):
            return halted
        for name, entry in allocations.items():
            if isinstance(entry, dict) and entry.get("halted") is True:
                key = str(name).strip().lower()
                if key:
                    halted.add(key)
    except Exception as exc:  # noqa: BLE001 — fail-safe by design
        logger.warning("halted_strategy_set_unreadable err=%s", exc)
    return halted



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

