#!/usr/bin/env python3
"""
TierManager — equity-tier strategy enable/disable (v9.1 ladder).

Reads current portfolio equity and config/tiers.json, determines which
tier the account is in, and writes runtime/tier_state.json with the
list of enabled strategies, the active risk profile, and any pending
demotion state.

V9.1 ladder: MICRO < STARTER < PRO_GROWTH < SCALE.  (Renamed PRO -> SCALE
in v9.1; the legacy top-tier name "PRO" is migrated transparently on
read.  See `_normalize_legacy_name`.)

Promotion gate: equity >= tier.min_equity_usd.
Demotion gate: equity <  current_tier.demotion_equity_usd (fallback:
              current_tier.min_equity_usd * 0.90).  MICRO never demotes.

Mid-session deferral: a demotion proposed while the market is open is
held until the next market close; the demotion_pending block in the
output describes the pending action so the operator can see it.

INPUTS:
  - runtime/portfolio_snapshot.json
  - config/tiers.json
  - runtime/tier_state.json (previous state, for hysteresis)

OUTPUT (runtime/tier_state.json):
  {
    "schema_version":             "tier_state.v2",
    "tier_name":                  "SCALE",
    "tier_description":           ...,
    "current_equity_usd":         float,
    "tier_min_equity":            float,
    "tier_max_equity":            float | null,
    "enabled_strategies":         [...],
    "allowed_instruments":        [...],
    "risk_profile":               {...},
    "previous_tier":              "PRO_GROWTH" | null,
    "promoted_at_utc":            ISO-8601 | null,
    "demotion_pending":           bool,
    "demotion_pending_to":        str | null,
    "demotion_pending_reason":    str | null,
    "demotion_pending_since_utc": ISO-8601 | null,
    "demotion_applies_at":        "next_market_open" | null,
    "ts_utc":                     ISO-8601
  }
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("chad.risk.tier_manager")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_DIR = REPO_ROOT / "config"

SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
TIERS_CONFIG_PATH = CONFIG_DIR / "tiers.json"
OUT_PATH = RUNTIME_DIR / "tier_state.json"


# v9.1 ladder ordering — higher rank = larger account.
TIER_RANK: Dict[str, int] = {
    "MICRO": 0,
    "STARTER": 1,
    "PRO_GROWTH": 2,
    "SCALE": 3,
}

# Legacy tier-name aliases.  Renamed PRO -> SCALE in v9.1.
_LEGACY_TIER_ALIASES: Dict[str, str] = {
    "PRO": "SCALE",
    "INSTITUTIONAL": "SCALE",
    "MID": "PRO_GROWTH",
    "SMALL": "STARTER",
}

# Canonical strategy universe used to expand the SCALE wildcard
# (`enabled_strategies: ["*"]`) at publish time.  Downstream consumers
# (e.g. dynamic_risk_allocator.load_tier_filter) match names against this
# set, so emitting the literal "*" would filter out every strategy.
# Kept in alphabetical order; matches the v9.1 16-strategy registry.
_CANONICAL_STRATEGY_NAMES: List[str] = [
    "alpha",
    "alpha_crypto",
    "alpha_futures",
    "alpha_intraday",
    "alpha_options",
    "beta",
    "beta_trend",
    "delta",
    "delta_pairs",
    "gamma",
    "gamma_futures",
    "gamma_reversion",
    "omega",
    "omega_macro",
    "omega_momentum_options",
    "omega_vol",
]


def _expand_enabled_strategies(raw: Any) -> List[str]:
    """Expand the config's `enabled_strategies`.  A literal `["*"]` is
    replaced with the canonical 16-strategy universe so that downstream
    set-membership filters (load_tier_filter) behave correctly.  Any
    other shape is normalised to a list of unique non-empty strings."""
    if not isinstance(raw, list):
        return []
    if any(isinstance(x, str) and x.strip() == "*" for x in raw):
        return list(_CANONICAL_STRATEGY_NAMES)
    return [str(x) for x in raw if isinstance(x, str) and x.strip()]


@dataclass
class TierRiskProfile:
    """Risk profile pulled from the active tier's `risk_profile` block.

    Numeric caps may be None — meaning "no cap at this tier" (e.g. SCALE
    leaves caps to the existing sizing pipeline)."""
    max_contracts_per_trade: Optional[int]
    max_risk_per_trade_usd: Optional[float]
    max_daily_loss_usd: Optional[float]
    max_weekly_loss_usd: Optional[float]
    max_trades_per_day: Optional[int]
    primary_session_only: bool
    flatten_before_eod: bool
    flatten_eod_minutes_before_close: Optional[int]
    stop_width_gate_enabled: bool


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _normalize_legacy_name(name: Optional[str]) -> Optional[str]:
    # renamed PRO -> SCALE in v9.1
    if name is None:
        return None
    return _LEGACY_TIER_ALIASES.get(name, name)


def _normalize_tiers_input(tiers: Any) -> Dict[str, Dict[str, Any]]:
    """Accept either v9.1 dict-form ({"MICRO": {...}}) or legacy list-form
    ([{"name": "MICRO", ...}]).  Returns a dict keyed by tier name with
    each entry's "name" field guaranteed to be set."""
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(tiers, dict):
        for name, spec in tiers.items():
            if not isinstance(spec, dict):
                continue
            entry = dict(spec)
            entry.setdefault("name", name)
            out[name] = entry
        return out
    if isinstance(tiers, list):
        for spec in tiers:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not name:
                continue
            out[name] = dict(spec)
        return out
    return out


def _ordered_tier_names(tiers: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return tier names sorted from lowest to highest band."""
    def _key(name: str) -> Tuple[int, float]:
        rank = TIER_RANK.get(name, 99)
        floor = float(tiers[name].get("min_equity_usd", 0) or 0)
        return (rank, floor)
    return sorted(tiers.keys(), key=_key)


def _naive_tier_for_equity(
    equity: float, tiers: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """Return the tier whose [min, max) band contains `equity`, or the
    top-most tier if equity is above all bands.  None if no tiers."""
    ordered = _ordered_tier_names(tiers)
    if not ordered:
        return None
    for name in ordered:
        spec = tiers[name]
        lo = float(spec.get("min_equity_usd", 0) or 0)
        hi_raw = spec.get("max_equity_usd")
        hi = float(hi_raw) if hi_raw is not None else float("inf")
        if lo <= equity < hi:
            return name
    return ordered[-1]


def _demotion_threshold(spec: Dict[str, Any]) -> float:
    """Read demotion_equity_usd from a tier spec; fall back to 90 % of
    min_equity_usd if the field is missing/null."""
    raw = spec.get("demotion_equity_usd")
    if raw is None:
        return float(spec.get("min_equity_usd", 0) or 0) * 0.90
    return float(raw)


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------

class TierManager:
    """Resolve current tier and risk profile for a given equity reading.

    The class is deliberately stateless beyond a single resolution: callers
    construct it with the current equity (and optionally the previous tier
    plus a `market_open` flag) and read attributes / call `get_risk_profile`
    / `state_payload` to publish.

    Demotion semantics
    ------------------
    * If the naive tier (equity band) is **higher** than the previous tier,
      the promotion is applied immediately.
    * If the naive tier is **lower**, a demotion is *proposed*.  A proposal
      becomes pending whenever the equity has dropped below the previous
      tier's `min_equity_usd` (the "warning band").  The proposed demotion
      is *applied* only when equity falls below the previous tier's
      `demotion_equity_usd` AND `_should_apply_demotion` returns True
      (i.e. the market is closed).
    * MICRO never demotes (it has no tier below it).
    """

    def __init__(
        self,
        equity: float,
        current_tier: Optional[str] = None,
        market_open: bool = True,
        tiers_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ):
        if tiers_config is None:
            cfg_path = config_path if config_path is not None else TIERS_CONFIG_PATH
            tiers_config = _read_json(cfg_path)
        self._tiers: Dict[str, Dict[str, Any]] = _normalize_tiers_input(
            tiers_config.get("tiers", {}) if isinstance(tiers_config, dict) else {}
        )
        self._equity: float = float(equity)
        self._current_tier: Optional[str] = _normalize_legacy_name(current_tier)
        self._market_open: bool = bool(market_open)

        self._tier_name: str = ""
        self._proposed_tier_name: str = ""
        self._demotion_pending: bool = False
        self._demotion_pending_to: Optional[str] = None
        self._demotion_pending_reason: Optional[str] = None
        self._compute()

    # ---- public attribute-style accessors ----

    @property
    def tier_name(self) -> str:
        return self._tier_name

    @property
    def proposed_tier_name(self) -> str:
        return self._proposed_tier_name

    @property
    def demotion_pending(self) -> bool:
        return self._demotion_pending

    @property
    def demotion_pending_to(self) -> Optional[str]:
        return self._demotion_pending_to

    @property
    def demotion_pending_reason(self) -> Optional[str]:
        return self._demotion_pending_reason

    # ---- public methods ----

    def get_risk_profile(self) -> TierRiskProfile:
        """Return the risk profile for the *resolved* tier.  Missing fields
        fall back to None for numeric caps and False for booleans; the
        method never raises."""
        spec = self._tiers.get(self._tier_name, {})
        rp = spec.get("risk_profile") if isinstance(spec, dict) else None
        if not isinstance(rp, dict):
            rp = {}

        def _opt_int(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        def _opt_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _bool(v: Any) -> bool:
            return bool(v) if v is not None else False

        return TierRiskProfile(
            max_contracts_per_trade=_opt_int(rp.get("max_contracts_per_trade")),
            max_risk_per_trade_usd=_opt_float(rp.get("max_risk_per_trade_usd")),
            max_daily_loss_usd=_opt_float(rp.get("max_daily_loss_usd")),
            max_weekly_loss_usd=_opt_float(rp.get("max_weekly_loss_usd")),
            max_trades_per_day=_opt_int(rp.get("max_trades_per_day")),
            primary_session_only=_bool(rp.get("primary_session_only")),
            flatten_before_eod=_bool(rp.get("flatten_before_eod")),
            flatten_eod_minutes_before_close=_opt_int(
                rp.get("flatten_eod_minutes_before_close")
            ),
            stop_width_gate_enabled=_bool(rp.get("stop_width_gate_enabled")),
        )

    # ---- core resolution ----

    @staticmethod
    def _should_apply_demotion(
        proposed_tier: str, current_tier: str, market_open: bool
    ) -> bool:
        """Return True if a tier change from `current_tier` to
        `proposed_tier` should take effect immediately.

        Promotions (proposed rank >= current rank) always apply.
        Demotions (proposed rank <  current rank) defer if the market is
        open; they apply when the market is closed."""
        proposed_rank = TIER_RANK.get(proposed_tier, -1)
        current_rank = TIER_RANK.get(current_tier, -1)
        if proposed_rank >= current_rank:
            return True
        return not market_open

    def _compute(self) -> None:
        if not self._tiers:
            LOG.error("tier_manager_no_tiers_configured")
            self._tier_name = self._current_tier or ""
            self._proposed_tier_name = self._tier_name
            return

        naive = _naive_tier_for_equity(self._equity, self._tiers)
        if naive is None:
            naive = _ordered_tier_names(self._tiers)[0]

        # First-time resolution (no previous tier): take naive directly.
        if not self._current_tier or self._current_tier not in self._tiers:
            self._tier_name = naive
            self._proposed_tier_name = naive
            self._demotion_pending = False
            self._demotion_pending_to = None
            self._demotion_pending_reason = None
            return

        current = self._current_tier
        current_rank = TIER_RANK.get(current, -1)
        naive_rank = TIER_RANK.get(naive, -1)

        # Promotion (or stable): apply naive immediately.  Note that this
        # check runs *before* any MICRO-specific handling — MICRO must be
        # able to promote upward when equity rises.
        if naive_rank >= current_rank:
            self._tier_name = naive
            self._proposed_tier_name = naive
            self._demotion_pending = False
            self._demotion_pending_to = None
            self._demotion_pending_reason = None
            return

        # MICRO can never demote (it has no band below it).  This branch
        # is reached only if `naive_rank < current_rank` while current is
        # MICRO — practically impossible since MICRO is rank 0 and there
        # is no rank -1 tier, but kept as a defensive guard.
        if current == "MICRO":
            self._tier_name = current
            self._proposed_tier_name = current
            self._demotion_pending = False
            self._demotion_pending_to = None
            self._demotion_pending_reason = None
            return

        # naive_rank < current_rank  →  demotion candidate.  Demotion is
        # pending whenever the equity has dropped below the current tier
        # band; whether it *applies* depends on the demotion gate AND the
        # mid-session deferral rule.
        current_spec = self._tiers[current]
        threshold = _demotion_threshold(current_spec)

        if self._equity < threshold:
            # Below the demotion gate — eligible for full apply.
            if self._should_apply_demotion(naive, current, self._market_open):
                self._tier_name = naive
                self._proposed_tier_name = naive
                self._demotion_pending = False
                self._demotion_pending_to = None
                self._demotion_pending_reason = None
            else:
                self._tier_name = current
                self._proposed_tier_name = naive
                self._demotion_pending = True
                self._demotion_pending_to = naive
                self._demotion_pending_reason = "mid_session_demotion_deferred"
        else:
            # Inside the warning band: equity below current.min but above
            # current.demotion_equity.  Demotion is queued but not yet
            # eligible to apply.
            self._tier_name = current
            self._proposed_tier_name = naive
            self._demotion_pending = True
            self._demotion_pending_to = naive
            self._demotion_pending_reason = "equity_below_tier_floor"

    # ---- payload assembly ----

    def state_payload(
        self,
        previous_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the dict that should be written to runtime/tier_state.json.

        `previous_state`, if supplied, is used to (a) carry forward the
        `promoted_at_utc` timestamp when the tier did not change and
        (b) preserve `demotion_pending_since_utc` when the demotion has
        been pending across cycles."""
        prev = previous_state or {}
        prev_tier_raw = prev.get("tier_name")
        prev_tier = _normalize_legacy_name(prev_tier_raw)
        prev_promoted = prev.get("promoted_at_utc")
        prev_demote_since = prev.get("demotion_pending_since_utc")

        promoted_at = prev_promoted
        if prev_tier != self._tier_name:
            promoted_at = _utc_now_iso()

        if self._demotion_pending:
            demote_since = prev_demote_since or _utc_now_iso()
            if (
                self._demotion_pending_reason == "mid_session_demotion_deferred"
            ):
                applies_at: Optional[str] = "next_market_open"
            else:
                applies_at = None
        else:
            demote_since = None
            applies_at = None

        spec = self._tiers.get(self._tier_name, {})

        return {
            "schema_version": "tier_state.v2",
            "tier_name": self._tier_name,
            "tier_description": spec.get("description", ""),
            "current_equity_usd": self._equity,
            "tier_min_equity": float(spec.get("min_equity_usd", 0) or 0),
            "tier_max_equity": (
                float(spec["max_equity_usd"])
                if spec.get("max_equity_usd") is not None
                else None
            ),
            "enabled_strategies": _expand_enabled_strategies(
                spec.get("enabled_strategies")
            ),
            "allowed_instruments": list(spec.get("allowed_instruments", [])),
            "risk_profile": asdict(self.get_risk_profile()),
            "previous_tier": prev_tier,
            "promoted_at_utc": promoted_at,
            "demotion_pending": self._demotion_pending,
            "demotion_pending_to": self._demotion_pending_to,
            "demotion_pending_reason": self._demotion_pending_reason,
            "demotion_pending_since_utc": demote_since,
            "demotion_applies_at": applies_at,
            "ts_utc": _utc_now_iso(),
        }


# ---------------------------------------------------------------------------
# Backwards-compatible functional helper used by the audit harness.
# ---------------------------------------------------------------------------

def _select_tier(
    equity: float,
    tiers: Any,
    previous_tier: Optional[str],
    hysteresis_pct: float,
) -> Dict[str, Any]:
    """Legacy helper retained for the behavioural-audit harness.

    Accepts either the v9.1 dict-form schema or the pre-v9.1 list-form.
    Always returns a tier spec dict that includes a "name" key.

    Demotion logic uses the demotion_equity_usd field when present and
    falls back to `min_equity_usd * (1 - hysteresis_pct/100)`.
    Legacy tier names in `previous_tier` are migrated transparently
    (e.g. "PRO" -> "SCALE")."""
    tier_map = _normalize_tiers_input(tiers)
    if not tier_map:
        return {"name": "MICRO"}

    prev = _normalize_legacy_name(previous_tier)
    naive = _naive_tier_for_equity(equity, tier_map)
    if naive is None:
        naive = _ordered_tier_names(tier_map)[0]

    if not prev or prev not in tier_map:
        return tier_map[naive]

    prev_rank = TIER_RANK.get(prev, -1)
    naive_rank = TIER_RANK.get(naive, -1)
    if naive_rank >= prev_rank:
        return tier_map[naive]

    # naive below previous → consult demotion threshold (with hysteresis
    # fallback when demotion_equity_usd is absent).
    prev_spec = tier_map[prev]
    raw_demotion = prev_spec.get("demotion_equity_usd")
    if raw_demotion is not None:
        threshold = float(raw_demotion)
    else:
        prev_min = float(prev_spec.get("min_equity_usd", 0) or 0)
        threshold = prev_min * (1.0 - float(hysteresis_pct) / 100.0)

    if equity >= threshold:
        LOG.info(
            "tier_hysteresis_held previous=%s naive=%s equity=%.2f floor=%.2f",
            prev, naive, equity, threshold,
        )
        return prev_spec
    return tier_map[naive]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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

    previous_state = _read_json(OUT_PATH)
    previous_tier = previous_state.get("tier_name")

    # Best-effort market-open guess for the CLI: if the operator does not
    # supply a flag, assume the market is closed so that pending demotions
    # are applied promptly on the next CLI run.  Live publishers should
    # pass `market_open` explicitly via the TierManager class API.
    tm = TierManager(
        equity=equity,
        current_tier=previous_tier,
        market_open=False,
        tiers_config=config,
    )

    if previous_tier and _normalize_legacy_name(previous_tier) != tm.tier_name:
        LOG.info(
            "tier_change previous=%s new=%s equity=%.2f",
            previous_tier, tm.tier_name, equity,
        )

    payload = tm.state_payload(previous_state=previous_state)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "tier_state_published tier=%s equity=$%.2f strategies=%d pending_demotion=%s",
        tm.tier_name, equity, len(payload["enabled_strategies"]),
        tm.demotion_pending,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
