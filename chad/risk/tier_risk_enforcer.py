#!/usr/bin/env python3
"""
TierRiskEnforcer — runtime enforcement of tier-derived risk caps.

Reads realized P&L from data/trades/trade_history_YYYYMMDD.ndjson
(schema closed_trade.v1) and applies the active TierRiskProfile to
decide whether a candidate trade is allowed.

Every call to `check()` atomically writes runtime/tier_enforcement_state.json
so external surfaces (operator dashboard, audit harness) can observe the
current enforcement posture without recomputing.

This module imports TierRiskProfile from chad.risk.tier_manager and does
not redefine it locally. It contains no broker or execution imports.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from chad.risk.tier_manager import TierRiskProfile

LOG = logging.getLogger("chad.risk.tier_risk_enforcer")

# Deterministic skip reason codes — referenced by tests and by
# downstream audit surfaces. Do not change without versioning.
SKIP_INVALID_CONTRACT_COUNT: str = "SKIP_INVALID_CONTRACT_COUNT"
SKIP_MAX_CONTRACTS_EXCEEDED: str = "SKIP_MAX_CONTRACTS_EXCEEDED"
SKIP_MAX_TRADES_REACHED: str = "SKIP_MAX_TRADES_REACHED"
SKIP_DAILY_LOSS_LIMIT: str = "SKIP_DAILY_LOSS_LIMIT"
SKIP_WEEKLY_LOSS_LIMIT: str = "SKIP_WEEKLY_LOSS_LIMIT"

STATE_FILENAME: str = "tier_enforcement_state.json"
STATE_TTL_SECONDS: int = 300
CLOSED_TRADE_SCHEMA: str = "closed_trade.v1"
LEDGER_FILENAME_PATTERN: str = "trade_history_{date}.ndjson"
WEEKLY_LOOKBACK_DAYS: int = 7


@dataclass
class StopWidthValidation:
    """Result of validating a proposed stop against the active risk budget."""
    stop_width_points: float
    stop_width_usd: float
    budget_usd: Optional[float]
    fits_budget: bool


@dataclass
class TierEnforcementDecision:
    """Result of a tier enforcement check against the candidate trade."""
    allowed: bool
    reason: Optional[str]
    daily_loss_today_usd: float
    weekly_loss_usd: float
    trades_today: int
    budget_remaining_usd: Optional[float]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    """Return a timezone-aware ISO-8601 UTC string ending in 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic JSON write — temp file + os.replace, parent created on demand.

    Mirrors the pattern used in chad/risk/quarantine_layer.atomic_write_json
    but tolerates `payload` as a Mapping (not strictly Dict).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime.

    Returns None on any failure.  Strings ending in 'Z' are coerced to
    '+00:00' so datetime.fromisoformat accepts them.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class TierRiskEnforcer:
    """Apply tier-derived enforcement to a candidate trade.

    Parameters
    ----------
    ledger_dir : Path
        Directory containing trade_history_YYYYMMDD.ndjson files.
    runtime_dir : Path
        Directory where tier_enforcement_state.json is written.
    tier_name : str
        Current active tier name (e.g. "MICRO", "STARTER", "PRO_GROWTH", "SCALE").
    tier_risk_profile : TierRiskProfile
        Resolved risk profile for the active tier.
    """

    def __init__(
        self,
        ledger_dir: Path,
        runtime_dir: Path,
        tier_name: str,
        tier_risk_profile: TierRiskProfile,
    ) -> None:
        self._ledger_dir: Path = Path(ledger_dir)
        self._runtime_dir: Path = Path(runtime_dir)
        self._tier_name: str = str(tier_name or "")
        self._profile: TierRiskProfile = tier_risk_profile
        # Flatten-window flag is toggled externally by the strategy when it
        # detects it is operating inside the configured EOD-flatten window.
        # The flag is reflected in the next runtime state write.
        self._flatten_triggered: bool = False

    # -----------------------------------------------------------------
    # Public state setters
    # -----------------------------------------------------------------

    def set_flatten_window_active(self, active: bool) -> None:
        """Mark whether enforcement is being invoked inside the EOD-flatten window.

        The strategy detects the flatten window using its own session-clock
        logic and propagates that signal here so the next state write
        records ``flatten_triggered`` truthfully.
        """
        self._flatten_triggered = bool(active)

    # -----------------------------------------------------------------
    # Core checks
    # -----------------------------------------------------------------

    def check(
        self,
        strategy: str,
        instrument: str,
        contracts: int,
    ) -> TierEnforcementDecision:
        """Evaluate a candidate trade against the active tier's caps.

        Always atomically writes runtime/tier_enforcement_state.json.

        Returns
        -------
        TierEnforcementDecision
            ``allowed=False`` with a deterministic reason if any cap is hit;
            otherwise ``allowed=True`` with ``reason=None``.
        """
        # Recompute realized loss aggregates fresh for every check; the
        # ledger is the source of truth.
        agg = self._aggregate_losses(strategy=strategy)
        trades_today = agg["trades_today"]
        daily_loss_today_usd = agg["daily_loss_today_usd"]
        weekly_loss_usd = agg["weekly_loss_usd"]

        max_contracts = self._profile.max_contracts_per_trade
        max_trades = self._profile.max_trades_per_day
        max_daily_loss = self._profile.max_daily_loss_usd
        max_weekly_loss = self._profile.max_weekly_loss_usd

        budget_remaining: Optional[float]
        if max_daily_loss is None:
            budget_remaining = None
        else:
            budget_remaining = float(max_daily_loss) - abs(float(daily_loss_today_usd))

        decision_allowed = True
        decision_reason: Optional[str] = None

        # 1. Invalid contract count is always rejected, regardless of tier.
        if contracts is None or contracts <= 0:
            decision_allowed = False
            decision_reason = SKIP_INVALID_CONTRACT_COUNT
        # 2. Max contracts per trade — only enforced when cap is finite.
        elif max_contracts is not None and contracts > int(max_contracts):
            decision_allowed = False
            decision_reason = SKIP_MAX_CONTRACTS_EXCEEDED
        # 3. Max trades per day.
        elif max_trades is not None and trades_today >= int(max_trades):
            decision_allowed = False
            decision_reason = SKIP_MAX_TRADES_REACHED
        # 4. Daily realized loss — block when |loss| >= limit.
        elif (
            max_daily_loss is not None
            and abs(float(daily_loss_today_usd)) >= float(max_daily_loss)
        ):
            decision_allowed = False
            decision_reason = SKIP_DAILY_LOSS_LIMIT
        # 5. Weekly realized loss — block when |loss| >= limit.
        elif (
            max_weekly_loss is not None
            and abs(float(weekly_loss_usd)) >= float(max_weekly_loss)
        ):
            decision_allowed = False
            decision_reason = SKIP_WEEKLY_LOSS_LIMIT

        decision = TierEnforcementDecision(
            allowed=decision_allowed,
            reason=decision_reason,
            daily_loss_today_usd=float(daily_loss_today_usd),
            weekly_loss_usd=float(weekly_loss_usd),
            trades_today=int(trades_today),
            budget_remaining_usd=(
                None if budget_remaining is None else float(budget_remaining)
            ),
        )

        self._write_state(decision)
        return decision

    def validate_stop_width(
        self,
        entry_price: float,
        stop_price: float,
        contracts: int,
        dollars_per_point: float,
    ) -> StopWidthValidation:
        """Check whether a proposed stop fits inside the tier's per-trade risk budget.

        Parameters
        ----------
        entry_price : float
            Candidate entry price.
        stop_price : float
            Candidate stop price.
        contracts : int
            Position size in contracts.
        dollars_per_point : float
            Instrument multiplier (e.g. 5.0 for MES, 2.0 for MNQ).

        Returns
        -------
        StopWidthValidation
            ``fits_budget=True`` if ``stop_width_usd <= max_risk_per_trade_usd``,
            or always True when the per-trade cap is None.
        """
        stop_width_points = abs(float(entry_price) - float(stop_price))
        stop_width_usd = stop_width_points * float(dollars_per_point) * float(contracts)
        budget = self._profile.max_risk_per_trade_usd
        if budget is None:
            fits = True
        else:
            fits = stop_width_usd <= float(budget)
        return StopWidthValidation(
            stop_width_points=float(stop_width_points),
            stop_width_usd=float(stop_width_usd),
            budget_usd=(None if budget is None else float(budget)),
            fits_budget=bool(fits),
        )

    # -----------------------------------------------------------------
    # Ledger reading
    # -----------------------------------------------------------------

    def _aggregate_losses(self, strategy: Optional[str] = None) -> Dict[str, float]:
        """Aggregate realized P&L across the daily and weekly windows.

        Reads up to ``WEEKLY_LOOKBACK_DAYS`` of trade_history_YYYYMMDD.ndjson
        files. Malformed or partially-written lines are skipped safely so
        enforcement is fail-open against ledger corruption rather than
        crashing the strategy pipeline.

        Returns a dict with keys:
          - trades_today (int)
          - daily_loss_today_usd (float, negative when net loss)
          - weekly_loss_usd (float, negative when net loss)
        """
        today_utc: date = _utc_now().date()
        week_start: date = today_utc - timedelta(days=WEEKLY_LOOKBACK_DAYS - 1)

        trades_today = 0
        daily_loss = 0.0
        weekly_loss = 0.0

        if not self._ledger_dir.exists():
            return {
                "trades_today": 0,
                "daily_loss_today_usd": 0.0,
                "weekly_loss_usd": 0.0,
            }

        for offset in range(WEEKLY_LOOKBACK_DAYS):
            day = today_utc - timedelta(days=offset)
            if day < week_start:
                break
            path = self._ledger_dir / LEDGER_FILENAME_PATTERN.format(
                date=day.strftime("%Y%m%d")
            )
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        payload = rec.get("payload") if isinstance(rec, dict) else None
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("schema_version") != CLOSED_TRADE_SCHEMA:
                            continue
                        if strategy and str(payload.get("strategy", "")).strip() != strategy:
                            continue
                        try:
                            pnl = float(payload.get("pnl"))
                        except (TypeError, ValueError):
                            continue
                        exit_dt = _parse_iso(payload.get("exit_time_utc"))
                        if exit_dt is None:
                            exit_dt = _parse_iso(rec.get("timestamp_utc"))
                        if exit_dt is None:
                            continue
                        exit_day = exit_dt.date()
                        if exit_day < week_start or exit_day > today_utc:
                            continue
                        if pnl < 0.0:
                            weekly_loss += pnl
                            if exit_day == today_utc:
                                daily_loss += pnl
                        if exit_day == today_utc:
                            trades_today += 1
            except Exception as exc:
                LOG.warning(
                    "tier_risk_enforcer ledger_read_failed path=%s err=%s",
                    path, exc,
                )
                continue

        return {
            "trades_today": trades_today,
            "daily_loss_today_usd": daily_loss,
            "weekly_loss_usd": weekly_loss,
        }

    # -----------------------------------------------------------------
    # Runtime state writer
    # -----------------------------------------------------------------

    def _write_state(self, decision: TierEnforcementDecision) -> None:
        """Atomically write runtime/tier_enforcement_state.json.

        Schema is fixed; see the spec in chad/risk/tier_risk_enforcer.py
        docstring. ``flatten_triggered`` reflects the most recent value
        set via ``set_flatten_window_active``.
        """
        max_daily = self._profile.max_daily_loss_usd
        max_weekly = self._profile.max_weekly_loss_usd
        max_trades = self._profile.max_trades_per_day
        max_contracts = self._profile.max_contracts_per_trade

        daily_hit = (
            max_daily is not None
            and abs(decision.daily_loss_today_usd) >= float(max_daily)
        )
        weekly_hit = (
            max_weekly is not None
            and abs(decision.weekly_loss_usd) >= float(max_weekly)
        )
        trade_hit = (
            max_trades is not None
            and decision.trades_today >= int(max_trades)
        )
        max_contracts_hit = decision.reason == SKIP_MAX_CONTRACTS_EXCEEDED

        payload: Dict[str, Any] = {
            "ts_utc": _iso_utc(_utc_now()),
            "ttl_seconds": STATE_TTL_SECONDS,
            "tier": self._tier_name,
            "trades_today": int(decision.trades_today),
            "max_trades_per_day": (None if max_trades is None else int(max_trades)),
            "daily_loss_today_usd": float(decision.daily_loss_today_usd),
            "max_daily_loss_usd": (None if max_daily is None else float(max_daily)),
            "weekly_loss_usd": float(decision.weekly_loss_usd),
            "max_weekly_loss_usd": (None if max_weekly is None else float(max_weekly)),
            "budget_remaining_today_usd": (
                None
                if decision.budget_remaining_usd is None
                else float(decision.budget_remaining_usd)
            ),
            "daily_loss_limit_hit": bool(daily_hit),
            "weekly_loss_limit_hit": bool(weekly_hit),
            "trade_count_limit_hit": bool(trade_hit),
            "max_contracts_limit_hit": bool(max_contracts_hit),
            "flatten_triggered": bool(self._flatten_triggered),
        }
        # max_contracts_per_trade is part of the profile but not in the
        # required state schema; the limit_hit flag captures the only
        # consumer signal that downstream surfaces need.
        _ = max_contracts

        _atomic_write_json(self._runtime_dir / STATE_FILENAME, payload)


__all__ = [
    "StopWidthValidation",
    "TierEnforcementDecision",
    "TierRiskEnforcer",
    "SKIP_INVALID_CONTRACT_COUNT",
    "SKIP_MAX_CONTRACTS_EXCEEDED",
    "SKIP_MAX_TRADES_REACHED",
    "SKIP_DAILY_LOSS_LIMIT",
    "SKIP_WEEKLY_LOSS_LIMIT",
    "STATE_FILENAME",
    "STATE_TTL_SECONDS",
]
