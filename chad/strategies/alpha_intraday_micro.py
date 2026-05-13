#!/usr/bin/env python3
"""
CHAD — alpha_intraday_micro

Tier-aware micro-futures intraday strategy operating on MES and MNQ.

Five setup families, each gated by session window, opening-range / VWAP /
overnight reference levels, the tier risk profile, and a per-trade stop
budget check:

  * ORB                    — opening-range breakout
  * VWAP_RECLAIM           — reclaim of an intraday VWAP
  * VWAP_REJECTION         — rejection at intraday VWAP
  * PULLBACK_CONTINUATION  — first pullback into a confirmed trend
  * SWEEP_REVERSAL         — liquidity sweep + snapback

All session-window arithmetic is computed in ``America/New_York``;
the EC2/system local timezone is never consulted.

The strategy depends on:
  * ``chad.risk.tier_manager.TierRiskProfile``         — caps it must respect
  * ``chad.risk.tier_risk_enforcer.TierRiskEnforcer``  — runtime gate writer
  * ``chad.strategies.alpha_intraday_micro_config``    — constants

It produces ``chad.types.TradeSignal`` objects — the canonical signal type
used by every other strategy in this codebase. No broker or execution
adapter is imported; this module is signal-emission only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from chad.risk.tier_manager import TierRiskProfile
from chad.risk.tier_risk_enforcer import (
    SKIP_INVALID_CONTRACT_COUNT,
    StopWidthValidation,
    TierRiskEnforcer,
)
from chad.strategies.alpha_intraday_micro_config import (
    HARD_EOD_EXIT_TIME,
    MES_DOLLARS_PER_POINT,
    MNQ_DOLLARS_PER_POINT,
    OPENING_RANGE_BARS,
    ORB_MAX_EXTENSION_PCT,
    ORB_PRIOR_DAY_CLEARANCE_ATR,
    ORB_VALID_UNTIL_MINUTES_AFTER_OPEN,
    ORB_VOLUME_MULTIPLIER,
    PRIMARY_SESSION_END,
    PRIMARY_SESSION_START,
    PULLBACK_MAX_RETRACE_RATIO,
    PULLBACK_MIN_TREND_BARS,
    R_TARGET_MAIN,
    R_TARGET_PARTIAL,
    SECONDARY_SESSION_END,
    SECONDARY_SESSION_START,
    SESSION_TIMEZONE,
    SETUP_PRIORITY,
    SKIP_DUPLICATE_SIGNAL,
    SKIP_EOD_FLATTEN_WINDOW,
    SKIP_INSUFFICIENT_BARS,
    SKIP_OUTSIDE_PRIMARY_WINDOW,
    SKIP_OUTSIDE_SECONDARY_WINDOW,
    SKIP_ORB_NO_CLEARANCE,
    SKIP_ORB_TOO_EXTENDED,
    SKIP_ORB_VOLUME_TOO_LOW,
    SKIP_PRIORITY_SUPPRESSED,
    SKIP_PULLBACK_TOO_DEEP,
    SKIP_PULLBACK_TREND_NOT_CONFIRMED,
    SKIP_STOP_TOO_WIDE,
    SKIP_SWEEP_NO_SNAPBACK,
    SKIP_SWEEP_TOO_DEEP,
    SKIP_SWEEP_TREND_DAY,
    SKIP_TIER_ENFORCEMENT,
    SKIP_VWAP_MESSY,
    SKIP_VWAP_NO_INITIAL_DRIVE,
    SWEEP_MAX_BREACH_PCT,
    SWEEP_SNAP_BARS,
    SWEEP_VALID_UNTIL_MINUTES_AFTER_OPEN,
    TREND_DAY_EXTENSION_PCT,
    VWAP_INITIAL_DRIVE_MIN_PCT,
    VWAP_MESSY_TEST_COUNT,
)
from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal

LOG = logging.getLogger("chad.strategies.alpha_intraday_micro")

UNIVERSE: Tuple[str, ...] = ("MES", "MNQ")
DOLLARS_PER_POINT: Dict[str, float] = {
    "MES": MES_DOLLARS_PER_POINT,
    "MNQ": MNQ_DOLLARS_PER_POINT,
}

SESSION_PRIMARY: str = "PRIMARY"
SESSION_SECONDARY: str = "SECONDARY"

DEFAULT_RUNTIME_DIR: Path = Path("/home/ubuntu/chad_finale/runtime")
DEFAULT_LEDGER_DIR: Path = Path("/home/ubuntu/chad_finale/data/trades")
# Regular-trading-hours open in America/New_York.  Setup-window arithmetic
# (ORB-valid-for-N-minutes, sweep-valid-for-N-minutes) is anchored here.
SESSION_OPEN_LOCAL: time = time(hour=9, minute=30)


def build_alpha_intraday_micro_config() -> StrategyConfig:
    """Return the StrategyConfig used to register this brain.

    The strategy is registered as additive — no other strategy's
    universe, weight, or load order is affected.
    """
    return StrategyConfig(
        name=StrategyName.ALPHA_INTRADAY_MICRO,
        enabled=True,
        target_universe=list(UNIVERSE),
        max_gross_exposure=None,
        notes="alpha_intraday_micro: tier-aware MES/MNQ intraday brain",
    )


# ----------------------------------------------------------------------
# Local utilities — kept private to this module.
# ----------------------------------------------------------------------

def _parse_hhmm(value: str) -> time:
    """Parse an "HH:MM" string into a naive ``datetime.time``."""
    hh, mm = value.split(":", 1)
    return time(hour=int(hh), minute=int(mm))


def _bar_volume(bar: Any) -> float:
    """Return the bar's ``volume`` field as float, 0.0 on any failure."""
    try:
        if isinstance(bar, Mapping):
            return float(bar.get("volume", 0.0) or 0.0)
        return float(getattr(bar, "volume", 0.0) or 0.0)
    except Exception:
        return 0.0


def _bar_field(bar: Any, name: str) -> float:
    """Return the named OHLC field of a bar as float, 0.0 on any failure."""
    try:
        if isinstance(bar, Mapping):
            return float(bar.get(name, 0.0) or 0.0)
        return float(getattr(bar, name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _bar_timestamp(bar: Any) -> str:
    """Return a string timestamp identifier for a bar.

    Falls back to an empty string if the bar carries no recognisable
    timestamp field — duplicate-suppression then degrades to comparing
    setup-family + side + instrument only, which is the safe behaviour.
    """
    if isinstance(bar, Mapping):
        for key in ("ts_utc", "timestamp", "ts", "time"):
            v = bar.get(key)
            if v:
                return str(v)
        return ""
    for key in ("ts_utc", "timestamp", "ts", "time"):
        v = getattr(bar, key, None)
        if v:
            return str(v)
    return ""


@dataclass(frozen=True)
class _DuplicateKey:
    """Composite identity for duplicate suppression.

    Two candidate signals collide when every field matches.  Bar
    timestamp is included so the same setup firing on a later bar is
    not treated as a duplicate.
    """
    instrument: str
    setup_family: str
    direction: str
    bar_ts: str


# ----------------------------------------------------------------------
# Strategy class
# ----------------------------------------------------------------------

class AlphaIntradayMicro:
    """Tier-aware intraday micro-futures strategy.

    Construct one instance per cycle; instance state is used only for
    intra-cycle priority suppression and duplicate suppression and is
    not persisted between cycles.
    """

    def __init__(
        self,
        tier_profile: TierRiskProfile,
        tier_enforcer: TierRiskEnforcer,
        runtime_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """Build a fresh strategy instance.

        Parameters
        ----------
        tier_profile : TierRiskProfile
            Resolved risk profile for the active tier.  Caps may be None.
        tier_enforcer : TierRiskEnforcer
            Pre-built enforcer.  Must already be wired to the canonical
            ledger + runtime directories.
        runtime_dir : Optional[Path]
            Override for the runtime directory.  Defaults to the canonical
            ``/home/ubuntu/chad_finale/runtime`` path.
        now : Optional[datetime]
            Timezone-aware injection point used by tests.  When omitted
            the strategy uses ``datetime.now(timezone.utc)``.
        """
        self._tier_profile: TierRiskProfile = tier_profile
        self._tier_enforcer: TierRiskEnforcer = tier_enforcer
        self._runtime_dir: Path = Path(runtime_dir) if runtime_dir else DEFAULT_RUNTIME_DIR
        if now is not None and now.tzinfo is None:
            raise ValueError("AlphaIntradayMicro.now must be timezone-aware")
        self._now_override: Optional[datetime] = now
        self._tz: ZoneInfo = ZoneInfo(SESSION_TIMEZONE)

        # Per-cycle state ---------------------------------------------------
        # Maps (instrument, session_window) -> highest priority that has
        # already fired in that window during this cycle.
        self._fired_priority: Dict[Tuple[str, str], int] = {}
        # Set of duplicate keys already emitted this cycle.
        self._emitted_keys: set = set()
        # Lightweight per-call diagnostic log; tests inspect this to verify
        # deterministic skip-reason codes were recorded.
        self._skip_log: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Public surface
    # -----------------------------------------------------------------

    @property
    def skip_log(self) -> List[Dict[str, Any]]:
        """Return the per-cycle diagnostic skip log (read-only view)."""
        return list(self._skip_log)

    def generate_signals(self, ctx: object) -> List[TradeSignal]:
        """Run all five setup evaluators across the active universe.

        Parameters
        ----------
        ctx : object
            CHAD ``MarketContext`` or any object exposing ``bars_1m``,
            ``bars``, and (optionally) supplemental dicts named
            ``prior_day_high``, ``prior_day_low``, ``overnight_high``,
            ``overnight_low``, and ``vwap``.

        Returns
        -------
        List[TradeSignal]
            Zero or more emitted signals in setup-priority order.
        """
        signals: List[TradeSignal] = []
        try:
            bars_map = getattr(ctx, "bars_1m", None) or getattr(ctx, "bars", None) or {}
            if not isinstance(bars_map, Mapping):
                return signals

            vwap_map = getattr(ctx, "vwap", {}) or {}
            prior_high_map = getattr(ctx, "prior_day_high", {}) or {}
            prior_low_map = getattr(ctx, "prior_day_low", {}) or {}
            on_high_map = getattr(ctx, "overnight_high", {}) or {}
            on_low_map = getattr(ctx, "overnight_low", {}) or {}

            for instrument in UNIVERSE:
                bars = bars_map.get(instrument)
                if not bars:
                    continue
                bars_seq: Sequence[Any] = bars  # type: ignore[assignment]
                vwap = _opt_float(vwap_map.get(instrument))
                prior_high = _opt_float(prior_high_map.get(instrument))
                prior_low = _opt_float(prior_low_map.get(instrument))
                on_high = _opt_float(on_high_map.get(instrument))
                on_low = _opt_float(on_low_map.get(instrument))

                for evaluator in (
                    lambda: self._evaluate_orb(
                        bars_seq,
                        vwap,
                        prior_high,
                        prior_low,
                        self._tier_profile,
                        instrument=instrument,
                    ),
                    lambda: self._evaluate_vwap_reclaim_rejection(
                        bars_seq,
                        vwap,
                        self._tier_profile,
                        instrument=instrument,
                    ),
                    lambda: self._evaluate_pullback_continuation(
                        bars_seq,
                        vwap,
                        self._tier_profile,
                        instrument=instrument,
                    ),
                    lambda: self._evaluate_sweep_reversal(
                        bars_seq,
                        on_high,
                        on_low,
                        vwap,
                        self._tier_profile,
                        instrument=instrument,
                    ),
                ):
                    try:
                        sig = evaluator()
                    except Exception as exc:
                        LOG.warning(
                            "alpha_intraday_micro: evaluator error instrument=%s err=%s",
                            instrument, exc,
                        )
                        sig = None
                    if sig is not None:
                        signals.append(sig)
        except Exception as exc:
            LOG.warning("alpha_intraday_micro: generate_signals exception=%s", exc)
            return []
        return signals

    # -----------------------------------------------------------------
    # Private setup evaluators
    # -----------------------------------------------------------------

    def _evaluate_orb(
        self,
        bars: Sequence[Any],
        vwap: Optional[float],
        prior_day_high: Optional[float],
        prior_day_low: Optional[float],
        tier_profile: TierRiskProfile,
        *,
        instrument: str = "MES",
    ) -> Optional[TradeSignal]:
        """Evaluate the opening-range-breakout setup for one instrument.

        Returns a TradeSignal if all gates pass, otherwise None with the
        deterministic skip reason recorded in ``skip_log``.
        """
        setup_family = "ORB"
        # 1) Session-window check
        window = self._session_window()
        if window != SESSION_PRIMARY:
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        # 2) EOD-flatten window before any signal emission
        if self._in_flatten_window(tier_profile):
            self._mark_flatten()
            self._log_skip(instrument, setup_family, SKIP_EOD_FLATTEN_WINDOW)
            return None
        # 3) ORB validity window — only fires in the first N minutes
        if self._minutes_since_open() > ORB_VALID_UNTIL_MINUTES_AFTER_OPEN:
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        # 4) Sufficient bars
        if len(bars) < OPENING_RANGE_BARS + 1:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None

        opening_bars = bars[:OPENING_RANGE_BARS]
        current_bar = bars[-1]
        or_high = max(_bar_field(b, "high") for b in opening_bars)
        or_low = min(_bar_field(b, "low") for b in opening_bars)
        avg_or_volume = sum(_bar_volume(b) for b in opening_bars) / max(
            1, len(opening_bars)
        )
        current_close = _bar_field(current_bar, "close")
        current_volume = _bar_volume(current_bar)

        # 5) Volume confirmation
        if avg_or_volume > 0 and current_volume < avg_or_volume * ORB_VOLUME_MULTIPLIER:
            self._log_skip(instrument, setup_family, SKIP_ORB_VOLUME_TOO_LOW)
            return None

        # 6) Direction + extension check
        direction: Optional[SignalSide] = None
        breakout_level = 0.0
        if current_close >= or_high:
            direction = SignalSide.BUY
            breakout_level = or_high
        elif current_close <= or_low:
            direction = SignalSide.SELL
            breakout_level = or_low
        else:
            self._log_skip(instrument, setup_family, SKIP_ORB_NO_CLEARANCE)
            return None

        if breakout_level > 0:
            extension = abs(current_close - breakout_level) / breakout_level
            if extension > ORB_MAX_EXTENSION_PCT:
                self._log_skip(instrument, setup_family, SKIP_ORB_TOO_EXTENDED)
                return None

        # 7) Prior-day clearance check (when reference data present)
        atr_proxy = max(0.0, or_high - or_low)
        if (
            direction == SignalSide.BUY
            and prior_day_high is not None
            and atr_proxy > 0
            and current_close < prior_day_high + atr_proxy * ORB_PRIOR_DAY_CLEARANCE_ATR
            and current_close < prior_day_high
        ):
            self._log_skip(instrument, setup_family, SKIP_ORB_NO_CLEARANCE)
            return None
        if (
            direction == SignalSide.SELL
            and prior_day_low is not None
            and atr_proxy > 0
            and current_close > prior_day_low - atr_proxy * ORB_PRIOR_DAY_CLEARANCE_ATR
            and current_close > prior_day_low
        ):
            self._log_skip(instrument, setup_family, SKIP_ORB_NO_CLEARANCE)
            return None

        entry = current_close
        stop = or_low if direction == SignalSide.BUY else or_high
        return self._finalize_setup(
            instrument=instrument,
            setup_family=setup_family,
            session_window=window,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            current_bar=current_bar,
            tier_profile=tier_profile,
            meta_extras={
                "opening_range_high": float(or_high),
                "opening_range_low": float(or_low),
                "vwap_at_signal": (None if vwap is None else float(vwap)),
            },
        )

    def _evaluate_vwap_reclaim_rejection(
        self,
        bars: Sequence[Any],
        vwap: Optional[float],
        tier_profile: TierRiskProfile,
        *,
        instrument: str = "MES",
    ) -> Optional[TradeSignal]:
        """Evaluate VWAP reclaim/rejection setups for one instrument.

        Returns a TradeSignal if all gates pass, otherwise None with the
        deterministic skip reason recorded in ``skip_log``.
        """
        # Determine setup family later (reclaim vs rejection) but use a
        # placeholder for early-exit skip logging.
        setup_family = "VWAP_RECLAIM"
        window = self._session_window()
        if window == "OUT":
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        if self._in_flatten_window(tier_profile):
            self._mark_flatten()
            self._log_skip(instrument, setup_family, SKIP_EOD_FLATTEN_WINDOW)
            return None
        if tier_profile.primary_session_only and window != SESSION_PRIMARY:
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_SECONDARY_WINDOW)
            return None
        if vwap is None or len(bars) < OPENING_RANGE_BARS + 1:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None

        # Initial-drive: distance from session-open to current as a pct.
        opening_bars = bars[:OPENING_RANGE_BARS]
        open_px = _bar_field(opening_bars[0], "open") or _bar_field(opening_bars[0], "close")
        current_bar = bars[-1]
        current_close = _bar_field(current_bar, "close")
        if open_px <= 0:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None
        drive = abs(current_close - open_px) / open_px
        if drive < VWAP_INITIAL_DRIVE_MIN_PCT:
            self._log_skip(instrument, setup_family, SKIP_VWAP_NO_INITIAL_DRIVE)
            return None

        # Messy VWAP test: count how many bars crossed VWAP recently.
        recent_bars = bars[-min(len(bars), 10):]
        crosses = 0
        last_side: Optional[str] = None
        for b in recent_bars:
            close_b = _bar_field(b, "close")
            side = "above" if close_b >= vwap else "below"
            if last_side is not None and side != last_side:
                crosses += 1
            last_side = side
        if crosses >= VWAP_MESSY_TEST_COUNT:
            self._log_skip(instrument, setup_family, SKIP_VWAP_MESSY)
            return None

        # Direction selection
        prev_close = _bar_field(bars[-2], "close") if len(bars) >= 2 else current_close
        direction: Optional[SignalSide] = None
        if prev_close < vwap and current_close >= vwap:
            direction = SignalSide.BUY
            setup_family = "VWAP_RECLAIM"
        elif prev_close > vwap and current_close <= vwap:
            direction = SignalSide.SELL
            setup_family = "VWAP_REJECTION"
        else:
            self._log_skip(instrument, setup_family, SKIP_VWAP_NO_INITIAL_DRIVE)
            return None

        # Stop just on the far side of VWAP by the bar range
        bar_range = max(_bar_field(current_bar, "high") - _bar_field(current_bar, "low"), 0.25)
        stop = (current_close - bar_range) if direction == SignalSide.BUY else (current_close + bar_range)
        return self._finalize_setup(
            instrument=instrument,
            setup_family=setup_family,
            session_window=window,
            direction=direction,
            entry_price=current_close,
            stop_price=stop,
            current_bar=current_bar,
            tier_profile=tier_profile,
            meta_extras={
                "opening_range_high": None,
                "opening_range_low": None,
                "vwap_at_signal": float(vwap),
            },
        )

    def _evaluate_pullback_continuation(
        self,
        bars: Sequence[Any],
        vwap: Optional[float],
        tier_profile: TierRiskProfile,
        *,
        instrument: str = "MES",
    ) -> Optional[TradeSignal]:
        """Evaluate the pullback-continuation setup.

        Looks for a confirmed N-bar trend followed by a pullback that
        retraces no more than ``PULLBACK_MAX_RETRACE_RATIO`` of the
        impulse before resuming. Returns None when any gate fails.
        """
        setup_family = "PULLBACK_CONTINUATION"
        window = self._session_window()
        if window == "OUT":
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        if self._in_flatten_window(tier_profile):
            self._mark_flatten()
            self._log_skip(instrument, setup_family, SKIP_EOD_FLATTEN_WINDOW)
            return None
        if tier_profile.primary_session_only and window != SESSION_PRIMARY:
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_SECONDARY_WINDOW)
            return None
        if len(bars) < PULLBACK_MIN_TREND_BARS + 2:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None

        impulse = bars[-(PULLBACK_MIN_TREND_BARS + 2): -1]
        last_bar = bars[-1]
        closes = [_bar_field(b, "close") for b in impulse]
        if len(closes) < PULLBACK_MIN_TREND_BARS:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None
        up_trend = all(closes[i] >= closes[i - 1] for i in range(1, len(closes)))
        down_trend = all(closes[i] <= closes[i - 1] for i in range(1, len(closes)))
        if not (up_trend or down_trend):
            self._log_skip(instrument, setup_family, SKIP_PULLBACK_TREND_NOT_CONFIRMED)
            return None
        impulse_high = max(_bar_field(b, "high") for b in impulse)
        impulse_low = min(_bar_field(b, "low") for b in impulse)
        impulse_range = max(impulse_high - impulse_low, 0.0)
        if impulse_range <= 0:
            self._log_skip(instrument, setup_family, SKIP_PULLBACK_TREND_NOT_CONFIRMED)
            return None
        last_close = _bar_field(last_bar, "close")
        if up_trend:
            retrace = (impulse_high - last_close) / impulse_range
            direction = SignalSide.BUY
            stop = impulse_low
        else:
            retrace = (last_close - impulse_low) / impulse_range
            direction = SignalSide.SELL
            stop = impulse_high
        if retrace > PULLBACK_MAX_RETRACE_RATIO:
            self._log_skip(instrument, setup_family, SKIP_PULLBACK_TOO_DEEP)
            return None
        return self._finalize_setup(
            instrument=instrument,
            setup_family=setup_family,
            session_window=window,
            direction=direction,
            entry_price=last_close,
            stop_price=stop,
            current_bar=last_bar,
            tier_profile=tier_profile,
            meta_extras={
                "opening_range_high": None,
                "opening_range_low": None,
                "vwap_at_signal": (None if vwap is None else float(vwap)),
            },
        )

    def _evaluate_sweep_reversal(
        self,
        bars: Sequence[Any],
        overnight_high: Optional[float],
        overnight_low: Optional[float],
        vwap: Optional[float],
        tier_profile: TierRiskProfile,
        *,
        instrument: str = "MES",
    ) -> Optional[TradeSignal]:
        """Evaluate the sweep-reversal setup against overnight references.

        Returns a TradeSignal when an overnight high/low is swept and price
        snaps back within ``SWEEP_SNAP_BARS``; returns None otherwise.
        """
        setup_family = "SWEEP_REVERSAL"
        window = self._session_window()
        if window == "OUT":
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        if self._in_flatten_window(tier_profile):
            self._mark_flatten()
            self._log_skip(instrument, setup_family, SKIP_EOD_FLATTEN_WINDOW)
            return None
        if self._minutes_since_open() > SWEEP_VALID_UNTIL_MINUTES_AFTER_OPEN:
            self._log_skip(instrument, setup_family, SKIP_OUTSIDE_PRIMARY_WINDOW)
            return None
        if overnight_high is None or overnight_low is None or len(bars) < SWEEP_SNAP_BARS + 1:
            self._log_skip(instrument, setup_family, SKIP_INSUFFICIENT_BARS)
            return None

        session_high = max(_bar_field(b, "high") for b in bars)
        session_low = min(_bar_field(b, "low") for b in bars)
        on_range = max(overnight_high - overnight_low, 1e-6)
        # Trend-day filter: when the session has already extended beyond
        # the overnight range by more than TREND_DAY_EXTENSION_PCT in
        # either direction, reversion attempts are skipped.
        if (
            (session_high - overnight_high) / on_range > TREND_DAY_EXTENSION_PCT
            and (overnight_low - session_low) / on_range < 0
        ):
            self._log_skip(instrument, setup_family, SKIP_SWEEP_TREND_DAY)
            return None

        last_bar = bars[-1]
        last_close = _bar_field(last_bar, "close")
        last_high = _bar_field(last_bar, "high")
        last_low = _bar_field(last_bar, "low")
        prior_bars = bars[-(SWEEP_SNAP_BARS + 1): -1]
        # Bullish reversal: sweep below overnight low + snap back.
        if last_low < overnight_low and any(_bar_field(b, "low") < overnight_low for b in prior_bars):
            breach = (overnight_low - last_low) / overnight_low if overnight_low else 0.0
            if breach > SWEEP_MAX_BREACH_PCT:
                self._log_skip(instrument, setup_family, SKIP_SWEEP_TOO_DEEP)
                return None
            if last_close < overnight_low:
                self._log_skip(instrument, setup_family, SKIP_SWEEP_NO_SNAPBACK)
                return None
            direction = SignalSide.BUY
            stop = last_low
        # Bearish reversal: sweep above overnight high + snap back.
        elif last_high > overnight_high and any(_bar_field(b, "high") > overnight_high for b in prior_bars):
            breach = (last_high - overnight_high) / overnight_high if overnight_high else 0.0
            if breach > SWEEP_MAX_BREACH_PCT:
                self._log_skip(instrument, setup_family, SKIP_SWEEP_TOO_DEEP)
                return None
            if last_close > overnight_high:
                self._log_skip(instrument, setup_family, SKIP_SWEEP_NO_SNAPBACK)
                return None
            direction = SignalSide.SELL
            stop = last_high
        else:
            self._log_skip(instrument, setup_family, SKIP_SWEEP_NO_SNAPBACK)
            return None

        return self._finalize_setup(
            instrument=instrument,
            setup_family=setup_family,
            session_window=window,
            direction=direction,
            entry_price=last_close,
            stop_price=stop,
            current_bar=last_bar,
            tier_profile=tier_profile,
            meta_extras={
                "opening_range_high": None,
                "opening_range_low": None,
                "vwap_at_signal": (None if vwap is None else float(vwap)),
            },
        )

    # -----------------------------------------------------------------
    # Internal pipeline (stop validation, enforcement, suppression)
    # -----------------------------------------------------------------

    def _finalize_setup(
        self,
        *,
        instrument: str,
        setup_family: str,
        session_window: str,
        direction: SignalSide,
        entry_price: float,
        stop_price: float,
        current_bar: Any,
        tier_profile: TierRiskProfile,
        meta_extras: Mapping[str, Any],
    ) -> Optional[TradeSignal]:
        """Run stop-width, enforcement, priority and duplicate gates and emit.

        This is the single chokepoint that builds the meta block and the
        TradeSignal. All setup evaluators converge here so the gate
        ordering is invariant across families.
        """
        contracts = self._contracts_for(tier_profile)
        if contracts <= 0:
            self._log_skip(instrument, setup_family, SKIP_INVALID_CONTRACT_COUNT)
            return None
        dollars_per_point = DOLLARS_PER_POINT.get(instrument, 1.0)

        validation: StopWidthValidation = self._tier_enforcer.validate_stop_width(
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            contracts=int(contracts),
            dollars_per_point=float(dollars_per_point),
        )
        if not validation.fits_budget:
            self._log_skip(instrument, setup_family, SKIP_STOP_TOO_WIDE)
            return None

        # Tier enforcement (counters, daily/weekly losses, max contracts)
        decision = self._tier_enforcer.check(
            strategy=str(StrategyName.ALPHA_INTRADAY_MICRO.value),
            instrument=instrument,
            contracts=int(contracts),
        )
        if not decision.allowed:
            self._log_skip(
                instrument,
                setup_family,
                SKIP_TIER_ENFORCEMENT,
                enforcement_reason=decision.reason,
            )
            return None

        # Setup priority suppression
        priority = SETUP_PRIORITY.get(setup_family, 99)
        key = (instrument, session_window)
        already = self._fired_priority.get(key)
        if already is not None and priority > already:
            self._log_skip(instrument, setup_family, SKIP_PRIORITY_SUPPRESSED)
            return None

        # Duplicate suppression
        dkey = _DuplicateKey(
            instrument=instrument,
            setup_family=setup_family,
            direction=str(direction.value),
            bar_ts=_bar_timestamp(current_bar),
        )
        if dkey in self._emitted_keys:
            self._log_skip(instrument, setup_family, SKIP_DUPLICATE_SIGNAL)
            return None

        # Build the meta block required by the spec.
        meta: Dict[str, Any] = {
            "setup_family": setup_family,
            "session_window": session_window,
            "opening_range_high": meta_extras.get("opening_range_high"),
            "opening_range_low": meta_extras.get("opening_range_low"),
            "vwap_at_signal": meta_extras.get("vwap_at_signal"),
            "stop_width_points": float(validation.stop_width_points),
            "stop_width_usd": float(validation.stop_width_usd),
            "risk_budget_usd": (
                None if validation.budget_usd is None else float(validation.budget_usd)
            ),
            "stop_fits_budget": bool(validation.fits_budget),
            "r_target_1": float(R_TARGET_PARTIAL),
            "r_target_2": float(R_TARGET_MAIN),
        }

        signal = TradeSignal(
            strategy=StrategyName.ALPHA_INTRADAY_MICRO,
            symbol=instrument,
            side=direction,
            size=float(contracts),
            confidence=0.65,
            asset_class=AssetClass.FUTURES,
            meta=meta,
        )

        # Record suppression state only after successful emission.
        self._fired_priority[key] = priority if already is None else min(already, priority)
        self._emitted_keys.add(dkey)
        return signal

    # -----------------------------------------------------------------
    # Session-window and clock helpers
    # -----------------------------------------------------------------

    def _now_utc(self) -> datetime:
        if self._now_override is not None:
            return self._now_override.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _now_local(self) -> datetime:
        return self._now_utc().astimezone(self._tz)

    def _session_window(self) -> str:
        """Return 'PRIMARY', 'SECONDARY', or 'OUT' based on America/New_York."""
        local = self._now_local().time()
        primary_start = _parse_hhmm(PRIMARY_SESSION_START)
        primary_end = _parse_hhmm(PRIMARY_SESSION_END)
        secondary_start = _parse_hhmm(SECONDARY_SESSION_START)
        secondary_end = _parse_hhmm(SECONDARY_SESSION_END)
        if primary_start <= local < primary_end:
            return SESSION_PRIMARY
        if secondary_start <= local < secondary_end:
            return SESSION_SECONDARY
        return "OUT"

    def _minutes_since_open(self) -> float:
        """Return minutes elapsed since 09:30 America/New_York (the RTH open)."""
        now = self._now_local()
        open_today = now.replace(
            hour=SESSION_OPEN_LOCAL.hour,
            minute=SESSION_OPEN_LOCAL.minute,
            second=0,
            microsecond=0,
        )
        return (now - open_today).total_seconds() / 60.0

    def _in_flatten_window(self, tier_profile: TierRiskProfile) -> bool:
        """Return True if EOD-flatten mode is active for this tier and clock.

        The flatten window opens ``flatten_eod_minutes_before_close`` minutes
        before ``HARD_EOD_EXIT_TIME`` and remains open up to and including
        that time. If the tier does not request EOD-flatten, returns False.
        """
        if not tier_profile.flatten_before_eod:
            return False
        flatten_minutes = tier_profile.flatten_eod_minutes_before_close
        if flatten_minutes is None:
            return False
        hard_eod = _parse_hhmm(HARD_EOD_EXIT_TIME)
        now = self._now_local()
        eod_dt = now.replace(
            hour=hard_eod.hour, minute=hard_eod.minute, second=0, microsecond=0
        )
        flatten_open = eod_dt - timedelta(minutes=int(flatten_minutes))
        return flatten_open <= now <= eod_dt

    def _mark_flatten(self) -> None:
        """Notify the enforcer that this evaluator is firing in flatten window."""
        self._tier_enforcer.set_flatten_window_active(True)

    # -----------------------------------------------------------------
    # Sizing + diagnostic helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _contracts_for(tier_profile: TierRiskProfile) -> int:
        """Pick the smallest valid contract count under the tier cap.

        The strategy always sizes to one contract by design — tier caps
        only ever shrink size further. SCALE (cap=None) keeps one
        contract; lower tiers also keep one because their per-trade
        budget would otherwise gate via stop-width validation.
        """
        cap = tier_profile.max_contracts_per_trade
        if cap is None:
            return 1
        return min(1, int(cap)) if cap > 0 else 0

    def _log_skip(
        self,
        instrument: str,
        setup_family: str,
        reason: str,
        enforcement_reason: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "instrument": instrument,
            "setup_family": setup_family,
            "reason": reason,
        }
        if enforcement_reason:
            entry["enforcement_reason"] = enforcement_reason
        self._skip_log.append(entry)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _opt_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to float, returning None when unavailable or NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check without importing math
        return None
    return f


def alpha_intraday_micro_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """StrategyEngine-compatible handler for alpha_intraday_micro.

    Builds a TierRiskEnforcer wired to the canonical ledger and runtime
    directories, instantiates AlphaIntradayMicro, and returns the signal
    list. Tier name and profile are pulled from ``ctx.tier_profile`` /
    ``ctx.tier_name`` when present so callers can inject a non-default
    profile; otherwise the strategy is skipped (no signals emitted) —
    the strategy is tier-gated by design and refuses to size without
    an explicit profile.
    """
    try:
        tier_profile = getattr(ctx, "tier_profile", None)
        tier_name = getattr(ctx, "tier_name", "")
        if not isinstance(tier_profile, TierRiskProfile):
            return []
        runtime_dir = DEFAULT_RUNTIME_DIR
        ledger_dir = DEFAULT_LEDGER_DIR
        enforcer = TierRiskEnforcer(
            ledger_dir=ledger_dir,
            runtime_dir=runtime_dir,
            tier_name=str(tier_name),
            tier_risk_profile=tier_profile,
        )
        strat = AlphaIntradayMicro(
            tier_profile=tier_profile,
            tier_enforcer=enforcer,
            runtime_dir=runtime_dir,
        )
        return strat.generate_signals(ctx)
    except Exception as exc:
        LOG.warning("alpha_intraday_micro_handler: exception=%s", exc)
        return []


__all__ = [
    "AlphaIntradayMicro",
    "alpha_intraday_micro_handler",
    "build_alpha_intraday_micro_config",
    "UNIVERSE",
]
