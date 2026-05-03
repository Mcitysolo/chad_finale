"""
CHAD Smart Strategy Throttle Gate
===================================
Performance-aware, time-window-based throttle.

Control stack position:
    Normal → Smart Throttle → Reduced Signal Rate →
    Temporary Pause → Edge Decay Halt

Key principle:
    - Winning strategies: ALLOW unconditionally
    - Losing/churning strategies: progressive graduated response
    - Exits, stop-losses, risk reductions, hedges: NEVER blocked

Throttle levels (returned per strategy per signal):
    ALLOW                    — no issue, proceed normally
    THROTTLE                 — reduce fresh-entry frequency (time window)
    CONFIDENCE_UPSHIFT       — require higher confidence for new entries
    PAUSE_TEMPORARILY        — block new entries for N minutes
    HALT_DEFER_TO_EDGE_DECAY — losses severe enough, defer to edge decay

Time windows (smoother than cycle counts):
    max_entries_per_15min: int — Level 1 throttle
    max_entries_per_hour: int  — Level 2 throttle
    pause_duration_minutes: int — Level 3 pause
    confidence_upshift_delta: float — added to base confidence requirement
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chad.strategy_throttle_gate")

REPO_ROOT = Path(__file__).parent.parent.parent
RUNTIME = REPO_ROOT / "runtime"
DATA = REPO_ROOT / "data"

# Minimum trades before throttle can fire
MIN_TRADES_TO_EVALUATE = 10

# Winning strategy — no throttle if all conditions met
WIN_RATE_HEALTHY = 0.55
MIN_PNL_HEALTHY = 0.0

# Level 1 — THROTTLE: reduce fresh-entry frequency
THROTTLE_WIN_RATE = 0.45
THROTTLE_MAX_ENTRIES_15MIN = 3
THROTTLE_MAX_ENTRIES_HOUR = 8

# Level 2 — CONFIDENCE_UPSHIFT: require higher confidence
CONFIDENCE_UPSHIFT_WIN_RATE = 0.40
CONFIDENCE_UPSHIFT_DELTA = 0.15

# Level 3 — PAUSE_TEMPORARILY: stop new entries for N minutes
PAUSE_WIN_RATE = 0.35
PAUSE_DURATION_MINUTES = 120
LOSS_STREAK_FOR_PAUSE = 4

# Level 4 — defer to edge decay (5 consecutive losses)
EDGE_DECAY_LOSS_STREAK = 5

# In-memory tracking, persisted to runtime/strategy_throttle_state.json
# so pause windows and entry-rate state survive process restart.
_ENTRY_TIMESTAMPS: Dict[str, List[float]] = {}
_PAUSED_UNTIL: Dict[str, float] = {}

_THROTTLE_STATE_PATH = (
    Path(__file__).parent.parent.parent
    / "runtime" / "strategy_throttle_state.json"
)


def _save_throttle_state() -> None:
    """Persist throttle state atomically.

    Only active pauses (until > now) and recent entry timestamps
    (within the last 2 hours) are written.
    """
    try:
        import os as _os
        now = time.time()
        state = {
            "schema_version": "strategy_throttle_state.v1",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "paused_until": {
                k: datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
                for k, v in _PAUSED_UNTIL.items()
                if v > now
            },
            "entry_timestamps": {
                k: [t for t in v if t > now - 7200]
                for k, v in _ENTRY_TIMESTAMPS.items()
                if any(t > now - 7200 for t in v)
            },
        }
        _THROTTLE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _tmp = _THROTTLE_STATE_PATH.with_suffix(".json.tmp")
        _tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _os.replace(str(_tmp), str(_THROTTLE_STATE_PATH))
    except Exception as e:
        logger.debug("strategy_throttle: state_save_failed err=%s", e)


def _load_throttle_state() -> None:
    """Load persisted throttle state on module import.

    Restores pause windows still in the future and entry timestamps
    from the last 2 hours. Silently ignores missing/corrupt state.
    """
    try:
        if not _THROTTLE_STATE_PATH.exists():
            return
        data = json.loads(
            _THROTTLE_STATE_PATH.read_text(encoding="utf-8")
        )
        now = time.time()
        for strategy, until_iso in (data.get("paused_until") or {}).items():
            try:
                until_ts = datetime.fromisoformat(
                    str(until_iso).replace("Z", "+00:00")
                ).timestamp()
                if until_ts > now:
                    _PAUSED_UNTIL[strategy] = until_ts
                    logger.info(
                        "strategy_throttle: restored pause "
                        "strategy=%s until=%s",
                        strategy, until_iso,
                    )
            except Exception:
                pass
        for strategy, timestamps in (data.get("entry_timestamps") or {}).items():
            try:
                valid = [float(t) for t in timestamps if float(t) > now - 7200]
                if valid:
                    _ENTRY_TIMESTAMPS[strategy] = valid
            except Exception:
                pass
    except Exception as e:
        logger.debug("strategy_throttle: state_load_failed err=%s", e)


_load_throttle_state()


class ThrottleLevel(str, Enum):
    ALLOW = "ALLOW"
    THROTTLE = "THROTTLE"
    CONFIDENCE_UPSHIFT = "CONFIDENCE_UPSHIFT"
    PAUSE_TEMPORARILY = "PAUSE_TEMPORARILY"
    HALT_DEFER_TO_EDGE_DECAY = "HALT_DEFER_TO_EDGE_DECAY"


@dataclass
class ThrottleDecision:
    level: ThrottleLevel
    strategy: str
    reason: str
    confidence_floor: float = 0.0
    pause_until_utc: Optional[str] = None
    win_rate_today: float = 0.0
    pnl_today: float = 0.0
    trades_today: int = 0
    loss_streak: int = 0


@dataclass
class StrategyStats:
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    loss_streak: int = 0
    win_rate_today: float = 0.5

    @property
    def has_enough_data(self) -> bool:
        return self.trades_today >= MIN_TRADES_TO_EVALUATE

    @property
    def is_winning(self) -> bool:
        return (
            self.win_rate_today >= WIN_RATE_HEALTHY
            and self.pnl_today >= MIN_PNL_HEALTHY
        )


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_strategy_stats_today() -> Dict[str, StrategyStats]:
    """
    Load today's closed trade stats per strategy from trade history.
    Returns dict of strategy_name -> StrategyStats.
    """
    stats: Dict[str, StrategyStats] = {}
    today = datetime.now(timezone.utc).date()

    try:
        trade_dir = DATA / "trades"
        for fname in sorted(trade_dir.glob("trade_history_*.ndjson"))[-3:]:
            try:
                with open(fname, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        payload = rec.get("payload", rec)
                        if not isinstance(payload, dict):
                            continue

                        exit_time = payload.get("exit_time_utc", "")
                        if not exit_time:
                            continue
                        try:
                            exit_dt = datetime.fromisoformat(
                                str(exit_time).replace("Z", "+00:00")
                            )
                            if exit_dt.date() != today:
                                continue
                        except Exception:
                            continue

                        strategy = str(
                            payload.get("strategy", "")
                        ).lower().strip()
                        if not strategy or strategy == "broker_sync":
                            continue

                        pnl = float(payload.get("pnl", 0.0) or 0.0)
                        if strategy not in stats:
                            stats[strategy] = StrategyStats()

                        s = stats[strategy]
                        s.trades_today += 1
                        s.pnl_today += pnl
                        if pnl > 0:
                            s.wins_today += 1
                            s.loss_streak = 0
                        else:
                            s.losses_today += 1
                            s.loss_streak += 1

            except Exception:
                continue

    except Exception as e:
        logger.debug("strategy_throttle: stats_load_failed err=%s", e)

    for s in stats.values():
        if s.trades_today > 0:
            s.win_rate_today = s.wins_today / s.trades_today

    return stats


def _check_reconciliation_ok() -> bool:
    try:
        d = _read_json(RUNTIME / "reconciliation_state.json")
        return str(d.get("status", "")).upper() == "GREEN"
    except Exception:
        return True


def _check_profit_lock_ok() -> bool:
    try:
        d = _read_json(RUNTIME / "profit_lock_state.json")
        mode = str(d.get("mode", "NORMAL")).upper()
        return mode in ("NORMAL", "WARN")
    except Exception:
        return True


def _is_blocking_signal(signal: Any) -> bool:
    """
    Return True if signal is an exit, stop-loss, hedge, or risk reduction.
    These are NEVER throttled regardless of strategy performance.
    """
    meta = getattr(signal, "meta", {}) or {}
    if isinstance(meta, dict):
        tags = meta.get("tags", meta.get("signal_tags", []))
        if isinstance(tags, list):
            blocking = {
                "exit", "stop_loss", "hedge", "liquidation",
                "close", "risk_reduction", "reconciliation_repair",
                "hedging",
            }
            if any(str(t).lower() in blocking for t in tags):
                return True
        if meta.get("exit") or meta.get("reason") == "max_hold_exit":
            return True

    direct_tags = getattr(signal, "tags", None)
    if direct_tags:
        blocking = {
            "exit", "stop_loss", "hedge", "liquidation",
            "close", "risk_reduction", "reconciliation_repair",
            "hedging",
        }
        try:
            if any(str(t).lower() in blocking for t in direct_tags):
                return True
        except Exception:
            pass

    side = str(getattr(signal, "side", "") or "")
    if hasattr(getattr(signal, "side", None), "value"):
        side = str(signal.side.value)
    side = side.upper()

    symbol = str(getattr(signal, "symbol", "") or "").upper()
    strategy = getattr(signal, "strategy", "")
    if hasattr(strategy, "value"):
        strategy = strategy.value
    strategy = str(strategy).lower()

    # SELL on existing long = closing position; treat as protective
    if side == "SELL":
        try:
            guard = _read_json(RUNTIME / "position_guard.json")
            key = f"{strategy}|{symbol}"
            entry = guard.get(key, {})
            if isinstance(entry, dict) and entry.get("open"):
                existing_side = str(entry.get("side", "")).upper()
                if existing_side == "BUY":
                    return True
        except Exception:
            pass

    return False


def _record_entry(strategy: str) -> None:
    """Record a new entry timestamp for time-window tracking."""
    now = time.time()
    if strategy not in _ENTRY_TIMESTAMPS:
        _ENTRY_TIMESTAMPS[strategy] = []
    cutoff = now - 7200
    _ENTRY_TIMESTAMPS[strategy] = [
        t for t in _ENTRY_TIMESTAMPS[strategy] if t > cutoff
    ]
    _ENTRY_TIMESTAMPS[strategy].append(now)
    try:
        _save_throttle_state()
    except Exception:
        pass


def _entries_in_window(strategy: str, window_seconds: int) -> int:
    now = time.time()
    cutoff = now - window_seconds
    timestamps = _ENTRY_TIMESTAMPS.get(strategy, [])
    return sum(1 for t in timestamps if t > cutoff)


def _is_paused(strategy: str) -> Tuple[bool, Optional[str]]:
    until = _PAUSED_UNTIL.get(strategy, 0.0)
    if until > time.time():
        dt = datetime.fromtimestamp(until, tz=timezone.utc)
        return True, dt.isoformat()
    return False, None


def _set_pause(strategy: str, duration_minutes: int) -> str:
    until = time.time() + (duration_minutes * 60)
    _PAUSED_UNTIL[strategy] = until
    dt = datetime.fromtimestamp(until, tz=timezone.utc)
    try:
        _save_throttle_state()
    except Exception:
        pass
    return dt.isoformat()


def evaluate_signal(
    signal: Any,
    strategy_stats: Dict[str, StrategyStats],
    recon_ok: bool,
    profit_lock_ok: bool,
) -> ThrottleDecision:
    """Evaluate throttle level for a single signal."""
    strategy = getattr(signal, "strategy", "")
    if hasattr(strategy, "value"):
        strategy = strategy.value
    strategy = str(strategy).lower().strip()

    # Exits and protective signals are NEVER throttled
    if _is_blocking_signal(signal):
        return ThrottleDecision(
            level=ThrottleLevel.ALLOW,
            strategy=strategy,
            reason="exit_or_protective_signal_never_throttled",
        )

    stats = strategy_stats.get(strategy, StrategyStats())

    # Not enough data → ALLOW
    if not stats.has_enough_data:
        return ThrottleDecision(
            level=ThrottleLevel.ALLOW,
            strategy=strategy,
            reason=f"insufficient_data_{stats.trades_today}_trades",
            trades_today=stats.trades_today,
        )

    # Winning strategy → ALLOW
    if stats.is_winning and recon_ok and profit_lock_ok:
        return ThrottleDecision(
            level=ThrottleLevel.ALLOW,
            strategy=strategy,
            reason="winning_strategy_unrestricted",
            win_rate_today=stats.win_rate_today,
            pnl_today=stats.pnl_today,
            trades_today=stats.trades_today,
        )

    # Active pause window
    paused, pause_until = _is_paused(strategy)
    if paused:
        return ThrottleDecision(
            level=ThrottleLevel.PAUSE_TEMPORARILY,
            strategy=strategy,
            reason="active_pause_window",
            pause_until_utc=pause_until,
            win_rate_today=stats.win_rate_today,
            pnl_today=stats.pnl_today,
            trades_today=stats.trades_today,
            loss_streak=stats.loss_streak,
        )

    # Level 4: Defer to edge decay
    if stats.loss_streak >= EDGE_DECAY_LOSS_STREAK:
        return ThrottleDecision(
            level=ThrottleLevel.HALT_DEFER_TO_EDGE_DECAY,
            strategy=strategy,
            reason=f"loss_streak_{stats.loss_streak}_defer_edge_decay",
            win_rate_today=stats.win_rate_today,
            pnl_today=stats.pnl_today,
            trades_today=stats.trades_today,
            loss_streak=stats.loss_streak,
        )

    # Level 3: PAUSE
    if (
        stats.loss_streak >= LOSS_STREAK_FOR_PAUSE
        or stats.win_rate_today <= PAUSE_WIN_RATE
    ):
        pause_until_iso = _set_pause(strategy, PAUSE_DURATION_MINUTES)
        logger.warning(
            "strategy_throttle PAUSE_TEMPORARILY strategy=%s "
            "win_rate=%.2f loss_streak=%d pnl=%.2f pause_until=%s",
            strategy, stats.win_rate_today, stats.loss_streak,
            stats.pnl_today, pause_until_iso,
        )
        return ThrottleDecision(
            level=ThrottleLevel.PAUSE_TEMPORARILY,
            strategy=strategy,
            reason=(
                f"win_rate_{stats.win_rate_today:.2f}_or_streak_"
                f"{stats.loss_streak}"
            ),
            pause_until_utc=pause_until_iso,
            win_rate_today=stats.win_rate_today,
            pnl_today=stats.pnl_today,
            trades_today=stats.trades_today,
            loss_streak=stats.loss_streak,
        )

    # Level 2: CONFIDENCE_UPSHIFT
    if stats.win_rate_today <= CONFIDENCE_UPSHIFT_WIN_RATE:
        signal_confidence = float(
            getattr(signal, "confidence", 0.5) or 0.5
        )
        required = 0.5 + CONFIDENCE_UPSHIFT_DELTA
        if signal_confidence < required:
            return ThrottleDecision(
                level=ThrottleLevel.CONFIDENCE_UPSHIFT,
                strategy=strategy,
                reason=(
                    f"win_rate_{stats.win_rate_today:.2f}_confidence_"
                    f"{signal_confidence:.2f}_below_{required:.2f}"
                ),
                confidence_floor=required,
                win_rate_today=stats.win_rate_today,
                pnl_today=stats.pnl_today,
                trades_today=stats.trades_today,
                loss_streak=stats.loss_streak,
            )

    # Level 1: THROTTLE (time-window rate limit)
    if stats.win_rate_today <= THROTTLE_WIN_RATE or stats.pnl_today < 0:
        entries_15min = _entries_in_window(strategy, 15 * 60)
        entries_1hr = _entries_in_window(strategy, 3600)

        if (
            entries_15min >= THROTTLE_MAX_ENTRIES_15MIN
            or entries_1hr >= THROTTLE_MAX_ENTRIES_HOUR
        ):
            return ThrottleDecision(
                level=ThrottleLevel.THROTTLE,
                strategy=strategy,
                reason=(
                    f"time_window_exceeded_15min={entries_15min}_"
                    f"1hr={entries_1hr}"
                ),
                win_rate_today=stats.win_rate_today,
                pnl_today=stats.pnl_today,
                trades_today=stats.trades_today,
                loss_streak=stats.loss_streak,
            )

    return ThrottleDecision(
        level=ThrottleLevel.ALLOW,
        strategy=strategy,
        reason="no_throttle_condition_met",
        win_rate_today=stats.win_rate_today,
        pnl_today=stats.pnl_today,
        trades_today=stats.trades_today,
    )


def run_throttle_gate(
    signals: List[Any],
) -> Tuple[List[Any], List[ThrottleDecision]]:
    """
    Run the Smart Strategy Throttle Gate over a list of signals.

    Returns:
        (allowed_signals, all_decisions)

    Winning strategies pass through unaffected.
    Losing strategies are progressively throttled.
    Exits and protective signals always pass.
    Gate failure always returns all signals (fail-open).
    """
    if not signals:
        return signals, []

    try:
        strategy_stats = _load_strategy_stats_today()
        recon_ok = _check_reconciliation_ok()
        profit_lock_ok = _check_profit_lock_ok()
    except Exception as e:
        logger.warning(
            "strategy_throttle: context_load_failed err=%s — ALLOW all", e
        )
        return signals, []

    allowed: List[Any] = []
    decisions: List[ThrottleDecision] = []
    throttle_counts: Dict[str, int] = {}

    for signal in signals:
        try:
            decision = evaluate_signal(
                signal=signal,
                strategy_stats=strategy_stats,
                recon_ok=recon_ok,
                profit_lock_ok=profit_lock_ok,
            )
            decisions.append(decision)

            level = decision.level

            if level == ThrottleLevel.ALLOW:
                allowed.append(signal)
                _record_entry(decision.strategy)

            elif level == ThrottleLevel.THROTTLE:
                throttle_counts[decision.strategy] = (
                    throttle_counts.get(decision.strategy, 0) + 1
                )
                logger.info(
                    "strategy_throttle THROTTLE strategy=%s "
                    "win_rate=%.2f pnl=%.2f reason=%s",
                    decision.strategy, decision.win_rate_today,
                    decision.pnl_today, decision.reason,
                )

            elif level == ThrottleLevel.CONFIDENCE_UPSHIFT:
                logger.info(
                    "strategy_throttle CONFIDENCE_UPSHIFT strategy=%s "
                    "floor=%.2f reason=%s",
                    decision.strategy, decision.confidence_floor,
                    decision.reason,
                )

            elif level == ThrottleLevel.PAUSE_TEMPORARILY:
                logger.warning(
                    "strategy_throttle PAUSED strategy=%s until=%s",
                    decision.strategy, decision.pause_until_utc,
                )

            elif level == ThrottleLevel.HALT_DEFER_TO_EDGE_DECAY:
                logger.warning(
                    "strategy_throttle DEFER_EDGE_DECAY strategy=%s streak=%d",
                    decision.strategy, decision.loss_streak,
                )
                # Actually write the halt to strategy_allocations.json so
                # edge-decay enforcement picks it up immediately rather
                # than waiting for the next edge_decay_monitor pass.
                try:
                    from chad.risk.edge_decay_monitor import (
                        set_strategy_halted as _set_strategy_halted,
                    )
                    _set_strategy_halted(
                        decision.strategy,
                        int(decision.loss_streak or 0),
                    )
                    logger.warning(
                        "strategy_throttle HALT_WRITTEN strategy=%s "
                        "streak=%d — written to strategy_allocations.json",
                        decision.strategy, decision.loss_streak,
                    )
                except Exception as _halt_err:
                    logger.warning(
                        "strategy_throttle: halt_write_failed err=%s",
                        _halt_err,
                    )
                # Still allow the signal — edge decay gate will enforce
                allowed.append(signal)

        except Exception as e:
            logger.debug(
                "strategy_throttle: eval_failed err=%s — ALLOW", e
            )
            allowed.append(signal)

    blocked = len(signals) - len(allowed)
    if blocked > 0:
        logger.warning(
            "strategy_throttle summary: %d/%d signals throttled",
            blocked, len(signals),
        )

    return allowed, decisions
