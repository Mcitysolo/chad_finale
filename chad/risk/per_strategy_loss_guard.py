"""
chad/risk/per_strategy_loss_guard.py

GAP-026: per-strategy daily realized PnL guard.

Behaviour summary
-----------------
* Report-only by default — every cycle the guard logs a WARNING for
  any strategy whose current-day realized PnL is at or below its
  configured loss limit. No signals are suppressed.
* When the operator opts in via the env flag
  CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1, the guard suppresses fresh
  entry signals from breached strategies for the rest of the UTC day.
* Exits, reductions, hedges, close intents, flip signals, and
  liquidation actions are NEVER suppressed. The guard's purpose is to
  stop *new risk* on a bleeding strategy — never to trap the book.
* The "today" PnL window is anchored to the active epoch start so a
  contaminated previous epoch (e.g. Delta Epoch 1 LEAN losses) cannot
  trigger Epoch 2 current-day suppression. The window also starts at
  the most recent UTC midnight, so today's losses are isolated even if
  the active epoch began many days ago.

The guard performs no runtime writes. Telemetry lives in the routing
diagnostics artifact and journald.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_PATH = REPO_ROOT / "config" / "per_strategy_loss_limits.json"
EPOCH_STATE_PATH = RUNTIME_DIR / "epoch_state.json"
TRADES_DIR = REPO_ROOT / "data" / "trades"

ENFORCEMENT_ENV = "CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE"

# Tags / intent values that always pass the guard untouched.
PROTECTIVE_TAGS: frozenset = frozenset(
    {
        "exit",
        "close",
        "reduce",
        "hedge",
        "stop_loss",
        "liquidation",
        "risk_reduction",
        "protective",
    }
)
PROTECTIVE_INTENTS: frozenset = frozenset(
    {"exit", "close", "reduce", "hedge"}
)
PROTECTIVE_SIDES: frozenset = frozenset({"EXIT", "CLOSE"})


@dataclass(frozen=True)
class GuardDecision:
    """Result of evaluating one signal/intent."""

    strategy: str
    suppressed: bool
    reason: str
    realized_pnl_today: float
    limit_usd: float
    enforce_mode: bool
    is_protective: bool


@dataclass(frozen=True)
class GuardConfig:
    default_mode: str
    enforcement_env: str
    default_limit_usd: float
    limits_usd: Dict[str, float]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GuardConfig":
        target = Path(path) if path is not None else CONFIG_PATH
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
        limits_raw = obj.get("limits_usd") or {}
        limits = {}
        for k, v in limits_raw.items():
            try:
                limits[str(k).strip().lower()] = float(v)
            except (TypeError, ValueError):
                continue
        try:
            default_limit = float(obj.get("default_limit_usd", -250.0))
        except (TypeError, ValueError):
            default_limit = -250.0
        return cls(
            default_mode=str(
                obj.get("default_mode") or "report_only"
            ).strip().lower(),
            enforcement_env=str(
                obj.get("enforcement_env") or ENFORCEMENT_ENV
            ),
            default_limit_usd=default_limit,
            limits_usd=limits,
        )

    def limit_for(self, strategy: str) -> float:
        s = (strategy or "").strip().lower()
        if s in self.limits_usd:
            return float(self.limits_usd[s])
        return float(self.default_limit_usd)


def is_enforce_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True iff CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE is truthy."""
    src = env if env is not None else os.environ
    raw = str(src.get(ENFORCEMENT_ENV, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_iso(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_epoch_start() -> Optional[datetime]:
    try:
        obj = json.loads(EPOCH_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _parse_iso(obj.get("epoch_started_at_utc"))


def _today_window_start(now: Optional[datetime] = None) -> datetime:
    """
    Return the larger of (UTC midnight today, epoch start). This keeps
    the PnL window strictly within the current UTC day AND within the
    active epoch — so historical losses from a contaminated prior
    epoch can never trigger today's suppression.
    """
    now_utc = now or datetime.now(timezone.utc)
    midnight = datetime(
        now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
    )
    epoch_start = _read_epoch_start()
    if epoch_start is None:
        return midnight
    return max(midnight, epoch_start)


def _trade_pnl(record: Mapping[str, Any]) -> float:
    """Best-effort PnL extraction from a closed-trade record."""
    for key in ("net_pnl", "pnl", "gross_pnl"):
        v = record.get(key)
        try:
            if v is not None and v != "":
                return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _trade_exit_ts(record: Mapping[str, Any]) -> Optional[datetime]:
    for key in ("exit_time_utc", "closed_at_utc", "ts_utc"):
        ts = _parse_iso(record.get(key))
        if ts is not None:
            return ts
    return None


def compute_today_realized_pnl(
    *,
    now: Optional[datetime] = None,
    trades_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Sum realized PnL per strategy from closed-trade records whose
    exit_time_utc falls within the (epoch_start, midnight)-anchored
    today window. Returns empty dict if no records / files missing.
    """
    window_start = _today_window_start(now)
    end = (now or datetime.now(timezone.utc))
    src_dir = Path(trades_dir) if trades_dir is not None else TRADES_DIR
    out: Dict[str, float] = {}
    if not src_dir.is_dir():
        return out

    # Only scan files plausibly within today's window — match by name
    # prefix where possible to keep this O(today's file).
    for path in sorted(src_dir.glob("trade_history_*.ndjson")):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            payload = rec.get("payload", rec)
            if not isinstance(payload, Mapping):
                continue
            ts = _trade_exit_ts(payload)
            if ts is None:
                continue
            if ts < window_start or ts > end:
                continue
            strat = str(payload.get("strategy") or "").strip().lower()
            if not strat:
                continue
            out[strat] = out.get(strat, 0.0) + _trade_pnl(payload)
    return out


def _signal_strategy(sig: Any) -> str:
    try:
        v = getattr(sig, "strategy", None)
        v = getattr(v, "value", v)
        if v is None and isinstance(sig, Mapping):
            v = sig.get("strategy")
        return str(v or "").strip().lower()
    except Exception:
        return ""


def is_protective_signal(sig: Any) -> bool:
    """
    True if the signal/intent is an exit, reduction, hedge, close, or
    other protective action. The guard MUST always allow these — it
    only ever blocks fresh entries.
    """
    try:
        side = str(getattr(sig, "side", "") or "").upper()
        if side in PROTECTIVE_SIDES:
            return True

        intent = getattr(sig, "intent", None)
        if isinstance(intent, str) and intent.lower() in PROTECTIVE_INTENTS:
            return True

        meta = getattr(sig, "meta", None)
        if isinstance(meta, Mapping):
            if meta.get("exit") or meta.get("close") or meta.get("reduce"):
                return True
            if str(meta.get("intent", "")).lower() in PROTECTIVE_INTENTS:
                return True
            if str(meta.get("reason", "")).lower() in {
                "max_hold_exit",
                "stop_loss",
                "liquidation",
                "risk_reduction",
            }:
                return True
            mtags = meta.get("tags") or meta.get("signal_tags") or []
            if any(
                str(t).strip().lower() in PROTECTIVE_TAGS for t in mtags
            ):
                return True

        tags = getattr(sig, "tags", None) or ()
        if any(
            str(t).strip().lower() in PROTECTIVE_TAGS for t in tags
        ):
            return True

        # Mapping-style signals.
        if isinstance(sig, Mapping):
            if (
                str(sig.get("intent", "")).lower() in PROTECTIVE_INTENTS
                or str(sig.get("side", "")).upper() in PROTECTIVE_SIDES
            ):
                return True
            mtags = sig.get("tags") or ()
            if any(
                str(t).strip().lower() in PROTECTIVE_TAGS for t in mtags
            ):
                return True
    except Exception:
        return False
    return False


class PerStrategyLossGuard:
    """
    Stateless per-cycle evaluator. Construct once per cycle, reuse
    `evaluate_signal()` across the routed signal list.

    Default mode: report_only (logs WARNING, suppresses nothing).
    Enforcement: CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1 — fresh entries
    from breached strategies are suppressed; protective actions always
    pass.
    """

    def __init__(
        self,
        *,
        config: Optional[GuardConfig] = None,
        env: Optional[Mapping[str, str]] = None,
        now: Optional[datetime] = None,
        trades_dir: Optional[Path] = None,
        realized_pnl_override: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.config: GuardConfig = config or GuardConfig.load()
        self.enforce: bool = is_enforce_enabled(env)
        self.realized_pnl: Dict[str, float] = (
            dict(realized_pnl_override)
            if realized_pnl_override is not None
            else compute_today_realized_pnl(
                now=now, trades_dir=trades_dir
            )
        )
        # Strategies whose today-window realized PnL is already at or
        # below their loss limit. Reported every cycle; suppressed only
        # when self.enforce is True.
        self.breached: Dict[str, dict] = {}
        for strat, pnl in self.realized_pnl.items():
            limit = self.config.limit_for(strat)
            if pnl <= limit:
                self.breached[strat] = {
                    "realized_pnl_today": float(pnl),
                    "limit_usd": float(limit),
                }

    def report(self, logger: Optional[logging.Logger] = None) -> None:
        """Emit the per-cycle WARNING line for breached strategies."""
        log = logger or LOG
        if not self.breached:
            return
        for strat, info in sorted(self.breached.items()):
            mode = "ENFORCING" if self.enforce else "REPORT_ONLY"
            log.warning(
                "PER_STRATEGY_LOSS_GUARD %s strategy=%s "
                "today_pnl=%.2f limit=%.2f",
                mode,
                strat,
                info["realized_pnl_today"],
                info["limit_usd"],
            )

    def should_suppress(
        self,
        signal: Any,
    ) -> GuardDecision:
        strat = _signal_strategy(signal)
        protective = is_protective_signal(signal)
        pnl = float(self.realized_pnl.get(strat, 0.0))
        limit = self.config.limit_for(strat)
        breached = strat in self.breached

        if protective:
            return GuardDecision(
                strategy=strat,
                suppressed=False,
                reason="protective_action_never_suppressed",
                realized_pnl_today=pnl,
                limit_usd=limit,
                enforce_mode=self.enforce,
                is_protective=True,
            )
        if not breached:
            return GuardDecision(
                strategy=strat,
                suppressed=False,
                reason="within_loss_limit",
                realized_pnl_today=pnl,
                limit_usd=limit,
                enforce_mode=self.enforce,
                is_protective=False,
            )
        # Breached and not protective.
        if not self.enforce:
            return GuardDecision(
                strategy=strat,
                suppressed=False,
                reason="report_only_mode",
                realized_pnl_today=pnl,
                limit_usd=limit,
                enforce_mode=False,
                is_protective=False,
            )
        return GuardDecision(
            strategy=strat,
            suppressed=True,
            reason="loss_limit_breached_enforced",
            realized_pnl_today=pnl,
            limit_usd=limit,
            enforce_mode=True,
            is_protective=False,
        )

    def filter_signals(
        self,
        signals: Iterable[Any],
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[List[Any], List[GuardDecision]]:
        """
        Apply the guard across an iterable of signals/intents.
        Returns (kept_signals, all_decisions). In report-only mode the
        kept list equals the input — only the decisions vary.
        """
        log = logger or LOG
        kept: List[Any] = []
        decisions: List[GuardDecision] = []
        for sig in signals or ():
            d = self.should_suppress(sig)
            decisions.append(d)
            if d.suppressed:
                log.warning(
                    "PER_STRATEGY_LOSS_GUARD suppress_entry "
                    "strategy=%s today_pnl=%.2f limit=%.2f",
                    d.strategy,
                    d.realized_pnl_today,
                    d.limit_usd,
                )
                continue
            kept.append(sig)
        return kept, decisions


__all__ = [
    "ENFORCEMENT_ENV",
    "GuardConfig",
    "GuardDecision",
    "PerStrategyLossGuard",
    "compute_today_realized_pnl",
    "is_enforce_enabled",
    "is_protective_signal",
]
