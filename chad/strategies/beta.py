#!/usr/bin/env python3
"""
chad/strategies/beta.py

Beta — CHAD's long-term institutional compounder.

Concept
-------
Beta does NOT trade actively. It slowly builds and holds positions in the
most-held large-cap U.S. equities reported across the top institutional
investors (Berkshire, Bridgewater, Renaissance, Citadel, BlackRock,
Vanguard, Appaloosa, Pershing Square) via quarterly SEC 13F filings.

- Target weights come from runtime/institutional_consensus.json, written
  weekly by scripts/update_institutional_consensus.py.
- Signals are emitted only when the current portfolio is materially
  underweight a consensus position.
- At most MAX_SIGNALS_PER_CYCLE signals per handler invocation; the
  live loop runs once per minute so this naturally caps activity.
- Funded by 30% of realized system profits via chad/risk/profit_router.py.

Regime behavior
---------------
Active in every regime — Beta is a long-term holder, not a regime trader.
The regime matrix enables "beta" across trending_bull / trending_bear /
volatile / unknown. The activation layer further drops signals in
"adverse" silently.

Fail-closed surfaces
--------------------
- If runtime/institutional_consensus.json is missing, stale, or empty ->
  handler returns [] and logs a warning.
- If ctx.portfolio or ctx.prices is unavailable -> returns [].
- Never raises from the handler.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from chad.types import (
    AssetClass,
    MarketContext,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSENSUS_PATH = REPO_ROOT / "runtime" / "institutional_consensus.json"


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaParams:
    """Tunables for Beta. Defaults are deliberately conservative."""

    # Consensus staleness cap. 13F cycles quarterly, so 45 days (one quarter's
    # reporting delay) is the soft cap. Past that, fall back to [] signals.
    max_consensus_age_days: int = 45

    # A position counts as "under-weight" if the actual weight is this much
    # below target (absolute, e.g. 0.02 == 2% of account).
    underweight_gap: float = 0.02

    # Max signals per handler call. The live loop runs once per minute.
    max_signals_per_cycle: int = 2

    # At most MAX_SIGNALS_PER_WEEK signals in a rolling 7-day window.
    max_signals_per_week: int = 3

    # Cap each target position at MAX_POSITION_WEIGHT of account equity.
    # Prevents a single large-cap with extreme institutional conviction from
    # dominating the book.
    max_position_weight: float = 0.02

    # Equity floor — do nothing if we have less than this in the account.
    min_equity: float = 5_000.0

    # Once we've held a position for this many days, stop trying to add to it
    # (Beta builds slowly but doesn't forever keep piling in).
    min_days_between_rebalance: int = 7

    # Currency-agnostic asset class heuristic for ETFs vs stocks.
    _etf_symbols: frozenset = field(default_factory=lambda: frozenset({
        "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG",
    }))


DEFAULT_PARAMS = BetaParams()


# ---------------------------------------------------------------------------
# In-memory throttle state
# ---------------------------------------------------------------------------

class _BetaState:
    """
    Rolling per-symbol and per-week signal throttle.

    Kept in-memory (no disk) so process restarts don't block legitimate
    signals, consistent with the other strategies' throttle patterns.
    """

    def __init__(self) -> None:
        # symbol -> last signal timestamp (UTC aware)
        self.last_signal_at: Dict[str, datetime] = {}
        # rolling log of recent signal timestamps, for the weekly cap
        self.recent_signals: List[datetime] = []

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=7)
        self.recent_signals = [t for t in self.recent_signals if t >= cutoff]

    def can_emit_symbol(self, sym: str, now: datetime, gap_days: int) -> bool:
        last = self.last_signal_at.get(sym)
        if last is None:
            return True
        return (now - last) >= timedelta(days=max(1, gap_days))

    def can_emit_weekly(self, now: datetime, cap: int) -> bool:
        self._prune(now)
        return len(self.recent_signals) < max(1, cap)

    def mark(self, sym: str, now: datetime) -> None:
        self.last_signal_at[sym] = now
        self.recent_signals.append(now)


_STATE = _BetaState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _asset_class(sym: str, p: BetaParams) -> AssetClass:
    return AssetClass.ETF if sym in p._etf_symbols else AssetClass.EQUITY


def _get_now(ctx: Any) -> datetime:
    now = getattr(ctx, "now", None)
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _get_equity(portfolio: Any) -> float:
    """Pull total equity from the portfolio snapshot (robust to shape variations)."""
    for attr in ("total_equity", "equity", "net_liq"):
        v = getattr(portfolio, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    extra = getattr(portfolio, "extra", None) or {}
    if isinstance(extra, dict):
        v = extra.get("equity") or extra.get("net_liq")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    cash = getattr(portfolio, "cash", None)
    return float(cash) if isinstance(cash, (int, float)) and cash > 0 else 0.0


def _current_position_weights(
    portfolio: Any,
    prices: Mapping[str, float],
    equity: float,
) -> Dict[str, float]:
    """Return symbol -> weight (notional/equity) for positions > 0."""
    if equity <= 0:
        return {}
    out: Dict[str, float] = {}
    positions = getattr(portfolio, "positions", None) or {}
    if not isinstance(positions, dict):
        return {}
    for sym, pos in positions.items():
        qty = float(getattr(pos, "quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        price = float(prices.get(sym) or getattr(pos, "avg_price", 0.0) or 0.0)
        if price <= 0:
            continue
        out[_norm_sym(sym)] = (qty * price) / equity
    return out


def _load_consensus(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    # Resolve at call time so tests can monkeypatch module-level CONSENSUS_PATH.
    p = path if path is not None else CONSENSUS_PATH
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOG.warning("beta: failed to load consensus file path=%s err=%s", p, exc)
        return None


def _consensus_is_fresh(consensus: Mapping[str, Any], max_age_days: int, now: datetime) -> bool:
    ts = consensus.get("updated_ts_utc")
    if not isinstance(ts, str):
        return False
    try:
        clean = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = now - dt
    return age <= timedelta(days=max(1, max_age_days))


def _prices_map(ctx: Any) -> Dict[str, float]:
    raw = getattr(ctx, "prices", None) or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
            if fv > 0 and fv == fv:
                out[_norm_sym(k)] = fv
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def build_beta_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.BETA,
        enabled=True,
        target_universe=None,  # universe is driven by consensus file, not static
        max_gross_exposure=None,
        notes="Beta (institutional-consensus compounder; SEC 13F driven).",
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def beta_handler(
    ctx: MarketContext,
    params: Optional[BetaParams] = None,
) -> Sequence[TradeSignal]:
    """
    Entry point called by the strategy engine each cycle.

    Strategy: for every symbol in the consensus weights, compute the gap
    between target weight and current portfolio weight. If the gap exceeds
    underweight_gap (2% by default), emit a small BUY sized to fill a
    slice of the gap, subject to per-cycle and weekly caps.

    Fail-closed: returns [] on any missing/stale/empty consensus data.
    """
    p = params or DEFAULT_PARAMS
    now = _get_now(ctx)

    # Portfolio / equity floor
    portfolio = getattr(ctx, "portfolio", None)
    if portfolio is None:
        return []
    equity = _get_equity(portfolio)
    if equity < p.min_equity:
        return []

    # Consensus file
    consensus = _load_consensus()
    if consensus is None:
        LOG.warning("beta: institutional_consensus.json missing — no signals")
        return []
    if not _consensus_is_fresh(consensus, p.max_consensus_age_days, now):
        LOG.warning(
            "beta: institutional_consensus stale (updated_ts_utc=%s, max_age=%dd)",
            consensus.get("updated_ts_utc"), p.max_consensus_age_days,
        )
        return []

    weights = consensus.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        LOG.warning("beta: consensus has no weights — no signals")
        return []

    # Weekly throttle
    if not _STATE.can_emit_weekly(now, p.max_signals_per_week):
        return []

    prices = _prices_map(ctx)
    if not prices:
        LOG.warning("beta: ctx.prices empty — cannot size positions")
        return []

    current_weights = _current_position_weights(portfolio, prices, equity)

    # Rank candidates by gap (target - current) descending; filter to those
    # above underweight_gap AND throttle-ok.
    candidates: List[Dict[str, Any]] = []
    for raw_sym, raw_target in weights.items():
        sym = _norm_sym(raw_sym)
        if not sym or sym.startswith("UNRESOLVED:"):
            continue
        try:
            target = float(raw_target)
        except (TypeError, ValueError):
            continue
        target_capped = min(target, p.max_position_weight)
        current = current_weights.get(sym, 0.0)
        gap = target_capped - current
        if gap <= p.underweight_gap:
            continue
        if not _STATE.can_emit_symbol(sym, now, p.min_days_between_rebalance):
            continue
        if sym not in prices:
            continue
        candidates.append({
            "symbol": sym,
            "target": target_capped,
            "current": current,
            "gap": gap,
            "price": prices[sym],
            "conviction": target,  # raw target as a conviction proxy
        })

    if not candidates:
        return []

    candidates.sort(key=lambda c: (c["conviction"], c["gap"]), reverse=True)

    # Size each BUY to fill ~half the gap. Half-gap sizing is deliberate —
    # Beta builds slowly; one signal never closes the full gap in one shot.
    signals: List[TradeSignal] = []
    for c in candidates[: p.max_signals_per_cycle]:
        fill_notional = 0.5 * c["gap"] * equity
        if fill_notional <= 0:
            continue
        size = fill_notional / c["price"]
        # Round down to integer shares to avoid fractional exec on equities
        size_int = int(size)
        if size_int < 1:
            continue
        sig = TradeSignal(
            strategy=StrategyName.BETA,
            symbol=c["symbol"],
            side=SignalSide.BUY,
            size=float(size_int),
            confidence=float(min(0.95, 0.60 + c["conviction"])),
            asset_class=_asset_class(c["symbol"], p),
            created_at=now,
            meta={
                "reason": "institutional_consensus_rebalance",
                "target_weight": round(c["target"], 4),
                "current_weight": round(c["current"], 4),
                "gap": round(c["gap"], 4),
                "conviction": round(c["conviction"], 4),
                "consensus_updated_ts_utc": consensus.get("updated_ts_utc"),
                "consensus_funds": consensus.get("funds_included") or [],
            },
        )
        signals.append(sig)
        _STATE.mark(c["symbol"], now)
        if not _STATE.can_emit_weekly(now, p.max_signals_per_week):
            break

    return signals


__all__ = [
    "Beta",
    "BetaParams",
    "DEFAULT_PARAMS",
    "build_beta_config",
    "beta_handler",
]


# ---------------------------------------------------------------------------
# Class-shaped facade (matches the original spec's sketch)
# ---------------------------------------------------------------------------

class Beta:
    """
    Thin object-oriented facade over beta_handler.

    The canonical integration point is the handler (wired into the
    StrategyEngine registry). This class exists so tests and ad-hoc tooling
    can instantiate a Beta and drive it without threading a ctx through the
    handler when all they want is "what signals would Beta produce now?".
    """

    SIGNAL_FAMILY = "institutional_consensus"
    STRATEGY_NAME = "beta"
    VENUE = "ibkr"

    def __init__(self, params: Optional[BetaParams] = None) -> None:
        self.params = params or DEFAULT_PARAMS

    def generate_signals(self, ctx: MarketContext) -> Sequence[TradeSignal]:
        return beta_handler(ctx, self.params)

    @staticmethod
    def reset_state() -> None:
        """Clear in-memory throttle state. Test/CLI only."""
        _STATE.last_signal_at.clear()
        _STATE.recent_signals.clear()
