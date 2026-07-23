"""chad/risk/position_exit_overlay.py — position-aware equity exit overlay (SHADOW-first).

Implements Q5(b) of ``ops/pending_actions/EXIT_AUDIT_equity_roundtrip_close_2026-07-13.md``.

Why this exists
---------------
The exit audit established that the strategy-native exits (gamma/alpha/delta/omega/
alpha_options) never fire at runtime because the strategy-facing ``portfolio.positions`` is
hardwired empty (no ``ContextBuilder().build()`` caller passes ``current_positions``), and the
only working close path — the reconciler — is *incidental* (it fires only when a **different**
strategy emits an opposing directional signal on the same symbol). There is no deterministic,
position-aware risk exit for equities. This overlay is that exit, at the portfolio layer.

Design (locked per the PA doc)
------------------------------
- Runs each cycle, reads the SAME authoritative open-position source the reconciler uses
  (``chad.core.position_reconciler.load_open_positions`` → ``runtime/position_guard.json``) —
  never the empty strategy context.
- Exit conditions v1 (config/position_exit_overlay.json, every threshold documented there),
  OR'd, first hit proposes a **reduce-only** close for the full held quantity of that entry:
    (1) ATR trailing stop  — from the data/bars/1d cache; trails the favorable extreme
                             (peak for longs / trough for shorts) by ``atr_trail_mult`` ATRs.
    (2) time-based max-hold — age from the guard ``opened_at``; per-asset-class limit.
    (3) hard stop-loss pct  — fixed backstop off the persisted entry anchor.
- SHADOW mode (default): evaluates every open position every cycle, logs a grep-able
  ``EXIT_OVERLAY_SHADOW verdict=WOULD_CLOSE|HOLD ...`` marker, appends evidence to
  ``data/exit_overlay/exit_overlay_YYYYMMDD.ndjson``, and **closes NOTHING**.
- ACTIVE mode (a separate future authorization per the PA doc's pre-registered criteria):
  emits reduce-only close intents through the SAME submit path as the reconciler
  (``apply_close_intents`` → ``paper_adapter.submit_strategy_trade_intents``), so the closes
  inherit the adapter's RTH gate (``ibkr_adapter._evaluate_rth_gate``), idempotency and
  margin-shadow observation for free — reused, never forked.
- Kill-switch ``CHAD_POSITION_EXIT_OVERLAY`` (OFF → inert entirely; SHADOW; ACTIVE) overrides
  the config ``mode``. Default (config) SHADOW.
- Fail behaviour: a cycle-level error logs ``EXIT_OVERLAY_ERROR`` and submits NOTHING — which
  is fail-open in shadow (nothing would submit anyway) and fail-closed in active (an
  evaluation error proposes no close).
- PHANTOM-AWARE: the overlay evaluates ONLY broker-confirmed positions. A guard entry whose
  ``broker_sync|<symbol>`` truth holds nothing on the same side (the Fault-C phantom class from
  the audit) is skipped with ``EXIT_OVERLAY_SKIP_UNCONFIRMED`` — so the overlay structurally
  cannot reproduce the rejected-close storm.

Isolation: the evaluation core imports only stdlib. The only chad import is the lazy
``apply_close_intents`` reuse inside the ACTIVE branch of the runner (and the default loaders),
so importing this module is light and has no cycle with the adapter.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MODE_OFF",
    "MODE_SHADOW",
    "MODE_ACTIVE",
    "MARKER_SHADOW",
    "MARKER_SKIP",
    "MARKER_ERROR",
    "MARKER_ACTIVE",
    "EVIDENCE_SCHEMA_VERSION",
    "ExitOverlayConfig",
    "ExitOverlayConfigError",
    "ExitOverlayVerdict",
    "ExitOverlayResult",
    "evaluate_positions",
    "PositionExitOverlay",
    "build_default_overlay",
    "load_overlay_config",
    "resolve_mode",
    "default_evidence_dir",
]

LOGGER = logging.getLogger("chad.risk.position_exit_overlay")

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ACTIVE = "active"
_VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

MARKER_SHADOW = "EXIT_OVERLAY_SHADOW"
MARKER_SKIP = "EXIT_OVERLAY_SKIP_UNCONFIRMED"
MARKER_ERROR = "EXIT_OVERLAY_ERROR"
MARKER_ACTIVE = "EXIT_OVERLAY_ACTIVE_CLOSE"
MARKER_HEARTBEAT = "EXIT_OVERLAY_HEARTBEAT"
MARKER_STAND_DOWN = "EXIT_OVERLAY_STAND_DOWN"
MARKER_BACKOFF = "EXIT_OVERLAY_SUBMIT_BACKOFF"

# W3B-6: v2 adds mark provenance (mark_ts_utc / mark_age_s / mark_source) —
# additive only. Before v2 the evidence carried price + eval wall-clock with no
# way to see how stale the mark was; PA_SIM_MARK_freshness_2026-07-20
# documented a 55.3s-stale ref ($1.88/sh divergence on a 273-sh UNH close)
# that was invisible in these records because the loaders discard the mark's
# own timestamp. v2 stamps it; it does not shrink it.
EVIDENCE_SCHEMA_VERSION = "exit_overlay.v2"
HEARTBEAT_SCHEMA_VERSION = "exit_overlay_heartbeat.v1"
CONFIG_SCHEMA_VERSION = "position_exit_overlay_config.v1"
SUBMIT_LEDGER_SCHEMA_VERSION = "position_exit_overlay_submit_ledger.v1"
STAND_DOWN_SCHEMA_VERSION = "exit_overlay_stand_down.v1"
KILL_SWITCH_ENV = "CHAD_POSITION_EXIT_OVERLAY"

# B-5 (FLIP-UNBLOCK 2026-07-17): bounded close-retry governor. ULTRA_CLOSE_AUDIT §B-5 proved a
# standing WOULD_CLOSE re-proposes the SAME close every ~72s cycle (385/session), and the adapter
# idempotency store has two holes — a rejected row is re-INSERTed (retry every cycle, forever) and
# a filled key reclaims after 900s (another close every 15 min). The overlay had NO feedback
# channel and NO cooldown, so a stuck close storms the broker unbounded. The governor caps submits
# per position_key with exponential backoff, then STANDS DOWN permanently for that key and raises a
# one-shot coach alert. Defaults are code-level constants (no config-file mutation).
SUBMIT_MAX_ATTEMPTS = 5
SUBMIT_BACKOFF_BASE_SECONDS = 300.0   # first retry gate: 5 min after the first submit
SUBMIT_BACKOFF_MAX_SECONDS = 3600.0   # cap the doubling interval at 1 hour

# XOV-2345: the overlay's per-position markers/evidence are emitted only when it
# HAS positions, so "evaluating nothing" and "not running at all" looked
# identical for 16h. The heartbeat is written on EVERY cycle in EVERY mode
# (including OFF and the error path) so a stalled watcher is always detectable
# by freshness alone. TTL is deliberately generous vs the ~1min cycle.
HEARTBEAT_TTL_SECONDS = 900
_DEFAULT_EVIDENCE_SUBDIR = ("data", "exit_overlay")
_DEFAULT_HEARTBEAT_RELPATH = ("runtime", "exit_overlay_heartbeat.json")
_DEFAULT_STATE_RELPATH = ("runtime", "position_exit_overlay_state.json")

# Asset classes this equity-scoped overlay acts on. Futures have their own exits
# (alpha_futures._evaluate_exit_signal, ops/micro_eod_flatten); options have the options
# monitor; crypto/forex are out of scope. Anything else is skipped (never closed).
_ACTED_ASSET_CLASSES = ("equity", "etf")

# ETF/ETP tickers — mirror chad/strategies/gamma.py:_asset_class plus the inverse/vol ETPs the
# equity book actually holds (SH/SVXY/PSQ/UVXY). Used only to label asset_class for the
# per-asset-class max-hold and for RTH-gate consistency (equity + etf are both RTH-gated).
_ETF_SYMBOLS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG",
    "SH", "SVXY", "PSQ", "UVXY", "VXX",
})

# Known CHAD futures roots — used only to route futures guard entries to the SKIP path so the
# equity overlay never touches them. (Guard entries do not carry a reliable asset_class field.)
_FUTURES_SYMBOLS = frozenset({
    "MES", "MNQ", "MCL", "MGC", "M6E", "M2K", "SILK6", "ZN", "ZB", "ES", "NQ", "CL", "GC",
})


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class ExitOverlayConfigError(ValueError):
    """Raised for a malformed config; callers fail-open (overlay not wired)."""


@dataclass(frozen=True)
class ExitOverlayConfig:
    """Frozen, validated overlay config. ``mode`` is the sole post-freeze mutable field."""

    mode: str
    atr_period: int
    atr_trail_mult: float
    hard_stop_loss_pct: float
    min_bars_for_atr: int
    max_hold_days: Mapping[str, float]

    @property
    def is_off(self) -> bool:
        return self.mode == MODE_OFF

    @property
    def is_active(self) -> bool:
        return self.mode == MODE_ACTIVE

    def max_hold_for(self, asset_class: str) -> Optional[float]:
        m = self.max_hold_days
        if asset_class in m:
            return float(m[asset_class])
        if "default" in m:
            return float(m["default"])
        return None


def _require_number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExitOverlayConfigError(f"{name}: expected number, got {type(value).__name__}")
    f = float(value)
    if not math.isfinite(f):
        raise ExitOverlayConfigError(f"{name}: number must be finite, got {value!r}")
    if positive and f <= 0.0:
        raise ExitOverlayConfigError(f"{name}: must be > 0, got {f}")
    return f


def load_overlay_config(payload: Mapping[str, Any]) -> ExitOverlayConfig:
    """Validate a config mapping into a frozen ExitOverlayConfig. Raises on any problem."""
    if not isinstance(payload, Mapping):
        raise ExitOverlayConfigError("config: expected an object")

    mode = str(payload.get("mode", MODE_SHADOW)).strip().lower()
    if mode not in _VALID_MODES:
        raise ExitOverlayConfigError(f"mode: expected one of {_VALID_MODES}, got {mode!r}")

    atr_period = int(_require_number(payload.get("atr_period"), "atr_period"))
    min_bars = int(_require_number(payload.get("min_bars_for_atr"), "min_bars_for_atr"))
    atr_trail_mult = _require_number(payload.get("atr_trail_mult"), "atr_trail_mult")
    hard_stop = _require_number(payload.get("hard_stop_loss_pct"), "hard_stop_loss_pct")
    if hard_stop >= 1.0:
        raise ExitOverlayConfigError(f"hard_stop_loss_pct: expected a fraction < 1.0, got {hard_stop}")

    raw_hold = payload.get("max_hold_days")
    if not isinstance(raw_hold, Mapping) or not raw_hold:
        raise ExitOverlayConfigError("max_hold_days: expected a non-empty object")
    max_hold: Dict[str, float] = {}
    for k, v in raw_hold.items():
        max_hold[str(k).strip().lower()] = _require_number(v, f"max_hold_days.{k}")

    return ExitOverlayConfig(
        mode=mode,
        atr_period=atr_period,
        atr_trail_mult=atr_trail_mult,
        hard_stop_loss_pct=hard_stop,
        min_bars_for_atr=min_bars,
        max_hold_days=max_hold,
    )


def resolve_mode(
    config_mode: str,
    env: Optional[Mapping[str, str]] = None,
    *,
    env_var: str = KILL_SWITCH_ENV,
) -> str:
    """Effective mode = env kill-switch (if set to a valid value) overriding the config mode.

    ``env_var`` defaults to the equity lane's switch; the crypto lane (UC1) passes its own
    (``CHAD_CRYPTO_EXIT_OVERLAY``) so the two lanes are independently inertable — killing
    crypto must never require killing equities, or vice versa.
    """
    env = os.environ if env is None else env
    raw = str(env.get(env_var, "") or "").strip().lower()
    if raw in _VALID_MODES:
        return raw
    return config_mode


# --------------------------------------------------------------------------- #
# Verdict + result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExitOverlayVerdict:
    """One position's evaluation outcome. ``verdict`` ∈ {WOULD_CLOSE, HOLD,
    SKIP_UNCONFIRMED, SKIP_NON_EQUITY, SKIP_NO_DATA, SKIP_EXCLUDED}."""

    verdict: str
    reason: str
    position_key: str
    symbol: str
    strategy: str
    side: str
    asset_class: str
    open_qty: float
    close_qty: float
    price: Optional[float]
    atr: Optional[float]
    atr_stop: Optional[float]
    entry_price: Optional[float]
    hard_stop_price: Optional[float]
    peak: Optional[float]
    trough: Optional[float]
    age_days: Optional[float]
    max_hold_days: Optional[float]
    broker_confirmed_qty: float
    ts_utc: str
    # W3B-6 (exit_overlay.v2): mark provenance. None when the verdict carries
    # no price (skips) or the caller supplied no mark metadata.
    mark_ts_utc: Optional[str] = None
    mark_age_s: Optional[float] = None
    mark_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "ts_utc": self.ts_utc,
            "verdict": self.verdict,
            "reason": self.reason,
            "position_key": self.position_key,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "asset_class": self.asset_class,
            "open_qty": self.open_qty,
            "close_qty": self.close_qty,
            "price": self.price,
            "atr": self.atr,
            "atr_stop": self.atr_stop,
            "entry_price": self.entry_price,
            "hard_stop_price": self.hard_stop_price,
            "peak": self.peak,
            "trough": self.trough,
            "age_days": self.age_days,
            "max_hold_days": self.max_hold_days,
            "broker_confirmed_qty": self.broker_confirmed_qty,
            "mark_ts_utc": self.mark_ts_utc,
            "mark_age_s": self.mark_age_s,
            "mark_source": self.mark_source,
        }

    def marker_line(self, mode: str) -> str:
        return (
            f"{MARKER_SHADOW} verdict={self.verdict} reason={self.reason} "
            f"symbol={self.symbol or '-'} strategy={self.strategy or '-'} "
            f"side={self.side or '-'} close_qty={_fmt(self.close_qty)} "
            f"price={_fmt(self.price)} atr_stop={_fmt(self.atr_stop)} "
            f"hard_stop={_fmt(self.hard_stop_price)} age={_fmt(self.age_days)} mode={mode}"
        )


@dataclass(frozen=True)
class ExitOverlayResult:
    """Pure evaluation output: verdicts, would-close intents, and the anchors to persist."""

    verdicts: List[ExitOverlayVerdict] = field(default_factory=list)
    close_intents: List[Dict[str, Any]] = field(default_factory=list)
    updated_anchors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evaluated: bool = True

    @property
    def would_close(self) -> List[ExitOverlayVerdict]:
        return [v for v in self.verdicts if v.verdict == "WOULD_CLOSE"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float]) -> str:
    return "null" if x is None else f"{x:.4f}"


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _iso(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _asset_class(symbol: str) -> str:
    if symbol in _FUTURES_SYMBOLS:
        return "futures"
    if symbol in _ETF_SYMBOLS:
        return "etf"
    return "equity"


def _atr(bars: Sequence[Mapping[str, Any]], period: int) -> Optional[float]:
    """Wilder-style ATR (EMA of true range), last value. None if insufficient valid bars.

    Mirrors the true-range + EMA construction in chad/strategies/gamma.py (_true_range/_atr)
    but is self-contained (no strategy import) so the risk module stays isolated.
    """
    if period < 1 or not bars:
        return None
    window = list(bars)[-(max(period * 3, period + 2)):]
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    for b in window:
        if not isinstance(b, Mapping):
            continue
        h = _f(b.get("high"))
        l = _f(b.get("low"))
        c = _f(b.get("close"))
        if h <= 0 or l <= 0 or c <= 0 or h < l:
            continue
        highs.append(h)
        lows.append(l)
        closes.append(c)
    if len(closes) < period + 1:
        return None
    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    alpha = 2.0 / (period + 1.0)
    e = tr[0]
    for v in tr[1:]:
        e = alpha * v + (1.0 - alpha) * e
    return e if e > 0 else None


def _default_excluded_symbols() -> "frozenset[str]":
    """Operator-owned / non-CHAD symbols the overlay must never propose a close for (B-6).

    Resolved from the reconciler's single-source exclusion set (``_EFFECTIVE_NON_CHAD_SYMBOLS``,
    itself derived from ``config/reconciliation_exclusions.json``) so there is no parallel config
    reader. Fail-open to empty on any resolution failure: the downstream ``apply_close_intents``
    chokepoint (position_reconciler.py) remains the backstop, so a failed import never sells an
    excluded symbol — it only loses the redundant overlay-level layer for that build.
    """
    try:
        from chad.core.position_reconciler import _EFFECTIVE_NON_CHAD_SYMBOLS
        return frozenset(str(s).strip().upper() for s in _EFFECTIVE_NON_CHAD_SYMBOLS)
    except Exception:  # noqa: BLE001 - the chokepoint is the backstop; never fatal
        return frozenset()


def _broker_signed_by_symbol(guard_state: Mapping[str, Any]) -> Dict[str, float]:
    """Signed broker-truth quantity per symbol from ``broker_sync|<symbol>`` guard entries.

    Aggregates the recorded ``quantity`` signed by side REGARDLESS of the ``open`` flag — that
    quantity *is* broker truth. This mirrors chad/core/position_guard._signed_qty and the
    broker-side aggregation in detect_guard_vs_broker_drift_v2 (WKF U3); reimplemented locally
    to keep this risk module free of a core-module import at load time.
    """
    out: Dict[str, float] = {}
    if not isinstance(guard_state, Mapping):
        return out
    for key, entry in guard_state.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if not key.startswith("broker_sync|"):
            continue
        sym = str(entry.get("symbol", "") or "").strip().upper()
        if not sym:
            continue
        qty = abs(_f(entry.get("quantity")))
        side = str(entry.get("side", "") or "").strip().upper()
        signed = -qty if side == "SELL" else qty
        out[sym] = out.get(sym, 0.0) + signed
    return out


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_days(entry: Mapping[str, Any], anchor: Mapping[str, Any], now: datetime) -> Optional[float]:
    opened = _parse_iso(entry.get("opened_at")) or _parse_iso(anchor.get("first_seen_utc"))
    if opened is None:
        return None
    return max(0.0, (now.astimezone(timezone.utc) - opened).total_seconds() / 86400.0)


def _stable_age_days(opened_at: Any, now: datetime) -> Optional[float]:
    """Age in days from a single already-resolved, rebuild-immune open time (B-3).

    Callers resolve ``opened_at`` from the FIFO lot / persisted anchor (never the guard's
    rewritten ``opened_at``) so the max-hold clock cannot be reset by paper_ledger_rebuild.
    """
    opened = _parse_iso(opened_at)
    if opened is None:
        return None
    return max(0.0, (now.astimezone(timezone.utc) - opened).total_seconds() / 86400.0)


def _submit_backoff_required_seconds(attempts: int) -> float:
    """Minimum seconds that must elapse since the last submit before attempt ``attempts+1``.

    Exponential from ``SUBMIT_BACKOFF_BASE_SECONDS``, doubling per prior attempt, capped at
    ``SUBMIT_BACKOFF_MAX_SECONDS``. attempts<=0 → 0 (the first submit is never gated).
    """
    if attempts <= 0:
        return 0.0
    return min(SUBMIT_BACKOFF_BASE_SECONDS * (2.0 ** (attempts - 1)), SUBMIT_BACKOFF_MAX_SECONDS)


def _submit_gate(record: Optional[Mapping[str, Any]], now: datetime) -> Tuple[str, Dict[str, Any]]:
    """Pure bounded-retry decision for ONE position_key's standing close (B-5).

    Given the persisted submit record (or ``None`` on first sight) and ``now``, return
    ``(decision, new_record)`` where ``decision`` ∈ {``"submit"``, ``"backoff"``, ``"stand_down"``}:

      * ``submit``     — allowed this cycle; ``attempts`` incremented, ``last_utc`` stamped.
      * ``backoff``    — inside the exponential cooldown; record unchanged (no submit).
      * ``stand_down`` — attempts ceiling reached (or already stood down); never submits again.

    Deterministic — every wall-clock comparison uses the injected ``now`` (no ambient clock),
    so the whole governor is unit-testable without patching time.
    """
    now_iso = _iso(now)
    attempts = int(record.get("attempts", 0)) if isinstance(record, Mapping) else 0
    first_utc = (record.get("first_utc") if isinstance(record, Mapping) else None) or now_iso
    last_utc = record.get("last_utc") if isinstance(record, Mapping) else None
    stood_down = bool(record.get("stood_down")) if isinstance(record, Mapping) else False
    alerted = bool(record.get("alerted")) if isinstance(record, Mapping) else False
    stood_down_utc = record.get("stood_down_utc") if isinstance(record, Mapping) else None

    new: Dict[str, Any] = {
        "attempts": attempts,
        "first_utc": first_utc,
        "last_utc": last_utc,
        "stood_down": stood_down,
        "alerted": alerted,
    }
    if stood_down_utc:
        new["stood_down_utc"] = stood_down_utc

    # Already stood down — terminal, never submits again for this key.
    if stood_down:
        return "stand_down", new

    # Ceiling reached — transition to stand-down (the caller raises the one-shot coach alert).
    if attempts >= SUBMIT_MAX_ATTEMPTS:
        new["stood_down"] = True
        new["stood_down_utc"] = now_iso
        return "stand_down", new

    # Exponential backoff gate (the first submit, attempts==0, is never gated).
    required = _submit_backoff_required_seconds(attempts)
    if required > 0.0 and last_utc:
        prev = _parse_iso(last_utc)
        if prev is not None:
            elapsed = (now.astimezone(timezone.utc) - prev).total_seconds()
            if elapsed < required:
                return "backoff", new

    # Allowed — count this submit.
    new["attempts"] = attempts + 1
    new["last_utc"] = now_iso
    return "submit", new


# --------------------------------------------------------------------------- #
# Core evaluation (pure)
# --------------------------------------------------------------------------- #
def evaluate_exit_conditions(
    *,
    side: str,
    price: float,
    entry_price: float,
    peak: float,
    trough: float,
    atr: Optional[float],
    age_days: Optional[float],
    max_hold_days: Optional[float],
    config: ExitOverlayConfig,
    prior_atr_stop: Optional[float] = None,
) -> Tuple[Optional[str], float, Optional[float]]:
    """Pure exit-condition kernel shared by the equity and crypto lanes.

    Fixed, documented order; first True wins:
      hard-stop (most urgent / largest loss) -> ATR trailing -> max-hold.

    Returns ``(fired_reason_or_None, hard_stop_price, atr_stop_or_None)`` where the returned
    ``atr_stop`` is the RATCHETED stop actually used to test ``atr_hit`` (see below). Extracted
    verbatim from the equity loop (UC1) so the Kraken lane cannot drift from the equity
    semantics — one kernel, one set of thresholds-per-lane, pinned by both lanes' tests.

    B-3 (FLIP-UNBLOCK 2026-07-17): monotonic trailing stop. A trailing stop must only move in
    the favourable direction — up for a long, down for a short. The raw ``peak - mult*atr``
    LOOSENS when ATR is revised upward (the audit measured the UNH stop drop $11 as ATR rose
    +45%, un-firing 744 standing closes). ``prior_atr_stop`` is the tightest stop persisted so
    far; when supplied the effective stop is ratcheted against it and can never loosen. The
    crypto lane omits the argument (default ``None``) → identical pre-B-3 behaviour. The ratchet
    also HOLDS through a transient ATR gap: if ``atr`` is momentarily unavailable but a prior
    ratchet exists, the stop is retained rather than dropped.
    """
    if side == "BUY":
        hard_stop_price = entry_price * (1.0 - config.hard_stop_loss_pct)
        atr_stop = (peak - config.atr_trail_mult * atr) if atr is not None else None
        if prior_atr_stop is not None:
            atr_stop = prior_atr_stop if atr_stop is None else max(atr_stop, prior_atr_stop)
        hard_hit = price <= hard_stop_price
        atr_hit = atr_stop is not None and price <= atr_stop
    else:  # SELL / short
        hard_stop_price = entry_price * (1.0 + config.hard_stop_loss_pct)
        atr_stop = (trough + config.atr_trail_mult * atr) if atr is not None else None
        if prior_atr_stop is not None:
            atr_stop = prior_atr_stop if atr_stop is None else min(atr_stop, prior_atr_stop)
        hard_hit = price >= hard_stop_price
        atr_hit = atr_stop is not None and price >= atr_stop
    hold_hit = max_hold_days is not None and age_days is not None and age_days >= max_hold_days

    fired: Optional[str] = None
    if hard_hit:
        fired = "hard_stop_loss"
    elif atr_hit:
        fired = "atr_trailing_stop"
    elif hold_hit:
        fired = "max_hold"
    return fired, hard_stop_price, atr_stop


def evaluate_positions(
    *,
    open_positions: Mapping[str, Mapping[str, Any]],
    guard_state: Mapping[str, Any],
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    price_by_symbol: Mapping[str, float],
    anchors: Mapping[str, Mapping[str, Any]],
    config: ExitOverlayConfig,
    now_utc: datetime,
    fifo_truth_by_key: Optional[Mapping[str, Mapping[str, Any]]] = None,
    excluded_symbols: Optional[AbstractSet[str]] = None,
    price_meta_by_symbol: Optional[Mapping[str, Mapping[str, Any]]] = None,
    advice_by_symbol: Optional[Mapping[str, Mapping[str, Any]]] = None,
    advice_mode: str = "off",
) -> ExitOverlayResult:
    """Deterministic core. Given the position/guard/bars/price/anchor snapshot and the config,
    return verdicts, reduce-only close-intent dicts (for WOULD_CLOSE), and the anchors to
    persist. Never performs I/O; never mutates its inputs; never submits anything.

    Reduce-only invariant: a proposed close qty never exceeds the broker-confirmed same-side
    quantity at evaluation time (``min(guard_qty, broker_held_same_side)``).

    B-6 (FLIP-UNBLOCK 2026-07-17): ``excluded_symbols`` are operator-owned / non-CHAD symbols
    (``_EFFECTIVE_NON_CHAD_SYMBOLS``) the overlay must never touch. Pre-B-6 the PA §3 CLAIMED the
    overlay "honors ``_EFFECTIVE_NON_CHAD_SYMBOLS``", but no exclusion term existed in the module —
    exclusions were honoured only accidentally, one layer downstream at the ``apply_close_intents``
    chokepoint, and the phantom guard even confirmed CHAD's position against the operator's own
    broker shares. Skipping here (BEFORE the phantom confirmation and any close intent) makes the
    doc's claim true and adds the second, module-level layer. ``None`` → no exclusions applied.

    B-3 (FLIP-UNBLOCK 2026-07-17): ``fifo_truth_by_key`` supplies a real cost basis and a
    STABLE open time per ``strategy|symbol`` position, read from the FIFO lot book
    (``trade_closer_state.json``). The guard carries no ``entry_price`` field at all, so before
    B-3 the anchor re-seeded ``entry_price = peak = trough = spot`` on every first sight — the
    hard stop was computed off a price that was never an entry. And ``opened_at`` is rewritten
    by ``paper_ledger_rebuild`` every ~1–2 days, so ``max_hold`` could never fire. FIFO truth
    fixes both: the cost basis comes from the lot ``fill_price``, and the age from the lot
    ``ts_utc`` (which the rebuild does not touch). Absent FIFO truth the code falls back to the
    prior behaviour so this is a strict superset. ``None`` → the map is empty.

    W4B-3 (J16): ``advice_by_symbol`` is the aggregated exit-advice map
    (``exit_advice.load_advice_by_symbol``) — the D4-dropped native urges, fresh
    and exclusion-filtered. The advice rule runs ONLY when the kernel fires
    nothing (hard_stop -> atr_trail -> max_hold keep absolute precedence), only
    for long (BUY-side) positions, and only when the strategy that OWNS the leg
    is among the advising strategies (D6: "the strategy that opened it wants
    out"). ``advice_mode``:
      off     -> rule skipped entirely (byte-identical pre-W4B-3);
      record  -> a firing emits an ADVICE_WOULD_CLOSE verdict (evidence only —
                 NO close intent, nothing submitted even in ACTIVE);
      consume -> a firing emits WOULD_CLOSE + a reduce-only close intent
                 (reason ``exit_overlay_strategy_advice``) through the same
                 clamp/governor/backstops as every overlay close. The flip to
                 consume is a SEPARATE operator GO (D6 rider).
    """
    ts = _iso(now_utc)
    fifo_truth = fifo_truth_by_key or {}
    broker_signed = _broker_signed_by_symbol(guard_state)
    verdicts: List[ExitOverlayVerdict] = []
    close_intents: List[Dict[str, Any]] = []
    updated_anchors: Dict[str, Dict[str, Any]] = {}

    for key, entry in sorted(open_positions.items()):
        if not isinstance(key, str) or key.startswith("_") or key.startswith("broker_sync|"):
            continue
        if not isinstance(entry, dict) or not entry.get("open"):
            continue

        symbol = str(entry.get("symbol", "") or "").strip().upper()
        side = str(entry.get("side", "") or "").strip().upper()
        if not symbol or side not in ("BUY", "SELL"):
            continue
        open_qty = abs(_f(entry.get("quantity")))
        strategy = key.split("|", 1)[0] if "|" in key else str(entry.get("strategy", "") or "")
        asset_class = _asset_class(symbol)

        def _mk(verdict: str, reason: str, **kw: Any) -> ExitOverlayVerdict:
            base: Dict[str, Any] = dict(
                verdict=verdict, reason=reason, position_key=key, symbol=symbol,
                strategy=strategy, side=side, asset_class=asset_class, open_qty=open_qty,
                close_qty=0.0, price=None, atr=None, atr_stop=None, entry_price=None,
                hard_stop_price=None, peak=None, trough=None, age_days=None,
                max_hold_days=config.max_hold_for(asset_class),
                broker_confirmed_qty=0.0, ts_utc=ts,
            )
            base.update(kw)
            return ExitOverlayVerdict(**base)

        # B-6: operator-owned / non-CHAD symbols are skipped FIRST — before the phantom
        # confirmation reads broker truth and before any close intent. The overlay never
        # proposes a close for them and never treats the operator's inventory as confirmation.
        if excluded_symbols and symbol in excluded_symbols:
            verdicts.append(_mk("SKIP_EXCLUDED", "operator_excluded_symbol"))
            continue

        # Scope guard — equity/ETF only. Futures/options/other skipped (they own their exits).
        if asset_class not in _ACTED_ASSET_CLASSES:
            verdicts.append(_mk("SKIP_NON_EQUITY", f"asset_class={asset_class}"))
            continue

        # PHANTOM-AWARE broker confirmation — the Fault-C guard against a rejected-close storm.
        signed = broker_signed.get(symbol, 0.0)
        broker_held = signed if side == "BUY" else -signed
        if broker_held <= 0.0:
            verdicts.append(_mk("SKIP_UNCONFIRMED", "no_broker_confirmed_same_side_qty",
                                broker_confirmed_qty=max(0.0, broker_held)))
            continue

        price = _f(price_by_symbol.get(symbol))
        bars = bars_by_symbol.get(symbol) or []
        # W3B-6: mark provenance travels with the price it describes.
        meta = (price_meta_by_symbol or {}).get(symbol) or {}
        mark_source: Optional[str] = str(meta.get("source")) if (price > 0.0 and meta.get("source")) else None
        mark_ts_utc: Optional[str] = str(meta.get("ts_utc")) if (price > 0.0 and meta.get("ts_utc")) else None
        if price <= 0.0:
            # last-resort: most-recent valid bar close — up to a day stale, so
            # it must never masquerade as a live mark (mark_source says so).
            for b in reversed(list(bars)):
                c = _f(b.get("close")) if isinstance(b, Mapping) else 0.0
                if c > 0.0:
                    price = c
                    mark_source = "bar_close_fallback"
                    raw_bar_ts = b.get("ts_utc") or b.get("date")
                    mark_ts_utc = str(raw_bar_ts) if raw_bar_ts else None
                    break
        if price <= 0.0:
            verdicts.append(_mk("SKIP_NO_DATA", "no_price", broker_confirmed_qty=broker_held))
            continue
        mark_age_s: Optional[float] = None
        if mark_ts_utc:
            mark_dt = _parse_iso(mark_ts_utc)
            if mark_dt is not None:
                mark_age_s = round((now_utc - mark_dt).total_seconds(), 1)

        # Anchor (entry/peak/trough) — seed on first sight, otherwise trail the extreme.
        prior = anchors.get(key) if isinstance(anchors.get(key), Mapping) else None
        fifo = fifo_truth.get(key) if isinstance(fifo_truth.get(key), Mapping) else None

        # entry_price: a persisted anchor wins (stable across cycles once set); else the FIFO
        # cost basis (real paid price); else spot as a last resort (documented fallback — the
        # guard has no price field, so this is all that is knowable for a lot with no FIFO row).
        entry_price = _f(prior.get("entry_price")) if prior else 0.0
        entry_price_source = "anchor" if entry_price > 0.0 else ""
        if entry_price <= 0.0 and fifo:
            entry_price = _f(fifo.get("entry_price"))
            if entry_price > 0.0:
                entry_price_source = "fifo_cost_basis"
        if entry_price <= 0.0:
            entry_price = price
            entry_price_source = "spot_fallback"

        peak = max(_f(prior.get("peak")) if prior else 0.0, price)
        trough_raw = _f(prior.get("trough")) if prior else 0.0
        trough = min(trough_raw, price) if trough_raw > 0.0 else price
        first_seen = (prior.get("first_seen_utc") if prior else None) or ts

        # Stable open time for max-hold: FIFO lot ts_utc (authoritative, rebuild-immune) >
        # a previously-persisted stable time > the anchor first_seen. The guard's own
        # ``opened_at`` is deliberately NOT preferred — paper_ledger_rebuild rewrites it every
        # 1–2 days, which is exactly why max_hold could never fire pre-B-3.
        opened_at = (
            (fifo.get("opened_at") if fifo else None)
            or (prior.get("opened_at_utc") if prior else None)
            or first_seen
        )

        max_hold = config.max_hold_for(asset_class)
        atr = _atr(bars, config.atr_period) if len(list(bars)) >= config.min_bars_for_atr else None
        age = _stable_age_days(opened_at, now_utc)
        prior_ratchet = _f(prior.get("atr_stop_ratchet")) if prior else 0.0

        fired, hard_stop_price, atr_stop = evaluate_exit_conditions(
            side=side, price=price, entry_price=entry_price, peak=peak, trough=trough,
            atr=atr, age_days=age, max_hold_days=max_hold, config=config,
            prior_atr_stop=(prior_ratchet if prior_ratchet > 0.0 else None),
        )

        anchor = {
            "entry_price": entry_price,
            "entry_price_source": entry_price_source,
            "peak": peak,
            "trough": trough,
            "first_seen_utc": first_seen,
            "opened_at_utc": opened_at,
            "last_seen_utc": ts,
            # Persist the ratcheted stop so it survives to the next cycle and can only tighten.
            "atr_stop_ratchet": atr_stop if atr_stop is not None else prior_ratchet,
        }
        updated_anchors[key] = anchor

        common = dict(
            price=price, atr=atr, atr_stop=atr_stop, entry_price=entry_price,
            hard_stop_price=hard_stop_price, peak=peak, trough=trough, age_days=age,
            broker_confirmed_qty=broker_held,
            mark_ts_utc=mark_ts_utc, mark_age_s=mark_age_s, mark_source=mark_source,
        )

        if fired is None:
            # W4B-3 (J16): the advice rule — evaluated only when no kernel
            # condition fired, long legs only, owning strategy only. Excluded
            # symbols can never reach here (SKIP_EXCLUDED above) and the
            # aggregator filters them independently (belt-and-braces).
            advice = (advice_by_symbol or {}).get(symbol) if advice_mode in ("record", "consume") else None
            if (
                advice
                and side == "BUY"
                and strategy
                and strategy in (advice.get("strategies") or ())
            ):
                advice_close_qty = min(open_qty, broker_held)
                if advice_close_qty > 0.0:
                    advice_common = dict(common)
                    advice_common["close_qty"] = advice_close_qty
                    if advice_mode == "consume":
                        verdicts.append(_mk("WOULD_CLOSE", "strategy_advice", **advice_common))
                        close_intents.append({
                            "symbol": symbol,
                            "action": "CLOSE",
                            "open_side": side,
                            "close_side": "SELL",
                            "quantity": float(advice_close_qty),
                            "reason": "exit_overlay_strategy_advice",
                            "position_key": key,
                            "strategy": strategy,
                        })
                    else:  # record: evidence only, never an intent
                        verdicts.append(_mk("ADVICE_WOULD_CLOSE", "strategy_advice", **advice_common))
                    continue
            verdicts.append(_mk("HOLD", "no_condition_met", **common))
            continue

        # Reduce-only: never exceed the broker-confirmed same-side qty.
        close_qty = min(open_qty, broker_held)
        if close_qty <= 0.0:
            verdicts.append(_mk("HOLD", "reduce_only_zero_qty", **common))
            continue

        common["close_qty"] = close_qty
        verdicts.append(_mk("WOULD_CLOSE", fired, **common))
        close_side = "SELL" if side == "BUY" else "BUY"
        close_intents.append({
            "symbol": symbol,
            "action": "CLOSE",
            "open_side": side,
            "close_side": close_side,
            "quantity": float(close_qty),
            "reason": f"exit_overlay_{fired}",
            "position_key": key,
            "strategy": strategy,
        })

    return ExitOverlayResult(
        verdicts=verdicts,
        close_intents=close_intents,
        updated_anchors=updated_anchors,
        evaluated=True,
    )


# --------------------------------------------------------------------------- #
# Runner (impure): loaders, evidence, anchor persistence, active submit
# --------------------------------------------------------------------------- #
BarsLoader = Callable[[Sequence[str]], Dict[str, List[Dict[str, Any]]]]
PriceLoader = Callable[[Sequence[str]], Dict[str, float]]
GuardLoader = Callable[[], Dict[str, Any]]
OpenPositionsLoader = Callable[[], Dict[str, Dict[str, Any]]]
# B-3: returns {"strategy|symbol": {"entry_price": float, "opened_at": iso}} from the FIFO
# lot book. Optional — an overlay built without one behaves exactly as pre-B-3.
FifoTruthLoader = Callable[[], Dict[str, Dict[str, Any]]]
# W4B-3 (J16): returns the aggregated fresh-advice map for a given now. Optional —
# an overlay built without one resolves the default aggregator lazily (and the
# aggregator itself is inert unless CHAD_EXIT_ADVICE != off).
AdviceLoader = Callable[[datetime], Dict[str, Dict[str, Any]]]


class PositionExitOverlay:
    """Wires the pure core to disk loaders, evidence, anchor persistence, and (only in ACTIVE)
    the reconciler submit path. ``run_cycle`` never raises into the trading loop."""

    def __init__(
        self,
        config: ExitOverlayConfig,
        *,
        evidence_path: Optional[Path],
        state_path: Optional[Path],
        guard_loader: GuardLoader,
        open_positions_loader: OpenPositionsLoader,
        bars_loader: BarsLoader,
        price_loader: PriceLoader,
        fifo_truth_loader: Optional[FifoTruthLoader] = None,
        heartbeat_path: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
        excluded_symbols: Optional[AbstractSet[str]] = None,
        advice_loader: Optional[AdviceLoader] = None,
    ) -> None:
        self._config = config
        self._evidence_path = evidence_path
        self._heartbeat_path = heartbeat_path
        self._state_path = state_path
        self._guard_loader = guard_loader
        self._open_positions_loader = open_positions_loader
        self._bars_loader = bars_loader
        self._price_loader = price_loader
        self._fifo_truth_loader = fifo_truth_loader
        self._advice_loader = advice_loader
        self._env = os.environ if env is None else env
        self._log = logger or LOGGER
        # B-6: operator-owned / non-CHAD exclusions. An explicit set (tests) wins; otherwise
        # resolved lazily+cached from the reconciler's single-source set on first cycle.
        self._excluded_symbols_override = excluded_symbols
        self._excluded_cache: Optional["frozenset[str]"] = None

    @property
    def mode(self) -> str:
        return resolve_mode(self._config.mode, self._env)

    def _excluded_symbols(self) -> "frozenset[str]":
        if self._excluded_symbols_override is not None:
            return frozenset(self._excluded_symbols_override)
        if self._excluded_cache is None:
            # Under pytest, default to NO exclusions unless a test passes an explicit set — so
            # fixtures may freely use real tickers (BAC/SPY/QQQ) that happen to be operator-owned.
            # Production resolves the reconciler's single-source set. Mirrors the module's
            # _under_active_pytest() leak-guard philosophy (build_default_overlay).
            self._excluded_cache = (
                frozenset() if _under_active_pytest() else _default_excluded_symbols()
            )
        return self._excluded_cache

    def run_cycle(self, paper_adapter: Any, *, now_utc: Optional[datetime] = None) -> ExitOverlayResult:
        mode = self.mode
        now = now_utc or datetime.now(timezone.utc)
        if mode == MODE_OFF:
            # Heartbeat even when OFF: "disabled" must stay distinguishable from
            # "dead". Freshness proves the wiring still runs; `mode` explains why
            # nothing is being evaluated.
            self._safe_heartbeat(mode, now, evaluated=0, would_close=0, healthy=True)
            return ExitOverlayResult(evaluated=False)
        try:
            guard_state = self._guard_loader() or {}
            open_positions = self._open_positions_loader() or {}
            symbols = sorted({
                str(v.get("symbol", "") or "").strip().upper()
                for k, v in open_positions.items()
                if isinstance(v, dict) and isinstance(k, str)
                and not k.startswith("_") and not k.startswith("broker_sync|")
            } - {""})
            bars = self._bars_loader(symbols) if symbols else {}
            # W3B-6: a loader may return (prices, meta) — meta carries per-symbol
            # mark provenance ({sym: {"ts_utc":..., "source":...}}). Plain-dict
            # loaders (all pre-v2 injections) keep working; their marks are
            # simply unstamped.
            loaded = self._price_loader(symbols) if symbols else {}
            if isinstance(loaded, tuple) and len(loaded) == 2:
                prices, price_meta = (loaded[0] or {}), (loaded[1] or {})
            else:
                prices, price_meta = (loaded or {}), {}
            anchors = self._load_anchors()
            fifo_truth = {}
            if self._fifo_truth_loader is not None:
                try:
                    fifo_truth = self._fifo_truth_loader() or {}
                except Exception:  # noqa: BLE001 - FIFO truth is an enrichment, never fatal
                    fifo_truth = {}

            # W4B-3 (J16): advice is loaded only when CHAD_EXIT_ADVICE != off.
            # A loader failure degrades to no-advice — never to a broken cycle.
            advice_mode = "off"
            advice_map: Dict[str, Dict[str, Any]] = {}
            try:
                from chad.core.exit_advice import (
                    load_advice_by_symbol,
                    resolve_exit_advice_mode,
                )
                advice_mode = resolve_exit_advice_mode(self._env)
                if advice_mode != "off":
                    if self._advice_loader is not None:
                        advice_map = self._advice_loader(now) or {}
                    else:
                        advice_map = load_advice_by_symbol(
                            now=now, excluded_symbols=self._excluded_symbols(),
                        ) or {}
            except Exception:  # noqa: BLE001 - advice is an enrichment, never fatal
                advice_mode, advice_map = "off", {}

            result = evaluate_positions(
                open_positions=open_positions,
                guard_state=guard_state,
                bars_by_symbol=bars,
                price_by_symbol=prices,
                price_meta_by_symbol=price_meta,
                anchors=anchors,
                config=self._config,
                now_utc=now,
                fifo_truth_by_key=fifo_truth,
                excluded_symbols=self._excluded_symbols(),
                advice_by_symbol=advice_map,
                advice_mode=advice_mode,
            )
        except Exception as exc:  # noqa: BLE001 - never fatal; submits nothing on error
            self._safe_error(exc)
            self._safe_heartbeat(mode, now, evaluated=0, would_close=0, healthy=False)
            return ExitOverlayResult(evaluated=False)

        # Observability (best-effort; a marker/evidence failure never affects submission).
        for v in result.verdicts:
            self._safe_marker(v, mode)
            self._safe_write_evidence(v)
        # B-3: every position that produced a verdict this cycle is still in the book
        # (WOULD_CLOSE/HOLD/SKIP_* all iterate live guard entries) — that is the live-key set
        # the merge prunes against.
        live_keys = frozenset(v.position_key for v in result.verdicts)
        self._save_anchors(result.updated_anchors, live_keys)
        # XOV-2345: unconditional — evaluated=0 is exactly the state that used to
        # be invisible (a live overlay whose position source had gone empty).
        self._safe_heartbeat(
            mode, now,
            evaluated=len(result.verdicts),
            would_close=len(result.close_intents),
            healthy=True,
        )

        # ACTIVE: run the bounded-retry governor + submit reduce-only closes through the
        # reconciler's proven path. Called on EVERY active cycle (even with zero close intents)
        # so the governor can prune ledger records for positions that have since closed (B-5).
        if mode == MODE_ACTIVE:
            self._submit_active(result, paper_adapter, now)
        return result

    # -- active submit -------------------------------------------------------- #
    def _submit_active(self, result: "ExitOverlayResult", paper_adapter: Any, now: datetime) -> None:
        """Gate the cycle's close intents through the bounded-retry governor (B-5), then submit
        the survivors reduce-only. Every step is best-effort; a governor or submit error must
        never propagate into the trading loop."""
        try:
            close_intents = list(result.close_intents or [])
            # would_close keys are the positions still trying to close this cycle. A record whose
            # key is absent (position closed / condition cleared) is pruned so a later genuine
            # close starts fresh. On a whole-book false-flat (no verdicts at all) prune NOTHING —
            # the same empty-book signature B-3 refuses to act on.
            would_close_keys = {
                str(c.get("position_key") or "") for c in close_intents if c.get("position_key")
            }
            book_non_empty = bool(result.verdicts)

            ledger = self._load_submit_ledger()
            if book_non_empty:
                ledger = {k: v for k, v in ledger.items() if k in would_close_keys}

            allowed: List[Dict[str, Any]] = []
            newly_stood_down: List[Dict[str, Any]] = []
            for c in close_intents:
                key = str(c.get("position_key") or "")
                if not key:
                    # No key to govern by — fall back to submitting (reclamp still applies).
                    allowed.append(c)
                    continue
                decision, record = _submit_gate(ledger.get(key), now)
                if decision == "submit":
                    ledger[key] = record
                    allowed.append(c)
                elif decision == "backoff":
                    ledger[key] = record
                    self._log.info(
                        "%s position_key=%s symbol=%s attempts=%s (cooling down — not submitting)",
                        MARKER_BACKOFF, key, c.get("symbol"), record.get("attempts"),
                    )
                else:  # stand_down
                    if not record.get("alerted"):
                        record["alerted"] = True
                        newly_stood_down.append({
                            "position_key": key,
                            "symbol": c.get("symbol"),
                            "strategy": c.get("strategy"),
                            "reason": c.get("reason"),
                            "attempts": record.get("attempts"),
                            "stood_down_utc": record.get("stood_down_utc"),
                        })
                    ledger[key] = record

            self._save_submit_ledger(ledger)
            if newly_stood_down:
                self._safe_stand_down_alert(newly_stood_down, now)

            if not allowed:
                return
            fresh_guard = self._guard_loader() or {}
            intents = _reduce_only_reclamp(allowed, fresh_guard)
            if not intents:
                return
            from chad.core.position_reconciler import apply_close_intents  # lazy: reuse, don't fork
            for c in intents:
                self._log.info(
                    "%s symbol=%s strategy=%s side=%s qty=%s reason=%s",
                    MARKER_ACTIVE, c.get("symbol"), c.get("strategy"),
                    c.get("close_side"), c.get("quantity"), c.get("reason"),
                )
            apply_close_intents(intents, paper_adapter)
        except Exception as exc:  # noqa: BLE001 - a submit error must not stop the loop
            self._safe_error(exc)

    # -- submit governor state (B-5) ----------------------------------------- #
    def _submit_ledger_path(self) -> Optional[Path]:
        """Sibling of the anchor state file — keeps the retry ledger out of B-3's anchor payload
        so ``_save_anchors`` stays untouched. ``None`` when the overlay has no state path (the
        governor then degrades to in-cycle-only: still bounded per cycle, just not across a
        process restart)."""
        if self._state_path is None:
            return None
        return self._state_path.with_name(self._state_path.stem + "_submit_ledger.json")

    def _load_submit_ledger(self) -> Dict[str, Dict[str, Any]]:
        path = self._submit_ledger_path()
        if path is None or not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            led = raw.get("ledger") if isinstance(raw, dict) else None
            if not isinstance(led, dict):
                return {}
            return {k: dict(v) for k, v in led.items() if isinstance(k, str) and isinstance(v, dict)}
        except Exception:  # noqa: BLE001 - a corrupt ledger must never stop a cycle
            return {}

    def _save_submit_ledger(self, ledger: Mapping[str, Mapping[str, Any]]) -> None:
        path = self._submit_ledger_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            payload = {
                "schema_version": SUBMIT_LEDGER_SCHEMA_VERSION,
                "ledger": {k: dict(v) for k, v in ledger.items()},
            }
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 - ledger persistence is best-effort
            pass

    def _safe_stand_down_alert(self, stood_down: List[Dict[str, Any]], now: datetime) -> None:
        """One-shot coach alert when a standing close hits the attempts ceiling (B-5).

        Emits a loud ``EXIT_OVERLAY_STAND_DOWN`` marker (scanned by health_monitor) and atomically
        publishes ``runtime/exit_overlay_stand_down.json`` so the alert spine can surface it by
        freshness. Best-effort — never raises. Dedupe is handled by the caller via the ledger
        ``alerted`` flag, so this fires exactly once per key."""
        for item in stood_down:
            try:
                self._log.error(
                    "%s position_key=%s symbol=%s reason=%s attempts=%s — STOOD DOWN after "
                    "%d attempts; a standing close could not clear. Operator action required "
                    "(check broker fill / reconciliation; the overlay will not retry this key).",
                    MARKER_STAND_DOWN, item.get("position_key"), item.get("symbol"),
                    item.get("reason"), item.get("attempts"), SUBMIT_MAX_ATTEMPTS,
                )
            except Exception:  # noqa: BLE001
                pass
        if self._state_path is None:
            return
        try:
            alert_path = self._state_path.with_name("exit_overlay_stand_down.json")
            alert_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": STAND_DOWN_SCHEMA_VERSION,
                "ts_utc": _iso(now),
                "max_attempts": SUBMIT_MAX_ATTEMPTS,
                "stood_down": stood_down,
            }
            tmp = alert_path.with_suffix(alert_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, alert_path)
        except Exception:  # noqa: BLE001 - stand-down alert file is best-effort
            pass

    # -- anchors -------------------------------------------------------------- #
    def _load_anchors(self) -> Dict[str, Dict[str, Any]]:
        if self._state_path is None or not self._state_path.is_file():
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            anchors = raw.get("anchors") if isinstance(raw, dict) else None
            return anchors if isinstance(anchors, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _save_anchors(
        self,
        updated_anchors: Mapping[str, Dict[str, Any]],
        live_keys: "frozenset[str]",
    ) -> None:
        """MERGE the freshly-evaluated anchors onto the persisted set (B-3).

        The pre-B-3 code REPLACED the file with ``updated_anchors``, which contains only the
        keys that reached a full evaluation this cycle. Every skip path (SKIP_UNCONFIRMED /
        SKIP_NON_EQUITY / SKIP_NO_DATA) omits its key, so a single skip cycle ERASED the
        anchor and the next cycle re-seeded ``entry = peak = trough = spot`` — the anchor wipe
        the audit proved live twice, both right after a 23:45 broker false-flat.

        Fix, per the audit's own prescription ("MERGE, not replace; prune only on a *confirmed*
        close"):
          * Start from the on-disk anchors so a skipped-but-open position keeps its history.
          * Prune a key only when the book is NON-empty and the key is absent from it (a real
            close). When ``live_keys`` is empty — the exact signature of a whole-book false-flat
            (guard read returned nothing) — prune NOTHING, so the wipe cannot recur.
          * Overlay the fresh anchors last (new peak/ratchet for evaluated positions).
        """
        if self._state_path is None:
            return
        try:
            merged: Dict[str, Dict[str, Any]] = dict(self._load_anchors())
            if live_keys:
                merged = {k: v for k, v in merged.items() if k in live_keys}
            merged.update(updated_anchors)
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            payload = {
                "schema_version": "position_exit_overlay_state.v1",
                "anchors": merged,
            }
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except Exception:  # noqa: BLE001 - anchor persistence is best-effort
            pass

    # -- observability -------------------------------------------------------- #
    def _safe_marker(self, verdict: ExitOverlayVerdict, mode: str) -> None:
        try:
            if verdict.verdict == "SKIP_UNCONFIRMED":
                self._log.warning(
                    "%s symbol=%s strategy=%s side=%s open_qty=%s reason=%s mode=%s",
                    MARKER_SKIP, verdict.symbol, verdict.strategy, verdict.side,
                    verdict.open_qty, verdict.reason, mode,
                    extra={"exit_overlay": verdict.to_dict()},
                )
            else:
                self._log.info(verdict.marker_line(mode), extra={"exit_overlay": verdict.to_dict()})
        except Exception:  # noqa: BLE001
            pass

    def _safe_error(self, exc: Exception) -> None:
        try:
            self._log.error(
                "%s detail=%s: %s (submitting nothing)", MARKER_ERROR, type(exc).__name__, exc
            )
        except Exception:  # noqa: BLE001
            pass

    def _safe_heartbeat(
        self,
        mode: str,
        now: datetime,
        *,
        evaluated: int,
        would_close: int,
        healthy: bool,
    ) -> None:
        """Emit the once-per-cycle liveness proof. Best-effort; never raises.

        Logs ``EXIT_OVERLAY_HEARTBEAT evaluated=N`` and atomically republishes
        ``runtime/exit_overlay_heartbeat.json`` so health_monitor's freshness
        rule (R14) can alert on a stalled overlay.
        """
        try:
            self._log.info(
                "%s evaluated=%d mode=%s would_close=%d healthy=%s",
                MARKER_HEARTBEAT, evaluated, mode, would_close, healthy,
            )
        except Exception:  # noqa: BLE001
            pass
        if self._heartbeat_path is None:
            return
        try:
            payload = {
                "schema_version": HEARTBEAT_SCHEMA_VERSION,
                "ts_utc": _iso(now),
                "ttl_seconds": HEARTBEAT_TTL_SECONDS,
                "mode": mode,
                "evaluated": int(evaluated),
                "would_close": int(would_close),
                "healthy": bool(healthy),
            }
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._heartbeat_path.with_suffix(self._heartbeat_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._heartbeat_path)
        except Exception:  # noqa: BLE001 - heartbeat persistence is best-effort
            pass

    def _safe_write_evidence(self, verdict: ExitOverlayVerdict) -> None:
        path = self._resolve_evidence_path(verdict)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(verdict.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
        except Exception:  # noqa: BLE001 - evidence is best-effort
            pass

    def _resolve_evidence_path(self, verdict: ExitOverlayVerdict) -> Optional[Path]:
        if self._evidence_path is None:
            return None
        base = self._evidence_path
        if base.suffix == ".ndjson":
            return base
        day = verdict.ts_utc[:10].replace("-", "") if verdict.ts_utc else "19700101"
        return base / f"exit_overlay_{day}.ndjson"


def _reduce_only_reclamp(close_intents: List[Dict[str, Any]], guard_state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Re-check the reduce-only invariant against fresh broker truth immediately before submit.

    Clamps each close qty to the broker-confirmed same-side quantity; drops any intent whose
    broker-held qty has since gone to zero (phantom) or whose side no longer matches truth.
    """
    broker_signed = _broker_signed_by_symbol(guard_state)
    out: List[Dict[str, Any]] = []
    for c in close_intents:
        symbol = str(c.get("symbol", "") or "").strip().upper()
        open_side = str(c.get("open_side", "") or "").strip().upper()
        signed = broker_signed.get(symbol, 0.0)
        broker_held = signed if open_side == "BUY" else -signed
        if broker_held <= 0.0:
            LOGGER.warning(
                "%s reclamp_drop symbol=%s open_side=%s broker_held=%s (phantom at submit)",
                MARKER_ERROR, symbol, open_side, broker_held,
            )
            continue
        qty = min(_f(c.get("quantity")), broker_held)
        if qty <= 0.0:
            continue
        clamped = dict(c)
        clamped["quantity"] = float(qty)
        out.append(clamped)
    return out


# --------------------------------------------------------------------------- #
# Default production loaders + factory
# --------------------------------------------------------------------------- #
def default_evidence_dir(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_EVIDENCE_SUBDIR)


def default_heartbeat_path(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_HEARTBEAT_RELPATH)


def _default_state_path(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_STATE_RELPATH)


def _default_bars_loader(repo_root: Path) -> BarsLoader:
    bars_dir = repo_root / "data" / "bars" / "1d"

    def _load(symbols: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for sym in symbols:
            fp = bars_dir / f"{sym}.json"
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            rows = raw.get("bars") if isinstance(raw, dict) else (raw if isinstance(raw, list) else None)
            if isinstance(rows, list):
                out[sym] = [r for r in rows if isinstance(r, dict)]
        return out

    return _load


def _default_price_loader(repo_root: Path) -> PriceLoader:
    cache_path = repo_root / "runtime" / "price_cache.json"

    def _load(symbols: Sequence[str]) -> Dict[str, float]:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        # TTL freshness — mirrors position_reconciler._load_fresh_cache_price.
        ts = _parse_iso(data.get("ts_utc"))
        if ts is not None:
            try:
                ttl = float(data.get("ttl_seconds") or 300.0)
            except (TypeError, ValueError):
                ttl = 300.0
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > ttl:
                return {}
        prices = data.get("prices") if isinstance(data, dict) else None
        if not isinstance(prices, dict):
            return {}
        out: Dict[str, float] = {}
        for sym in symbols:
            v = _f(prices.get(sym))
            if v > 0.0:
                out[sym] = v
        # W3B-6: the cache's ts_utc was parsed for the TTL check and then
        # DISCARDED — the 55s-stale-mark class was invisible. Return it as
        # per-symbol mark provenance (one file-level stamp; the cache has no
        # per-symbol timestamps — that, too, is now honest in the evidence).
        raw_ts = str(data.get("ts_utc") or "")
        meta = {sym: {"ts_utc": raw_ts, "source": "price_cache"} for sym in out} if raw_ts else {}
        return out, meta

    return _load


def _default_guard_loader() -> GuardLoader:
    def _load() -> Dict[str, Any]:
        try:
            from chad.core.position_guard import STATE_PATH
            if not STATE_PATH.is_file():
                return {}
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    return _load


def _default_open_positions_loader() -> OpenPositionsLoader:
    def _load() -> Dict[str, Dict[str, Any]]:
        try:
            from chad.core.position_reconciler import load_open_positions
            return load_open_positions()
        except Exception:  # noqa: BLE001
            return {}

    return _load


def _fifo_truth_from_state(data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build ``{strategy|symbol: {entry_price, opened_at}}`` from a trade_closer_state mapping.

    Cost basis = quantity-weighted mean of the open lots' ``fill_price`` (the true FIFO basis);
    open time = the earliest lot ``ts_utc`` (rebuild-immune). Pure so it is unit-testable
    without touching disk. A row missing usable lots is skipped rather than emitted with a
    zero basis — an absent entry falls the caller back to prior/spot behaviour.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, Mapping):
        return out
    for row in data.get("queues") or []:
        if not isinstance(row, Mapping):
            continue
        strategy = str(row.get("strategy", "") or "").strip()
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if not strategy or not symbol:
            continue
        wsum = 0.0
        qsum = 0.0
        earliest: Optional[str] = None
        earliest_dt: Optional[datetime] = None
        for lot in row.get("lots") or []:
            if not isinstance(lot, Mapping):
                continue
            q = abs(_f(lot.get("quantity")))
            px = _f(lot.get("fill_price"))
            if q <= 0.0 or px <= 0.0:
                continue
            wsum += px * q
            qsum += q
            ts_raw = lot.get("ts_utc")
            ts_dt = _parse_iso(ts_raw)
            if ts_dt is not None and (earliest_dt is None or ts_dt < earliest_dt):
                earliest_dt = ts_dt
                earliest = str(ts_raw)
        if qsum <= 0.0:
            continue
        entry: Dict[str, Any] = {"entry_price": wsum / qsum}
        if earliest:
            entry["opened_at"] = earliest
        out[f"{strategy}|{symbol}"] = entry
    return out


def _default_fifo_truth_loader(repo_root: Path) -> FifoTruthLoader:
    state_path = repo_root / "runtime" / "trade_closer_state.json"

    def _load() -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - missing/corrupt state → no FIFO enrichment
            return {}
        return _fifo_truth_from_state(data)

    return _load


def _under_active_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def build_default_overlay(
    *,
    repo_root: Path,
    config_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    state_path: Optional[Path] = None,
    heartbeat_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
    price_loader: Optional[PriceLoader] = None,
) -> Optional["PositionExitOverlay"]:
    """Build the production overlay from ``config/position_exit_overlay.json``. FAIL-OPEN: on any
    config problem it logs ``EXIT_OVERLAY_ERROR`` and returns ``None`` (overlay not wired → the
    trading loop is byte-identical). Mirrors margin_shadow_gate.build_default_shadow_gate,
    including the leak guard that forbids composing the real ``data/exit_overlay`` /
    ``runtime/position_exit_overlay_state.json`` paths under a running pytest.
    """
    log = logger or LOGGER
    cfg_path = config_path or (repo_root / "config" / "position_exit_overlay.json")
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        config = load_overlay_config(payload)
    except (ExitOverlayConfigError, OSError, ValueError) as exc:
        log.error(
            "%s config_load_failed path=%s detail=%s: %s (fail-open: overlay NOT wired)",
            MARKER_ERROR, cfg_path, type(exc).__name__, exc,
        )
        return None

    if evidence_dir is not None:
        ev: Optional[Path] = evidence_dir
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_overlay: evidence_dir is REQUIRED-explicit under "
            f"pytest — pass evidence_dir=tmp_path (never the real "
            f"{default_evidence_dir(repo_root)} path). [exit-overlay test-write leak guard]"
        )
    else:
        ev = default_evidence_dir(repo_root)

    if state_path is not None:
        st: Optional[Path] = state_path
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_overlay: state_path is REQUIRED-explicit under pytest "
            f"— pass state_path=tmp_path/... (never the real {_default_state_path(repo_root)})."
        )
    else:
        st = _default_state_path(repo_root)

    if heartbeat_path is not None:
        hb: Optional[Path] = heartbeat_path
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_overlay: heartbeat_path is REQUIRED-explicit "
            f"under pytest — pass heartbeat_path=tmp_path/... (never the real "
            f"{default_heartbeat_path(repo_root)}). [exit-overlay test-write leak guard]"
        )
    else:
        hb = default_heartbeat_path(repo_root)

    return PositionExitOverlay(
        config,
        evidence_path=ev,
        heartbeat_path=hb,
        state_path=st,
        guard_loader=_default_guard_loader(),
        open_positions_loader=_default_open_positions_loader(),
        bars_loader=_default_bars_loader(repo_root),
        # W3B-7: an injected loader (portfolio-marks composition, flag-gated in
        # live_loop) overrides the default; None keeps the price_cache loader.
        price_loader=price_loader or _default_price_loader(repo_root),
        fifo_truth_loader=_default_fifo_truth_loader(repo_root),
        env=env,
        logger=log,
    )
