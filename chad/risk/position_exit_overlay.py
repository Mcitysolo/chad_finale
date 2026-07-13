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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

EVIDENCE_SCHEMA_VERSION = "exit_overlay.v1"
CONFIG_SCHEMA_VERSION = "position_exit_overlay_config.v1"
KILL_SWITCH_ENV = "CHAD_POSITION_EXIT_OVERLAY"

_DEFAULT_EVIDENCE_SUBDIR = ("data", "exit_overlay")
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


def resolve_mode(config_mode: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Effective mode = env kill-switch (if set to a valid value) overriding the config mode."""
    env = os.environ if env is None else env
    raw = str(env.get(KILL_SWITCH_ENV, "") or "").strip().lower()
    if raw in _VALID_MODES:
        return raw
    return config_mode


# --------------------------------------------------------------------------- #
# Verdict + result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExitOverlayVerdict:
    """One position's evaluation outcome. ``verdict`` ∈ {WOULD_CLOSE, HOLD,
    SKIP_UNCONFIRMED, SKIP_NON_EQUITY, SKIP_NO_DATA}."""

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


# --------------------------------------------------------------------------- #
# Core evaluation (pure)
# --------------------------------------------------------------------------- #
def evaluate_positions(
    *,
    open_positions: Mapping[str, Mapping[str, Any]],
    guard_state: Mapping[str, Any],
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    price_by_symbol: Mapping[str, float],
    anchors: Mapping[str, Mapping[str, Any]],
    config: ExitOverlayConfig,
    now_utc: datetime,
) -> ExitOverlayResult:
    """Deterministic core. Given the position/guard/bars/price/anchor snapshot and the config,
    return verdicts, reduce-only close-intent dicts (for WOULD_CLOSE), and the anchors to
    persist. Never performs I/O; never mutates its inputs; never submits anything.

    Reduce-only invariant: a proposed close qty never exceeds the broker-confirmed same-side
    quantity at evaluation time (``min(guard_qty, broker_held_same_side)``).
    """
    ts = _iso(now_utc)
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
        if price <= 0.0:
            # last-resort: most-recent valid bar close
            for b in reversed(list(bars)):
                c = _f(b.get("close")) if isinstance(b, Mapping) else 0.0
                if c > 0.0:
                    price = c
                    break
        if price <= 0.0:
            verdicts.append(_mk("SKIP_NO_DATA", "no_price", broker_confirmed_qty=broker_held))
            continue

        # Anchor (entry/peak/trough) — seed on first sight, otherwise trail the extreme.
        prior = anchors.get(key) if isinstance(anchors.get(key), Mapping) else None
        entry_price = _f(prior.get("entry_price")) if prior else 0.0
        if entry_price <= 0.0:
            entry_price = price
        peak = max(_f(prior.get("peak")) if prior else 0.0, price)
        trough_raw = _f(prior.get("trough")) if prior else 0.0
        trough = min(trough_raw, price) if trough_raw > 0.0 else price
        first_seen = (prior.get("first_seen_utc") if prior else None) or ts
        anchor = {
            "entry_price": entry_price,
            "peak": peak,
            "trough": trough,
            "first_seen_utc": first_seen,
            "last_seen_utc": ts,
        }
        updated_anchors[key] = anchor

        atr = _atr(bars, config.atr_period) if len(list(bars)) >= config.min_bars_for_atr else None
        age = _age_days(entry, anchor, now_utc)
        max_hold = config.max_hold_for(asset_class)

        # Condition evaluation — fixed, documented order; first True wins.
        # Order: hard-stop (most urgent / largest loss) -> ATR trailing -> max-hold.
        if side == "BUY":
            hard_stop_price = entry_price * (1.0 - config.hard_stop_loss_pct)
            atr_stop = (peak - config.atr_trail_mult * atr) if atr is not None else None
            hard_hit = price <= hard_stop_price
            atr_hit = atr_stop is not None and price <= atr_stop
        else:  # SELL / short
            hard_stop_price = entry_price * (1.0 + config.hard_stop_loss_pct)
            atr_stop = (trough + config.atr_trail_mult * atr) if atr is not None else None
            hard_hit = price >= hard_stop_price
            atr_hit = atr_stop is not None and price >= atr_stop
        hold_hit = max_hold is not None and age is not None and age >= max_hold

        fired = None
        if hard_hit:
            fired = "hard_stop_loss"
        elif atr_hit:
            fired = "atr_trailing_stop"
        elif hold_hit:
            fired = "max_hold"

        common = dict(
            price=price, atr=atr, atr_stop=atr_stop, entry_price=entry_price,
            hard_stop_price=hard_stop_price, peak=peak, trough=trough, age_days=age,
            broker_confirmed_qty=broker_held,
        )

        if fired is None:
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
        env: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._evidence_path = evidence_path
        self._state_path = state_path
        self._guard_loader = guard_loader
        self._open_positions_loader = open_positions_loader
        self._bars_loader = bars_loader
        self._price_loader = price_loader
        self._env = os.environ if env is None else env
        self._log = logger or LOGGER

    @property
    def mode(self) -> str:
        return resolve_mode(self._config.mode, self._env)

    def run_cycle(self, paper_adapter: Any, *, now_utc: Optional[datetime] = None) -> ExitOverlayResult:
        mode = self.mode
        if mode == MODE_OFF:
            return ExitOverlayResult(evaluated=False)
        now = now_utc or datetime.now(timezone.utc)
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
            prices = self._price_loader(symbols) if symbols else {}
            anchors = self._load_anchors()

            result = evaluate_positions(
                open_positions=open_positions,
                guard_state=guard_state,
                bars_by_symbol=bars,
                price_by_symbol=prices,
                anchors=anchors,
                config=self._config,
                now_utc=now,
            )
        except Exception as exc:  # noqa: BLE001 - never fatal; submits nothing on error
            self._safe_error(exc)
            return ExitOverlayResult(evaluated=False)

        # Observability (best-effort; a marker/evidence failure never affects submission).
        for v in result.verdicts:
            self._safe_marker(v, mode)
            self._safe_write_evidence(v)
        self._save_anchors(result.updated_anchors)

        # ACTIVE: submit reduce-only closes through the reconciler's proven path.
        if mode == MODE_ACTIVE and result.close_intents:
            self._submit_active(result.close_intents, paper_adapter)
        return result

    # -- active submit -------------------------------------------------------- #
    def _submit_active(self, close_intents: List[Dict[str, Any]], paper_adapter: Any) -> None:
        try:
            fresh_guard = self._guard_loader() or {}
            intents = _reduce_only_reclamp(close_intents, fresh_guard)
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

    def _save_anchors(self, anchors: Mapping[str, Dict[str, Any]]) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            payload = {
                "schema_version": "position_exit_overlay_state.v1",
                "anchors": dict(anchors),
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
        return out

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


def _under_active_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def build_default_overlay(
    *,
    repo_root: Path,
    config_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    state_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
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

    return PositionExitOverlay(
        config,
        evidence_path=ev,
        state_path=st,
        guard_loader=_default_guard_loader(),
        open_positions_loader=_default_open_positions_loader(),
        bars_loader=_default_bars_loader(repo_root),
        price_loader=_default_price_loader(repo_root),
        env=env,
        logger=log,
    )
