"""chad/validation/backtest_engine.py — Phase 4 Stage-1 historical backtest engine (SSOT §3.5 / Part 2).

Replays a **REPLAYABLE** strategy head's decision logic over the audited daily-bar
corpus into synthetic trades, then feeds those trades through the Phase-1
:mod:`~chad.validation.scoring_spine`, the Phase-2
:mod:`~chad.validation.cost_model` (commission + spread + slippage on every leg,
plus the pessimistic intrabar rule), and the Phase-2
:mod:`~chad.validation.splits` (train/val/OOS windowing). It computes NO verdict
thresholds — that is Phase 5. It produces scored tracks + metadata only.

The feature-parity gate comes first (SSOT §V1, critical ordering)
-----------------------------------------------------------------
:func:`run_backtest` refuses to score any head that
:func:`chad.validation.feature_parity.is_backtestable` rejects. A
``NOT_REPLAYABLE`` / ``UNKNOWN`` / (by default) ``APPROXIMABLE`` head is returned
with ``backtested=False`` and a ``skip_reason`` — **never** silently degraded into
a fabricated score. This is the wall between "we can honestly validate this" and
"we cannot".

Strict no-lookahead, enforced structurally (SSOT §3.8)
------------------------------------------------------
A decision at bar ``t`` may see ONLY bars ``≤ t``. This is not a convention here —
it is a *mechanism*: the decision function is handed a :class:`BarWindow` that
exposes bars ``0..t`` and raises :class:`LookaheadError` on any attempt to read
bar ``t+1`` or beyond. An entry fills at the NEXT bar's open (the realistic
post-close fill), and exits resolve via the pessimistic
:func:`~chad.validation.cost_model.resolve_intrabar` (stop assumed first on an
ambiguous bar). The determinism + no-lookahead guarantees are asserted by a
"poisoned future" test: mutating bars after index ``k`` must not change any
decision at an index ``≤ k``.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library +
sibling ``chad.validation`` modules only. It imports strategy *decision functions*
that the CALLER supplies (a REPLAYABLE head's logic, or a test stub); it never
imports a live strategy/broker/runtime module itself, so the transitive-import
isolation test (``tests/validation/test_isolation.py``) stays green.

Sentinel / raise convention (mirrors the rest of the harness): degenerate-but-valid
data never raises — too few bars, or a head that never trades, yields a result with
empty tracks + a ``warning`` and ``backtested`` reflecting whether scoring ran. Only
invalid *configuration* or *malformed input* raises ``ValueError`` (non-positive
sizing, a bar failing the Phase-0 data-quality audit as ``FAIL``, a decision
function that is not callable, a malformed :class:`Signal`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Optional, Sequence

from chad.validation.bar_audit import Status, audit_bar_file, audit_symbol
from chad.validation.cost_model import (
    DEFAULT_COST_CONFIG,
    CostConfig,
    InstrumentClass,
    LiquidityTier,
    Trade,
    apply_costs,
    resolve_intrabar,
)
from chad.validation.feature_parity import (
    FeatureParityResult,
    ParityStatus,
    is_backtestable,
)
from chad.validation.scoring_spine import (
    DEFAULT_PERIODS_PER_YEAR,
    ScoreResult,
    score_returns,
    score_trades,
)
from chad.validation.splits import Partition, generate_walk_forward, partition

__all__ = [
    "LookaheadError",
    "Bar",
    "BarWindow",
    "Signal",
    "ExecutionSpec",
    "SyntheticTrade",
    "DecisionRecord",
    "TrackScore",
    "BacktestResult",
    "DEFAULT_EXECUTION_SPEC",
    "prepare_bars",
    "load_bars_file",
    "replay_decisions",
    "run_backtest",
]


# --------------------------------------------------------------------------- #
# The structural no-lookahead guard.
# --------------------------------------------------------------------------- #
class LookaheadError(Exception):
    """Raised when a decision at bar ``t`` attempts to read bar ``t+1`` or beyond.

    The engine's core anti-lookahead defence (SSOT §3.8): a :class:`BarWindow`
    refuses future access rather than silently returning it, so a leak fails loudly
    and a test can assert the refusal.
    """


@dataclass(frozen=True)
class Bar:
    """One normalized daily OHLCV bar with its position ``index`` in the series.

    Constructed from the audited corpus schema (``open``/``high``/``low``/``close``/
    ``volume``/``ts_utc``). Frozen so a decision function cannot mutate history.
    """

    index: int
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_ohlc(self) -> dict[str, float]:
        """The mapping shape :func:`resolve_intrabar` expects (``high``/``low`` etc.)."""
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BarWindow:
    """A read-only view of bars ``0..t`` handed to a decision function at bar ``t``.

    Structural no-lookahead (SSOT §3.8): the window is constructed with ONLY bars
    ``0..t`` — it physically does not hold any bar past ``t`` (even the private
    ``_bars`` slot is truncated), so a decision function *cannot* read the future by
    any means, public or private. On the public surface, an explicit future access
    (a positive index ``> t``, or a negative index reaching before ``0``) raises
    :class:`LookaheadError` for a loud, diagnosable leak attempt; slices and
    iteration range over ``0..t`` only.
    """

    __slots__ = ("_bars", "_t")

    def __init__(self, bars: tuple[Bar, ...], t: int) -> None:
        if not isinstance(t, int) or isinstance(t, bool):
            raise ValueError(f"t must be an int, got {t!r}")
        if t < 0 or t >= len(bars):
            raise ValueError(f"t={t} out of range for {len(bars)} bars")
        # Truncate to the visible past so the future is physically absent from the
        # object — the no-lookahead guarantee is structural, not by-convention.
        self._bars: tuple[Bar, ...] = tuple(bars[: t + 1])
        self._t = t

    @property
    def t(self) -> int:
        """The current decision index (inclusive upper bound of what is visible)."""
        return self._t

    @property
    def current(self) -> Bar:
        """The bar at ``t`` — the bar on whose close the decision is made."""
        return self._bars[self._t]

    def __len__(self) -> int:
        return self._t + 1

    def __iter__(self):
        return iter(self._bars[: self._t + 1])

    def __getitem__(self, key: int | slice) -> Bar | list[Bar]:
        if isinstance(key, slice):
            # ``_bars`` physically holds only bars 0..t, so a slice cannot reach the
            # future; standard slicing (incl. reverse slices like ``w[::-1]``) is
            # correct and leak-free without any manual index clamping.
            return list(self._bars[key])
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(f"bar index must be int or slice, got {type(key).__name__}")
        idx = key if key >= 0 else self._t + 1 + key
        if idx < 0:
            raise LookaheadError(f"index {key} reaches before the start of history")
        if idx > self._t:
            raise LookaheadError(
                f"index {key} (resolved {idx}) is in the future at decision bar t={self._t}"
            )
        return self._bars[idx]

    def closes(self) -> tuple[float, ...]:
        """Closing prices ``0..t`` (a fresh tuple; no future leaks)."""
        return tuple(b.close for b in self._bars[: self._t + 1])

    def highs(self) -> tuple[float, ...]:
        return tuple(b.high for b in self._bars[: self._t + 1])

    def lows(self) -> tuple[float, ...]:
        return tuple(b.low for b in self._bars[: self._t + 1])

    def opens(self) -> tuple[float, ...]:
        return tuple(b.open for b in self._bars[: self._t + 1])

    def volumes(self) -> tuple[float, ...]:
        return tuple(b.volume for b in self._bars[: self._t + 1])


# --------------------------------------------------------------------------- #
# Decision + execution inputs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Signal:
    """A head's decision at a bar: enter ``direction`` with an absolute ``stop``/``target``.

    ``direction`` is ``"long"`` or ``"short"``. ``stop`` and ``target`` are absolute
    price levels (both strictly positive). The engine fills the entry at the NEXT
    bar's open and resolves the exit against subsequent bars with the pessimistic
    intrabar rule. A decision function returns ``None`` to stay flat.
    """

    direction: str
    stop: float
    target: float
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction!r}")
        for name in ("stop", "target"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a real number, got {v!r}")
            if not (float(v) > 0.0):
                raise ValueError(f"{name} must be > 0, got {v}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "stop": self.stop,
            "target": self.target,
            "label": self.label,
        }


# The decision function contract a REPLAYABLE head (or test stub) implements.
DecideFn = Callable[[BarWindow], Optional[Signal]]


@dataclass(frozen=True)
class ExecutionSpec:
    """How a synthetic fill is sized + classified for costing (SSOT §3.5).

    Deterministic instrument metadata used only to charge costs — it never affects
    the decision logic (which sees the :class:`BarWindow` alone). Defaults model a
    single liquid equity share.
    """

    instrument_class: InstrumentClass = InstrumentClass.STK
    quantity: float = 1.0
    multiplier: float = 1.0
    liquidity_tier: LiquidityTier = LiquidityTier.LIQUID

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_class, InstrumentClass):
            raise ValueError(f"instrument_class must be an InstrumentClass, got {self.instrument_class!r}")
        if not isinstance(self.liquidity_tier, LiquidityTier):
            raise ValueError(f"liquidity_tier must be a LiquidityTier, got {self.liquidity_tier!r}")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity!r}")
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, (int, float)) or self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_class": self.instrument_class.value,
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "liquidity_tier": self.liquidity_tier.value,
        }


DEFAULT_EXECUTION_SPEC: ExecutionSpec = ExecutionSpec()


# --------------------------------------------------------------------------- #
# Output records — flat, serialisable, embedded verbatim by later reports.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionRecord:
    """A decision-time record (index + emitted signal), free of any future data.

    The poison test compares these across a clean vs future-poisoned run: every
    record with ``index ≤ k`` must be byte-identical, proving no-lookahead. The
    record intentionally carries ONLY the decision (direction/stop/target), never a
    price sourced from a future bar.
    """

    index: int
    direction: Optional[str]
    stop: Optional[float]
    target: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "direction": self.direction, "stop": self.stop, "target": self.target}


@dataclass(frozen=True)
class SyntheticTrade:
    """One closed round-trip trade produced by the replay, with costs applied.

    ``gross_pnl`` is direction-signed ``(exit-entry)*qty*mult`` before friction;
    ``net_pnl`` subtracts the Phase-2 cost haircut; ``ret`` is ``net_pnl /
    entry_notional``. ``ambiguous_exit`` flags a pessimistic (stop-assumed-first)
    resolution; ``exit_reason`` is ``"stop"``/``"target"``/``"end_of_data"``.
    """

    entry_index: int
    exit_index: int
    entry_ts: str
    exit_ts: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    multiplier: float
    gross_pnl: float
    total_cost: float
    net_pnl: float
    ret: float
    ambiguous_exit: bool
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "multiplier": self.multiplier,
            "gross_pnl": self.gross_pnl,
            "total_cost": self.total_cost,
            "net_pnl": self.net_pnl,
            "ret": self.ret,
            "ambiguous_exit": self.ambiguous_exit,
            "exit_reason": self.exit_reason,
        }


@dataclass(frozen=True)
class TrackScore:
    """Scored performance of one track (overall / train / val / oos).

    ``trade_score`` is :func:`score_trades` over the track's net PnLs; ``return_score``
    is :func:`score_returns` over the per-trade net returns (each trade = one
    "period", annualized by the realized trades-per-year, ``periods_per_year``).
    Both are the SSOT single scoring spine — no scoring arithmetic lives here.
    """

    name: str
    n_trades: int
    periods_per_year: float
    trade_score: dict[str, Any]
    return_score: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_trades": self.n_trades,
            "periods_per_year": self.periods_per_year,
            "trade_score": self.trade_score,
            "return_score": self.return_score,
        }


@dataclass(frozen=True)
class BacktestResult:
    """The full Stage-1 replay result for one head (scored tracks + metadata, no verdict).

    ``backtested`` is ``True`` only when the feature-parity gate passed AND there was
    enough data to score; otherwise ``skip_reason`` explains why (a parity status or
    a data-degeneracy tag) and ``tracks`` is empty. ``NOT_REPLAYABLE``/``UNKNOWN``
    heads are ALWAYS ``backtested=False`` with no scores — never faked.
    """

    head: str
    status: str
    backtested: bool
    skip_reason: Optional[str]
    symbol: Optional[str]
    n_bars: int
    label_horizon: int
    trades: tuple[SyntheticTrade, ...]
    n_decisions: int
    n_signals: int
    ambiguous_exit_count: int
    excluded_trade_count: int
    tracks: tuple[TrackScore, ...]
    partition: Optional[dict[str, Any]]
    n_walk_forward_windows: int
    span_years: Optional[float]
    execution_spec: dict[str, Any]
    cost_config_echo: dict[str, Any]
    parity_reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "status": self.status,
            "backtested": self.backtested,
            "skip_reason": self.skip_reason,
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "label_horizon": self.label_horizon,
            "trades": [t.to_dict() for t in self.trades],
            "n_decisions": self.n_decisions,
            "n_signals": self.n_signals,
            "ambiguous_exit_count": self.ambiguous_exit_count,
            "excluded_trade_count": self.excluded_trade_count,
            "tracks": [t.to_dict() for t in self.tracks],
            "partition": self.partition,
            "n_walk_forward_windows": self.n_walk_forward_windows,
            "span_years": self.span_years,
            "execution_spec": self.execution_spec,
            "cost_config_echo": self.cost_config_echo,
            "parity_reasons": list(self.parity_reasons),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Bar preparation (audited; no bar is backtested until it passes Phase 0).
# --------------------------------------------------------------------------- #
def _coerce_price(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"bar {name} must be a real number, got {value!r}")
    return float(value)


def _to_bar(raw: Mapping[str, Any], index: int) -> Bar:
    try:
        ts = raw["ts_utc"]
        o = raw["open"]
        h = raw["high"]
        low = raw["low"]
        c = raw["close"]
        v = raw["volume"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"bar at index {index} missing required OHLCV/ts field: {exc}") from exc
    return Bar(
        index=index,
        ts=str(ts),
        open=_coerce_price("open", o),
        high=_coerce_price("high", h),
        low=_coerce_price("low", low),
        close=_coerce_price("close", c),
        volume=_coerce_price("volume", v),
    )


def prepare_bars(
    raw_bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str = "?",
    audit: bool = True,
) -> list[Bar]:
    """Normalize + (by default) Phase-0 audit a raw bar list into ordered :class:`Bar`s.

    Bars are sorted ascending by ``ts_utc`` and re-indexed ``0..n-1`` so the engine's
    index space is contiguous and chronological. When ``audit=True`` (default), the
    Phase-0 :func:`~chad.validation.bar_audit.audit_symbol` runs first and a ``FAIL``
    status raises ``ValueError`` — nothing is backtested on unaudited/failed bars
    (SSOT Phase 0). ``WARN``/``CLEAN`` proceed. Deterministic; never mutates input.
    """
    if audit:
        report = audit_symbol(symbol, list(raw_bars))
        if report.status is Status.FAIL:
            fail_codes = [f.code for f in report.findings if f.severity is Status.FAIL]
            raise ValueError(
                f"refusing to backtest symbol {symbol!r}: Phase-0 data-quality audit "
                f"is FAIL ({', '.join(fail_codes) or 'unspecified'})"
            )
    bars = [_to_bar(raw, i) for i, raw in enumerate(raw_bars)]
    bars.sort(key=lambda b: b.ts)
    return [
        Bar(index=i, ts=b.ts, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
        for i, b in enumerate(bars)
    ]


def load_bars_file(path: str, *, audit: bool = True) -> list[Bar]:
    """Load + prepare bars from a corpus ``*.json`` file (``{"bars": [...]}``).

    Runs the Phase-0 file audit (:func:`~chad.validation.bar_audit.audit_bar_file`)
    as the gate when ``audit=True``; a ``FAIL`` raises ``ValueError``. Reads the file
    read-only; never writes. Imports :mod:`json` locally to keep the module surface
    minimal.
    """
    import json

    if audit:
        report = audit_bar_file(path)
        if report.status is Status.FAIL:
            fail_codes = [f.code for f in report.findings if f.severity is Status.FAIL]
            raise ValueError(
                f"refusing to backtest {path!r}: Phase-0 data-quality audit is FAIL "
                f"({', '.join(fail_codes) or 'unspecified'})"
            )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"{path!r} has no 'bars' array")
    symbol = payload.get("symbol") if isinstance(payload.get("symbol"), str) else "?"
    # The file already passed (or skipped) the audit above; prepare without re-auditing.
    return prepare_bars(raw, symbol=symbol, audit=False)


# --------------------------------------------------------------------------- #
# The event loop — decisions + raw fills, both a pure function of (bars, decide).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _RawFill:
    """An un-costed round trip (prices from bars only) — costed later by the caller."""

    entry_index: int
    exit_index: int
    direction: str
    entry_price: float
    exit_price: float
    ambiguous_exit: bool
    exit_reason: str


def _simulate(
    bars: tuple[Bar, ...],
    decide: DecideFn,
) -> tuple[list[_RawFill], list[DecisionRecord], int, list[str]]:
    """Run the single-position event loop → (raw fills, decisions, ambiguous_count, warnings).

    Deterministic and pure in ``(bars, decide)``: the decision at bar ``t`` sees only
    a :class:`BarWindow` bounded at ``t``; an entry fills at ``bars[t+1].open``; an
    open position's exit is resolved on each subsequent bar with the pessimistic
    :func:`resolve_intrabar`; any still-open position is liquidated at the final
    bar's close (``exit_reason="end_of_data"``). Decisions are recorded ONLY at bars
    where the head is consulted (i.e. flat), each carrying decision-time data only —
    so the record stream depends solely on the past (asserted by the poison test).
    """
    n = len(bars)
    fills: list[_RawFill] = []
    decisions: list[DecisionRecord] = []
    warnings: list[str] = []
    ambiguous_count = 0

    # Open position state (None = flat).
    pos_dir: Optional[str] = None
    pos_entry_index = -1
    pos_entry_price = 0.0
    pos_stop = 0.0
    pos_target = 0.0

    for t in range(n):
        bar = bars[t]

        # 1) If in a position, attempt to exit on THIS bar (pessimistic intrabar).
        if pos_dir is not None:
            res = resolve_intrabar(bar.as_ohlc(), pos_stop, pos_target)
            if res.outcome != "none":
                if res.fill_price is None:  # explicit guard (money path; not an -O-strippable assert)
                    raise ValueError(
                        "resolve_intrabar returned a non-'none' outcome without a fill_price"
                    )
                if res.ambiguous:
                    ambiguous_count += 1
                fills.append(
                    _RawFill(
                        entry_index=pos_entry_index,
                        exit_index=t,
                        direction=pos_dir,
                        entry_price=pos_entry_price,
                        exit_price=res.fill_price,
                        ambiguous_exit=res.ambiguous,
                        exit_reason=res.outcome,
                    )
                )
                pos_dir = None

        # 2) If flat, consult the head using ONLY bars ≤ t; fill any entry at t+1 open.
        if pos_dir is None:
            window = BarWindow(bars, t)
            signal = decide(window)
            if signal is not None and not isinstance(signal, Signal):
                raise ValueError(
                    f"decide() must return a Signal or None, got {type(signal).__name__}"
                )
            decisions.append(
                DecisionRecord(
                    index=t,
                    direction=None if signal is None else signal.direction,
                    stop=None if signal is None else signal.stop,
                    target=None if signal is None else signal.target,
                )
            )
            if signal is not None:
                if t + 1 < n:
                    pos_dir = signal.direction
                    pos_entry_index = t + 1
                    pos_entry_price = bars[t + 1].open
                    pos_stop = signal.stop
                    pos_target = signal.target
                # else: a signal on the last bar cannot fill (no next open) — dropped.

    # 3) Liquidate a still-open position at the final bar's close.
    if pos_dir is not None and n > 0:
        last = bars[n - 1]
        # Only a genuine open position (entered before the last bar) reaches here.
        if pos_entry_index <= n - 1:
            fills.append(
                _RawFill(
                    entry_index=pos_entry_index,
                    exit_index=n - 1,
                    direction=pos_dir,
                    entry_price=pos_entry_price,
                    exit_price=last.close,
                    ambiguous_exit=False,
                    exit_reason="end_of_data",
                )
            )
            warnings.append(
                f"1 position still open at end of data; liquidated at final close "
                f"(entry_index={pos_entry_index})"
            )

    return fills, decisions, ambiguous_count, warnings


def replay_decisions(decide: DecideFn, bars: Sequence[Bar]) -> tuple[DecisionRecord, ...]:
    """Return the head's decision stream over ``bars`` (no costs, no scoring).

    The pure no-lookahead surface: every :class:`DecisionRecord` depends only on bars
    at or before its index (each decision sees a :class:`BarWindow` bounded at ``t``).
    Used by the poison test and internally by :func:`run_backtest`. ``decide`` must be
    callable; ``bars`` must be a sequence of :class:`Bar`.
    """
    if not callable(decide):
        raise ValueError("decide must be callable")
    bt = _as_bar_tuple(bars)
    _, decisions, _, _ = _simulate(bt, decide)
    return tuple(decisions)


# --------------------------------------------------------------------------- #
# Scoring helpers — annualization + track scoring via the shared spine only.
# --------------------------------------------------------------------------- #
def _as_bar_tuple(bars: Sequence[Bar]) -> tuple[Bar, ...]:
    bt = tuple(bars)
    for b in bt:
        if not isinstance(b, Bar):
            raise ValueError(f"bars must be Bar instances, got {type(b).__name__}")
    return bt


def _parse_date(ts: str) -> Optional[date]:
    """Parse a leading ``YYYY-MM-DD`` from ``ts`` (corpus schema); ``None`` if unparseable."""
    head = ts[:10]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _span_years(bars: tuple[Bar, ...]) -> Optional[float]:
    """Calendar span of the bar series in years (``None`` if dates unparseable/degenerate)."""
    if len(bars) < 2:
        return None
    first = _parse_date(bars[0].ts)
    last = _parse_date(bars[-1].ts)
    if first is None or last is None:
        return None
    days = (last - first).days
    if days <= 0:
        return None
    return days / 365.25


def _periods_per_year(n_trades: int, span_years: Optional[float]) -> float:
    """Annualization basis for the per-trade return series: realized trades / year.

    Falls back to the canonical daily default (:data:`DEFAULT_PERIODS_PER_YEAR`, the
    spine's 252) when the span is unknown or degenerate. Floored at a tiny positive
    value so :func:`score_returns` (which requires a positive ``periods_per_year``)
    never receives zero.
    """
    if span_years is not None and span_years > 0.0 and n_trades > 0:
        return max(n_trades / span_years, 1e-9)
    return float(DEFAULT_PERIODS_PER_YEAR)


def _score_track(
    name: str,
    trades: Sequence[SyntheticTrade],
    periods_per_year: float,
) -> TrackScore:
    """Score one track's trades through the shared spine (trade stats + return series)."""
    pnls = [t.net_pnl for t in trades]
    rets = [t.ret for t in trades]
    trade_score: ScoreResult = score_trades(pnls, label=name)
    if rets:
        return_score: ScoreResult = score_returns(
            rets, periods_per_year=periods_per_year, label=name
        )
    else:
        # No trades → an all-None returns result (honest sentinel, never a fake zero).
        return_score = score_returns([], periods_per_year=periods_per_year, label=name)
    return TrackScore(
        name=name,
        n_trades=len(trades),
        periods_per_year=periods_per_year,
        trade_score=trade_score.to_dict(),
        return_score=return_score.to_dict(),
    )


# --------------------------------------------------------------------------- #
# The public entry point.
# --------------------------------------------------------------------------- #
def _skip_result(
    head: str,
    parity: FeatureParityResult,
    *,
    reason: str,
    symbol: Optional[str],
    n_bars: int,
    label_horizon: int,
    execution_spec: ExecutionSpec,
    cost_config: CostConfig,
    warnings: Sequence[str] = (),
) -> BacktestResult:
    """Build a not-backtested result (parity gate or degenerate data) — never scored."""
    return BacktestResult(
        head=head,
        status=parity.status.value,
        backtested=False,
        skip_reason=reason,
        symbol=symbol,
        n_bars=n_bars,
        label_horizon=label_horizon,
        trades=(),
        n_decisions=0,
        n_signals=0,
        ambiguous_exit_count=0,
        excluded_trade_count=0,
        tracks=(),
        partition=None,
        n_walk_forward_windows=0,
        span_years=None,
        execution_spec=execution_spec.to_dict(),
        cost_config_echo=cost_config.to_dict(),
        parity_reasons=parity.reasons,
        warnings=tuple(warnings),
    )


def run_backtest(
    head: str,
    decide: DecideFn,
    bars: Sequence[Bar],
    *,
    parity: FeatureParityResult,
    label_horizon: int,
    symbol: Optional[str] = None,
    execution_spec: ExecutionSpec = DEFAULT_EXECUTION_SPEC,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    oos_frac: float = 0.2,
    embargo: int = 0,
    wf_train_size: Optional[int] = None,
    wf_test_size: Optional[int] = None,
    allow_approximable: bool = False,
) -> BacktestResult:
    """Replay a REPLAYABLE head over ``bars`` → costed synthetic trades → scored tracks.

    The feature-parity gate (SSOT §V1) comes first: unless
    :func:`is_backtestable(parity, allow_approximable=…)` is true, the head is NOT
    scored — a ``BacktestResult`` with ``backtested=False`` and ``skip_reason`` set to
    the parity status is returned. This is the mechanism that makes a ``NOT_REPLAYABLE``
    / ``UNKNOWN`` head impossible to silently score.

    Pipeline for a backtestable head: :func:`_simulate` (structural no-lookahead,
    pessimistic intrabar) → :func:`~chad.validation.cost_model.apply_costs` on every
    round trip → :func:`~chad.validation.scoring_spine` for the overall track and for
    each Phase-2 :func:`~chad.validation.splits.partition` train/val/OOS slice (trades
    assigned by entry index; ``label_horizon`` drives the purge/embargo). The
    walk-forward window count is recorded as metadata. No verdict thresholds (Phase 5).

    Deterministic. Guards degenerate data (``< 2`` bars, or a head that never trades)
    with warnings + empty tracks. Raises ``ValueError`` only on invalid config /
    malformed input (non-callable ``decide``, non-``Bar`` bars, negative
    ``label_horizon``, FAILED bar audit via :func:`prepare_bars`).
    """
    if not isinstance(parity, FeatureParityResult):
        raise ValueError(f"parity must be a FeatureParityResult, got {type(parity).__name__}")
    if not callable(decide):
        raise ValueError("decide must be callable")
    if isinstance(label_horizon, bool) or not isinstance(label_horizon, int) or label_horizon < 0:
        raise ValueError(f"label_horizon must be an int >= 0, got {label_horizon!r}")
    if head != parity.head:
        raise ValueError(f"head {head!r} does not match parity.head {parity.head!r}")

    bt = _as_bar_tuple(bars)
    n = len(bt)

    # --- GATE: feature parity FIRST. A non-backtestable head is never scored. ---- #
    if not is_backtestable(parity, allow_approximable=allow_approximable):
        return _skip_result(
            head, parity,
            reason=parity.status.value,
            symbol=symbol, n_bars=n, label_horizon=label_horizon,
            execution_spec=execution_spec, cost_config=cost_config,
        )

    # --- Degenerate data guard (backtestable head, but too few bars to score). --- #
    if n < 2:
        return _skip_result(
            head, parity,
            reason="insufficient_bars",
            symbol=symbol, n_bars=n, label_horizon=label_horizon,
            execution_spec=execution_spec, cost_config=cost_config,
            warnings=(f"only {n} bar(s); need >= 2 to backtest",),
        )

    # --- Replay → raw fills + decision stream. ----------------------------------- #
    raw_fills, decisions, ambiguous_count, sim_warnings = _simulate(bt, decide)
    n_signals = sum(1 for d in decisions if d.direction is not None)

    # --- Cost every round trip (Phase-2 chokepoint) → SyntheticTrade. ------------ #
    trades: list[SyntheticTrade] = []
    for f in raw_fills:
        sign = 1.0 if f.direction == "long" else -1.0
        qty = execution_spec.quantity
        mult = execution_spec.multiplier
        gross = sign * (f.exit_price - f.entry_price) * qty * mult
        costed = apply_costs(
            Trade(
                instrument_class=execution_spec.instrument_class,
                quantity=qty,
                entry_price=f.entry_price,
                exit_price=f.exit_price,
                liquidity_tier=execution_spec.liquidity_tier,
                multiplier=mult,
                gross_pnl=gross,
            ),
            cost_config,
        )
        entry_notional = f.entry_price * qty * mult
        # gross_pnl is always supplied to the Trade above, so cost_model always
        # populates net_pnl; the fallback is unreachable defence kept for the type.
        net = costed.net_pnl if costed.net_pnl is not None else gross - costed.total_cost
        trades.append(
            SyntheticTrade(
                entry_index=f.entry_index,
                exit_index=f.exit_index,
                entry_ts=bt[f.entry_index].ts,
                exit_ts=bt[f.exit_index].ts,
                direction=f.direction,
                entry_price=f.entry_price,
                exit_price=f.exit_price,
                quantity=qty,
                multiplier=mult,
                gross_pnl=gross,
                total_cost=costed.total_cost,
                net_pnl=net,
                ret=(net / entry_notional) if entry_notional > 0.0 else 0.0,
                ambiguous_exit=f.ambiguous_exit,
                exit_reason=f.exit_reason,
            )
        )

    warnings = list(sim_warnings)
    if not trades:
        warnings.append("head produced no closed trades over this bar series")

    span_years = _span_years(bt)
    ppy = _periods_per_year(len(trades), span_years)

    # --- Phase-2 splits: partition bars, assign trades to train/val/OOS. --------- #
    part: Partition = partition(
        n, label_horizon=label_horizon,
        train_frac=train_frac, val_frac=val_frac, oos_frac=oos_frac, embargo=embargo,
    )
    train_idx = set(part.train_indices)
    val_idx = set(part.val_indices)
    oos_idx = set(part.oos_indices)
    train_trades: list[SyntheticTrade] = []
    val_trades: list[SyntheticTrade] = []
    oos_trades: list[SyntheticTrade] = []
    excluded = 0
    for t in trades:
        # A trade is attributed to a split ONLY when BOTH its entry AND its exit fall
        # inside that split's usable index set. A synthetic hold is unbounded (it exits
        # on stop/target or is liquidated at end-of-data), so a trade entered in one
        # split can exit in a later one; attributing by entry alone would fold that
        # later split's price move into this split's score and destroy IS/OOS
        # independence. A trade that straddles a boundary (or enters a purge/embargo
        # gap) is therefore excluded from every split track — counted, never leaked.
        if t.entry_index in train_idx and t.exit_index in train_idx:
            train_trades.append(t)
        elif t.entry_index in val_idx and t.exit_index in val_idx:
            val_trades.append(t)
        elif t.entry_index in oos_idx and t.exit_index in oos_idx:
            oos_trades.append(t)
        else:
            excluded += 1  # purge/embargo-gap entry OR a hold straddling a split boundary

    tracks = (
        _score_track("overall", trades, ppy),
        _score_track("train", train_trades, ppy),
        _score_track("val", val_trades, ppy),
        _score_track("oos", oos_trades, ppy),
    )

    # --- Walk-forward window count (metadata for W_min, SSOT §4.3). -------------- #
    ts_wf = wf_test_size if wf_test_size is not None else max(1, n // 6)
    tr_wf = wf_train_size if wf_train_size is not None else max(1, n // 3)
    wf_windows = generate_walk_forward(
        n, train_size=tr_wf, test_size=ts_wf, label_horizon=label_horizon, embargo=embargo,
    )

    return BacktestResult(
        head=head,
        status=parity.status.value,
        backtested=True,
        skip_reason=None,
        symbol=symbol,
        n_bars=n,
        label_horizon=label_horizon,
        trades=tuple(trades),
        n_decisions=len(decisions),
        n_signals=n_signals,
        ambiguous_exit_count=ambiguous_count,
        excluded_trade_count=excluded,
        tracks=tracks,
        partition=part.to_dict(),
        n_walk_forward_windows=len(wf_windows),
        span_years=span_years,
        execution_spec=execution_spec.to_dict(),
        cost_config_echo=cost_config.to_dict(),
        parity_reasons=parity.reasons,
        warnings=tuple(warnings),
    )
