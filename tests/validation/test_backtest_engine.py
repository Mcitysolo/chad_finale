"""Tests for chad/validation/backtest_engine.py — Phase 4 Stage-1 replay (SSOT §3.5/§3.8).

Fixture-only, stub-strategy tests (never a real strategy's internals). They pin the
engine's contract:
  * a deterministic stub over a fixture bar series → the EXACT expected synthetic
    trades, with the Phase-2 cost haircut applied to precise numbers;
  * strict no-lookahead, proven two ways — a :class:`BarWindow` that raises on future
    access, and a "poisoned future" replay whose past decisions are byte-identical;
  * a NOT_REPLAYABLE / UNKNOWN / (default) APPROXIMABLE head is SKIPPED, never scored;
  * the Phase-2 splits and the pessimistic intrabar rule are actually exercised.
"""

from __future__ import annotations

import datetime
import json

import pytest

from chad.validation.backtest_engine import (
    DEFAULT_EXECUTION_SPEC,
    Bar,
    BarWindow,
    BacktestResult,
    ExecutionSpec,
    LookaheadError,
    Signal,
    load_bars_file,
    prepare_bars,
    replay_decisions,
    run_backtest,
)
from chad.validation.cost_model import (
    DEFAULT_COST_CONFIG,
    InstrumentClass,
    LiquidityTier,
    Trade,
    apply_costs,
)
from chad.validation.feature_parity import (
    FeatureParityResult,
    ParityStatus,
    classify_source,
)

_START = datetime.date(2025, 1, 6)  # a Monday
BARS_ONLY_SRC = "def decide(w):\n    return [b.close for b in w.bars]"


def _raw(closes, *, start=_START):
    """Deterministic raw bars from a close series (o=c, h=c+1, l=c-1, vol=1e6)."""
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        out.append({
            "open": c, "high": c + 1.0, "low": c - 1.0, "close": c,
            "volume": 1_000_000.0,
            "ts_utc": (start + datetime.timedelta(days=i)).isoformat(),
        })
    return out


def _replayable(head: str = "stub") -> FeatureParityResult:
    r = classify_source(head, BARS_ONLY_SRC)
    assert r.status is ParityStatus.REPLAYABLE
    return r


def _momentum(w: BarWindow):
    """Go long when the current close exceeds the prior close; ±2 stop/target band."""
    if len(w) < 2:
        return None
    c = w.current.close
    prev = w[len(w) - 2].close
    if c > prev:
        return Signal(direction="long", stop=c - 2.0, target=c + 2.0)
    return None


# --------------------------------------------------------------------------- #
# prepare_bars — normalize, sort, reindex, Phase-0 audit gate.
# --------------------------------------------------------------------------- #
def test_prepare_bars_sorts_and_reindexes() -> None:
    raw = _raw([100, 101, 102])
    raw.reverse()  # feed out of chronological order
    bars = prepare_bars(raw, symbol="X")
    assert [b.index for b in bars] == [0, 1, 2]
    assert [b.ts for b in bars] == sorted(b["ts_utc"] for b in raw)
    assert bars[0].close == 100.0 and bars[-1].close == 102.0


def test_prepare_bars_refuses_failed_audit() -> None:
    bad = _raw([100, 101])
    bad[1]["high"] = 90.0  # high < low → OHLC FAIL
    with pytest.raises(ValueError, match="FAIL"):
        prepare_bars(bad, symbol="BADOHLC", audit=True)
    # audit=False bypasses the gate (caller's explicit responsibility).
    assert len(prepare_bars(bad, symbol="BADOHLC", audit=False)) == 2


def test_load_bars_file_roundtrip(tmp_path) -> None:
    f = tmp_path / "SYN.json"
    f.write_text(json.dumps({"symbol": "SYN", "bars": _raw([10, 11, 12])}), encoding="utf-8")
    bars = load_bars_file(str(f))
    assert [b.close for b in bars] == [10.0, 11.0, 12.0]


def test_load_bars_file_refuses_failed_audit(tmp_path) -> None:
    raw = _raw([10, 11])
    raw[0]["low"] = 999.0  # low > high → OHLC FAIL
    f = tmp_path / "BAD.json"
    f.write_text(json.dumps({"symbol": "BAD", "bars": raw}), encoding="utf-8")
    with pytest.raises(ValueError, match="FAIL"):
        load_bars_file(str(f), audit=True)


# --------------------------------------------------------------------------- #
# BarWindow — structural no-lookahead.
# --------------------------------------------------------------------------- #
def test_barwindow_bounds_and_future_raises() -> None:
    bars = tuple(prepare_bars(_raw([100, 101, 102, 103, 104]), symbol="X"))
    w = BarWindow(bars, 2)
    assert len(w) == 3
    assert w.current is bars[2]
    assert w.t == 2
    assert w[0] is bars[0] and w[2] is bars[2]
    assert w[-1] is bars[2]                       # negative maps to t
    assert [b.index for b in w] == [0, 1, 2]      # iteration bounded
    assert w.closes() == (100.0, 101.0, 102.0)
    assert [b.index for b in w[0:99]] == [0, 1, 2]   # slice clamps to t
    assert [b.index for b in w[::-1]] == [2, 1, 0]   # reverse slice is correct (not [])
    assert [b.index for b in w[w.t::-1]] == [2, 1, 0]
    with pytest.raises(LookaheadError):
        _ = w[3]                                   # explicit future access
    with pytest.raises(LookaheadError):
        _ = w[-99]                                 # before start of history


def test_barwindow_physically_holds_no_future() -> None:
    """The window is truncated at t, so even private access cannot reach the future."""
    bars = tuple(prepare_bars(_raw([100, 101, 102, 103, 104]), symbol="X"))
    w = BarWindow(bars, 2)
    # The private slot holds ONLY bars 0..t — no future bar is present to leak.
    assert len(w._bars) == 3  # type: ignore[attr-defined]
    with pytest.raises(IndexError):
        _ = w._bars[3]        # type: ignore[attr-defined]  # would be bar t+1 if held


def test_barwindow_rejects_bad_construction() -> None:
    bars = tuple(prepare_bars(_raw([100, 101, 102]), symbol="X"))
    with pytest.raises(ValueError):
        BarWindow(bars, 3)   # t out of range
    with pytest.raises(ValueError):
        BarWindow(bars, -1)


# --------------------------------------------------------------------------- #
# Deterministic stub → EXACT trades + EXACT costs.
# --------------------------------------------------------------------------- #
def test_single_trade_exact_prices_and_costs() -> None:
    # bar0 c=100; bar1 o=100..target 110 not yet; bar2 hits target 110.
    raw = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6, "ts_utc": "2025-01-06"},
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1e6, "ts_utc": "2025-01-07"},
        {"open": 104.0, "high": 112.0, "low": 103.0, "close": 110.0, "volume": 1e6, "ts_utc": "2025-01-08"},
    ]
    bars = prepare_bars(raw, symbol="EXACT")

    def once(w: BarWindow):
        return Signal("long", stop=95.0, target=110.0) if w.t == 0 else None

    res = run_backtest("stub", once, bars, parity=_replayable(), label_horizon=0, symbol="EXACT")
    assert res.backtested is True
    assert len(res.trades) == 1
    t = res.trades[0]
    assert (t.entry_index, t.exit_index) == (1, 2)
    assert t.entry_price == 100.0                    # entry at NEXT bar's open
    assert t.exit_price == 110.0                     # target level
    assert t.exit_reason == "target"
    assert t.ambiguous_exit is False
    assert t.gross_pnl == pytest.approx(10.0)

    # Costs match apply_costs on the equivalent Trade exactly (single haircut path).
    expected = apply_costs(
        Trade(InstrumentClass.STK, 1.0, 100.0, 110.0, LiquidityTier.LIQUID, 1.0, gross_pnl=10.0),
        DEFAULT_COST_CONFIG,
    )
    assert t.total_cost == pytest.approx(2.063)      # 2.00 comm + 0.042 spread + 0.021 slip
    assert t.net_pnl == pytest.approx(7.937)
    assert t.net_pnl == pytest.approx(expected.net_pnl)
    assert t.ret == pytest.approx(7.937 / 100.0)
    # Costs are genuinely applied: net strictly less than gross.
    assert t.net_pnl < t.gross_pnl
    # And the overall scored track flows through the shared spine.
    assert res.tracks[0].name == "overall"
    assert res.tracks[0].trade_score["total_pnl"] == pytest.approx(7.937)


def test_entry_fills_at_next_open() -> None:
    bars = prepare_bars(_raw([100, 200, 201, 202]), symbol="X")

    def once(w: BarWindow):
        return Signal("long", stop=1.0, target=1e9) if w.t == 0 else None  # never exits

    res = run_backtest("stub", once, bars, parity=_replayable(), label_horizon=0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.entry_index == 1
    assert tr.entry_price == bars[1].open  # filled at the bar AFTER the decision


def test_end_of_data_liquidation() -> None:
    bars = prepare_bars(_raw([100, 101, 102, 103]), symbol="X")

    def once(w: BarWindow):
        return Signal("long", stop=1.0, target=1e9) if w.t == 0 else None  # never hits

    res = run_backtest("stub", once, bars, parity=_replayable(), label_horizon=0)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "end_of_data"
    assert res.trades[0].exit_index == len(bars) - 1
    assert res.trades[0].exit_price == bars[-1].close
    assert any("still open at end of data" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# Pessimistic intrabar rule (V2).
# --------------------------------------------------------------------------- #
def test_pessimistic_intrabar_prefers_stop() -> None:
    # bar1 range [97,103] contains BOTH stop=98 and target=102 → pessimistic stop.
    raw = [
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1e6, "ts_utc": "2025-01-06"},
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1e6, "ts_utc": "2025-01-07"},
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1e6, "ts_utc": "2025-01-08"},
    ]
    bars = prepare_bars(raw, symbol="AMBI")

    def once(w: BarWindow):
        return Signal("long", stop=98.0, target=102.0) if w.t == 0 else None

    res = run_backtest("stub", once, bars, parity=_replayable(), label_horizon=0)
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "stop"
    assert res.trades[0].ambiguous_exit is True
    assert res.ambiguous_exit_count == 1


# --------------------------------------------------------------------------- #
# No-lookahead — poisoned future must not change past decisions.
# --------------------------------------------------------------------------- #
def test_poisoned_future_does_not_change_past_decisions() -> None:
    closes = [100, 101, 100, 102, 103, 101, 104, 105, 103, 106, 107, 105]
    k = 5
    clean = prepare_bars(_raw(closes), symbol="X")

    poisoned_closes = list(closes)
    for i in range(k + 1, len(closes)):
        poisoned_closes[i] = 9999.0 if i % 2 == 0 else 0.01  # wild future divergence
    poisoned = prepare_bars(_raw(poisoned_closes), symbol="X", audit=False)

    d_clean = replay_decisions(_momentum, clean)
    d_pois = replay_decisions(_momentum, poisoned)

    past_clean = [d.to_dict() for d in d_clean if d.index <= k]
    past_pois = [d.to_dict() for d in d_pois if d.index <= k]
    assert past_clean == past_pois, "future bars leaked into a past decision"


def test_cheating_decide_that_reads_future_raises() -> None:
    bars = prepare_bars(_raw([100, 101, 102, 103]), symbol="X")

    def cheater(w: BarWindow):
        _ = w[w.t + 1]  # structurally blocked: reach one bar into the future
        return None

    with pytest.raises(LookaheadError):
        run_backtest("cheat", cheater, bars, parity=_replayable("cheat"), label_horizon=0)


def test_run_backtest_is_deterministic() -> None:
    bars = prepare_bars(_raw([100, 101, 100, 103, 104, 102, 105, 106]), symbol="X")
    a = run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=1).to_dict()
    b = run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=1).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# The gate — NOT_REPLAYABLE / UNKNOWN / APPROXIMABLE are skipped, never scored.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src,status", [
    ("def decide(ctx):\n    return ctx.news_intel", "NOT_REPLAYABLE"),
    ("def decide(ctx):\n    return ctx.opaque_thing(ctx.mystery)", "UNKNOWN"),
])
def test_non_replayable_head_is_skipped_not_scored(src, status) -> None:
    bars = prepare_bars(_raw([100, 101, 102, 103]), symbol="X")
    parity = classify_source("h", src)
    assert parity.status.value == status
    res = run_backtest("h", _momentum, bars, parity=parity, label_horizon=1)
    assert res.backtested is False
    assert res.skip_reason == status
    assert res.trades == ()
    assert res.tracks == ()
    assert res.status == status
    # The parity reasons are carried through for the report (honest provenance).
    assert res.parity_reasons == parity.reasons


def test_approximable_head_skipped_by_default_scored_when_allowed() -> None:
    bars = prepare_bars(_raw([100, 101, 100, 103, 104, 102, 105, 106]), symbol="X")
    parity = classify_source("h", "def decide(ctx):\n    return ctx.bars, ctx.regime")
    assert parity.status is ParityStatus.APPROXIMABLE
    skipped = run_backtest("h", _momentum, bars, parity=parity, label_horizon=1)
    assert skipped.backtested is False and skipped.skip_reason == "APPROXIMABLE"
    allowed = run_backtest(
        "h", _momentum, bars, parity=parity, label_horizon=1, allow_approximable=True
    )
    assert allowed.backtested is True and allowed.skip_reason is None


# --------------------------------------------------------------------------- #
# Splits actually applied (Phase-2).
# --------------------------------------------------------------------------- #
def test_splits_are_applied_and_trades_attributed() -> None:
    closes = [100 + (i % 3) - (i % 2) + i * 0.5 for i in range(24)]  # wiggly uptrend
    bars = prepare_bars(_raw(closes), symbol="X", audit=False)
    res = run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=2)
    assert res.backtested is True
    assert res.partition is not None
    names = [tr.name for tr in res.tracks]
    assert names == ["overall", "train", "val", "oos"]

    overall = next(tr for tr in res.tracks if tr.name == "overall")
    train = next(tr for tr in res.tracks if tr.name == "train")
    oos = next(tr for tr in res.tracks if tr.name == "oos")
    # Split trade counts + excluded (straddlers/purge-gap) partition the overall trades.
    # Leak-free by construction: an attributed trade has BOTH endpoints in its split;
    # anything crossing a boundary lands in `excluded`, never in a split track (C1 fix).
    # The dedicated straddle test below proves the exclusion behaviour directly.
    attributed = sum(tr.n_trades for tr in res.tracks[1:])
    assert attributed + res.excluded_trade_count == overall.n_trades
    assert {"train_indices", "val_indices", "oos_indices"} <= set(res.partition)
    assert train.n_trades <= overall.n_trades and oos.n_trades <= overall.n_trades
    assert res.n_walk_forward_windows >= 0


def test_boundary_straddling_trade_is_excluded_not_leaked() -> None:
    """A trade entered in-sample but exiting out-of-sample must land in NO split (C1)."""
    # A never-exiting long entered at bar 1 is liquidated at end-of-data (last bar),
    # so it straddles from train into oos.
    bars = prepare_bars(_raw([100 + i for i in range(20)]), symbol="X", audit=False)

    def once(w: BarWindow):
        return Signal("long", stop=1.0, target=1e9) if w.t == 1 else None  # never hits

    res = run_backtest("stub", once, bars, parity=_replayable(), label_horizon=2)
    assert len(res.trades) == 1
    tr = res.trades[0]
    part = res.partition
    # It enters in train but exits at the final (oos) bar → straddles the boundary.
    assert tr.entry_index in set(part["train_indices"])
    assert tr.exit_index == len(bars) - 1
    assert tr.exit_index not in set(part["train_indices"])
    # Therefore it is excluded from every split track, not folded into train's PnL.
    assert res.excluded_trade_count == 1
    for name in ("train", "val", "oos"):
        track = next(t for t in res.tracks if t.name == name)
        assert track.n_trades == 0
    # The overall track still sees it (overall is not split-scoped).
    assert next(t for t in res.tracks if t.name == "overall").n_trades == 1


# --------------------------------------------------------------------------- #
# Degenerate-but-valid data never raises.
# --------------------------------------------------------------------------- #
def test_too_few_bars_is_guarded() -> None:
    bars = prepare_bars(_raw([100]), symbol="X", audit=False)
    res = run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=0)
    assert res.backtested is False
    assert res.skip_reason == "insufficient_bars"
    assert res.tracks == ()


def test_head_that_never_trades_scores_empty_tracks() -> None:
    bars = prepare_bars(_raw([100, 101, 102, 103]), symbol="X")

    def flat(w: BarWindow):
        return None

    res = run_backtest("stub", flat, bars, parity=_replayable(), label_horizon=0)
    assert res.backtested is True          # it ran; it simply produced nothing
    assert res.trades == ()
    assert any("no closed trades" in w for w in res.warnings)
    overall = res.tracks[0]
    assert overall.n_trades == 0
    assert overall.return_score["sharpe"] is None  # honest sentinel, not a fake zero


# --------------------------------------------------------------------------- #
# Input validation (config/malformed → ValueError; data never raises).
# --------------------------------------------------------------------------- #
def test_signal_validation() -> None:
    with pytest.raises(ValueError):
        Signal("sideways", 1.0, 2.0)
    with pytest.raises(ValueError):
        Signal("long", -1.0, 2.0)


def test_signal_to_dict() -> None:
    d = Signal("long", 95.0, 110.0, label="x").to_dict()
    assert d == {"direction": "long", "stop": 95.0, "target": 110.0, "label": "x"}


def test_execution_spec_validation() -> None:
    with pytest.raises(ValueError):
        ExecutionSpec(quantity=0.0)
    with pytest.raises(ValueError):
        ExecutionSpec(multiplier=-1.0)


def test_run_backtest_rejects_head_parity_mismatch() -> None:
    bars = prepare_bars(_raw([100, 101, 102]), symbol="X")
    parity = _replayable("other")
    with pytest.raises(ValueError, match="does not match"):
        run_backtest("mismatch", _momentum, bars, parity=parity, label_horizon=0)


def test_run_backtest_rejects_bad_label_horizon() -> None:
    bars = prepare_bars(_raw([100, 101, 102]), symbol="X")
    with pytest.raises(ValueError):
        run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=-1)


def test_result_is_json_serialisable() -> None:
    bars = prepare_bars(_raw([100, 101, 100, 103, 104, 102, 105, 106]), symbol="X")
    res = run_backtest("stub", _momentum, bars, parity=_replayable(), label_horizon=1)
    dumped = json.dumps(res.to_dict())
    assert '"backtested": true' in dumped


def test_futures_multiplier_scales_pnl_and_costs() -> None:
    """A FUT ExecutionSpec with a multiplier scales gross PnL (proves exec spec is wired)."""
    bars = prepare_bars(
        [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6, "ts_utc": "2025-01-06"},
            {"open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1e6, "ts_utc": "2025-01-07"},
            {"open": 104.0, "high": 112.0, "low": 103.0, "close": 110.0, "volume": 1e6, "ts_utc": "2025-01-08"},
        ],
        symbol="FUT",
    )

    def once(w: BarWindow):
        return Signal("long", stop=95.0, target=110.0) if w.t == 0 else None

    spec = ExecutionSpec(
        instrument_class=InstrumentClass.FUT, quantity=2.0, multiplier=5.0,
        liquidity_tier=LiquidityTier.MID,
    )
    res = run_backtest(
        "stub", once, bars, parity=_replayable(), label_horizon=0, execution_spec=spec,
    )
    t = res.trades[0]
    assert t.gross_pnl == pytest.approx((110.0 - 100.0) * 2.0 * 5.0)  # 100.0
    assert t.total_cost > 0.0
    assert res.execution_spec["instrument_class"] == "FUT"
