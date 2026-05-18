"""GAP-001 corrected-scope (Phase-48): close-intent chokepoint exclusion.

Phase-46 found that the original GAP-001 fix (f02b0ed) only filtered
operator-excluded symbols at the reconciler upstream (line 115 of
chad/core/position_reconciler.py:reconcile_positions_with_signals), but
two other close-intent surfaces remained unprotected:

  1. apply_close_intents (chad/core/position_reconciler.py:232) — fed
     by both the reconciler path AND the regime_reduction path
     (chad/risk/regime_reduction.py → chad/core/live_loop.py:1509).
     The regime_reduction path did not honour the operator-exclusion
     SSOT and was the source of the 6 post-fix SPY/reconciler $100
     placeholder rejected fills observed on 2026-05-18.

  2. enforce_flip_close_first (chad/core/flip_executor.py:107) — the
     BG11 flip-close-first path that submits close intents directly
     to paper_adapter, bypassing apply_close_intents entirely.

Phase-48 adds two surgical chokepoint guards (one per surface). These
tests verify both guards behaviourally, per the 10 cases (a)-(j)
specified verbatim in the Phase-47 fix_spec.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

from chad.core import position_reconciler  # noqa: E402
from chad.core import flip_executor as fx  # noqa: E402
from chad.execution.net_exposure_gate import (  # noqa: E402
    GateAction,
    GateDecision,
)
from chad.risk.regime_reduction import generate_partial_close_intents  # noqa: E402


# ---------------------------------------------------------------------------
# Constants & test helpers
# ---------------------------------------------------------------------------

# Pulled from the runtime exclusion SSOT so the tests track the live set
# rather than a hand-maintained copy.
EXCLUDED = sorted(position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS)

# Sentinel guaranteed NOT to be in any operator-exclusion config.
_NON_EXCLUDED_SENTINEL = "ZZTESTSYM"
assert _NON_EXCLUDED_SENTINEL not in position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS


class _RecordingAdapter:
    """Captures every submit_strategy_trade_intents call so we can assert
    whether the broker was hit at all. Returns an empty submitted-list
    so we never traverse the downstream evidence-write / guard-mutation
    path (those are unit-tested elsewhere)."""

    def __init__(self) -> None:
        self.calls: List[List[Any]] = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        return []


def _close_intent(
    symbol: str,
    *,
    strategy: str = "reconciler",
    position_key: str | None = None,
    open_side: str = "BUY",
    qty: float = 10.0,
    reason: str | None = None,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "action": "CLOSE",
        "open_side": open_side,
        "close_side": "SELL" if open_side == "BUY" else "BUY",
        "quantity": qty,
        "reason": reason or f"test_close_{strategy}",
        "position_key": position_key or f"{strategy}|{symbol}",
        "strategy": strategy,
    }


def _flip_decision(
    *,
    symbol: str,
    existing_strategy: str = "reconciler",
    existing_side: str = "BUY",
    new_strategy: str = "alpha",
) -> GateDecision:
    return GateDecision(
        action=GateAction.FLIP_ALLOWED,
        reason="gap_001_phase_48_test",
        signal_index=0,
        symbol=symbol,
        strategy=new_strategy,
        conflicting_strategy=existing_strategy,
        conflicting_side=existing_side,
        flip_close_strategy=existing_strategy,
        flip_close_symbol=symbol,
        flip_close_side=existing_side,
    )


def _flipped_signal(strategy: str, symbol: str, new_side: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol=symbol,
        side=new_side,
        confidence=0.85,
        meta={},
    )


@pytest.fixture
def patched_guard(monkeypatch):
    """In-memory position-guard substitute for flip_executor — never
    touches the on-disk state file."""
    state: Dict[str, Dict[str, Any]] = {}
    saved: List[Dict[str, Any]] = []

    def _fake_load_state():
        return {k: dict(v) for k, v in state.items()}

    def _fake_save_state(s):
        saved.append({k: dict(v) for k, v in s.items() if isinstance(v, dict)})
        state.clear()
        state.update(s)

    monkeypatch.setattr(fx, "_load_state", _fake_load_state)
    monkeypatch.setattr(fx, "save_state", _fake_save_state)
    return SimpleNamespace(state=state, saved=saved)


# ===========================================================================
# apply_close_intents chokepoint — cases (a)-(f), (i), (j)
# ===========================================================================

def test_case_a_regime_reduction_fed_spy_close_skipped(caplog):
    """(a) regime_reduction-fed SPY close intent → skipped with
    APPLY_CLOSE_INTENTS_SKIP_EXCLUDED log; paper_adapter NOT called for
    SPY; mixed-batch remainder (non-excluded sentinel) IS processed."""
    adapter = _RecordingAdapter()
    intents = [
        _close_intent("SPY", strategy="reconciler",
                      position_key="reconciler|SPY", qty=30.0),
        _close_intent(_NON_EXCLUDED_SENTINEL, strategy="alpha", qty=5.0),
    ]

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(intents, adapter)

    # SPY: skipped → adapter not invoked for SPY
    submitted_symbols = [
        getattr(c[0], "symbol", None) for c in adapter.calls
        if c
    ]
    assert "SPY" not in submitted_symbols, (
        "SPY close intent must not reach paper_adapter when SPY is in "
        f"_EFFECTIVE_NON_CHAD_SYMBOLS. submitted={submitted_symbols}"
    )
    # Non-excluded remainder still processed.
    assert _NON_EXCLUDED_SENTINEL in submitted_symbols, (
        "Mixed-batch remainder must still reach paper_adapter; "
        f"submitted={submitted_symbols}"
    )
    # Structured skip log emitted for SPY.
    skip_records = [
        r for r in caplog.records
        if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
        and "symbol=SPY" in r.getMessage()
    ]
    assert len(skip_records) == 1, (
        f"Expected exactly one APPLY_CLOSE_INTENTS_SKIP_EXCLUDED for SPY; "
        f"got {len(skip_records)}: {[r.getMessage() for r in skip_records]}"
    )


def test_case_b_legitimate_single_symbol_passes_unchanged():
    """(b) Legitimate non-excluded single-symbol close intent → passes
    through; paper_adapter receives the intent unchanged."""
    adapter = _RecordingAdapter()
    intents = [
        _close_intent(_NON_EXCLUDED_SENTINEL, strategy="alpha", qty=7.0),
    ]

    position_reconciler.apply_close_intents(intents, adapter)

    assert len(adapter.calls) == 1, (
        f"Adapter must be invoked exactly once for non-excluded symbol; "
        f"calls={adapter.calls}"
    )
    submitted = adapter.calls[0]
    assert len(submitted) == 1
    assert submitted[0].symbol == _NON_EXCLUDED_SENTINEL


def test_case_c_already_filtered_reconciler_path_idempotent(caplog):
    """(c) Already-filtered reconciler path: when reconcile_positions_with_
    signals filters SPY at line 115, apply_close_intents receives a
    SPY-less list → the chokepoint guard never fires; behavior is
    bit-identical to pre-fix for this caller."""
    adapter = _RecordingAdapter()
    # Reconciler caller hands a list that has already had SPY pruned.
    intents = [
        _close_intent(_NON_EXCLUDED_SENTINEL, strategy="alpha", qty=3.0),
    ]

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(intents, adapter)

    # Chokepoint guard fires 0 times (no excluded symbol in input).
    skip_records = [
        r for r in caplog.records
        if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
    ]
    assert skip_records == [], (
        f"Chokepoint guard must NOT fire when no excluded symbol is in the "
        f"batch (reconciler upstream already filtered). Got: "
        f"{[r.getMessage() for r in skip_records]}"
    )
    # Non-excluded close still reaches the adapter.
    assert len(adapter.calls) == 1


def test_case_d_mixed_batch_only_excluded_skipped(caplog):
    """(d) Mixed batch {SPY, AAPL, sentinel-non-excluded, sentinel-2} →
    excluded ones skipped with their own log lines; non-excluded ones
    reach the adapter; final adapter receives 2 calls only."""
    adapter = _RecordingAdapter()
    intents = [
        _close_intent("SPY", strategy="reconciler",
                      position_key="reconciler|SPY"),
        _close_intent("AAPL", strategy="reconciler",
                      position_key="reconciler|AAPL"),
        _close_intent(_NON_EXCLUDED_SENTINEL, strategy="alpha", qty=4.0),
        _close_intent("ZZSENTINEL2", strategy="beta", qty=2.0),
    ]
    assert "ZZSENTINEL2" not in position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(intents, adapter)

    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
    ]
    assert len(skip_msgs) == 2, f"Expected 2 skip logs; got {skip_msgs}"
    assert any("symbol=SPY" in m for m in skip_msgs)
    assert any("symbol=AAPL" in m for m in skip_msgs)

    submitted_symbols = [
        c[0].symbol for c in adapter.calls if c
    ]
    assert submitted_symbols == [_NON_EXCLUDED_SENTINEL, "ZZSENTINEL2"], (
        f"Adapter must receive only the non-excluded symbols, in order. "
        f"Got: {submitted_symbols}"
    )


def test_case_e_empty_list_early_return(caplog):
    """(e) Empty close_intents list → unchanged early return at line 240;
    no log noise; no broker call."""
    adapter = _RecordingAdapter()

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents([], adapter)

    assert adapter.calls == []
    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
    ]
    assert skip_msgs == []


def test_case_f_all_excluded_batch_all_skipped(caplog):
    """(f) All-excluded batch {SPY, AAPL, MSFT} → all 3 skipped; adapter
    never invoked; 3 skip log lines."""
    adapter = _RecordingAdapter()
    intents = [
        _close_intent(sym, strategy="reconciler",
                      position_key=f"reconciler|{sym}")
        for sym in ("SPY", "AAPL", "MSFT")
    ]

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(intents, adapter)

    assert adapter.calls == [], (
        f"Adapter must not be invoked for all-excluded batch. calls={adapter.calls}"
    )
    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
    ]
    assert len(skip_msgs) == 3, f"Expected 3 skip logs; got {skip_msgs}"
    for sym in ("SPY", "AAPL", "MSFT"):
        assert any(f"symbol={sym}" in m for m in skip_msgs), (
            f"Missing skip log for {sym}; got {skip_msgs}"
        )


def test_case_i_regression_behavioral_regime_reduction_to_chokepoint(caplog):
    """(i) Regression behavioral: simulate the exact production
    preconditions (open 'reconciler|SPY' + regime transition →
    generate_partial_close_intents yields SPY, apply_close_intents
    skips it, no SPY adapter call, no $100 SPY fill)."""
    open_positions = {
        "reconciler|SPY": {
            "open": True,
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 30.0,
            "strategy": "reconciler",
        },
        f"alpha|{_NON_EXCLUDED_SENTINEL}": {
            "open": True,
            "symbol": _NON_EXCLUDED_SENTINEL,
            "side": "BUY",
            "quantity": 10.0,
            "strategy": "alpha",
        },
    }
    # Mimics the regime_reduction call inside live_loop.py:1502-1509.
    close_intents = generate_partial_close_intents(
        open_positions,
        reduction_pct=1.0,
        reason="regime_change_confident_to_cautious",
    )
    assert any(c["symbol"] == "SPY" for c in close_intents), (
        "Regression precondition: generate_partial_close_intents must "
        "include SPY (it does not consult the exclusion SSOT) — this is "
        "the surface the chokepoint guard exists to neutralise."
    )

    adapter = _RecordingAdapter()
    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(close_intents, adapter)

    submitted_symbols = [
        c[0].symbol for c in adapter.calls if c
    ]
    assert "SPY" not in submitted_symbols, (
        "Regression check: the regime_reduction → apply_close_intents "
        f"path must NOT submit SPY to the broker. submitted={submitted_symbols}"
    )
    # Non-excluded position still gets its close.
    assert _NON_EXCLUDED_SENTINEL in submitted_symbols
    # Audit-trail skip log present for SPY.
    assert any(
        "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
        and "symbol=SPY" in r.getMessage()
        for r in caplog.records
    )


def test_case_j_idempotency_double_invocation(caplog):
    """(j) Idempotency: apply_close_intents twice on the same SPY list →
    identical skip behavior each time; no state mutation; no broker
    call either time."""
    adapter = _RecordingAdapter()
    intents = [
        _close_intent("SPY", strategy="reconciler",
                      position_key="reconciler|SPY"),
    ]

    with caplog.at_level(logging.WARNING, logger="chad.core.position_reconciler"):
        position_reconciler.apply_close_intents(intents, adapter)
        first_call_count = len(adapter.calls)
        first_skip_count = sum(
            1 for r in caplog.records
            if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
        )
        # The input list must not have been mutated.
        assert intents[0]["symbol"] == "SPY"
        assert intents[0]["strategy"] == "reconciler"

        position_reconciler.apply_close_intents(intents, adapter)
        second_call_count = len(adapter.calls)
        second_skip_count = sum(
            1 for r in caplog.records
            if "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED" in r.getMessage()
        )

    assert first_call_count == 0 and second_call_count == 0, (
        f"Adapter must remain uncalled across both invocations. "
        f"first={first_call_count} second={second_call_count}"
    )
    assert second_skip_count == 2 * first_skip_count == 2, (
        f"Skip log must fire once per invocation. "
        f"first={first_skip_count} second={second_skip_count}"
    )


# ===========================================================================
# flip_executor chokepoint — cases (g), (h)
# ===========================================================================

def test_case_g_flip_executor_spy_skipped(patched_guard, caplog):
    """(g) flip_executor flip_close_symbol='SPY' with a legacy
    reconciler|SPY open position → BG11_FLIP_SKIP_EXCLUDED audit row;
    paper_adapter never invoked; flipped signal dropped."""
    patched_guard.state["reconciler|SPY"] = {
        "open": True, "strategy": "reconciler", "symbol": "SPY",
        "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
    }

    decisions = [
        _flip_decision(
            symbol="SPY", existing_strategy="reconciler",
            existing_side="BUY", new_strategy="alpha",
        ),
    ]
    flipped = _flipped_signal("alpha", "SPY", "SELL")
    adapter = _RecordingAdapter()

    with caplog.at_level(logging.WARNING, logger="chad.core.flip_executor"):
        out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert adapter.calls == [], (
        "paper_adapter must not be invoked for operator-excluded flip "
        f"close. calls={adapter.calls}"
    )
    skip_events = [r for r in audit if r.get("event") == "BG11_FLIP_SKIP_EXCLUDED"]
    assert len(skip_events) == 1, (
        f"Expected exactly one BG11_FLIP_SKIP_EXCLUDED audit row; got {audit}"
    )
    assert skip_events[0]["symbol"] == "SPY"
    assert skip_events[0]["existing_strategy"] == "reconciler"
    assert skip_events[0]["new_strategy"] == "alpha"
    assert skip_events[0]["source"] == position_reconciler._EXCLUSION_SOURCE
    # Conservative semantics: flipped signal is dropped (operator-excluded
    # symbols must not be opened OR closed by CHAD).
    assert flipped not in out, (
        "Flipped signal targeting an operator-excluded symbol must be "
        "dropped — CHAD must not open positions in excluded symbols either."
    )
    # Position-guard state untouched.
    assert patched_guard.state["reconciler|SPY"]["open"] is True
    assert patched_guard.saved == []


def test_case_h_flip_executor_non_excluded_unchanged(patched_guard):
    """(h) flip_executor flip_close_symbol=<non-excluded equity> →
    behavior unchanged from baseline BG11 flow (close submitted,
    flipped signal allowed when close confirmed)."""
    sym = "IWM"  # non-excluded equity; MES would also exercise the
    # same control-flow path but flip_executor hardcodes
    # AssetClass.EQUITY in _build_flip_close_intent so an equity is the
    # cleanest behavioral test here.
    assert sym not in position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS
    patched_guard.state[f"reconciler|{sym}"] = {
        "open": True, "strategy": "reconciler", "symbol": sym,
        "side": "BUY", "quantity": 10.0, "last_state": "OPEN",
    }

    decisions = [
        _flip_decision(
            symbol=sym, existing_strategy="reconciler",
            existing_side="BUY", new_strategy="alpha",
        ),
    ]
    flipped = _flipped_signal("alpha", sym, "SELL")

    class _OkAdapter:
        def __init__(self) -> None:
            self.calls: List[List[Any]] = []

        def submit_strategy_trade_intents(self, intents):
            self.calls.append(list(intents))
            return [
                SimpleNamespace(
                    symbol=getattr(it, "symbol", "?"),
                    side=getattr(it, "side", "?"),
                    quantity=getattr(it, "quantity", 0.0),
                    status="filled",
                    submitted_at=datetime.now(timezone.utc),
                    limit_price=100.0,
                    asset_class="EQUITY",
                )
                for it in intents
            ]

    adapter = _OkAdapter()
    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    # Baseline BG11 path: adapter called for the close, flipped signal
    # survives, and the BG11_FLIP_CLOSE_CONFIRMED audit row is present.
    assert len(adapter.calls) == 1, (
        f"Baseline BG11 path must submit the close exactly once for "
        f"non-excluded symbol. calls={adapter.calls}"
    )
    assert adapter.calls[0][0].symbol == sym
    assert flipped in out, (
        "Non-excluded flip with broker-confirmed close must allow the "
        "flipped signal through."
    )
    assert any(r["event"] == "BG11_FLIP_CLOSE_CONFIRMED" for r in audit), (
        f"Expected BG11_FLIP_CLOSE_CONFIRMED in audit for non-excluded "
        f"symbol. audit={audit}"
    )
    # No BG11_FLIP_SKIP_EXCLUDED for non-excluded symbol.
    assert not any(r.get("event") == "BG11_FLIP_SKIP_EXCLUDED" for r in audit)
