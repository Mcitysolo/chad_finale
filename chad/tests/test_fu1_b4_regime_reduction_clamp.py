"""
FU1-B4 (FLIP-UNBLOCK 2026-07-17): regime-reduction broker clamp.

ULTRA_CLOSE_AUDIT §B-4 proved the regime-reduction leg reproduces INCIDENT-0713 exactly:
``generate_partial_close_intents`` sized the reduction off the *ledger* end with NO broker-truth
clamp of any kind. When the ledger end is inflated relative to broker truth — e.g. a just-filled
close the guard has not yet reflected — the reduction sells into a smaller real end (the audit's
409-sold-vs-273-held) and flips the position short. INCIDENT-0713 was the same shape: SELL 1340
sized to an inflated FIFO 1260 against a real 700-long TLT.

These tests pin the clamp:

  * an oversell is clamped to the broker-held same-side quantity (never short);
  * a phantom / already-flat / opposite-side broker end drops the reduction entirely;
  * the exact TLT arithmetic (inflated ledger, smaller broker) can never go net short;
  * broker_sync|* / bookkeeping keys are never treated as reducible positions;
  * broker truth is derived correctly from broker_sync|* guard entries (open flag ignored);
  * the legacy (no broker truth) call path is byte-for-byte unchanged.
"""
from __future__ import annotations

import pytest

from chad.risk.regime_reduction import (
    broker_signed_qty_by_symbol,
    generate_partial_close_intents,
    handle_regime_transition,
)


# --------------------------------------------------------------------------- #
# broker_signed_qty_by_symbol — broker truth derivation
# --------------------------------------------------------------------------- #
def test_broker_signed_reads_long_and_short_from_broker_sync():
    guard = {
        "gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0, "open": True},
        "broker_sync|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0, "open": True},
        "broker_sync|TLT": {"symbol": "TLT", "side": "SELL", "quantity": 640.0, "open": True},
    }
    signed = broker_signed_qty_by_symbol(guard)
    assert signed["UNH"] == pytest.approx(273.0)   # long → positive
    assert signed["TLT"] == pytest.approx(-640.0)  # short → negative
    # the strategy leg (gamma|UNH) is NOT broker truth and must not be counted
    assert set(signed) == {"UNH", "TLT"}


def test_broker_signed_ignores_open_flag_zero_qty_mirror():
    # A stale broker_sync|TLT open=False qty=0.0 correctly contributes 0 (the live case the
    # audit flagged as 3.7 days stale but harmless because qty=0).
    guard = {"broker_sync|TLT": {"symbol": "TLT", "side": "SELL", "quantity": 0.0, "open": False}}
    assert broker_signed_qty_by_symbol(guard).get("TLT", 0.0) == pytest.approx(0.0)


def test_broker_signed_handles_junk():
    assert broker_signed_qty_by_symbol(None) == {}
    assert broker_signed_qty_by_symbol({"broker_sync|X": "not-a-dict"}) == {}
    assert broker_signed_qty_by_symbol({"broker_sync|": {"symbol": "", "quantity": 5}}) == {}


# --------------------------------------------------------------------------- #
# The clamp — never short
# --------------------------------------------------------------------------- #
def test_incident_0713_oversell_is_clamped_to_broker_held():
    # Ledger says 1260 (inflated FIFO), broker really holds 700 long. A 100% reduction would
    # SELL 1260 into a 700-long → flip short 560. Clamped: SELL exactly 700 → flat, never short.
    open_positions = {"agent|TLT": {"symbol": "TLT", "side": "BUY", "quantity": 1260.0}}
    broker = {"TLT": 700.0}
    intents = generate_partial_close_intents(
        open_positions, 1.0, "test", broker_signed_by_symbol=broker
    )
    assert len(intents) == 1
    assert intents[0]["quantity"] == pytest.approx(700.0)
    assert intents[0]["close_side"] == "SELL"


def test_409_vs_273_regime_leg_never_exceeds_broker_held():
    # The audit's exact scenario: guard still shows UNH 273 open (confirmation gap), regime
    # proposes int(273*0.5)=136. With broker truth the leg is capped at broker-held; it can
    # never — by itself — sell more than the broker holds, so it can never flip short.
    open_positions = {"gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0}}
    # Broker end already reduced to 100 (e.g. an earlier leg partially closed). 136 > 100.
    intents = generate_partial_close_intents(
        open_positions, 0.5, "regime_change", broker_signed_by_symbol={"UNH": 100.0}
    )
    assert intents[0]["quantity"] == pytest.approx(100.0)  # clamped down from 136


def test_phantom_broker_end_drops_the_reduction():
    # Broker holds nothing on that symbol (already flat / phantom ledger). No reduction at all.
    open_positions = {"gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0}}
    intents = generate_partial_close_intents(
        open_positions, 0.5, "regime_change", broker_signed_by_symbol={}
    )
    assert intents == []


def test_opposite_side_broker_end_drops_the_reduction():
    # Ledger claims a long, but the broker is actually SHORT the symbol. Selling would deepen
    # the short — must drop, never propose.
    open_positions = {"gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0}}
    intents = generate_partial_close_intents(
        open_positions, 0.5, "regime_change", broker_signed_by_symbol={"UNH": -50.0}
    )
    assert intents == []


def test_short_position_clamped_against_broker_short_qty():
    # Symmetric: a SELL-side (short) position reduces by BUYing to cover; the cover qty is
    # clamped to the broker-held short size so it can never over-cover into a long.
    open_positions = {"omega|SPY": {"symbol": "SPY", "side": "SELL", "quantity": 500.0}}
    intents = generate_partial_close_intents(
        open_positions, 1.0, "test", broker_signed_by_symbol={"SPY": -120.0}
    )
    assert intents[0]["close_side"] == "BUY"
    assert intents[0]["quantity"] == pytest.approx(120.0)  # clamped to broker short size


def test_reduction_under_broker_held_is_not_clamped():
    # When the reduction is genuinely within broker truth, nothing changes.
    open_positions = {"alpha|SPY": {"symbol": "SPY", "side": "BUY", "quantity": 100.0}}
    intents = generate_partial_close_intents(
        open_positions, 0.5, "test", broker_signed_by_symbol={"SPY": 999.0}
    )
    assert intents[0]["quantity"] == pytest.approx(50.0)


def test_clamp_to_below_one_unit_drops_intent():
    # Broker holds < 1 whole unit on that side → int(broker_held) == 0 → drop (never a
    # fractional or zero close).
    open_positions = {"alpha|SPY": {"symbol": "SPY", "side": "BUY", "quantity": 100.0}}
    intents = generate_partial_close_intents(
        open_positions, 0.5, "test", broker_signed_by_symbol={"SPY": 0.4}
    )
    assert intents == []


# --------------------------------------------------------------------------- #
# Bookkeeping-key hygiene
# --------------------------------------------------------------------------- #
def test_broker_sync_and_underscore_keys_are_never_reduced():
    open_positions = {
        "alpha|SPY": {"symbol": "SPY", "side": "BUY", "quantity": 100.0},
        "broker_sync|SPY": {"symbol": "SPY", "side": "BUY", "quantity": 100.0, "open": True},
        "_meta": {"symbol": "SPY", "side": "BUY", "quantity": 100.0},
    }
    intents = generate_partial_close_intents(
        open_positions, 0.5, "test", broker_signed_by_symbol={"SPY": 999.0}
    )
    # Only the real strategy leg produces an intent — the mirror and meta keys are skipped, so
    # broker truth is never double-counted as a reducible lot.
    assert len(intents) == 1
    assert intents[0]["position_key"] == "alpha|SPY"


# --------------------------------------------------------------------------- #
# Backward compatibility — no broker truth == pre-B-4 behaviour
# --------------------------------------------------------------------------- #
def test_legacy_unclamped_path_is_unchanged():
    open_positions = {
        "alpha|SPY": {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "quantity": 100.0},
        "gamma|QQQ": {"strategy": "gamma", "symbol": "QQQ", "side": "SELL", "quantity": 50.0},
    }
    # No broker_signed_by_symbol → clamp disabled, identical to the pre-B-4 result.
    intents = generate_partial_close_intents(open_positions, 0.5, "test_reason")
    by_symbol = {i["symbol"]: i["quantity"] for i in intents}
    assert by_symbol == {"SPY": pytest.approx(50.0), "QQQ": pytest.approx(25.0)}


# --------------------------------------------------------------------------- #
# handle_regime_transition threads broker truth
# --------------------------------------------------------------------------- #
def test_handle_regime_transition_threads_broker_clamp():
    open_positions = {"gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0}}
    # Broker only holds 40 long → the 136-share adverse reduction is clamped to 40.
    result = handle_regime_transition(
        from_regime="trending_bull",
        to_regime="adverse",
        open_positions=open_positions,
        broker_signed_by_symbol={"UNH": 40.0},
    )
    assert result["decision"]["should_reduce"] is True
    assert len(result["close_intents"]) == 1
    assert result["close_intents"][0]["quantity"] == pytest.approx(40.0)


def test_handle_regime_transition_phantom_drops_all_closes():
    open_positions = {"gamma|UNH": {"symbol": "UNH", "side": "BUY", "quantity": 273.0}}
    result = handle_regime_transition(
        from_regime="trending_bull",
        to_regime="adverse",
        open_positions=open_positions,
        broker_signed_by_symbol={},  # broker holds nothing → nothing to reduce
    )
    assert result["decision"]["should_reduce"] is True
    assert result["close_intents"] == []


def test_handle_regime_transition_no_broker_arg_is_legacy():
    open_positions = {"alpha|SPY": {"symbol": "SPY", "side": "BUY", "quantity": 100.0}}
    result = handle_regime_transition(
        from_regime="trending_bull", to_regime="adverse", open_positions=open_positions,
    )
    assert len(result["close_intents"]) == 1
    assert result["close_intents"][0]["quantity"] == pytest.approx(50.0)
