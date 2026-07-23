"""W4A-5 — per-intent fuse gate (IBKR stage-3 + Kraken mirror).

The prime-invariant tests (PLAN_W4A §8.1): every exit shape passes under every
mode, including enforce with a covering bucket tripped. Plus the margin-gate
contract: shadow is byte-identical (block=None), enforce blocks only a covering
entry, fail-open on any error.
"""

from __future__ import annotations

import json
import types
from datetime import datetime, timezone

from chad.risk.fuse_gate import FuseGate, is_exit_like

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)


def _ibkr_intent(strategy="gamma", symbol="PSQ", side="BUY", meta=None, tags=None):
    return types.SimpleNamespace(
        strategy=strategy, symbol=symbol, side=side, quantity=5.0,
        sec_type="STK", meta=meta or {}, tags=tags or [],
    )


def _kraken_intent(strategy="alpha_crypto", pair="XBTUSD", side="buy",
                   markers=(), idempotency_key=None, reduce_only=False):
    return types.SimpleNamespace(
        strategy=strategy, pair=pair, side=side, volume=0.01,
        markers=markers, idempotency_key=idempotency_key,
        reduce_only=reduce_only,
    )


def _tripped_state(fuse_id="family:gamma", kind="family"):
    return {
        "schema_version": "fuse_box_state.v1",
        "fuses": [
            {"fuse_id": fuse_id, "kind": kind, "tripped": True,
             "trip_rule": "consecutive_losers", "consecutive_losers": 3},
        ],
    }


def _gate(state, env, tmp_path):
    return FuseGate(state=state, env=env, evidence_dir=tmp_path / "ev", now=NOW)


# --------------------------------------------------------------------------- #
# is_exit_like — every close shape (§8.1)
# --------------------------------------------------------------------------- #

def test_exit_like_side_exit():
    assert is_exit_like(_ibkr_intent(side="EXIT"))
    assert is_exit_like(_ibkr_intent(side="CLOSE"))


def test_exit_like_close_stamp():
    """W4B-2 close-provenance stamp (belt over structural bypass)."""
    assert is_exit_like(_ibkr_intent(meta={"action": "CLOSE"}))
    assert is_exit_like(_ibkr_intent(meta={"close_origin": "apply_close_intents",
                                           "action": "CLOSE"}))


def test_exit_like_protective_tags():
    for tag in ("reduce", "hedge", "stop_loss", "liquidation"):
        assert is_exit_like(_ibkr_intent(tags=[tag])), tag
        assert is_exit_like(_ibkr_intent(meta={"tags": [tag]})), tag


def test_exit_like_protective_reason():
    assert is_exit_like(_ibkr_intent(meta={"reason": "max_hold_exit"}))
    assert is_exit_like(_ibkr_intent(meta={"reason": "stop_loss"}))


def test_entry_is_not_exit_like():
    assert not is_exit_like(_ibkr_intent(side="BUY"))
    assert not is_exit_like(_ibkr_intent(side="SELL", meta={"reason": "signal"}))


# --------------------------------------------------------------------------- #
# Prime invariant: a close ALWAYS passes, even enforce + tripped
# --------------------------------------------------------------------------- #

def test_enforce_tripped_still_passes_every_exit_shape(tmp_path):
    env = {"CHAD_FUSE_LC2": "enforce", "CHAD_FUSE_LC3": "enforce"}
    gate = _gate(_tripped_state(), env, tmp_path)
    exits = [
        _ibkr_intent(side="EXIT"),
        _ibkr_intent(side="CLOSE"),
        _ibkr_intent(meta={"action": "CLOSE"}),
        _ibkr_intent(tags=["reduce"]),
        _ibkr_intent(tags=["stop_loss"]),
        _ibkr_intent(meta={"reason": "liquidation"}),
    ]
    for ex in exits:
        assert gate.should_block(ex) is None, ex


# --------------------------------------------------------------------------- #
# Shadow byte-identical / enforce blocks / off passes
# --------------------------------------------------------------------------- #

def test_off_blocks_nothing(tmp_path):
    gate = _gate(_tripped_state(), {}, tmp_path)
    assert gate.should_block(_ibkr_intent()) is None
    assert not gate.any_active


def test_shadow_never_blocks_but_emits(tmp_path):
    gate = _gate(_tripped_state(), {"CHAD_FUSE_LC2": "shadow"}, tmp_path)
    assert gate.should_block(_ibkr_intent()) is None  # byte-identical
    rows = [
        json.loads(l)
        for l in (tmp_path / "ev" / "fuse_box_20260723.ndjson")
        .read_text().splitlines()
    ]
    assert rows[0]["event"] == "would_block"
    assert rows[0]["fuse_id"] == "family:gamma"
    assert rows[0]["mode"] == "shadow"


def test_enforce_blocks_covering_entry(tmp_path):
    gate = _gate(_tripped_state(), {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    verdict = gate.should_block(_ibkr_intent())
    assert verdict is not None
    assert verdict.blocked and verdict.fuse_id == "family:gamma"
    rows = [
        json.loads(l)
        for l in (tmp_path / "ev" / "fuse_box_20260723.ndjson")
        .read_text().splitlines()
    ]
    assert rows[0]["event"] == "block"


def test_enforce_does_not_block_uncovered_entry(tmp_path):
    """A tripped family:gamma must not block an alpha entry."""
    gate = _gate(_tripped_state(), {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block(_ibkr_intent(strategy="alpha", symbol="SPY")) is None


def test_lc3_flag_governs_symbol_bucket_only(tmp_path):
    """A tripped symbol bucket blocks only when LC3 (not LC2) is enforce."""
    state = _tripped_state(fuse_id="symbol:TLT", kind="symbol")
    intent = _ibkr_intent(strategy="gamma", symbol="TLT")
    # LC2 enforce, LC3 off → symbol bucket does NOT block.
    g1 = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert g1.should_block(intent) is None
    # LC3 enforce → blocks.
    g2 = _gate(state, {"CHAD_FUSE_LC3": "enforce"}, tmp_path / "b")
    assert g2.should_block(intent) is not None


def test_setup_bucket_matches_intent_meta(tmp_path):
    state = _tripped_state(fuse_id="setup:alpha_intraday:ORB", kind="setup")
    intent = _ibkr_intent(strategy="alpha_intraday", symbol="SPY",
                          meta={"setup_family": "ORB"})
    gate = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block(intent) is not None
    # different setup family → no block
    other = _ibkr_intent(strategy="alpha_intraday", symbol="SPY",
                         meta={"setup_family": "VWAP_RECLAIM"})
    assert gate.should_block(other) is None


# --------------------------------------------------------------------------- #
# Fail-open
# --------------------------------------------------------------------------- #

def test_fail_open_on_broken_state(tmp_path):
    """A malformed fuses row must never block (fail-open)."""
    bad = {"fuses": [{"garbage": True}]}
    gate = _gate(bad, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block(_ibkr_intent()) is None


def test_missing_state_blocks_nothing(tmp_path):
    gate = FuseGate(state_path=tmp_path / "nope.json",
                    env={"CHAD_FUSE_LC2": "enforce"},
                    evidence_dir=tmp_path / "ev", now=NOW)
    assert gate.should_block(_ibkr_intent()) is None


# --------------------------------------------------------------------------- #
# Kraken mirror
# --------------------------------------------------------------------------- #

def test_kraken_family_block_on_alpha(tmp_path):
    """family:alpha covers alpha_crypto — an equity-driven family trip blocks
    a crypto entry for the same family (intended cross-instrument semantics)."""
    state = _tripped_state(fuse_id="family:alpha", kind="family")
    gate = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    v = gate.should_block_kraken(_kraken_intent(strategy="alpha_crypto"))
    assert v is not None and v.fuse_id == "family:alpha"


def test_kraken_reduce_only_exempt(tmp_path):
    state = _tripped_state(fuse_id="family:alpha", kind="family")
    gate = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block_kraken(
        _kraken_intent(strategy="alpha_crypto", reduce_only=True)
    ) is None


def test_kraken_overlay_marker_exempt(tmp_path):
    state = _tripped_state(fuse_id="family:alpha", kind="family")
    gate = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block_kraken(
        _kraken_intent(strategy="alpha_crypto",
                       markers=("CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE", "atr_trail"))
    ) is None


def test_kraken_crypto_exit_key_exempt(tmp_path):
    state = _tripped_state(fuse_id="family:alpha", kind="family")
    gate = _gate(state, {"CHAD_FUSE_LC2": "enforce"}, tmp_path)
    assert gate.should_block_kraken(
        _kraken_intent(strategy="alpha_crypto",
                       idempotency_key="crypto_exit|alpha_crypto|BTC-USD|sell|0.01")
    ) is None


def test_kraken_shadow_never_blocks(tmp_path):
    state = _tripped_state(fuse_id="family:alpha", kind="family")
    gate = _gate(state, {"CHAD_FUSE_LC2": "shadow"}, tmp_path)
    assert gate.should_block_kraken(_kraken_intent(strategy="alpha_crypto")) is None
    rows = [
        json.loads(l)
        for l in (tmp_path / "ev" / "fuse_box_20260723.ndjson")
        .read_text().splitlines()
    ]
    assert rows[0]["lane"] == "kraken" and rows[0]["event"] == "would_block"
