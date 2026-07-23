"""W4A-7 — D5 RIDER (audits/W4A_GO_RECORD.md §2): the named regression that
proves a tripped LC5 emergency halt STILL permits (i) overlay closes and
(ii) flatten-all. "Exits always free" is a TEST here, not a placement
argument.

The prime invariant (PLAN_W4A §8.1) is enforced structurally — overlay,
reconciler and flatten closes never traverse the stage-3 fuse gate or the
Kraken risk path. These tests assert BOTH the structural fact (the close
paths don't consult the fuse gate) and the predicate belt (should_block_lc5_
entry / should_block_kraken exempt every exit shape even at emergency).
"""

from __future__ import annotations

import inspect
import types
from datetime import datetime, timezone

from chad.risk.fuse_gate import FuseGate

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)

# An LC5 state in full emergency, LC5 enforcing.
EMERGENCY_STATE = {
    "schema_version": "fuse_box_state.v1",
    "fuses": [],
    "lc5": {"factor": 0.25, "emergency": True, "staleness": "fresh"},
}
ENFORCE_ALL = {"CHAD_FUSE_LC2": "enforce", "CHAD_FUSE_LC3": "enforce",
               "CHAD_FUSE_LC5": "enforce"}


def _gate(tmp_path):
    return FuseGate(state=EMERGENCY_STATE, env=ENFORCE_ALL,
                    evidence_dir=tmp_path / "ev", now=NOW)


def _ibkr(strategy="gamma", symbol="PSQ", side="BUY", meta=None, tags=None):
    return types.SimpleNamespace(
        strategy=strategy, symbol=symbol, side=side, quantity=5.0,
        sec_type="STK", meta=meta or {}, tags=tags or [],
    )


# --------------------------------------------------------------------------- #
# (i) Emergency permits overlay/strategy closes — every exit shape
# --------------------------------------------------------------------------- #

def test_lc5_emergency_halt_permits_overlay_closes(tmp_path):
    """A −15% emergency must NOT block any close. Every exit shape passes both
    the LC5 emergency block and the LC2/LC3 gate while enforcing."""
    gate = _gate(tmp_path)
    assert gate.lc5_emergency() is True  # emergency really is active

    overlay_and_strategy_exits = [
        _ibkr(side="EXIT"),
        _ibkr(side="CLOSE"),
        # W4B-2 close-provenance stamp — the overlay/reconciler close shape.
        _ibkr(meta={"action": "CLOSE", "close_origin": "apply_close_intents"}),
        _ibkr(meta={"reason": "max_hold_exit"}),
        _ibkr(meta={"reason": "stop_loss"}),
        _ibkr(meta={"reason": "liquidation"}),
        _ibkr(tags=["reduce"]),
        _ibkr(tags=["hedge"]),
    ]
    for ex in overlay_and_strategy_exits:
        # LC5 emergency entry-block must exempt it...
        assert gate.should_block_lc5_entry(ex) is False, ex
        # ...and the LC2/LC3 gate must too (belt-and-braces).
        assert gate.should_block(ex) is None, ex


def test_lc5_emergency_blocks_a_fresh_entry(tmp_path):
    """Control: the SAME emergency DOES block a genuine new entry — proving the
    exemptions above are real carve-outs, not a dead gate."""
    gate = _gate(tmp_path)
    assert gate.should_block_lc5_entry(_ibkr(side="BUY")) is True


def test_lc5_emergency_permits_crypto_reduce_only_close(tmp_path):
    """The Kraken lane close shapes (reduce_only / overlay marker /
    crypto_exit| key) pass the mirror at emergency too."""
    gate = _gate(tmp_path)
    reduce_only = types.SimpleNamespace(
        strategy="alpha_crypto", pair="XBTUSD", side="sell", volume=0.01,
        markers=(), idempotency_key=None, reduce_only=True,
    )
    overlay = types.SimpleNamespace(
        strategy="alpha_crypto", pair="XBTUSD", side="sell", volume=0.01,
        markers=("CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE", "atr"),
        idempotency_key=None, reduce_only=False,
    )
    assert gate.should_block_kraken(reduce_only) is None
    assert gate.should_block_kraken(overlay) is None


# --------------------------------------------------------------------------- #
# (ii) Emergency permits flatten-all — structural proof
# --------------------------------------------------------------------------- #

def test_lc5_emergency_halt_permits_flatten_all():
    """Flatten-all (scripts/flatten_all.py) runs in its OWN process and reaches
    the brokers via apply_close_intents / _close_intent_to_ibkr direct — it
    never imports live_loop and never constructs a FuseGate. Structural proof:
    the flatten module's source references neither the stage-3 gate nor the
    fuse gate, so no fuse (LC2/LC3/LC5 emergency included) can be on its path."""
    import scripts.flatten_all as fa

    src = inspect.getsource(fa)
    # The flatten close path must not consult the fuse gate in any form.
    assert "FuseGate" not in src
    assert "fuse_gate" not in src
    assert "should_block_lc5_entry" not in src
    # It reaches closes via the reconciler direct path (bypasses stage-3).
    assert "apply_close_intents" in src


def test_reconciler_close_path_has_no_fuse_gate():
    """apply_close_intents / _close_intent_to_ibkr (the overlay + flatten +
    reconciler close authority) must not consult the fuse gate — structural
    guarantee that closes bypass stage-3 enforcement entirely."""
    import chad.core.position_reconciler as pr

    src = inspect.getsource(pr)
    assert "FuseGate" not in src
    assert "should_block" not in src


def test_kraken_exit_overlay_close_path_has_no_fuse_gate():
    """The crypto exit overlay's close submission must not consult the fuse
    gate either (reduce-only closes are sacrosanct)."""
    import chad.risk.crypto_exit_overlay as ceo

    src = inspect.getsource(ceo)
    assert "FuseGate" not in src
