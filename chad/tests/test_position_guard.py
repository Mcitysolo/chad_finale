"""ISSUE-56 regression tests: broker_sync anchor yields to strategy ownership."""
from dataclasses import dataclass

import pytest

from chad.core import position_guard


@dataclass
class _Intent:
    strategy: str
    symbol: str
    side: str
    quantity: float


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", state_path)
    return state_path


def _seed(path, state):
    import json
    path.write_text(json.dumps(state), encoding="utf-8")


def _load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def test_mark_position_open_closes_existing_broker_sync(tmp_state):
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
        }
    })
    position_guard.mark_position_open(_Intent("alpha", "SPY", "BUY", 30.0))
    state = _load(tmp_state)
    assert state["alpha|SPY"]["open"] is True
    assert state["alpha|SPY"]["quantity"] == 30.0
    assert state["broker_sync|SPY"]["open"] is False
    assert state["broker_sync|SPY"]["closed_by"] == "strategy_ownership_assumed"


def test_mark_position_open_no_anchor_noop_on_broker_sync(tmp_state):
    _seed(tmp_state, {})
    position_guard.mark_position_open(_Intent("alpha", "SPY", "BUY", 30.0))
    state = _load(tmp_state)
    assert "alpha|SPY" in state
    assert "broker_sync|SPY" not in state


def test_mark_position_open_sequential_strategies_same_symbol(tmp_state):
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
        }
    })
    position_guard.mark_position_open(_Intent("alpha", "SPY", "BUY", 15.0))
    position_guard.mark_position_open(_Intent("delta", "SPY", "BUY", 15.0))
    state = _load(tmp_state)
    assert state["broker_sync|SPY"]["open"] is False
    assert state["alpha|SPY"]["open"] is True
    assert state["delta|SPY"]["open"] is True


def test_mark_position_open_closed_broker_sync_untouched(tmp_state):
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": False, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "closed_by": "prior_close",
        }
    })
    position_guard.mark_position_open(_Intent("alpha", "SPY", "BUY", 30.0))
    state = _load(tmp_state)
    assert state["broker_sync|SPY"]["open"] is False
    assert state["broker_sync|SPY"]["closed_by"] == "prior_close"
    assert state["alpha|SPY"]["open"] is True


def test_replace_position_closes_existing_broker_sync(tmp_state):
    """Same-side full-attribution via replace_position: residual=0 → soft-close."""
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
        },
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "SELL", "quantity": 30.0, "last_state": "OPEN",
        },
    })
    position_guard.replace_position(_Intent("alpha", "SPY", "BUY", 30.0))
    state = _load(tmp_state)
    assert state["broker_sync|SPY"]["open"] is False
    assert state["broker_sync|SPY"]["closed_by"] == "strategy_ownership_assumed"
    assert state["alpha|SPY"]["open"] is True
    assert state["alpha|SPY"]["side"] == "BUY"
    assert state["alpha|SPY"]["last_state"] == "FLIPPED"


def _seed_trade_closer(path, queues):
    import json
    path.write_text(json.dumps({"queues": queues, "processed_fill_ids": []}),
                    encoding="utf-8")


def test_rebuild_clears_broker_sync_when_strategy_entry_added(tmp_path, monkeypatch):
    """
    When _rebuild_guard_from_paper_ledger writes a strategy-named entry
    for a symbol that already has a broker_sync|<symbol> entry, the
    broker_sync anchor is closed.
    """
    import logging
    from chad.core import live_loop

    guard_path = tmp_path / "position_guard.json"
    tc_path = tmp_path / "trade_closer_state.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH", tc_path)

    _seed(guard_path, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
        }
    })
    _seed_trade_closer(tc_path, [{
        "strategy": "delta", "symbol": "SPY", "sec_type": "STK",
        "lots": [{"side": "BUY", "quantity": 30.0,
                  "fill_price": 700.0, "lot_ts_utc": "2026-04-20T00:00:00Z",
                  "fill_id": "test-fill-1"}],
    }])

    live_loop._rebuild_guard_from_paper_ledger(logging.getLogger("test"))
    state = _load(guard_path)
    assert state["delta|SPY"]["open"] is True
    assert state["delta|SPY"]["quantity"] == 30.0
    assert state["broker_sync|SPY"]["open"] is False
    assert state["broker_sync|SPY"]["closed_by"] == "strategy_ownership_assumed"


def test_rebuild_preserves_broker_sync_when_no_strategy_entry_for_symbol(tmp_path, monkeypatch):
    """
    broker_sync entries for symbols without a corresponding strategy queue
    are preserved (close-sweep whitelist protects them).
    """
    import logging
    from chad.core import live_loop

    guard_path = tmp_path / "position_guard.json"
    tc_path = tmp_path / "trade_closer_state.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH", tc_path)

    _seed(guard_path, {
        "broker_sync|GLD": {
            "open": True, "strategy": "broker_sync", "symbol": "GLD",
            "side": "SELL", "quantity": 1108.0, "last_state": "OPEN",
        }
    })
    _seed_trade_closer(tc_path, [{
        "strategy": "alpha", "symbol": "SPY", "sec_type": "STK",
        "lots": [{"side": "BUY", "quantity": 10.0,
                  "fill_price": 700.0, "lot_ts_utc": "2026-04-20T00:00:00Z",
                  "fill_id": "test-fill-2"}],
    }])

    live_loop._rebuild_guard_from_paper_ledger(logging.getLogger("test"))
    state = _load(guard_path)
    assert state["broker_sync|GLD"]["open"] is True
    assert state["broker_sync|GLD"]["quantity"] == 1108.0
    assert state["alpha|SPY"]["open"] is True


def test_mark_position_open_partial_attribution_reduces_broker_sync(tmp_state):
    """ISSUE-56 v2: partial same-side attribution reduces broker_sync, keeps open."""
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 60.0, "last_state": "OPEN",
        }
    })
    position_guard.mark_position_open(_Intent("delta", "SPY", "BUY", 10.0))
    state = _load(tmp_state)
    assert state["broker_sync|SPY"]["open"] is True
    assert state["broker_sync|SPY"]["quantity"] == 50.0
    assert state["broker_sync|SPY"]["closed_by"] == "partial_attribution_residual"
    assert state["delta|SPY"]["open"] is True
    assert state["delta|SPY"]["quantity"] == 10.0
    # Publisher sum: 50 (broker_sync residual) + 10 (delta) = 60 = broker truth


def test_mark_position_open_opposite_side_does_not_reduce(tmp_state):
    """ISSUE-56 v2: opposite-side strategy is a flip intent, not attribution."""
    _seed(tmp_state, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 60.0, "last_state": "OPEN",
        }
    })
    position_guard.mark_position_open(_Intent("delta", "SPY", "SELL", 10.0))
    state = _load(tmp_state)
    assert state["broker_sync|SPY"]["open"] is True
    assert state["broker_sync|SPY"]["quantity"] == 60.0
    assert "closed_by" not in state["broker_sync|SPY"]
    assert state["delta|SPY"]["open"] is True
    assert state["delta|SPY"]["side"] == "SELL"


def test_close_stale_position_from_broker_truth_targets_only_named_entry(tmp_state):
    """Stale broker-truth close must mutate only the (strategy, symbol) key.

    Models the GAP-A001 follow-up: alpha_options|SPY is stale legacy state
    while the SPY STK position and other strategies on SPY remain valid.
    """
    _seed(tmp_state, {
        "alpha_options|SPY": {
            "open": True, "strategy": "alpha_options", "symbol": "SPY",
            "side": "BUY", "quantity": 0.5, "last_state": "MAINTAINED",
        },
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 30.0, "last_state": "OPEN",
        },
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 6.0, "last_state": "MAINTAINED",
        },
        "alpha_options|QQQ": {
            "open": True, "strategy": "alpha_options", "symbol": "QQQ",
            "side": "BUY", "quantity": 1.0, "last_state": "OPEN",
        },
    })
    evidence = {
        "broker_account": "DUK902770",
        "broker_sec_types_seen_for_symbol": ["STK"],
        "no_matching_sec_types": ["OPT", "BAG"],
        "operator_timestamp_utc": "2026-05-08T00:00:00+00:00",
        "source": "manual_operator_reconciliation_after_gap_a001",
    }

    result = position_guard.close_stale_position_from_broker_truth(
        "alpha_options", "SPY",
        reason="broker_truth_no_matching_options_position",
        evidence=evidence,
    )
    assert result is True

    state = _load(tmp_state)

    # Targeted entry closed with full audit trail, no fake fill.
    closed = state["alpha_options|SPY"]
    assert closed["open"] is False
    assert closed["last_state"] == "CLOSED"
    assert closed["closed_by"] == "broker_truth_no_matching_options_position"
    assert closed["closed_reason"] == "stale_guard_entry"
    assert closed["closed_evidence"] == evidence
    assert "closed_fill_id" not in closed, "must not fabricate a fill id"

    # Unrelated SPY stock and other-strategy entries are untouched.
    assert state["broker_sync|SPY"]["open"] is True
    assert state["broker_sync|SPY"]["quantity"] == 30.0
    assert "closed_by" not in state["broker_sync|SPY"]
    assert state["alpha|SPY"]["open"] is True
    assert state["alpha|SPY"]["quantity"] == 6.0
    assert "closed_by" not in state["alpha|SPY"]

    # Unrelated alpha_options strategy on a different symbol is untouched.
    assert state["alpha_options|QQQ"]["open"] is True
    assert "closed_by" not in state["alpha_options|QQQ"]


def test_close_stale_position_from_broker_truth_missing_key_is_noop(tmp_state):
    """No mutation and no exception if the key does not exist."""
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 6.0, "last_state": "OPEN",
        },
    })
    result = position_guard.close_stale_position_from_broker_truth(
        "alpha_options", "SPY",
        reason="broker_truth_no_matching_options_position",
        evidence={"source": "test"},
    )
    assert result is False
    state = _load(tmp_state)
    assert "alpha_options|SPY" not in state
    assert state["alpha|SPY"]["open"] is True


def test_rebuild_partial_attribution_multi_strategy(tmp_path, monkeypatch):
    """ISSUE-56 v2: rebuild reduces broker_sync across multiple strategy claims."""
    import logging
    from chad.core import live_loop

    guard_path = tmp_path / "position_guard.json"
    tc_path = tmp_path / "trade_closer_state.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH", tc_path)

    _seed(guard_path, {
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 60.0, "last_state": "OPEN",
        }
    })
    _seed_trade_closer(tc_path, [
        {
            "strategy": "alpha", "symbol": "SPY", "sec_type": "STK",
            "lots": [{"side": "BUY", "quantity": 20.0, "fill_price": 700.0,
                      "lot_ts_utc": "2026-04-20T00:00:00Z", "fill_id": "a1"}],
        },
        {
            "strategy": "delta", "symbol": "SPY", "sec_type": "STK",
            "lots": [{"side": "BUY", "quantity": 10.0, "fill_price": 700.0,
                      "lot_ts_utc": "2026-04-20T00:00:00Z", "fill_id": "d1"}],
        },
    ])

    live_loop._rebuild_guard_from_paper_ledger(logging.getLogger("test"))
    state = _load(guard_path)
    assert state["alpha|SPY"]["open"] is True
    assert state["alpha|SPY"]["quantity"] == 20.0
    assert state["delta|SPY"]["open"] is True
    assert state["delta|SPY"]["quantity"] == 10.0
    assert state["broker_sync|SPY"]["open"] is True
    assert state["broker_sync|SPY"]["quantity"] == 30.0
    assert state["broker_sync|SPY"]["closed_by"] == "partial_attribution_residual"
    # Publisher sum: 30 (broker_sync residual) + 20 (alpha) + 10 (delta) = 60 = broker truth
