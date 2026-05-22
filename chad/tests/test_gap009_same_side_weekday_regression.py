"""GAP-009 weekday same-side regression — `is_same_side_open` over-block guard.

These regressions pin the three failure modes the 2026-05-07 audit identified
as the root cause of strategies producing only 5/16 fills over 7 days:

  1. Absent record (no prior entry for strategy|symbol) must NOT be flagged
     as same-side — a brand-new valid entry must pass the guard.
  2. Stale `open=False` entries (paper_ledger_rebuild / mark_position_closed)
     must NOT block, even when their `last_state` is still `"OPEN"` (which
     happens for `broker_truth_rebuild` / `paper_ledger_rebuild`).
  3. Cross-strategy isolation: same symbol under a different strategy is
     a different key (`<strategy>|<symbol>`), so opening on alpha must not
     block delta on the same symbol.

A fourth pin keeps futures-meta intents (contract_month, etc.) on the
non-blocking path so the live-loop M6E/MCL/MGC fills observed under
the trending_bull regime keep flowing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pytest

from chad.core import position_guard


@dataclass
class _Intent:
    strategy: str
    symbol: str
    side: str
    quantity: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", state_path)
    return state_path


def _seed(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_absent_record_allows_new_entry(tmp_state):
    """Brand-new strategy|symbol intent with no prior record must pass the guard."""
    _seed(tmp_state, {})

    intent = _Intent("alpha_futures", "MGC", "SELL", 1.0)
    assert position_guard.is_same_side_open(intent) is False
    assert position_guard.get_open_position(intent) is None
    assert position_guard.has_open_position(intent) is False
    assert position_guard.is_flip_signal(intent) is False


def test_closed_record_open_false_does_not_block(tmp_state):
    """Stale `open=False` entry must NOT trigger same-side suppression.

    Pin against the 2026-05-07 audit finding: paper_ledger_rebuild closes the
    entry but historically left `last_state: "OPEN"` in place. The guard reads
    `record.get("open") is True`; the residual `last_state` must not flip
    behaviour back to blocking.
    """
    _seed(tmp_state, {
        "delta|MGC": {
            "open": False,
            "strategy": "delta",
            "symbol": "MGC",
            "side": "SELL",
            "quantity": 1.0,
            # Notice: paper_ledger_rebuild stamps this even after close.
            "last_state": "OPEN",
            "closed_by": "paper_ledger_rebuild",
        },
    })

    intent = _Intent("delta", "MGC", "SELL", 1.0)
    assert position_guard.is_same_side_open(intent) is False
    assert position_guard.get_open_position(intent) is None

    # And the on-disk record must not be rewritten by the read-only check.
    after = _load(tmp_state)
    assert after["delta|MGC"]["open"] is False
    assert after["delta|MGC"]["last_state"] == "OPEN"


def test_different_strategy_same_symbol_does_not_block(tmp_state):
    """`<strategy>|<symbol>` keying isolates strategies sharing a symbol."""
    _seed(tmp_state, {
        "alpha|MSFT": {
            "open": True,
            "strategy": "alpha",
            "symbol": "MSFT",
            "side": "BUY",
            "quantity": 3.0,
            "last_state": "OPEN",
        },
    })

    # Same symbol, same side, but a different strategy → different key.
    cross = _Intent("delta", "MSFT", "BUY", 2.0)
    assert position_guard.is_same_side_open(cross) is False
    assert position_guard.get_open_position(cross) is None

    # The originating strategy's own intent is still blocked.
    own = _Intent("alpha", "MSFT", "BUY", 3.0)
    assert position_guard.is_same_side_open(own) is True


def test_open_true_same_side_blocks_true_duplicate(tmp_state):
    """Sanity pin: a genuine same-side same-strategy duplicate stays blocked."""
    _seed(tmp_state, {
        "gamma_futures|MCL": {
            "open": True,
            "strategy": "gamma_futures",
            "symbol": "MCL",
            "side": "SELL",
            "quantity": 737.0,
            "last_state": "MAINTAINED",
        },
    })
    duplicate = _Intent("gamma_futures", "MCL", "SELL", 2.0)
    assert position_guard.is_same_side_open(duplicate) is True


def test_futures_meta_intent_no_record_passes_guard(tmp_state):
    """A futures intent carrying contract_month meta still passes when no
    prior entry exists — meta must not introduce false same-side matches."""
    _seed(tmp_state, {})
    intent = _Intent(
        "omega_macro",
        "M6E",
        "BUY",
        2.0,
        meta={
            "contract_month": "202606",
            "contract_month_source": "chad.market_data.futures_contract_resolver",
        },
    )
    assert position_guard.is_same_side_open(intent) is False
    assert position_guard.get_open_position(intent) is None


def test_closed_record_with_open_record_other_side_allows_flip(tmp_state):
    """An open record on the opposite side is a flip, not a same-side block."""
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True,
            "strategy": "alpha",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 100.0,
            "last_state": "OPEN",
        },
    })
    flip = _Intent("alpha", "SPY", "SELL", 100.0)
    assert position_guard.is_same_side_open(flip) is False
    assert position_guard.is_flip_signal(flip) is True
