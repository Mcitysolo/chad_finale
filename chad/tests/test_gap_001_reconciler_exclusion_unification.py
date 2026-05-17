"""GAP-001 regression guard.

Defect (Phase-4 audit): chad/core/position_reconciler.py only consulted its
own hardcoded floor ``KNOWN_NON_CHAD_SYMBOLS = {"AAPL","MSFT"}`` and ignored
the operator-owned exclusion policy in
``config/reconciliation_exclusions.json`` (loaded by
``chad/ops/reconciliation_publisher.py``). The reconciler therefore emitted
SPY close intents from ``delta|SPY`` guard entries while SPY is documented
as a pre-existing broker position — visible as the long-running
``fill_price=100`` / ``pnl_untrusted=True`` SPY/SELL stream in
``data/fills/FILLS_*.ndjson``.

Remediation (Phase-6): the reconciler now builds ``_EFFECTIVE_NON_CHAD_SYMBOLS``
by union of (a) its own floor, (b) the publisher's loaded
``KNOWN_NON_CHAD_SYMBOLS``, and (c) the keys of the publisher's
``EXCLUSION_POLICY``. The skip predicate at the head of
``reconcile_positions_with_signals`` uses that effective set, with case
normalization, so operator-excluded symbols can never reach the close-intent
emit block.

These tests fail loudly if any change re-narrows the reconciler's effective
exclusion set or reintroduces the SPY close-emission path.
"""
from __future__ import annotations

from typing import List

import chad.core.position_reconciler as position_reconciler
from chad.core.position_reconciler import reconcile_positions_with_signals
from chad.ops.reconciliation_publisher import (
    EXCLUSION_POLICY as PUBLISHER_EXCLUSION_POLICY,
    KNOWN_NON_CHAD_SYMBOLS as PUBLISHER_NON_CHAD,
)


GAP_001_OPERATOR_EXCLUSIONS = frozenset(
    {"AAPL", "BAC", "CVX", "LLY", "MSFT", "NVDA", "PEP", "QQQ", "SPY"}
)


class _FakeSignal:
    """Minimal stand-in matching the attribute surface read by the
    reconciler (symbol, side, and a size attribute).

    ``chad/core/position_reconciler.py:_signal_size`` probes
    ``size``/``qty``/``quantity``/``net_size`` — providing ``size`` is
    sufficient and stable.
    """

    def __init__(self, symbol: str, side: str, size: float = 1.0) -> None:
        self.symbol = symbol
        self.side = side
        self.size = size


def test_effective_set_contains_full_operator_exclusions():
    eff = position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS
    missing = sorted(sym for sym in GAP_001_OPERATOR_EXCLUSIONS if sym not in eff)
    assert not missing, (
        "_EFFECTIVE_NON_CHAD_SYMBOLS must cover the full operator exclusion "
        f"set; missing={missing} (regression of GAP-001)"
    )
    assert position_reconciler._EXCLUSION_SOURCE == "unified_publisher_config", (
        "_EXCLUSION_SOURCE must report unified_publisher_config when the "
        "publisher import succeeds — local_floor_fallback indicates a broken "
        "import chain that would re-expose GAP-001."
    )


def test_effective_set_superset_of_publisher():
    eff = position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS
    normalized_publisher = frozenset(str(s).upper() for s in PUBLISHER_NON_CHAD)
    normalized_policy = frozenset(str(s).upper() for s in PUBLISHER_EXCLUSION_POLICY)
    assert normalized_publisher <= eff, (
        "_EFFECTIVE_NON_CHAD_SYMBOLS must be a superset of "
        "reconciliation_publisher.KNOWN_NON_CHAD_SYMBOLS"
    )
    assert normalized_policy <= eff, (
        "_EFFECTIVE_NON_CHAD_SYMBOLS must be a superset of the publisher's "
        "EXCLUSION_POLICY keys"
    )


def test_reconciler_emits_no_close_for_spy_gap_001():
    """Exact GAP-001 scenario: open delta|SPY BUY 30 + opposing SELL signal."""
    open_positions = {
        "delta|SPY": {
            "open": True,
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 30.0,
            "strategy": "delta",
            "position_key": "delta|SPY",
        }
    }
    routed_signals: List = [_FakeSignal("SPY", "SELL", size=30.0)]
    closes = reconcile_positions_with_signals(
        open_positions=open_positions,
        routed_signals=routed_signals,
        prices={},
    )
    spy_closes = [c for c in closes if c.get("symbol") == "SPY"]
    assert spy_closes == [], (
        "GAP-001: reconciler must not emit close intents for SPY (operator "
        f"exclusion). Got: {spy_closes}"
    )


def test_reconciler_still_closes_non_excluded():
    """Negative control — a fabricated non-excluded symbol must still flip."""
    sym = "ZZTESTSYM"
    assert sym not in position_reconciler._EFFECTIVE_NON_CHAD_SYMBOLS, (
        "Test sentinel symbol unexpectedly in exclusion set — choose a "
        "different sentinel."
    )
    open_positions = {
        f"delta|{sym}": {
            "open": True,
            "symbol": sym,
            "side": "BUY",
            "quantity": 30.0,
            "strategy": "delta",
            "position_key": f"delta|{sym}",
        }
    }
    routed_signals: List = [_FakeSignal(sym, "SELL", size=30.0)]
    closes = reconcile_positions_with_signals(
        open_positions=open_positions,
        routed_signals=routed_signals,
        prices={},
    )
    target = [c for c in closes if c.get("symbol") == sym]
    assert len(target) == 1, (
        "Reconciler must still emit a close intent for a non-excluded symbol "
        f"on a clear signal flip. Got: {closes}"
    )
    assert target[0].get("close_side") == "SELL", target[0]
    assert target[0].get("open_side") == "BUY", target[0]
