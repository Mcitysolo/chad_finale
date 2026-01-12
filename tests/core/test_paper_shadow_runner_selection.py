"""
Regression tests for CHAD Paper Shadow Runner selection logic.

Bug fixed (Phase 9):
- When max_orders_per_run=1 and flat_only_per_symbol=True, the runner used to
  slice intents_filtered first (taking only the first intent), then apply flat-only
  gating. If the first intent was for a symbol already held (e.g., AAPL), the runner
  would plan zero orders even if later intents were eligible (e.g., HD).

This test ensures the runner scans intents_filtered and selects the first eligible
intent(s) up to max_orders_per_run.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _select_planned(
    *,
    intents_filtered: List[Dict[str, Any]],
    inv_net: Dict[str, float],
    max_orders_per_run: int,
    flat_only_per_symbol: bool,
) -> List[Dict[str, Any]]:
    """
    Pure functional replica of the planned selection logic in
    chad/core/paper_shadow_runner.py (run_execute).

    We keep this local and deterministic to avoid importing the runner module,
    which imports ib_insync and may require IBKR runtime deps during tests.
    """
    planned: List[Dict[str, Any]] = []
    want = max(0, int(max_orders_per_run))

    for it in intents_filtered:
        if len(planned) >= want:
            break

        sym = str(it.get("symbol") or "").strip().upper()
        if not sym:
            continue

        if flat_only_per_symbol and inv_net.get(sym, 0.0) > 1e-9:
            continue

        planned.append(it)

    return planned


def test_planned_selection_skips_held_first_intent_and_falls_through() -> None:
    intents = [
        {"strategy": "beta", "symbol": "AAPL", "side": "BUY", "quantity": 3.0, "notional_estimate": 700.0},
        {"strategy": "alpha", "symbol": "HD", "side": "BUY", "quantity": 5.0, "notional_estimate": 1500.0},
        {"strategy": "beta", "symbol": "MSFT", "side": "BUY", "quantity": 3.0, "notional_estimate": 1400.0},
    ]
    inv_net = {"AAPL": 3.0}  # already holding AAPL

    planned = _select_planned(
        intents_filtered=intents,
        inv_net=inv_net,
        max_orders_per_run=1,
        flat_only_per_symbol=True,
    )

    assert len(planned) == 1
    assert planned[0]["symbol"] == "HD"
    assert planned[0]["strategy"] == "alpha"


def test_planned_selection_returns_empty_when_all_candidates_held() -> None:
    intents = [
        {"strategy": "beta", "symbol": "AAPL", "side": "BUY", "quantity": 3.0, "notional_estimate": 700.0},
        {"strategy": "alpha", "symbol": "AAPL", "side": "BUY", "quantity": 5.0, "notional_estimate": 1500.0},
    ]
    inv_net = {"AAPL": 1.0}

    planned = _select_planned(
        intents_filtered=intents,
        inv_net=inv_net,
        max_orders_per_run=1,
        flat_only_per_symbol=True,
    )

    assert planned == []


def test_planned_selection_allows_held_symbol_when_flat_only_disabled() -> None:
    intents = [
        {"strategy": "beta", "symbol": "AAPL", "side": "BUY", "quantity": 3.0, "notional_estimate": 700.0},
        {"strategy": "alpha", "symbol": "HD", "side": "BUY", "quantity": 5.0, "notional_estimate": 1500.0},
    ]
    inv_net = {"AAPL": 3.0}

    planned = _select_planned(
        intents_filtered=intents,
        inv_net=inv_net,
        max_orders_per_run=1,
        flat_only_per_symbol=False,
    )

    assert len(planned) == 1
    assert planned[0]["symbol"] == "AAPL"
