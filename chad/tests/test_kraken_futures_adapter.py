"""Phase C Item 1B — Kraken Futures adapter scaffold tests.

No live private Kraken Futures network calls. No credentials required.
"""

from __future__ import annotations

import sys

import pytest

from chad.exchanges.kraken_futures_client import (
    KrakenFuturesClient,
    KrakenFuturesOrderRequest,
)
from chad.execution.kraken_futures_adapter import (
    KrakenFuturesAdapter,
    KrakenFuturesIntent,
)


# ---------------------------------------------------------------------------
# 1-4. normalize_symbol mapping
# ---------------------------------------------------------------------------

def test_normalize_symbol_btc_usd() -> None:
    assert KrakenFuturesAdapter.normalize_symbol("BTC-USD") == "PF_XBTUSD"


def test_normalize_symbol_eth_usd() -> None:
    assert KrakenFuturesAdapter.normalize_symbol("ETH-USD") == "PF_ETHUSD"


def test_normalize_symbol_sol_usd() -> None:
    assert KrakenFuturesAdapter.normalize_symbol("SOL-USD") == "PF_SOLUSD"


def test_normalize_symbol_pf_passthrough() -> None:
    assert KrakenFuturesAdapter.normalize_symbol("PF_XBTUSD") == "PF_XBTUSD"


def test_normalize_symbol_xbt_alias() -> None:
    assert KrakenFuturesAdapter.normalize_symbol("XBT-USD") == "PF_XBTUSD"


def test_normalize_symbol_unmapped_raises() -> None:
    with pytest.raises(ValueError):
        KrakenFuturesAdapter.normalize_symbol("DOGE-USD")


# ---------------------------------------------------------------------------
# 5. build_order_request maps intent to request
# ---------------------------------------------------------------------------

def test_build_order_request_maps_intent_fields() -> None:
    adapter = KrakenFuturesAdapter(dry_run=True)
    intent = KrakenFuturesIntent(
        symbol="ETH-USD",
        side="SELL",
        quantity=0.25,
        order_type="LMT",
        limit_price=3100.5,
        reduce_only=True,
        strategy="phase_c_test",
        reason="unit_test",
    )
    req = adapter.build_order_request(intent)
    assert isinstance(req, KrakenFuturesOrderRequest)
    assert req.symbol == "PF_ETHUSD"
    assert req.side == "sell"
    assert req.order_type == "lmt"
    assert req.size == 0.25
    assert req.limit_price == 3100.5
    assert req.reduce_only is True


# ---------------------------------------------------------------------------
# 6. submit_intent dry-run returns dry-run result
# ---------------------------------------------------------------------------

def test_submit_intent_dry_run_returns_dry_run_result() -> None:
    adapter = KrakenFuturesAdapter(dry_run=True)
    intent = KrakenFuturesIntent(
        symbol="BTC-USD",
        side="buy",
        quantity=0.01,
        order_type="mkt",
        strategy="phase_c_test",
    )
    result = adapter.submit_intent(intent)
    assert result.ok is True
    assert result.dry_run is True
    assert result.status == "dry_run_accepted"
    assert result.request["symbol"] == "PF_XBTUSD"


# ---------------------------------------------------------------------------
# 7. invalid symbol intent fails validation
# ---------------------------------------------------------------------------

def test_invalid_symbol_intent_fails_validation() -> None:
    adapter = KrakenFuturesAdapter(dry_run=True)
    intent = KrakenFuturesIntent(
        symbol="DOGE-USD",
        side="buy",
        quantity=0.01,
        order_type="mkt",
        strategy="phase_c_test",
    )
    result = adapter.submit_intent(intent)
    assert result.ok is False
    assert result.status == "intent_invalid"
    assert result.error is not None and "kraken_futures_symbol_unmapped" in result.error


# ---------------------------------------------------------------------------
# 8. adapter default dry_run=True
# ---------------------------------------------------------------------------

def test_adapter_default_dry_run_true() -> None:
    adapter = KrakenFuturesAdapter()
    assert adapter.dry_run is True


def test_adapter_default_client_has_no_credentials() -> None:
    adapter = KrakenFuturesAdapter()
    intent = KrakenFuturesIntent(
        symbol="BTC-USD",
        side="buy",
        quantity=0.01,
        order_type="mkt",
    )
    result = adapter.submit_intent(intent)
    assert result.dry_run is True
    assert result.ok is True


# ---------------------------------------------------------------------------
# 9. adapter does not import live execution pipeline
# ---------------------------------------------------------------------------

def test_adapter_does_not_pull_in_live_execution_modules() -> None:
    """Importing the adapter must not load chad.core.live_loop or
    chad.execution.execution_pipeline transitively."""
    forbidden = {
        "chad.core.live_loop",
        "chad.execution.execution_pipeline",
    }
    # Drop the adapter and any forbidden modules so we can re-import cleanly.
    to_drop = [
        name for name in list(sys.modules)
        if name == "chad.execution.kraken_futures_adapter"
        or name == "chad.exchanges.kraken_futures_client"
        or name in forbidden
    ]
    for name in to_drop:
        sys.modules.pop(name, None)

    import importlib

    importlib.import_module("chad.execution.kraken_futures_adapter")
    loaded = set(sys.modules)
    for name in forbidden:
        assert name not in loaded, f"Adapter import pulled in {name}"


def test_adapter_source_has_no_live_pipeline_imports() -> None:
    """Check the adapter's AST imports, not the source text — the docstring
    legitimately names live_loop / execution_pipeline to declare scope."""
    import ast
    import chad.execution.kraken_futures_adapter as mod

    src_path = mod.__file__
    assert src_path is not None
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    forbidden = {
        "chad.core.live_loop",
        "chad.execution.execution_pipeline",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    for name in forbidden:
        assert name not in imported, f"Adapter imports {name}"
    for name in imported:
        assert not name.startswith("chad.strategies"), f"Adapter imports {name}"
