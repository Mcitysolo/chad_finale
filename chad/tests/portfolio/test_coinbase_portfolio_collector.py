from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.portfolio.coinbase_portfolio_collector import (
    CoinbaseDataClient,
    CoinbaseExchangeClient,
    CoinbasePortfolioCalculator,
    CoinbasePortfolioSnapshot,
    PortfolioSnapshotWriter,
)


class _FakeExchangeClient(CoinbaseExchangeClient):
    """
    Test double for CoinbaseExchangeClient that returns a fixed set of accounts
    without doing any network I/O.
    """

    def __init__(self, accounts: List[Dict[str, Any]]) -> None:
        # Bypass real parent __init__
        self._accounts = accounts

    def list_accounts(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        return self._accounts


class _FakeDataClient(CoinbaseDataClient):
    """
    Test double for CoinbaseDataClient that returns fixed spot prices.
    """

    def __init__(self, prices: Dict[str, Decimal]) -> None:
        # Bypass real parent __init__
        self._prices = {k.upper(): v for k, v in prices.items()}

    def get_spot_price_usd(self, currency: str) -> Decimal | None:  # type: ignore[override]
        return self._prices.get(currency.upper())


def test_portfolio_calculator_combines_balances() -> None:
    """
    CoinbasePortfolioCalculator should correctly value balances in USD using
    the provided spot prices and aggregate per-currency and total equity.
    """
    accounts = [
        {"currency": "BTC", "balance": Decimal("0.5")},
        {"currency": "ETH", "balance": Decimal("1")},
        {"currency": "USD", "balance": Decimal("1000")},
        # Zero-balance account should be ignored
        {"currency": "SOL", "balance": Decimal("0")},
    ]
    prices = {
        "BTC": Decimal("30000"),  # 0.5 * 30000 = 15000
        "ETH": Decimal("2000"),   # 1 * 2000 = 2000
        # USD handled as 1:1 implicitly by calculator
    }

    exchange_client = _FakeExchangeClient(accounts=accounts)
    data_client = _FakeDataClient(prices=prices)
    calculator = CoinbasePortfolioCalculator(exchange_client, data_client)

    snapshot: CoinbasePortfolioSnapshot = calculator.compute_portfolio()

    # Total: 15000 (BTC) + 2000 (ETH) + 1000 (USD) = 18000
    assert snapshot.total_usd == Decimal("18000")

    # Per-currency valuations should be present
    assert snapshot.per_currency["BTC"] == Decimal("15000")
    assert snapshot.per_currency["ETH"] == Decimal("2000")
    assert snapshot.per_currency["USD"] == Decimal("1000")
    # SOL had zero balance -> should not appear
    assert "SOL" not in snapshot.per_currency


def test_portfolio_snapshot_writer_creates_new_file(tmp_path: Path) -> None:
    """
    When no existing snapshot file is present, PortfolioSnapshotWriter should
    create a new file with ibkr_equity = 0.0 and coinbase_equity set from
    the provided USD amount.
    """
    snapshot_path = tmp_path / "portfolio_snapshot.json"
    writer = PortfolioSnapshotWriter(snapshot_path)

    writer.write_with_coinbase_equity(coinbase_equity_usd=Decimal("123.45"))

    assert snapshot_path.is_file()
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert data["ibkr_equity"] == 0.0
    # JSON float can introduce tiny rounding differences; use approx
    assert data["coinbase_equity"] == pytest.approx(123.45)


def test_portfolio_snapshot_writer_preserves_ibkr_equity(tmp_path: Path) -> None:
    """
    When an existing snapshot file is present, PortfolioSnapshotWriter should
    preserve ibkr_equity and only update coinbase_equity.
    """
    snapshot_path = tmp_path / "portfolio_snapshot.json"
    original_payload = {
        "ibkr_equity": 777777.77,
        "coinbase_equity": 1.23,
    }
    snapshot_path.write_text(
        json.dumps(original_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    writer = PortfolioSnapshotWriter(snapshot_path)
    writer.write_with_coinbase_equity(coinbase_equity_usd=Decimal("250.0"))

    data = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert data["ibkr_equity"] == pytest.approx(777777.77)
    assert data["coinbase_equity"] == pytest.approx(250.0)
