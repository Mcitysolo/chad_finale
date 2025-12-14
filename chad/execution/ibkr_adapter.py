#!/usr/bin/env python3
"""
chad/execution/ibkr_adapter.py

IBKR execution adapter for CHAD.

Phase 4 goal:
    - Provide a clean, typed interface for submitting orders to IBKR
      (paper or live), via ib_insync.
    - Support DRY_RUN mode where no real orders are sent; instead,
      all intents are logged.
    - Be robust against connection issues and transient errors.

Important:
    - This adapter is architected for production, but by default should be
      used in DRY_RUN mode until IB Gateway (or TWS) is confirmed stable
      on the configured host/port.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from ib_insync import IB, Contract, Stock, Order  # type: ignore[import]

from chad.types import RoutedSignal, SignalSide, AssetClass, StrategyName

LOGGER = logging.getLogger("chad.execution.ibkr")


@dataclass(frozen=True)
class IbkrConfig:
    """
    Configuration for the IBKR adapter.

    host:
        IP / hostname where IB Gateway or TWS is listening.
    port:
        API port configured in IB Gateway / TWS (e.g., 4002).
    client_id:
        Unique clientId for this CHAD instance.
    dry_run:
        If True, no real orders are sent; all intents are logged only.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 99
    dry_run: bool = True


@dataclass
class SubmittedOrder:
    """
    Representation of an order that was (or would have been) sent to IBKR.
    """

    symbol: str
    side: str
    quantity: float
    strategy: List[str]
    dry_run: bool
    submitted_at: datetime
    ib_order_id: Optional[int] = None


class IbkrAdapter:
    """
    IBKR execution adapter using ib_insync.

    Usage pattern:

        config = IbkrConfig(dry_run=True)
        adapter = IbkrAdapter(config)

        adapter.ensure_connected()
        adapter.submit_routed_signals(routed_signals)
        adapter.shutdown()

    In DRY_RUN mode:
        - ensure_connected() is a no-op (no connection attempted)
        - submit_routed_signals() logs intents only
    """

    def __init__(self, config: IbkrConfig | None = None) -> None:
        self._config = config or IbkrConfig()
        self._ib: Optional[IB] = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def ensure_connected(self) -> None:
        """
        Ensure there is an active connection to IBKR.

        In DRY_RUN mode this does nothing to avoid unnecessary coupling
        to IBKR during development or testing.

        Raises:
            RuntimeError if connection fails in non-dry-run mode.
        """
        if self._config.dry_run:
            # We deliberately avoid connecting in dry-run mode.
            LOGGER.info(
                "ibkr.dry_run_connection_skipped",
                extra={"host": self._config.host, "port": self._config.port},
            )
            return

        if self._ib is None:
            self._ib = IB()

        if self._ib.isConnected():
            return

        try:
            LOGGER.info(
                "ibkr.connecting",
                extra={"host": self._config.host, "port": self._config.port, "client_id": self._config.client_id},
            )
            self._ib.connect(
                self._config.host,
                self._config.port,
                clientId=self._config.client_id,
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "ibkr.connection_failed",
                extra={"error": str(exc)},
            )
            raise RuntimeError(f"Failed to connect to IBKR at {self._config.host}:{self._config.port}") from exc

        if not self._ib.isConnected():
            raise RuntimeError("IBKR connection did not become active")

        LOGGER.info(
            "ibkr.connected",
            extra={"accounts": self._ib.managedAccounts()},
        )

    def shutdown(self) -> None:
        """
        Disconnect cleanly from IBKR.

        Safe to call multiple times.
        """
        if self._ib is not None and self._ib.isConnected():
            try:
                self._ib.disconnect()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "ibkr.disconnect_error",
                    extra={"error": str(exc)},
                )
        self._ib = None

    # ------------------------------------------------------------------ #
    # Order submission
    # ------------------------------------------------------------------ #

    def submit_routed_signals(self, signals: Iterable[RoutedSignal]) -> List[SubmittedOrder]:
        """
        Submit (or log) orders corresponding to routed signals.

        In DRY_RUN mode:
            - Returns SubmittedOrder entries with dry_run=True.
            - No external side effects beyond logging.

        In live mode (dry_run=False):
            - Ensures IBKR connection is active.
            - Submits market orders for supported asset classes.
            - Returns SubmittedOrder entries including IB order IDs.
        """
        submitted: List[SubmittedOrder] = []

        for r in signals:
            try:
                so = self._submit_single_routed(r)
                if so is not None:
                    submitted.append(so)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "ibkr.submit_routed_signal_failed",
                    extra={
                        "symbol": r.symbol,
                        "side": r.side.value,
                        "net_size": r.net_size,
                        "error": str(exc),
                    },
                )

        return submitted

    def _submit_single_routed(self, routed: RoutedSignal) -> Optional[SubmittedOrder]:
        """
        Handle one RoutedSignal.

        Currently supports:
            - AssetClass.EQUITY
            - AssetClass.ETF

        Other asset classes are ignored for now (logged + skipped).
        """
        if routed.net_size <= 0.0:
            LOGGER.info(
                "ibkr.skip_non_positive_size",
                extra={"symbol": routed.symbol, "net_size": routed.net_size},
            )
            return None

        asset_class = routed.asset_class
        if asset_class not in (AssetClass.EQUITY, AssetClass.ETF):
            LOGGER.info(
                "ibkr.skip_unsupported_asset_class",
                extra={"symbol": routed.symbol, "asset_class": asset_class.value},
            )
            return None

        side = routed.side
        qty = float(routed.net_size)
        symbol = routed.symbol
        strategies = [s.value for s in routed.source_strategies]

        submitted_at = datetime.now(timezone.utc)

        # DRY_RUN: log only.
        if self._config.dry_run:
            LOGGER.info(
                "ibkr.dry_run_order",
                extra={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": qty,
                    "strategies": strategies,
                },
            )
            return SubmittedOrder(
                symbol=symbol,
                side=side.value,
                quantity=qty,
                strategy=strategies,
                dry_run=True,
                submitted_at=submitted_at,
                ib_order_id=None,
            )

        # Live mode: ensure connection and place order.
        self.ensure_connected()
        assert self._ib is not None  # for type-checkers

        contract = self._build_stock_contract(symbol)
        order = self._build_market_order(side, qty)

        LOGGER.info(
            "ibkr.place_order",
            extra={
                "symbol": symbol,
                "side": side.value,
                "quantity": qty,
                "strategies": strategies,
            },
        )
        trade = self._ib.placeOrder(contract, order)
        ib_order_id = trade.order.orderId

        return SubmittedOrder(
            symbol=symbol,
            side=side.value,
            quantity=qty,
            strategy=strategies,
            dry_run=False,
            submitted_at=submitted_at,
            ib_order_id=ib_order_id,
        )

    # ------------------------------------------------------------------ #
    # Contract & order helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_stock_contract(symbol: str) -> Contract:
        """
        Build a simple SMART-routed stock/ETF contract.

        This is sufficient for US stocks/ETFs; more complex routing
        (region-specific, forex, etc.) will be handled in later phases.
        """
        # Using Stock shortcut from ib_insync
        return Stock(symbol, "SMART", "USD")

    @staticmethod
    def _build_market_order(side: SignalSide, quantity: float) -> Order:
        """
        Build a basic market order from a RoutedSignal side and quantity.
        """
        action = "BUY" if side == SignalSide.BUY else "SELL"
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        return order
