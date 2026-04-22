"""
chad/execution/ems.py

Phase-8 Session 9 (A1): Execution Management System — explicit Protocol
+ venue-specific implementations.

The EMS owns **signal → order transformation**: instrument resolution,
venue selection, sizing (R3/R5/R6), order-type selection (E4), and
intent construction. It does NOT own order lifecycle — that is OMS
territory (see chad/execution/oms.py).

Pure relocation note
--------------------

Like ``oms.py``, this file introduces the EMS **surface** without moving
the existing implementations. IbkrEMS and KrakenEMS delegate to the
functions that already live in ``chad/execution/execution_pipeline.py``
(``build_execution_plan``, ``build_ibkr_intents_from_plan``,
``build_kraken_intents_from_routed_signals``). External behaviour is
identical; the new Protocol just gives callers a stable contract to
depend on while the implementation migrates.

Separation of concerns
----------------------

After this session the dependency shape is::

    strategies → signals → EMS.build_intents → intents
                                    │
                                    │ (via routing_gates + vote_collector
                                    │  orchestrated by execution_pipeline)
                                    ▼
                         EMS.build_order_request → OrderRequest
                                    │
                                    ▼
                               OMS.submit → OrderResult

The EMS ends at ``OrderRequest``; the OMS begins at ``OrderRequest``.
The thin orchestrator in ``execution_pipeline.py`` glues the two with
the Session-2/7/8 routing gates and the Session-5 vote collector
sandwiched between.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, Optional, Protocol, runtime_checkable

from chad.execution.oms import OrderRequest

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMS Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EMSInterface(Protocol):
    """Minimum contract for an Execution Management System.

    Implementations are venue-specific. IbkrEMS handles equity / ETF /
    futures / forex via IBKR; KrakenEMS handles crypto via Kraken. A
    future mode-specific SimulatedEMS (for A3 backtest unification)
    would share the same surface, swapping only the sizing / gate
    policies appropriate for historical data.
    """

    def build_order_request(self, intent: Any) -> OrderRequest:
        """Turn an EMS-produced StrategyTradeIntent into an
        OrderRequest ready for OMS submission."""
        ...

    def select_venue(self, intent: Any) -> str:
        """Return 'ibkr', 'kraken', or 'simulated' for this intent."""
        ...


# ---------------------------------------------------------------------------
# IbkrEMS
# ---------------------------------------------------------------------------


class IbkrEMS:
    """EMSInterface over the existing IBKR plan / intent builders.

    Delegation target: chad/execution/execution_pipeline.py ::
        build_execution_plan
        build_ibkr_intents_from_plan

    The Session-6/7/8 wiring already inside those functions (E4 order
    type selection, R3/R5/R6 sizing, A4/E5 threading) is inherited for
    free — this class does not re-implement any of it.
    """

    venue = "ibkr"

    def build_intents_from_plan(
        self,
        plan: Any,
        *,
        default_sec_type: str = "STK",
        default_exchange: str = "SMART",
        default_currency: str = "USD",
    ) -> List[Any]:
        """Call into the existing IBKR intent builder.

        Local import breaks the circular dependency that would
        otherwise occur: execution_pipeline.py imports from oms.py and
        ems.py via the thin-orchestrator shims, while ems.py needs the
        builders.
        """
        from chad.execution.execution_pipeline import build_ibkr_intents_from_plan

        return build_ibkr_intents_from_plan(
            plan,
            default_sec_type=default_sec_type,
            default_exchange=default_exchange,
            default_currency=default_currency,
        )

    def build_plan(
        self,
        routed_signals: Iterable[Any],
        prices: Mapping[str, float],
    ) -> Any:
        """Call into the existing planner."""
        from chad.execution.execution_pipeline import build_execution_plan

        return build_execution_plan(routed_signals, prices)

    def build_order_request(self, intent: Any) -> OrderRequest:
        """Translate a StrategyTradeIntent into an OrderRequest.

        The intent already carries the post-R3/R5/R6 quantity, the E4
        order_type, the limit price, and the order_urgency. This method
        packages those into the venue-agnostic OrderRequest the OMS
        accepts.
        """
        try:
            qty_raw = getattr(intent, "quantity", 0)
            qty = int(float(qty_raw)) if qty_raw is not None else 0
        except (TypeError, ValueError):
            qty = 0
        try:
            lp_raw = getattr(intent, "limit_price", None)
            limit_price = float(lp_raw) if lp_raw is not None else 0.0
        except (TypeError, ValueError):
            limit_price = 0.0
        order_type = str(getattr(intent, "order_type", "LMT") or "LMT")
        urgency = str(getattr(intent, "order_urgency", "normal") or "normal")
        return OrderRequest(
            intent=intent,
            venue=self.venue,
            order_type=order_type,
            limit_price=limit_price,
            quantity=qty,
            aggressive=(urgency == "high"),
            idempotency_key=None,
        )

    def select_venue(self, intent: Any) -> str:
        return self.venue


# ---------------------------------------------------------------------------
# KrakenEMS
# ---------------------------------------------------------------------------


class KrakenEMS:
    """EMSInterface over the existing Kraken intent builder.

    Delegation target: chad/execution/execution_pipeline.py ::
        build_kraken_intents_from_routed_signals
        _build_kraken_intent_from_routed_signal
    """

    venue = "kraken"

    def build_intents_from_signals(
        self,
        routed_signals: Iterable[Any],
        prices: Mapping[str, float],
        *,
        dynamic_cap_for_crypto: Optional[float] = None,
    ) -> List[Any]:
        from chad.execution.execution_pipeline import (
            build_kraken_intents_from_routed_signals,
        )

        return build_kraken_intents_from_routed_signals(
            routed_signals, prices, dynamic_cap_for_crypto=dynamic_cap_for_crypto
        )

    def build_order_request(self, intent: Any) -> OrderRequest:
        """Translate a Kraken StrategyTradeIntent into an OrderRequest.

        Kraken intents use ``volume`` (base-currency size) instead of
        ``quantity``, and ``price`` instead of ``limit_price``. Both are
        mapped into the venue-agnostic OrderRequest fields.
        """
        try:
            vol_raw = getattr(intent, "volume", 0.0)
            # Volume on Kraken is a fractional base-currency amount; we
            # preserve as an int (rounded down) to satisfy the
            # OrderRequest.quantity type. OMS implementations read the
            # underlying intent if they need the fractional value.
            qty = int(float(vol_raw)) if vol_raw is not None else 0
        except (TypeError, ValueError):
            qty = 0
        try:
            price_raw = getattr(intent, "price", None)
            limit_price = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            limit_price = 0.0
        order_type = str(getattr(intent, "ordertype", "limit") or "limit").upper()
        urgency = str(getattr(intent, "order_urgency", "normal") or "normal")
        return OrderRequest(
            intent=intent,
            venue=self.venue,
            order_type=order_type,
            limit_price=limit_price,
            quantity=qty,
            aggressive=(urgency == "high"),
            idempotency_key=getattr(intent, "idempotency_key", None),
        )

    def select_venue(self, intent: Any) -> str:
        return self.venue


# ---------------------------------------------------------------------------
# Convenience — split routed signals between IBKR and Kraken EMS
# ---------------------------------------------------------------------------


def split_signals_by_asset_class_for_ems(routed_signals: Iterable[Any]):
    """Thin wrapper preserving the pre-Session-9 split helper.

    Re-exports ``split_signals_by_asset_class`` from
    execution_pipeline so callers that want to use EMS explicitly can
    import the splitter from the EMS module.
    """
    from chad.execution.execution_pipeline import split_signals_by_asset_class

    return split_signals_by_asset_class(routed_signals)


__all__ = [
    "EMSInterface",
    "IbkrEMS",
    "KrakenEMS",
    "split_signals_by_asset_class_for_ems",
]
