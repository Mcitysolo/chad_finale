"""Tests for Phase-8 Session 9 (A1) OMS/EMS separation.

Pure-relocation contract:
  * OMSInterface / EMSInterface Protocols exist and are satisfied.
  * Status vocabulary from the pre-Session-9 world is preserved.
  * Backward-compat imports via chad.execution.execution_pipeline work.
  * paper_exec_evidence_writer sees the same SubmittedOrder structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List

import pytest

from chad.execution.ems import (
    EMSInterface,
    IbkrEMS,
    KrakenEMS,
)
from chad.execution.oms import (
    IbkrOMS,
    KrakenOMS,
    NullOMS,
    OMSInterface,
    OrderRequest,
    OrderResult,
    PRESERVED_STATUS_STRINGS,
    STATUS_DRY_RUN,
    STATUS_DUPLICATE_BLOCKED,
    STATUS_ERROR,
    STATUS_SUBMITTED,
    STATUS_UNKNOWN,
    STATUS_WHAT_IF,
)


# ---------------------------------------------------------------------------
# Fake adapter / executor test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeSubmittedOrder:
    """Mirror of ibkr_adapter.SubmittedOrder for test doubles — the
    only attributes the OMS wrapper actually reads."""

    symbol: str = "SPY"
    side: str = "BUY"
    quantity: float = 10.0
    strategy: Any = None
    dry_run: bool = False
    submitted_at: datetime = None
    ib_order_id: int = 12345
    status: str = "submitted"
    limit_price: float = 500.0
    error: str = ""


class _FakeIbkrAdapter:
    """Minimal IbkrAdapter stand-in returning a prebuilt SubmittedOrder."""

    def __init__(self, submitted: _FakeSubmittedOrder = None) -> None:
        self._submitted = submitted or _FakeSubmittedOrder()
        self.calls: List[Any] = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        return [self._submitted]


class _FakeKrakenExecutor:
    """Minimal KrakenExecutor stand-in returning a namespace response."""

    def __init__(self, response=None) -> None:
        self._response = response or SimpleNamespace(
            status="submitted",
            broker_order_id="KRAKEN-12345",
            error="",
        )
        self.calls: List[Any] = []

    def execute_with_risk(self, intent):
        self.calls.append(intent)
        return self._response


@dataclass
class _FakeIntent:
    symbol: str = "SPY"
    quantity: float = 10.0
    limit_price: float = 500.0
    order_type: str = "LMT"
    order_urgency: str = "normal"


@dataclass
class _FakeKrakenIntent:
    pair: str = "XBT/USD"
    volume: float = 0.01
    price: float = 80000.0
    ordertype: str = "limit"
    order_urgency: str = "normal"
    idempotency_key: str = "k1"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_oms_interface_satisfied_by_ibkr_oms():
    oms = IbkrOMS(_FakeIbkrAdapter())
    assert isinstance(oms, OMSInterface)


def test_oms_interface_satisfied_by_kraken_oms():
    oms = KrakenOMS(_FakeKrakenExecutor())
    assert isinstance(oms, OMSInterface)


def test_oms_interface_satisfied_by_null_oms():
    assert isinstance(NullOMS(), OMSInterface)


def test_ems_interface_satisfied_by_ibkr_ems():
    assert isinstance(IbkrEMS(), EMSInterface)


def test_ems_interface_satisfied_by_kraken_ems():
    assert isinstance(KrakenEMS(), EMSInterface)


# ---------------------------------------------------------------------------
# Status vocabulary preservation
# ---------------------------------------------------------------------------


def test_order_result_status_strings_preserved():
    """The set of status strings we commit to must include every string
    used by the pre-Session-9 SubmittedOrder. If a future session tries
    to rename one of these silently, this test will fail loudly."""
    expected = frozenset(
        {
            "submitted",
            "dry_run",
            "what-if",
            "duplicate_blocked",
            "error",
            "unknown",
        }
    )
    assert PRESERVED_STATUS_STRINGS == expected


def test_preserved_status_constants_have_exact_string_values():
    assert STATUS_SUBMITTED == "submitted"
    assert STATUS_DRY_RUN == "dry_run"
    assert STATUS_WHAT_IF == "what-if"
    assert STATUS_DUPLICATE_BLOCKED == "duplicate_blocked"
    assert STATUS_ERROR == "error"
    assert STATUS_UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# Submission translation
# ---------------------------------------------------------------------------


def test_ibkr_oms_submit_preserves_submitted_order_on_raw():
    """paper_exec_evidence_writer reads the SubmittedOrder — the OMS
    wrapper must put it on ``result.raw`` exactly as the adapter
    returned it."""
    submitted = _FakeSubmittedOrder(
        status="submitted",
        ib_order_id=77,
        quantity=5.0,
        limit_price=123.45,
        submitted_at=datetime.now(timezone.utc),
    )
    oms = IbkrOMS(_FakeIbkrAdapter(submitted=submitted))
    request = OrderRequest(
        intent=_FakeIntent(), venue="ibkr", order_type="LMT",
        limit_price=123.45, quantity=5,
    )
    result = oms.submit(request)
    # Status preserved byte-for-byte.
    assert result.status == "submitted"
    # order_id coerced to str.
    assert result.order_id == "77"
    # Original SubmittedOrder preserved on raw.
    assert result.raw is submitted
    assert result.raw.limit_price == 123.45
    assert result.raw.quantity == 5.0


def test_ibkr_oms_submit_handles_exception_as_error():
    class _BoomAdapter:
        def submit_strategy_trade_intents(self, intents):
            raise RuntimeError("simulated broker error")

    oms = IbkrOMS(_BoomAdapter())
    request = OrderRequest(intent=_FakeIntent(), venue="ibkr", quantity=1)
    result = oms.submit(request)
    assert result.status == STATUS_ERROR
    assert "simulated broker error" in result.rejection_reason


def test_ibkr_oms_submit_handles_empty_result_as_no_result():
    class _EmptyAdapter:
        def submit_strategy_trade_intents(self, intents):
            return []

    oms = IbkrOMS(_EmptyAdapter())
    request = OrderRequest(intent=_FakeIntent(), venue="ibkr", quantity=1)
    result = oms.submit(request)
    assert result.status == "no_result"


def test_ibkr_oms_passes_intent_through_unchanged():
    adapter = _FakeIbkrAdapter()
    oms = IbkrOMS(adapter)
    intent = _FakeIntent(symbol="QQQ", quantity=50)
    request = OrderRequest(intent=intent, venue="ibkr", quantity=50)
    oms.submit(request)
    # The adapter received the exact intent object from the request.
    assert adapter.calls[0][0] is intent


def test_kraken_oms_submit_translates_response():
    response = SimpleNamespace(
        status="submitted",
        broker_order_id="OXXX-YYY",
        error="",
    )
    oms = KrakenOMS(_FakeKrakenExecutor(response=response))
    req = OrderRequest(
        intent=_FakeKrakenIntent(),
        venue="kraken",
        order_type="LIMIT",
        limit_price=80000.0,
        quantity=0,
    )
    result = oms.submit(req)
    assert result.status == "submitted"
    assert result.order_id == "OXXX-YYY"
    assert result.venue == "kraken"
    assert result.raw is response


def test_kraken_oms_handles_none_response():
    class _NoneExecutor:
        def execute_with_risk(self, intent):
            return None

    oms = KrakenOMS(_NoneExecutor())
    req = OrderRequest(intent=_FakeKrakenIntent(), venue="kraken")
    assert oms.submit(req).status == "no_result"


# ---------------------------------------------------------------------------
# EMS translation
# ---------------------------------------------------------------------------


def test_ibkr_ems_build_order_request_maps_fields():
    ems = IbkrEMS()
    intent = _FakeIntent(
        symbol="SPY", quantity=42, limit_price=499.5,
        order_type="LMT", order_urgency="high",
    )
    req = ems.build_order_request(intent)
    assert req.venue == "ibkr"
    assert req.order_type == "LMT"
    assert req.quantity == 42
    assert req.limit_price == 499.5
    assert req.aggressive is True
    assert req.intent is intent


def test_ibkr_ems_normal_urgency_is_not_aggressive():
    req = IbkrEMS().build_order_request(_FakeIntent(order_urgency="normal"))
    assert req.aggressive is False


def test_kraken_ems_build_order_request_maps_volume_to_quantity():
    ems = KrakenEMS()
    intent = _FakeKrakenIntent(
        pair="XBT/USD", volume=2.5, price=80500.0,
        ordertype="limit", order_urgency="high",
    )
    req = ems.build_order_request(intent)
    assert req.venue == "kraken"
    assert req.order_type == "LIMIT"
    assert req.limit_price == 80500.0
    # int-coerced volume (fractional values would become 0 here — the
    # caller must size appropriately or read the raw intent).
    assert req.quantity == 2
    assert req.aggressive is True
    assert req.idempotency_key == "k1"


def test_ems_select_venue_returns_class_venue():
    assert IbkrEMS().select_venue(_FakeIntent()) == "ibkr"
    assert KrakenEMS().select_venue(_FakeKrakenIntent()) == "kraken"


# ---------------------------------------------------------------------------
# Orchestrator is a thin composer — imports check
# ---------------------------------------------------------------------------


def test_backward_compat_shim_reexports_from_execution_pipeline():
    """Existing callers that import OMS/EMS names from execution_pipeline
    must continue to work during the transition."""
    from chad.execution.execution_pipeline import (  # noqa: F401
        EMSInterface as _EMSI,
        IbkrEMS as _IE,
        IbkrOMS as _IO,
        KrakenEMS as _KE,
        KrakenOMS as _KO,
        NullOMS as _NO,
        OMSInterface as _OMSI,
        OrderRequest as _OR,
        OrderResult as _ORes,
        PRESERVED_STATUS_STRINGS as _PSS,
        STATUS_SUBMITTED as _SS,
    )
    # Same objects as the direct imports (re-export not re-definition).
    from chad.execution.ems import EMSInterface as _EMSI2
    from chad.execution.oms import OMSInterface as _OMSI2

    assert _EMSI is _EMSI2
    assert _OMSI is _OMSI2
    assert _PSS == PRESERVED_STATUS_STRINGS


def test_null_oms_returns_dry_run_status():
    """NullOMS is the reference 'no-op' implementation — the Session 10
    SimulatedOMS will build on it."""
    oms = NullOMS()
    req = OrderRequest(intent=_FakeIntent(), venue="null", quantity=10, limit_price=100.0)
    res = oms.submit(req)
    assert res.status == STATUS_DRY_RUN
    assert res.fill_quantity == 10
    assert res.fill_price == 100.0


# ---------------------------------------------------------------------------
# Observability preservation
# ---------------------------------------------------------------------------


def test_paper_exec_evidence_writer_can_read_raw_submitted_order():
    """Regression guard: after the OMS wrapping, the raw SubmittedOrder
    is still accessible on OrderResult.raw. paper_exec_evidence_writer
    reads SubmittedOrder fields directly; as long as they live on .raw
    downstream code keeps working."""
    submitted = _FakeSubmittedOrder(
        symbol="AAPL", side="BUY", quantity=100.0,
        status="submitted", ib_order_id=12345,
        limit_price=175.25,
    )
    result = OrderResult(
        order_id="12345", status="submitted",
        fill_price=175.25, fill_quantity=100,
        venue="ibkr", raw=submitted,
    )
    # Simulate what paper_exec_evidence_writer does.
    assert result.raw.symbol == "AAPL"
    assert result.raw.side == "BUY"
    assert result.raw.status == "submitted"
    assert result.raw.ib_order_id == 12345
