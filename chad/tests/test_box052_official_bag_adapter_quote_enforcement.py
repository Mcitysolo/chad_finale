"""Official Matrix Box 052 — BAG adapter quote enforcement invariants.

Concentrates the enforcement-rejection contract for the BAG (combo) order
path inside ``chad.execution.ibkr_adapter`` so that the
**adapter rejects unsafe BAG prices and consumes only broker-facing
per-share values**.

Two defense layers are asserted:

  Layer 1 — ``_resolve_bag_lmt_discipline``:
    * NaN / +inf / -inf / non-numeric ``intent.limit_price`` is dropped
      by ``_positive_float`` and falls into meta hydration; if meta is
      also unsafe the function returns ``None`` and emits
      ``BAG_INTENT_SKIPPED_NO_LIMIT_PRICE``.
    * NaN / inf / non-numeric / zero / negative ``net_debit_estimate``
      (either in ``meta["net_debit_estimate"]`` or
      ``meta["spread_spec"]``) does NOT hydrate; the skip path fires.
    * When a positive finite per-share price IS derivable, the value is
      forwarded **verbatim** to ``intent.limit_price`` (no ×100, no ÷100).
      This is the per-share / contract-dollar boundary established by
      Box 051; Box 052 re-asserts it under the quote-enforcement lens.

  Layer 2 — ``_OrderFactory.build`` (defense-in-depth before any order
  reaches IBKR):
    * Raises ``ValidationError("Limit order requires positive
      limit_price")`` on NaN / non-positive ``limit_price`` for LMT
      orders — this catches any path that bypasses
      ``_resolve_bag_lmt_discipline`` (e.g. non-BAG callers, future
      regressions).

Quote-staleness / midpoint policy:
  The offline BAG quote-check engine in ``chad.options.quote_check`` is
  intentionally **unwired** at the adapter (per its module docstring,
  lines 19-22). The adapter does NOT consume broker bid/ask/last for
  BAG; its enforcement is **structural** — if the strategy producer
  (``alpha_options.py``) cannot derive a positive per-share
  ``net_debit_estimate``, no BAG order is built. This test file locks
  that contract so any future wiring of live quotes into the adapter
  must update this file deliberately.

This test does NOT exercise broker, runtime, or live state. It is a
pure-unit invariant test of the adapter's BAG-quote rejection contract.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from chad.execution.ibkr_adapter import (
    IbkrConfig,
    NormalizedIntent,
    ValidationError,
    _OrderFactory,
    _resolve_bag_lmt_discipline,
)
from chad.options.spread_spec import OptionsSpreadSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bag_intent(
    *,
    limit_price: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
    order_type: str = "LMT",
    symbol: str = "SPY",
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy="alpha_options",
        symbol=symbol,
        sec_type="BAG",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type=order_type,
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options",
        source_strategies=("alpha_options",),
        created_at=datetime.now(timezone.utc),
        limit_price=limit_price,
        meta=dict(meta or {}),
    )


def _spec(net_debit_estimate: Optional[float]) -> OptionsSpreadSpec:
    return OptionsSpreadSpec(
        symbol="SPY",
        expiry="20260618",
        long_strike=737.0,
        short_strike=744.0,
        long_right="C",
        short_right="C",
        ratio_long=1,
        ratio_short=1,
        exchange="SMART",
        currency="USD",
        spread_type="BULL_CALL",
        max_loss_per_contract=350.0,
        net_debit_estimate=net_debit_estimate,
    )


class _StubOrder:
    """Stand-in for ib_async.order.Order, mirroring the attributes
    ``_OrderFactory.build`` sets — avoids any IBKR import in this
    test file. Pattern lifted from test_ibkr_adapter_tick_snap.py.
    """

    action = ""
    orderType = ""
    totalQuantity = 0.0
    tif = ""
    outsideRth = False
    whatIf = False
    account = ""
    lmtPrice = 0.0


def _stub_contract_classes():
    return (object, object, object, _StubOrder, object)


# ---------------------------------------------------------------------------
# Layer 1 — discipline rejects unsafe limit_price values (NaN / inf / -inf)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_lp",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_bag_discipline_rejects_nonfinite_limit_price_with_no_meta(
    bad_lp: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-finite ``intent.limit_price`` plus no usable meta MUST emit the
    skip marker and return ``None`` — adapter never builds the order.
    """
    intent = _make_bag_intent(limit_price=bad_lp, meta={})
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


def test_bag_discipline_rejects_nonfinite_limit_price_but_hydrates_from_safe_meta() -> None:
    """If ``intent.limit_price`` is NaN but ``meta["net_debit_estimate"]``
    holds a valid per-share value, the safe value is used; the unsafe
    intent value is dropped silently by ``_positive_float``.
    """
    intent = _make_bag_intent(
        limit_price=float("nan"), meta={"net_debit_estimate": 3.75}
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(3.75)


# ---------------------------------------------------------------------------
# Layer 1 — discipline rejects unsafe net_debit_estimate values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_meta_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -1.50,
        "not-a-number",
        None,
    ],
)
def test_bag_discipline_skip_when_meta_net_debit_estimate_unsafe(
    bad_meta_value: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Unsafe ``meta["net_debit_estimate"]`` MUST NOT hydrate; the skip
    marker fires and no order is built.
    """
    intent = _make_bag_intent(
        limit_price=None, meta={"net_debit_estimate": bad_meta_value}
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), 0.0, -2.0],
)
def test_bag_discipline_skip_when_typed_spread_spec_net_debit_estimate_unsafe(
    bad_value: float, caplog: pytest.LogCaptureFixture
) -> None:
    """Unsafe ``OptionsSpreadSpec.net_debit_estimate`` MUST NOT hydrate."""
    intent = _make_bag_intent(
        limit_price=None, meta={"spread_spec": _spec(bad_value)}
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


def test_bag_discipline_skip_when_dict_spread_spec_net_debit_estimate_inf(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dict-shaped ``spread_spec`` with non-finite net_debit_estimate also
    fails closed via the skip path.
    """
    intent = _make_bag_intent(
        limit_price=None,
        meta={"spread_spec": {"net_debit_estimate": float("inf")}},
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


# ---------------------------------------------------------------------------
# Layer 1 — broker-facing per-share invariant under enforcement lens
# ---------------------------------------------------------------------------


def test_bag_discipline_consumes_per_share_value_verbatim_no_unit_conversion() -> None:
    """When a safe positive per-share value is derivable, the adapter
    forwards it to ``intent.limit_price`` **verbatim** — no ×100 / ÷100.
    This is the broker-facing per-share invariant under enforcement.
    """
    per_share = 4.25
    intent = _make_bag_intent(
        limit_price=None, meta={"net_debit_estimate": per_share}
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == per_share
    assert isinstance(result.limit_price, float)
    assert math.isfinite(result.limit_price)
    assert result.limit_price > 0.0


def test_bag_discipline_does_not_silently_divide_contract_dollar_misuse() -> None:
    """Quote-enforcement contract: the adapter does NOT silently convert a
    contract-dollar-looking value (e.g. 350.0) into per-share by dividing
    by 100. Unit responsibility lives with the producer
    (``alpha_options.py``). This test pins the adapter's trust contract
    so any future "auto-÷100" patch must be deliberate.
    """
    contract_dollar_misuse = 350.0
    intent = _make_bag_intent(
        limit_price=None, meta={"net_debit_estimate": contract_dollar_misuse}
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == contract_dollar_misuse


# ---------------------------------------------------------------------------
# Layer 2 — _OrderFactory defense-in-depth on NaN / non-positive LMT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_lp",
    [float("nan"), 0.0, -1.0, -1e6],
)
def test_order_factory_rejects_unsafe_lmt_price(
    bad_lp: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: even if a non-BAG path or future regression
    bypasses ``_resolve_bag_lmt_discipline``, ``_OrderFactory.build``
    refuses to construct an LMT order with NaN or non-positive price.
    """
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = NormalizedIntent(
        strategy="alpha_options",
        symbol="SPY",
        sec_type="BAG",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options",
        source_strategies=("alpha_options",),
        created_at=datetime.now(timezone.utc),
        limit_price=bad_lp,
        meta={},
    )
    with pytest.raises(ValidationError, match="positive limit_price"):
        factory.build(intent, what_if=True)


def test_order_factory_accepts_safe_per_share_lmt_for_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive finite per-share LMT yields a constructed order with the
    exact value placed on ``order.lmtPrice`` (BAG is not in the FUT
    tick table → no snap occurs; per-share is forwarded verbatim).
    """
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    per_share = 3.50
    intent = NormalizedIntent(
        strategy="alpha_options",
        symbol="SPY",
        sec_type="BAG",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options",
        source_strategies=("alpha_options",),
        created_at=datetime.now(timezone.utc),
        limit_price=per_share,
        meta={},
    )
    prepared = factory.build(intent, what_if=True)
    assert prepared.order.orderType == "LMT"
    assert prepared.order.lmtPrice == pytest.approx(per_share)


# ---------------------------------------------------------------------------
# Non-BAG quote enforcement isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sec_type", ["STK", "FUT", "OPT", "CASH"])
def test_non_bag_intent_quote_enforcement_unchanged_by_bag_discipline(
    sec_type: str,
) -> None:
    """Non-BAG intents pass through ``_resolve_bag_lmt_discipline``
    untouched — the BAG quote-enforcement rules MUST NOT impose any
    transform on STK / FUT / OPT / CASH limit prices.
    """
    intent = NormalizedIntent(
        strategy="alpha",
        symbol="SPY",
        sec_type=sec_type,
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=10.0,
        notional_estimate=0.0,
        asset_class="equity" if sec_type == "STK" else "futures",
        source_strategies=("alpha",),
        created_at=datetime.now(timezone.utc),
        limit_price=500.0,
        meta={},
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is intent
    assert result.limit_price == 500.0
    assert result.order_type == "LMT"


def test_non_bag_unsafe_lmt_still_caught_by_order_factory_defense_in_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 2 is sec_type-agnostic: a non-BAG intent with NaN limit_price
    is still rejected by ``_OrderFactory.build``. This proves the
    defense-in-depth gate protects ALL LMT order paths, not only BAG.
    """
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = NormalizedIntent(
        strategy="alpha",
        symbol="SPY",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=10.0,
        notional_estimate=0.0,
        asset_class="equity",
        source_strategies=("alpha",),
        created_at=datetime.now(timezone.utc),
        limit_price=float("nan"),
        meta={},
    )
    with pytest.raises(ValidationError, match="positive limit_price"):
        factory.build(intent, what_if=True)


# ---------------------------------------------------------------------------
# Quote-staleness / midpoint wiring contract
# ---------------------------------------------------------------------------


def test_quote_check_engine_is_unwired_at_adapter_documented_contract() -> None:
    """Lock the documented contract: the offline BAG quote-check engine
    (``chad.options.quote_check``) is NOT consumed by the adapter. The
    adapter's BAG rejection is structural (producer-derived per-share
    debit), not midpoint-based. Any future wiring of live broker quotes
    into the adapter MUST update Box 052 deliberately.

    This test does not import any quote_check API into the adapter;
    instead it asserts that ``chad.execution.ibkr_adapter`` does not
    reference the quote_check module — a structural anchor that will
    fail when the wiring lands.
    """
    import chad.execution.ibkr_adapter as adapter

    # The adapter module does not import quote_check at import time.
    # (When wiring lands, this assertion is the canonical signal to
    # update the enforcement contract and Box 052 evidence.)
    src_path = adapter.__file__
    assert src_path is not None
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "from chad.options.quote_check" not in src
    assert "import chad.options.quote_check" not in src
