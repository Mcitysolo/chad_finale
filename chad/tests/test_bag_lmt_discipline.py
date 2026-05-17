"""
Phase D Item 2 Tier 2 — BAG LMT discipline contract tests.

Covers ``chad.execution.ibkr_adapter._resolve_bag_lmt_discipline``:

  - Non-BAG intents are returned unchanged (no-op).
  - BAG with order_type != "LMT" is coerced to LMT and emits the
    ``BAG_MKT_COERCED_TO_LMT`` marker (with symbol + original order_type).
  - BAG with missing/non-positive limit_price hydrates from each of the
    three documented meta sources, in the documented priority order, and
    emits the ``BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE`` marker:
        1. ``meta["net_debit_estimate"]``
        2. ``meta["spread_spec"].net_debit_estimate``  (typed
           OptionsSpreadSpec)
        3. ``meta["spread_spec"]["net_debit_estimate"]``  (Mapping)
  - BAG with no derivable positive limit_price returns ``None`` (signals
    skip to the caller) and emits the ``BAG_INTENT_SKIPPED_NO_LIMIT_PRICE``
    marker. The function MUST NOT build or place any order on that path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

import pytest

from chad.execution.ibkr_adapter import (
    NormalizedIntent,
    _resolve_bag_lmt_discipline,
)
from chad.options.spread_spec import OptionsSpreadSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    sec_type: str = "BAG",
    order_type: str = "LMT",
    limit_price: Optional[float] = None,
    meta: Optional[Mapping[str, Any]] = None,
    symbol: str = "SPY",
    strategy: str = "alpha_options",
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy=strategy,
        symbol=symbol,
        sec_type=sec_type,
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type=order_type,
        quantity=1.0,
        notional_estimate=0.0,
        asset_class="options" if sec_type == "BAG" else "equity",
        source_strategies=(strategy,),
        created_at=datetime.now(timezone.utc),
        limit_price=limit_price,
        meta=dict(meta or {}),
    )


def _valid_spread_spec(net_debit_estimate: Optional[float] = 350.0) -> OptionsSpreadSpec:
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
        max_loss_per_contract=700.0,
        net_debit_estimate=net_debit_estimate,
        spread_id="abc-123",
        dte=32,
    )


# ---------------------------------------------------------------------------
# Non-BAG passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sec_type", ["STK", "FUT", "OPT", "CASH"])
def test_non_bag_intent_returned_unchanged(sec_type: str) -> None:
    intent = _make_intent(sec_type=sec_type, order_type="MKT", limit_price=None)
    result = _resolve_bag_lmt_discipline(intent)
    assert result is intent
    assert result.order_type == "MKT"
    assert result.limit_price is None


def test_non_bag_mkt_intent_is_not_coerced_or_logged(caplog: pytest.LogCaptureFixture) -> None:
    intent = _make_intent(sec_type="STK", order_type="MKT", limit_price=None)
    with caplog.at_level(logging.INFO, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is intent
    text = caplog.text
    assert "BAG_MKT_COERCED_TO_LMT" not in text
    assert "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE" not in text
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" not in text


# ---------------------------------------------------------------------------
# Rule A — MKT -> LMT coercion
# ---------------------------------------------------------------------------


def test_bag_mkt_is_coerced_to_lmt_when_limit_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(order_type="MKT", limit_price=4.25)
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result is not intent  # frozen dataclass replaced
    assert result.order_type == "LMT"
    assert result.limit_price == pytest.approx(4.25)
    assert "BAG_MKT_COERCED_TO_LMT" in caplog.text
    assert "symbol=SPY" in caplog.text
    assert "original_order_type=MKT" in caplog.text


def test_bag_already_lmt_is_not_re_logged_as_coerced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(order_type="LMT", limit_price=4.25)
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is intent
    assert "BAG_MKT_COERCED_TO_LMT" not in caplog.text


# ---------------------------------------------------------------------------
# Rules B/C — hydrate limit_price from net_debit_estimate
# ---------------------------------------------------------------------------


def test_bag_hydrates_limit_price_from_meta_net_debit_estimate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"net_debit_estimate": 3.75},
    )
    with caplog.at_level(logging.INFO, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.order_type == "LMT"
    assert result.limit_price == pytest.approx(3.75)
    assert isinstance(result.limit_price, float)
    assert "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE" in caplog.text
    assert "symbol=SPY" in caplog.text


def test_bag_hydrates_limit_price_from_typed_spread_spec(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = _valid_spread_spec(net_debit_estimate=4.10)
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"spread_spec": spec},
    )
    with caplog.at_level(logging.INFO, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(4.10)
    assert "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE" in caplog.text


def test_bag_hydrates_limit_price_from_dict_spread_spec(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"spread_spec": {"net_debit_estimate": 2.55}},
    )
    with caplog.at_level(logging.INFO, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(2.55)
    assert "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE" in caplog.text


def test_bag_zero_limit_price_triggers_hydration() -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=0.0,
        meta={"net_debit_estimate": 1.25},
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(1.25)


def test_bag_negative_limit_price_triggers_hydration() -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=-2.0,
        meta={"net_debit_estimate": 1.80},
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(1.80)


def test_bag_meta_net_debit_estimate_takes_priority_over_spread_spec() -> None:
    spec = _valid_spread_spec(net_debit_estimate=9.99)
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"net_debit_estimate": 1.11, "spread_spec": spec},
    )
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert result.limit_price == pytest.approx(1.11)


def test_bag_existing_positive_limit_price_is_preserved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=5.25,
        meta={"net_debit_estimate": 1.00},  # MUST NOT override existing positive lp
    )
    with caplog.at_level(logging.INFO, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is intent
    assert result.limit_price == pytest.approx(5.25)
    assert "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE" not in caplog.text


# ---------------------------------------------------------------------------
# Rule D — skip path
# ---------------------------------------------------------------------------


def test_bag_skips_when_no_limit_and_no_meta(caplog: pytest.LogCaptureFixture) -> None:
    intent = _make_intent(order_type="LMT", limit_price=None, meta={})
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text
    assert "symbol=SPY" in caplog.text
    assert "strategy=alpha_options" in caplog.text


def test_bag_skips_when_meta_net_debit_estimate_non_positive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"net_debit_estimate": 0.0},
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


def test_bag_skips_when_typed_spread_spec_has_none_net_debit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = _valid_spread_spec(net_debit_estimate=None)
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"spread_spec": spec},
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


def test_bag_skips_when_dict_spread_spec_has_non_numeric_net_debit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(
        order_type="LMT",
        limit_price=None,
        meta={"spread_spec": {"net_debit_estimate": "not-a-number"}},
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text


def test_bag_mkt_with_no_limit_price_logs_both_coerce_and_skip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    intent = _make_intent(order_type="MKT", limit_price=None, meta={})
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _resolve_bag_lmt_discipline(intent)
    assert result is None
    # Both markers should be present — coercion happens first, then the
    # hydration attempt fails and the skip marker is emitted.
    assert "BAG_MKT_COERCED_TO_LMT" in caplog.text
    assert "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE" in caplog.text
    assert "original_order_type=MKT" in caplog.text


# ---------------------------------------------------------------------------
# Immutability / side-effect guards
# ---------------------------------------------------------------------------


def test_bag_helper_does_not_mutate_input_intent_meta() -> None:
    meta: Dict[str, Any] = {"net_debit_estimate": 2.10, "spread_type": "BULL_CALL"}
    intent = _make_intent(order_type="MKT", limit_price=None, meta=meta)
    snapshot = dict(intent.meta)
    result = _resolve_bag_lmt_discipline(intent)
    assert result is not None
    assert dict(intent.meta) == snapshot
    assert intent.order_type == "MKT"
    assert intent.limit_price is None
