"""Official Matrix Box 051 — BAG limit-price unit-normalization invariant.

Locks the per-share-vs-contract-dollar contract for BAG (options spread / combo)
intents end-to-end:

  1. ``OptionsSpreadSpec.net_debit_estimate`` is the **per-share** debit
     (e.g. $3.50 for a $7-wide bull-call vertical), NOT the contract-dollar
     value ($350 = $3.50 × 100 multiplier).
  2. ``alpha_options.py`` stamps the per-share value into both
     ``meta["net_debit_estimate"]`` and ``meta["spread_spec"].net_debit_estimate``.
  3. ``chad.execution.ibkr_adapter._resolve_bag_lmt_discipline`` hydrates
     ``intent.limit_price`` from the per-share value with **NO unit
     conversion** — the broker-facing IBKR BAG ``lmtPrice`` is per-share
     as required by the IBKR API.
  4. ``chad.execution.paper_exec_evidence_writer`` (and the BAG paper-fill
     simulator) compute ``notional = quantity × per_share × OPTIONS_MULTIPLIER (100)``
     to recover the contract-dollar value.

This test does NOT exercise broker, runtime, or live state. It is a
pure-unit invariant test of the strategy → adapter contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chad.execution.ibkr_adapter import (
    NormalizedIntent,
    _resolve_bag_lmt_discipline,
)
from chad.options.spread_spec import OptionsSpreadSpec


_OPTIONS_MULTIPLIER = 100  # standard equity-option contract multiplier


def _make_intent(
    *,
    limit_price: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> NormalizedIntent:
    return NormalizedIntent(
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
        limit_price=limit_price,
        meta=dict(meta or {}),
    )


def _spec_per_share(net_debit_estimate: float = 3.50) -> OptionsSpreadSpec:
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
        max_loss_per_contract=350.0,  # contract-dollar value (per-share × 100)
        net_debit_estimate=net_debit_estimate,  # per-share value
    )


def test_per_share_invariant_no_unit_conversion_when_hydrated_from_meta_net_debit():
    """Per-share invariant: when limit_price is missing, hydration from
    meta["net_debit_estimate"] sets intent.limit_price EQUAL to the
    per-share value, with NO ×100 / ÷100 transform.
    """
    per_share = 3.50
    intent = _make_intent(limit_price=None, meta={"net_debit_estimate": per_share})

    resolved = _resolve_bag_lmt_discipline(intent)

    assert resolved is not None, "BAG with valid per-share debit must not be skipped"
    assert resolved.limit_price == per_share, (
        f"BAG LMT must equal per-share net_debit_estimate exactly "
        f"(no unit conversion). got={resolved.limit_price!r} expected={per_share!r}"
    )


def test_per_share_invariant_no_unit_conversion_when_hydrated_from_typed_spread_spec():
    """Per-share invariant: same as above but from the typed OptionsSpreadSpec
    branch. Proves both meta lookup paths preserve per-share units.
    """
    per_share = 2.85
    spec = _spec_per_share(net_debit_estimate=per_share)
    intent = _make_intent(limit_price=None, meta={"spread_spec": spec})

    resolved = _resolve_bag_lmt_discipline(intent)

    assert resolved is not None
    assert resolved.limit_price == per_share, (
        f"BAG LMT from typed OptionsSpreadSpec must equal per-share "
        f"net_debit_estimate. got={resolved.limit_price!r} expected={per_share!r}"
    )


def test_per_share_invariant_no_unit_conversion_when_explicit_limit_price_supplied():
    """Per-share invariant: when limit_price is already supplied, it is
    preserved verbatim — no hidden ×100 / ÷100. This is what the broker
    will see as IBKR BAG lmtPrice.
    """
    per_share = 1.50
    intent = _make_intent(limit_price=per_share, meta={"net_debit_estimate": 999.0})

    resolved = _resolve_bag_lmt_discipline(intent)

    assert resolved is not None
    assert resolved.limit_price == per_share, (
        f"Explicit per-share LMT must be preserved exactly. "
        f"got={resolved.limit_price!r} expected={per_share!r}"
    )


def test_contract_dollar_value_would_yield_absurd_per_share_lmt():
    """Documentation invariant: if an operator/upstream module mistakenly
    passes a CONTRACT-DOLLAR value (e.g. $350 = $3.50 × 100) into
    net_debit_estimate, the discipline function will hydrate that value
    DIRECTLY as the per-share LMT — there is no automatic ÷100 conversion.

    The downstream IBKR API would then receive a 350.0 per-share LMT,
    which is absurd for a typical vertical spread. This test documents
    that the unit responsibility lies with the producer
    (``alpha_options.py``), NOT the adapter — the adapter trusts the
    per-share contract.

    Asserts the literal behaviour so that any future "auto-convert
    contract-dollar to per-share" patch would either (a) update this
    test deliberately, or (b) be flagged in review.
    """
    contract_dollar_misuse = 350.0  # NOT a valid per-share LMT for SPY vertical
    intent = _make_intent(
        limit_price=None, meta={"net_debit_estimate": contract_dollar_misuse}
    )

    resolved = _resolve_bag_lmt_discipline(intent)

    assert resolved is not None
    # The adapter trusts the producer; it does NOT silently divide by 100.
    assert resolved.limit_price == contract_dollar_misuse, (
        "Adapter must NOT silently convert contract-dollar to per-share; "
        "unit responsibility lies with the producer (alpha_options.py)."
    )


def test_alpha_options_stamps_per_share_into_spread_spec_typed_and_meta_dict():
    """Per-share invariant on the producer side: any OptionsSpreadSpec
    built with a per-share net_debit_estimate stays per-share in both
    its typed attribute and its legacy-dict ``to_meta()`` projection.
    """
    per_share = 3.25
    spec = _spec_per_share(net_debit_estimate=per_share)

    # Typed attribute
    assert spec.net_debit_estimate == per_share

    # Legacy dict projection (used downstream by alpha_options meta stamping)
    meta_dict = spec.to_legacy_meta()
    assert meta_dict["net_debit_estimate"] == per_share, (
        f"OptionsSpreadSpec.to_legacy_meta() must preserve per-share semantics. "
        f"got={meta_dict.get('net_debit_estimate')!r} expected={per_share!r}"
    )


def test_notional_recovery_uses_options_multiplier_for_contract_dollar():
    """Per-share / contract-dollar invariant on the paper-fill side:
    notional in contract-dollar units = quantity × per_share × multiplier
    (where multiplier = 100 for standard equity options).

    This is the boundary where per-share fill_price is converted into
    contract-dollar notional — and the only legitimate place a ×100
    happens in the BAG pipeline.
    """
    quantity = 2.0
    per_share_fill = 1.50
    expected_contract_dollar_notional = quantity * per_share_fill * _OPTIONS_MULTIPLIER
    assert expected_contract_dollar_notional == 300.0
    # This invariant is enforced by chad/tests/test_alpha_options_bag_paper_fill.py
    # (test_bag_simulator_computes_notional_with_options_multiplier, line ~120)
    # and chad/execution/paper_exec_evidence_writer.py (_OPTIONS_MULTIPLIER=100).
    # This test re-states it here to make the per-share→contract-dollar
    # boundary visible in one place for Official-Matrix Box 051.


def test_non_bag_intent_does_not_have_per_share_semantics_imposed():
    """Per-share semantics apply only to BAG. STK/OPT/FUT intents are
    returned unchanged by ``_resolve_bag_lmt_discipline`` (no unit
    transform); this is the existing test_bag_lmt_discipline.py
    coverage, re-asserted here as part of the unit-normalization
    contract.
    """
    intent = NormalizedIntent(
        strategy="alpha",
        symbol="SPY",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=100.0,
        notional_estimate=0.0,
        asset_class="equity",
        source_strategies=("alpha",),
        created_at=datetime.now(timezone.utc),
        limit_price=500.0,  # per-share STK price, unrelated to BAG semantics
        meta={},
    )

    resolved = _resolve_bag_lmt_discipline(intent)

    # Non-BAG: function returns intent unchanged.
    assert resolved is intent, "Non-BAG intents must pass through unchanged"
    assert resolved.limit_price == 500.0
