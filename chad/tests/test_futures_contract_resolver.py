#!/usr/bin/env python3
"""
chad/tests/test_futures_contract_resolver.py

GAP-A006: Verify that futures intents carry an explicit contract_month
before reaching the IBKR adapter, so the hot path never falls into the
prohibited reqContractDetails branch.

Covers:
  1. Deterministic resolver returns YYYYMM for MES/MNQ/MGC/MCL.
  2. build_ibkr_intents_from_plan attaches contract_month + source +
     resolved_at_utc into intent.meta for FUT planned orders.
  3. IbkrAdapter._intent_from_trade_intent propagates raw_intent.meta
     into NormalizedIntent.meta.
  4. _resolve_future returns a contract when contract_month is present
     and still raises ContractResolutionError when it is missing
     (preserves the P0-1 safety contract).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from chad.execution.execution_pipeline import (
    ExecutionPlan,
    PlannedOrder,
    build_ibkr_intents_from_plan,
)
from chad.execution.ibkr_adapter import (
    ContractResolutionError,
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
    _ContractResolver,
)
from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRStrategyTradeIntent
from chad.market_data.futures_contract_resolver import (
    SOURCE_NAME as FUTURES_RESOLVER_SOURCE,
    resolve_contract_month,
)
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# 1) Resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["MES", "MNQ", "MGC", "MCL"])
def test_resolver_returns_yyyymm_for_supported_symbols(symbol: str) -> None:
    """Schedule covers all four production futures symbols."""
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    month = resolve_contract_month(symbol, now=now)
    assert month is not None
    assert len(month) == 6 and month.isdigit()
    # Must be at least the current month
    assert month >= now.strftime("%Y%m")


def test_resolver_unsupported_symbol_returns_none() -> None:
    assert resolve_contract_month("ZZZ") is None


def test_resolver_lowercase_symbol_normalises() -> None:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    assert resolve_contract_month("mes", now=now) is not None


def test_resolver_does_not_return_expired_mcl_on_2026_05_19() -> None:
    """MCLM6 (delivery June 2026) last traded 2026-05-18 per IBKR. On
    2026-05-19 the resolver must not return '202606' or the live loop
    will submit to an expired contract and IBKR rejects with Error 201."""
    now = datetime(2026, 5, 19, 2, 30, tzinfo=timezone.utc)
    month = resolve_contract_month("MCL", now=now)
    assert month is not None
    assert month != "202606", (
        f"MCL must not return expired MCLM6 contract on 2026-05-19; got {month}"
    )
    assert month >= "202607"


def test_resolver_non_mcl_symbols_unchanged_on_2026_05_19() -> None:
    """Equity-index and metal micro-futures keep their standard
    'expires in delivery month' calendar; their 202606 entry is still
    valid on 2026-05-19."""
    now = datetime(2026, 5, 19, 2, 30, tzinfo=timezone.utc)
    for sym in ("MES", "MNQ", "MGC", "M2K", "MYM", "ZN", "ZB", "M6E"):
        assert resolve_contract_month(sym, now=now) == "202606", (
            f"{sym} schedule must still resolve to 202606 on 2026-05-19"
        )


# ---------------------------------------------------------------------------
# 2) Intent builder attaches meta for futures
# ---------------------------------------------------------------------------


def _planned_futures_order(symbol: str, price: float) -> PlannedOrder:
    return PlannedOrder(
        symbol=symbol,
        side=SignalSide.BUY,
        size=1.0,
        asset_class=AssetClass.FUTURES,
        price=price,
        notional=price * 5.0,
        primary_strategy=StrategyName.ALPHA_FUTURES,
        contributing_strategies=(StrategyName.ALPHA_FUTURES,),
    )


@pytest.fixture
def _disable_routing_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the routing gates so we can test intent metadata in isolation.

    The contract_month attachment happens before the gate; rejecting on
    gate failure would mask the meta fields we care about here.
    """
    import chad.execution.execution_pipeline as ep

    monkeypatch.setattr(ep, "run_all_gates", lambda **kwargs: (True, ""))


@pytest.fixture(autouse=True)
def _isolate_stop_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    # NEW-GAP-033b / Box 030: force the stop-bus check to False so that the
    # production short-circuit in build_ibkr_intents_from_plan does not consult
    # live runtime/stop_bus.json. Production behavior is correct and unchanged;
    # this fixture only isolates the unit under test from a real-world latency
    # excursion that would otherwise legitimately return [] and mask the
    # contract_month / registry assertions we care about.
    import chad.risk.stop_bus_state as sbs
    import chad.execution.execution_pipeline as ep

    monkeypatch.setattr(sbs, "is_stop_bus_active", lambda *_a, **_kw: False)
    monkeypatch.setattr(ep, "_stop_bus_active", lambda: False)


@pytest.mark.parametrize("symbol", ["MES", "MGC"])
def test_build_intents_attaches_contract_month_for_futures(
    symbol: str, _disable_routing_gates: None
) -> None:
    # Prices well outside any reasonable daily move would still pass since
    # the gate is patched out. Pick something sensible to make notional
    # sane in case the planner-side checks tighten later.
    price = 7000.0 if symbol == "MES" else 4500.0
    plan = ExecutionPlan(orders=[_planned_futures_order(symbol, price=price)])
    intents = build_ibkr_intents_from_plan(plan)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.sec_type == "FUT"
    assert isinstance(intent.meta, dict)
    cm = intent.meta.get("contract_month")
    assert cm and len(cm) == 6 and cm.isdigit()
    assert intent.meta.get("contract_month_source") == FUTURES_RESOLVER_SOURCE
    assert intent.meta.get("contract_month_resolved_at_utc")


def test_build_intents_does_not_attach_contract_month_for_equities(
    _disable_routing_gates: None,
) -> None:
    plan = ExecutionPlan(
        orders=[
            PlannedOrder(
                symbol="SPY",
                side=SignalSide.BUY,
                size=10.0,
                asset_class=AssetClass.ETF,
                price=730.0,
                notional=7300.0,
                primary_strategy=StrategyName.BETA,
                contributing_strategies=(StrategyName.BETA,),
            )
        ]
    )
    intents = build_ibkr_intents_from_plan(plan)
    assert len(intents) == 1
    assert "contract_month" not in (intents[0].meta or {})


# ---------------------------------------------------------------------------
# 3) Adapter normalisation propagates meta
# ---------------------------------------------------------------------------


def test_intent_from_trade_intent_propagates_meta() -> None:
    raw = IBKRStrategyTradeIntent(
        strategy="alpha_futures",
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=29000.0,
        limit_price=5800.0,
        meta={
            "contract_month": "202606",
            "contract_month_source": FUTURES_RESOLVER_SOURCE,
        },
    )
    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    normalized = adapter._intent_from_trade_intent(raw)
    assert isinstance(normalized, NormalizedIntent)
    assert normalized.meta.get("contract_month") == "202606"
    assert normalized.meta.get("contract_month_source") == FUTURES_RESOLVER_SOURCE


# ---------------------------------------------------------------------------
# 4) _resolve_future safety contract
# ---------------------------------------------------------------------------


class _FakeIB:
    """Minimal IBLike stand-in: presence (not None) is enough to trigger
    the hot-path branch that requires explicit contract_month."""

    def isConnected(self) -> bool:
        return True


def _make_normalized_futures_intent(
    meta: Dict[str, Any],
    *,
    symbol: str = "MES",
    exchange: str = "CME",
    currency: str = "USD",
    strategy: str = "alpha_futures",
    limit_price: float = 5800.0,
    notional_estimate: float = 29000.0,
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy=strategy,
        symbol=symbol,
        sec_type="FUT",
        exchange=exchange,
        currency=currency,
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=notional_estimate,
        asset_class="futures",
        source_strategies=(strategy,),
        created_at=datetime.now(timezone.utc),
        limit_price=limit_price,
        meta=meta,
    )


def test_resolve_future_accepts_explicit_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent({"contract_month": "202606"})
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_resolve_future_raises_when_contract_month_missing() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent({})
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)


# ---------------------------------------------------------------------------
# 5) omega_macro futures resolver coverage (M6E / ZN / ZB)
# ---------------------------------------------------------------------------


def test_ibkr_adapter_resolves_m6e_future_with_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="M6E",
        exchange="CME",
        strategy="omega_macro",
    )
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("symbol") == "M6E"
    assert resolved.summary.get("sec_type") == "FUT"
    assert resolved.summary.get("exchange") == "CME"
    assert resolved.summary.get("currency") == "USD"
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_ibkr_adapter_resolves_zn_future_with_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="ZN",
        exchange="CBOT",
        strategy="omega_macro",
    )
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("symbol") == "ZN"
    assert resolved.summary.get("sec_type") == "FUT"
    assert resolved.summary.get("exchange") == "CBOT"
    assert resolved.summary.get("currency") == "USD"
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_ibkr_adapter_resolves_zb_future_with_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="ZB",
        exchange="CBOT",
        strategy="omega_macro",
    )
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("symbol") == "ZB"
    assert resolved.summary.get("sec_type") == "FUT"
    assert resolved.summary.get("exchange") == "CBOT"
    assert resolved.summary.get("currency") == "USD"
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_ibkr_adapter_m6e_missing_contract_month_still_raises() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {}, symbol="M6E", exchange="CME", strategy="omega_macro"
    )
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)


def test_ibkr_adapter_unknown_future_still_raises() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="ZZZ",
        exchange="CME",
        strategy="omega_macro",
    )
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)


# ---------------------------------------------------------------------------
# 6) gamma_futures M2K resolver coverage
# ---------------------------------------------------------------------------


def test_ibkr_adapter_resolves_m2k_future_with_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="M2K",
        exchange="CME",
        strategy="gamma_futures",
    )
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("symbol") == "M2K"
    assert resolved.summary.get("sec_type") == "FUT"
    assert resolved.summary.get("exchange") == "CME"
    assert resolved.summary.get("currency") == "USD"
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_ibkr_adapter_m2k_missing_contract_month_raises() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {}, symbol="M2K", exchange="CME", strategy="gamma_futures"
    )
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)


def test_ibkr_adapter_resolves_mym_future_with_contract_month() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {"contract_month": "202606"},
        symbol="MYM",
        exchange="CBOT",
        strategy="gamma_futures",
    )
    resolved = resolver._resolve_future(_FakeIB(), intent)
    assert resolved.summary.get("symbol") == "MYM"
    assert resolved.summary.get("sec_type") == "FUT"
    assert resolved.summary.get("exchange") == "CBOT"
    assert resolved.summary.get("currency") == "USD"
    assert resolved.summary.get("contract_month") == "202606"
    assert resolved.summary.get("resolution") == "explicit"


def test_ibkr_adapter_mym_missing_contract_month_raises() -> None:
    cfg = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))
    intent = _make_normalized_futures_intent(
        {}, symbol="MYM", exchange="CBOT", strategy="gamma_futures"
    )
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)
