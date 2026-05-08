#!/usr/bin/env python3
"""
chad/tests/test_ibkr_qualify_cache.py

Tests for the IBKR qualified-contract cache that protects the live loop
from repeated qualifyContracts() round-trips and broker timeouts.

Covers:
- STK same-contract cache hit on second call.
- FUT same contract_month cache hit on second call.
- FUT different contract_month uses a different cache key.
- Cache disabled via TTL=0 still calls qualify every time.
- qualify failure does not poison the cache.
- Missing futures contract_month still raises ContractResolutionError.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List

import pytest

from chad.execution import ibkr_adapter as adapter_mod
from chad.execution.ibkr_adapter import (
    ContractResolutionError,
    IbkrAdapter,
    IbkrConfig,
    _ContractResolver,
    _QualifyCache,
    _resolve_qualify_cache_ttl_seconds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeIB:
    """Minimal IB stand-in that records qualifyContracts() invocations.

    Returns each input contract with a synthetic conId attached, so the
    cache stores a non-None result.
    """

    def __init__(self) -> None:
        self.qualify_calls: List[Any] = []
        self._next_con_id = 1000

    def qualifyContracts(self, *contracts: Any) -> List[Any]:
        self.qualify_calls.extend(contracts)
        out: List[Any] = []
        for c in contracts:
            qualified = SimpleNamespace(
                symbol=getattr(c, "symbol", ""),
                secType=getattr(c, "secType", ""),
                exchange=getattr(c, "exchange", ""),
                currency=getattr(c, "currency", ""),
                lastTradeDateOrContractMonth=getattr(c, "lastTradeDateOrContractMonth", ""),
                strike=getattr(c, "strike", 0.0),
                right=getattr(c, "right", ""),
                multiplier=getattr(c, "multiplier", ""),
                conId=self._next_con_id,
            )
            self._next_con_id += 1
            out.append(qualified)
        return out


def _stk_contract(symbol: str = "AAPL") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        secType="STK",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="",
        strike=0.0,
        right="",
        multiplier="",
    )


def _fut_contract(symbol: str = "MES", month: str = "202606") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        secType="FUT",
        exchange="CME",
        currency="USD",
        lastTradeDateOrContractMonth=month,
        strike=0.0,
        right="",
        multiplier="5",
    )


def _build_adapter(monkeypatch: pytest.MonkeyPatch, *, ttl_env: str = "86400") -> IbkrAdapter:
    monkeypatch.setenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", ttl_env)
    cfg = IbkrConfig(host="127.0.0.1", port=4002, client_id=999, dry_run=True)
    return IbkrAdapter(config=cfg)


# ---------------------------------------------------------------------------
# TTL resolution
# ---------------------------------------------------------------------------


def test_default_ttl_is_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", raising=False)
    assert _resolve_qualify_cache_ttl_seconds() == 24 * 3600.0


def test_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", "60")
    assert _resolve_qualify_cache_ttl_seconds() == 60.0


def test_ttl_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", "0")
    assert _resolve_qualify_cache_ttl_seconds() == 0.0


def test_ttl_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", "not-a-number")
    assert _resolve_qualify_cache_ttl_seconds() == 24 * 3600.0


# ---------------------------------------------------------------------------
# _QualifyCache unit-level behavior
# ---------------------------------------------------------------------------


def test_cache_disabled_with_zero_ttl() -> None:
    cache = _QualifyCache(ttl_seconds=0.0, now_fn=lambda: datetime.now(timezone.utc))
    assert not cache.enabled
    contract = _stk_contract()
    cache.store(contract, contract)
    assert cache.get(contract) is None


def test_cache_hit_within_ttl() -> None:
    cache = _QualifyCache(ttl_seconds=60.0, now_fn=lambda: datetime.now(timezone.utc))
    c = _stk_contract("MSFT")
    qualified = SimpleNamespace(symbol="MSFT", conId=42)
    cache.store(c, qualified)
    assert cache.get(c) is qualified


def test_cache_expires_after_ttl() -> None:
    clock = {"t": datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)}
    cache = _QualifyCache(ttl_seconds=10.0, now_fn=lambda: clock["t"])
    c = _stk_contract("NVDA")
    cache.store(c, SimpleNamespace(conId=1))
    # advance past TTL
    clock["t"] = datetime(2026, 5, 8, 12, 0, 11, tzinfo=timezone.utc)
    assert cache.get(c) is None


# ---------------------------------------------------------------------------
# IbkrAdapter._qualify_if_possible integration
# ---------------------------------------------------------------------------


def test_stk_same_contract_hits_cache_on_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(monkeypatch)
    ib = _FakeIB()
    c = _stk_contract("AAPL")

    first = adapter._qualify_if_possible(ib, c)
    second = adapter._qualify_if_possible(ib, c)

    assert len(ib.qualify_calls) == 1, "second call must hit cache"
    assert getattr(first, "conId", None) == getattr(second, "conId", None)


def test_fut_same_contract_month_hits_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(monkeypatch)
    ib = _FakeIB()
    c1 = _fut_contract("MES", "202606")
    c2 = _fut_contract("MES", "202606")

    adapter._qualify_if_possible(ib, c1)
    adapter._qualify_if_possible(ib, c2)

    assert len(ib.qualify_calls) == 1


def test_fut_different_contract_month_uses_different_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _build_adapter(monkeypatch)
    ib = _FakeIB()
    june = _fut_contract("MES", "202606")
    sept = _fut_contract("MES", "202609")

    adapter._qualify_if_possible(ib, june)
    adapter._qualify_if_possible(ib, sept)

    assert len(ib.qualify_calls) == 2, "different months must miss cache independently"


def test_cache_disabled_via_ttl_zero_still_calls_qualify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _build_adapter(monkeypatch, ttl_env="0")
    ib = _FakeIB()
    c = _stk_contract("TSLA")

    adapter._qualify_if_possible(ib, c)
    adapter._qualify_if_possible(ib, c)
    adapter._qualify_if_possible(ib, c)

    assert len(ib.qualify_calls) == 3, "TTL=0 must bypass cache entirely"


def test_qualify_failure_does_not_poison_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(monkeypatch)

    class _BoomIB:
        def __init__(self) -> None:
            self.calls = 0

        def qualifyContracts(self, *contracts: Any) -> List[Any]:
            self.calls += 1
            raise TimeoutError("simulated broker timeout")

    ib = _BoomIB()
    c = _stk_contract("GOOGL")

    out1 = adapter._qualify_if_possible(ib, c)
    out2 = adapter._qualify_if_possible(ib, c)

    # Fallback returns the original (unqualified) contract.
    assert out1 is c
    assert out2 is c
    # Both calls must hit the underlying qualify path (no negative caching).
    assert ib.calls == 2


def test_qualify_returning_empty_list_does_not_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _build_adapter(monkeypatch)

    class _EmptyIB:
        def __init__(self) -> None:
            self.calls = 0

        def qualifyContracts(self, *contracts: Any) -> List[Any]:
            self.calls += 1
            return []

    ib = _EmptyIB()
    c = _stk_contract("EMPTY")

    out1 = adapter._qualify_if_possible(ib, c)
    out2 = adapter._qualify_if_possible(ib, c)

    assert out1 is c and out2 is c
    assert ib.calls == 2, "empty qualify result must not be cached"


# ---------------------------------------------------------------------------
# Preserved behavior: futures resolution still demands explicit contract_month
# ---------------------------------------------------------------------------


def test_missing_futures_contract_month_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS", "86400")
    cfg = IbkrConfig(host="127.0.0.1", port=4002, client_id=999, dry_run=False)
    resolver = _ContractResolver(cfg, lambda: datetime.now(timezone.utc))

    intent = SimpleNamespace(
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        currency="USD",
        meta={},  # no contract_month / lastTradeDateOrContractMonth
    )

    fake_ib = _FakeIB()
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(fake_ib, intent)
