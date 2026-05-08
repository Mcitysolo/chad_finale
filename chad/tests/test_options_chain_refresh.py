#!/usr/bin/env python3
"""
chad/tests/test_options_chain_refresh.py

Tests for ISSUE-50: the options-chain refresh service must never block
indefinitely when IBKR data servers are unavailable or slow.

These tests construct a fake IB-like object and never connect to a real
broker. They verify:

- ``_resolve_contract_details_timeout`` honors ``CHAD_OPTIONS_CHAIN_TIMEOUT_SECONDS``
  and falls back safely on missing/invalid values.
- ``_fetch_chain_via_contract_details`` raises a typed ``OptionsChainTimeoutError``
  when the underlying IBKR call exceeds the configured bound.
- ``run`` writes a valid empty cache artifact (with ``ts_utc`` and an
  ``error`` field) when every symbol times out, and exits with rc != 0
  rather than hanging.
- Normal-success behavior is preserved: when the fake IB returns
  contract details, the cache file is written with populated chains.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.market_data import options_chain_refresh as ocr


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeContract:
    def __init__(
        self,
        *,
        expiry: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> None:
        self.lastTradeDateOrContractMonth = expiry
        self.strike = float(strike)
        self.right = right
        self.exchange = exchange


class _FakeDetail:
    def __init__(self, contract: _FakeContract) -> None:
        self.contract = contract


class _FakeIB:
    """
    Minimal IB-like object. ``mode='timeout'`` makes the contract-details
    fetch hang past any reasonable timeout (caller is expected to bound
    it via ``asyncio.wait_for``). ``mode='success'`` returns a small,
    valid set of details immediately.
    """

    def __init__(self, *, mode: str, spot: float = 500.0) -> None:
        self.mode = mode
        self.spot = spot
        self.connected = False
        self.cancelled = False
        self.errorEvent = _FakeEvent()

    # --- connection lifecycle -------------------------------------------------
    def connect(self, host: str, port: int, *, clientId: int, timeout: int, readonly: bool) -> None:  # noqa: N803
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    # --- market data ---------------------------------------------------------
    def qualifyContracts(self, *contracts: Any) -> List[Any]:
        return list(contracts)

    def reqMktData(self, *_args: Any, **_kwargs: Any) -> Any:
        ticker = types.SimpleNamespace(
            last=self.spot,
            close=self.spot,
            bid=self.spot - 0.1,
            ask=self.spot + 0.1,
        )
        return ticker

    def cancelMktData(self, *_args: Any, **_kwargs: Any) -> None:
        self.cancelled = True

    def sleep(self, _seconds: float) -> None:
        return None

    # --- chain metadata ------------------------------------------------------
    async def reqContractDetailsAsync(self, _template: Any) -> List[_FakeDetail]:
        if self.mode == "timeout":
            await asyncio.sleep(3600)  # caller MUST bound this
            return []
        # success mode: emit a few contracts spanning two near expiries
        # straddling spot. Use distant-enough dates to pass the DTE filter.
        from datetime import date, timedelta


        today = date.today()
        e1 = (today + timedelta(days=25)).strftime("%Y%m%d")
        e2 = (today + timedelta(days=40)).strftime("%Y%m%d")
        details: List[_FakeDetail] = []
        for exp in (e1, e2):
            for k in (self.spot - 5, self.spot, self.spot + 5):
                for right in ("C", "P"):
                    details.append(
                        _FakeDetail(_FakeContract(expiry=exp, strike=k, right=right))
                    )
        return details

    # --- event loop ----------------------------------------------------------
    def run(self, awaitable: Any) -> Any:
        return asyncio.new_event_loop().run_until_complete(awaitable)


class _FakeEvent:
    def __iadd__(self, _handler: Any) -> "_FakeEvent":
        return self


# ---------------------------------------------------------------------------
# Timeout resolver
# ---------------------------------------------------------------------------


def test_resolve_timeout_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, raising=False)
    assert ocr._resolve_contract_details_timeout() == float(
        ocr.DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC
    )


def test_resolve_timeout_default_is_30_seconds() -> None:
    assert ocr.DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC == 30


def test_resolve_timeout_honors_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    assert ocr._resolve_contract_details_timeout() == 5.0


@pytest.mark.parametrize("bad_value", ["", "   ", "abc", "0", "-1", "nan", "inf"])
def test_resolve_timeout_falls_back_on_invalid_env(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, bad_value)
    assert ocr._resolve_contract_details_timeout() == float(
        ocr.DEFAULT_CONTRACT_DETAILS_TIMEOUT_SEC
    )


# ---------------------------------------------------------------------------
# Per-call fetch timeout
# ---------------------------------------------------------------------------


def test_fetch_chain_via_contract_details_raises_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    fake_ib = _FakeIB(mode="timeout")

    with pytest.raises(ocr.OptionsChainTimeoutError) as excinfo:
        ocr._fetch_chain_via_contract_details(fake_ib, "SPY", spot=500.0)

    assert excinfo.value.symbol == "SPY"
    assert excinfo.value.timeout_seconds == pytest.approx(0.05)


def test_fetch_chain_via_contract_details_success_returns_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    fake_ib = _FakeIB(mode="success", spot=500.0)
    expirations, strikes, exchange = ocr._fetch_chain_via_contract_details(
        fake_ib, "SPY", spot=500.0
    )
    assert expirations, "expected at least one expiry"
    assert strikes, "expected at least one strike"
    assert exchange == "SMART"


# ---------------------------------------------------------------------------
# Required test name: end-to-end clean timeout via run()
# ---------------------------------------------------------------------------


def _install_fake_ib_insync(
    monkeypatch: pytest.MonkeyPatch, fake_ib: _FakeIB
) -> None:
    """Inject a minimal IB substitute under both ib_insync and ib_async.

    GAP-A019: Phase 1 migrations swap production imports from ib_insync to
    ib_async one file at a time. To keep this fake fixture usable across
    both pre- and post-migration source, the same module object is bound
    under both names in sys.modules. Lazy imports inside production
    (``from ib_insync import IB`` or ``from ib_async import IB``) both
    resolve to the same fake and never open a real socket.
    """

    fake_module = types.ModuleType("ib_insync")

    def _ib_factory() -> _FakeIB:
        return fake_ib

    class _Stock:
        def __init__(self, symbol: str, exchange: str, currency: str) -> None:
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency
            self.conId = 12345

    class _Option:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_module.IB = _ib_factory  # type: ignore[attr-defined]
    fake_module.Stock = _Stock  # type: ignore[attr-defined]
    fake_module.Option = _Option  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ib_insync", fake_module)
    monkeypatch.setitem(sys.modules, "ib_async", fake_module)


def test_options_chain_refresh_times_out_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    The required test for ISSUE-50.

    With a fake IB whose contract-details call hangs forever, ``run``
    must:
      * not hang indefinitely (timeout should fire in well under 1s),
      * exit with a non-zero return code,
      * write a valid options_chains_cache.json containing ``ts_utc``,
        an empty ``chains`` map, and an ``error`` field describing the
        timeout.
    """
    # Tight timeout so the test is fast.
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")

    # Redirect the cache file to a tmp path so we do not touch runtime state.
    cache_path = tmp_path / "options_chains_cache.json"
    monkeypatch.setattr(ocr, "CACHE_PATH", cache_path)

    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib_insync(monkeypatch, fake_ib)

    # Bound the whole call so a regression that re-introduces the hang
    # fails the test rather than wedging the suite.
    import threading

    rc_holder: Dict[str, int] = {}
    err_holder: Dict[str, BaseException] = {}

    def _target() -> None:
        try:
            rc_holder["rc"] = ocr.run(["SPY"])
        except BaseException as exc:  # noqa: BLE001
            err_holder["err"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=15.0)
    assert not t.is_alive(), "options chain refresh hung past timeout — ISSUE-50 regression"
    assert "err" not in err_holder, f"unexpected exception: {err_holder.get('err')!r}"
    assert rc_holder.get("rc", 0) != 0, "expected non-zero rc when all symbols timeout"

    assert cache_path.is_file(), "expected an artifact to be written even on timeout"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "ts_utc" in payload and isinstance(payload["ts_utc"], str)
    assert payload.get("chains") == {}
    assert "error" in payload and "timeout" in payload["error"].lower()


def test_options_chain_refresh_success_writes_populated_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    cache_path = tmp_path / "options_chains_cache.json"
    monkeypatch.setattr(ocr, "CACHE_PATH", cache_path)

    fake_ib = _FakeIB(mode="success", spot=500.0)
    _install_fake_ib_insync(monkeypatch, fake_ib)

    rc = ocr.run(["SPY"])
    assert rc == 0
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "ts_utc" in payload
    chains = payload.get("chains", {})
    assert "SPY" in chains
    spy = chains["SPY"]
    assert spy["expirations"], "expected at least one expiry"
    assert spy["strikes"], "expected at least one strike"
    # Success path should NOT produce an error field.
    assert "error" not in payload


def test_install_fake_ib_intercepts_both_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared fake must intercept ib_insync AND ib_async imports.

    GAP-A019 phase 1 migrations swap production imports between the two
    libraries one file at a time. A single fixture call has to satisfy
    both lazy ``from ib_insync import IB`` (pre-migration) and
    ``from ib_async import IB`` (post-migration) so future batches do
    not regress against real-broker test runs.
    """
    import importlib

    fake_ib = _FakeIB(mode="success", spot=100.0)
    _install_fake_ib_insync(monkeypatch, fake_ib)

    insync = importlib.import_module("ib_insync")
    asyncmod = importlib.import_module("ib_async")

    # Same fake module bound under both names — single source of truth
    # so attribute lookups stay in lockstep.
    assert insync is asyncmod

    # IB() factory returns the injected fake under either name.
    assert insync.IB() is fake_ib
    assert asyncmod.IB() is fake_ib

    # Stock / Option symbols resolve under either name (covers the
    # ``from ib_async import Stock, Option`` pattern used by
    # options_chain_refresh.py).
    assert insync.Stock("SPY", "SMART", "USD").symbol == "SPY"
    assert asyncmod.Stock("SPY", "SMART", "USD").symbol == "SPY"
    assert asyncmod.Option(symbol="SPY", strike=500.0).symbol == "SPY"
