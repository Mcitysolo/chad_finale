#!/usr/bin/env python3
"""
chad/tests/test_pr05_options_chain_secdef.py

PR-05 — options-chain discovery migrated from bulk ``reqContractDetails``
(pacing-throttled, timing out at 30s and producing an empty cache) to
``reqSecDefOptParams`` (a single chain-structure metadata round-trip that
pacing does not throttle).

These tests pin the new discovery contract WITHOUT changing what the
consumers (alpha_options, omega_momentum_options) receive:

  * happy path with a mocked reqSecDefOptParams result,
  * empty-chains response → typed failure (not a fake-healthy cache),
  * per-exchange filtering (prefer SMART; merge-all fallback),
  * strike-window filter (~10% of spot),
  * expiration filter (DTE-bound + nearest cap),
  * per-call timeout surfaces OptionsChainTimeoutError,
  * retry/backoff recovers a transient (flaky) failure,
  * failure-artifact emission with the unchanged v1 schema,
  * cache-write schema preservation (options_chain_cache.v2 + every field
    the consumers read),
  * consumer compatibility: the produced per-symbol chain round-trips
    through OptionsChain.from_dict and feeds select_vertical_spread.

Stdlib-only; never opens a real IBKR socket.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.market_data import options_chain_refresh as ocr


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __iadd__(self, _handler: Any) -> "_FakeEvent":
        return self


class _FakeOptionChain:
    """ib_async OptionChain stand-in (one reqSecDefOptParams result row)."""

    def __init__(
        self,
        *,
        exchange: str = "SMART",
        expirations: Optional[List[str]] = None,
        strikes: Optional[List[float]] = None,
        trading_class: str = "SPY",
        multiplier: str = "100",
    ) -> None:
        self.exchange = exchange
        self.expirations = list(expirations or [])
        self.strikes = list(strikes or [])
        self.tradingClass = trading_class
        self.multiplier = multiplier


def _near_expiries() -> List[str]:
    today = date.today()
    return [
        (today + timedelta(days=25)).strftime("%Y%m%d"),
        (today + timedelta(days=40)).strftime("%Y%m%d"),
    ]


class _FakeIB:
    """Mode-driven IB stand-in for the reqSecDefOptParams discovery path.

    ``mode='success'`` returns the chains passed via ``chains``.
    ``mode='timeout'`` hangs forever (caller MUST bound it).
    ``mode='flaky'`` hangs the first ``flaky_failures`` calls then succeeds.
    """

    def __init__(
        self,
        *,
        mode: str = "success",
        spot: float = 500.0,
        chains: Optional[List[_FakeOptionChain]] = None,
        flaky_failures: int = 0,
    ) -> None:
        self.mode = mode
        self.spot = spot
        self.connected = False
        self.errorEvent = _FakeEvent()
        self.flaky_failures_remaining = int(flaky_failures)
        self.sleep_calls: List[float] = []
        self.secdef_calls = 0
        if chains is not None:
            self._chains = chains
        else:
            e1, e2 = _near_expiries()
            self._chains = [
                _FakeOptionChain(
                    exchange="SMART",
                    expirations=[e1, e2],
                    strikes=[spot - 5, spot, spot + 5],
                )
            ]

    # connection lifecycle
    def connect(self, host: str, port: int, *, clientId: int, timeout: int, readonly: bool) -> None:  # noqa: N803
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    # market data / qualification
    def qualifyContracts(self, *contracts: Any) -> List[Any]:
        return list(contracts)

    def reqMktData(self, *_args: Any, **_kwargs: Any) -> Any:
        return types.SimpleNamespace(
            last=self.spot, close=self.spot,
            bid=self.spot - 0.1, ask=self.spot + 0.1,
        )

    def cancelMktData(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(float(seconds))

    # chain metadata
    async def reqSecDefOptParamsAsync(self, **_kwargs: Any) -> List[_FakeOptionChain]:
        self.secdef_calls += 1
        if self.mode == "timeout":
            await asyncio.sleep(3600)
            return []
        if self.mode == "flaky" and self.flaky_failures_remaining > 0:
            self.flaky_failures_remaining -= 1
            await asyncio.sleep(3600)
            return []
        if self.mode == "empty":
            return []
        return list(self._chains)

    def run(self, awaitable: Any) -> Any:
        return asyncio.new_event_loop().run_until_complete(awaitable)


def _install_fake_ib(monkeypatch: pytest.MonkeyPatch, fake: _FakeIB) -> None:
    fake_module = types.ModuleType("ib_insync")

    def _factory() -> _FakeIB:
        return fake

    class _Stock:
        def __init__(self, symbol: str, exchange: str, currency: str) -> None:
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency
            self.conId = 756733  # SPY conId-like sentinel

    fake_module.IB = _factory  # type: ignore[attr-defined]
    fake_module.Stock = _Stock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ib_insync", fake_module)
    monkeypatch.setitem(sys.modules, "ib_async", fake_module)


@pytest.fixture
def _isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(ocr, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(ocr, "CACHE_PATH", tmp_path / "options_chains_cache.json")
    return tmp_path


def _run_in_thread(target_fn, timeout: float = 20.0):
    box: Dict[str, Any] = {}

    def _go() -> None:
        try:
            box["rc"] = target_fn()
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), "refresh hung past the test bound — timeout regression"
    if "err" in box:
        raise AssertionError(f"unexpected exception in ocr.run: {box['err']!r}")
    return box.get("rc")


# ---------------------------------------------------------------------------
# 1. Happy path — reqSecDefOptParams discovery
# ---------------------------------------------------------------------------


def test_happy_path_uses_secdef_opt_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    fake_ib = _FakeIB(mode="success", spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    expirations, strikes, exchange = ocr._fetch_chain_via_secdef_opt_params(
        fake_ib, "SPY", spot=500.0
    )
    assert fake_ib.secdef_calls == 1, "must call reqSecDefOptParams exactly once"
    assert expirations == sorted(_near_expiries())
    assert strikes == [495.0, 500.0, 505.0]
    assert exchange == "SMART"


# ---------------------------------------------------------------------------
# 2. Empty chains response
# ---------------------------------------------------------------------------


def test_empty_chains_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    fake_ib = _FakeIB(mode="empty", spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    with pytest.raises(RuntimeError) as excinfo:
        ocr._fetch_chain_via_secdef_opt_params(fake_ib, "SPY", spot=500.0)
    assert "reqSecDefOptParams returned empty" in str(excinfo.value)


def test_chains_present_but_no_strikes_in_window_yields_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # reqSecDefOptParams returns a chain, but every strike is far from spot.
    e1, e2 = _near_expiries()
    chains = [_FakeOptionChain(exchange="SMART", expirations=[e1, e2],
                               strikes=[100.0, 110.0, 120.0])]
    fake_ib = _FakeIB(mode="success", spot=500.0, chains=chains)
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    _install_fake_ib(monkeypatch, fake_ib)

    expirations, strikes, _ = ocr._fetch_chain_via_secdef_opt_params(
        fake_ib, "SPY", spot=500.0
    )
    assert expirations, "expirations still pass the DTE filter"
    assert strikes == [], "no strike within +/-10% of spot survives the window"


# ---------------------------------------------------------------------------
# 3. Per-exchange filtering
# ---------------------------------------------------------------------------


def test_per_exchange_prefers_smart() -> None:
    e1, e2 = _near_expiries()
    chains = [
        _FakeOptionChain(exchange="CBOE", expirations=[e1], strikes=[499.0]),
        _FakeOptionChain(exchange="SMART", expirations=[e1, e2],
                         strikes=[495.0, 500.0, 505.0]),
        _FakeOptionChain(exchange="AMEX", expirations=[e2], strikes=[501.0]),
    ]
    selected, exchange = ocr._select_exchange_chains(chains)
    assert exchange == "SMART"
    assert len(selected) == 1
    assert selected[0].exchange == "SMART"


def test_per_exchange_merges_when_smart_absent() -> None:
    e1, e2 = _near_expiries()
    chains = [
        _FakeOptionChain(exchange="CBOE", expirations=[e1], strikes=[499.0]),
        _FakeOptionChain(exchange="AMEX", expirations=[e2], strikes=[501.0]),
    ]
    selected, exchange = ocr._select_exchange_chains(chains)
    assert exchange == "CBOE", "label is the first concrete exchange seen"
    assert len(selected) == 2, "all exchanges merged when SMART is absent"


def test_smart_only_strikes_used_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-SMART exchange exposes a strike inside the window that SMART does
    # not list; selecting SMART must NOT pull in the CBOE-only strike.
    e1, e2 = _near_expiries()
    chains = [
        _FakeOptionChain(exchange="CBOE", expirations=[e1], strikes=[498.5]),
        _FakeOptionChain(exchange="SMART", expirations=[e1, e2],
                         strikes=[495.0, 500.0, 505.0]),
    ]
    fake_ib = _FakeIB(mode="success", spot=500.0, chains=chains)
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    _install_fake_ib(monkeypatch, fake_ib)
    _, strikes, exchange = ocr._fetch_chain_via_secdef_opt_params(
        fake_ib, "SPY", spot=500.0
    )
    assert exchange == "SMART"
    assert 498.5 not in strikes
    assert strikes == [495.0, 500.0, 505.0]


# ---------------------------------------------------------------------------
# 4. Strike-window filter (~10% of spot)
# ---------------------------------------------------------------------------


def test_strike_window_filter_keeps_only_within_10pct() -> None:
    spot = 500.0
    # 450 and 550 are the exact +/-10% boundaries (inclusive); 449.9 / 550.1
    # are just outside and must be dropped.
    raw = [400.0, 449.9, 450.0, 495.0, 500.0, 505.0, 550.0, 550.1, 600.0]
    kept = ocr._filter_strikes(raw, spot)
    assert kept == [450.0, 495.0, 500.0, 505.0, 550.0]


def test_strike_window_filter_no_spot_returns_all_sorted() -> None:
    raw = [505.0, 495.0, 500.0]
    assert ocr._filter_strikes(raw, 0.0) == [495.0, 500.0, 505.0]


# ---------------------------------------------------------------------------
# 5. Expiration filter (DTE-bound + nearest cap)
# ---------------------------------------------------------------------------


def test_expiration_filter_drops_past_and_far_dte() -> None:
    today = date(2026, 5, 29)
    yesterday = (today - timedelta(days=1)).strftime("%Y%m%d")
    near = (today + timedelta(days=30)).strftime("%Y%m%d")
    edge = (today + timedelta(days=ocr.MAX_EXPIRY_DTE)).strftime("%Y%m%d")
    leap = (today + timedelta(days=ocr.MAX_EXPIRY_DTE + 1)).strftime("%Y%m%d")
    kept = ocr._filter_expirations(
        [yesterday, near, edge, leap, "garbage"], today=today
    )
    assert kept == sorted([near, edge])
    assert yesterday not in kept and leap not in kept


def test_expiration_filter_caps_at_max_expiries() -> None:
    today = date(2026, 5, 29)
    # 30 weekly expirations within 90 days → only the nearest MAX_EXPIRIES kept.
    exps = [
        (today + timedelta(days=d)).strftime("%Y%m%d")
        for d in range(1, 88, 3)
    ]
    kept = ocr._filter_expirations(exps, today=today)
    assert len(kept) == ocr.MAX_EXPIRIES
    assert kept == sorted(exps)[: ocr.MAX_EXPIRIES]


# ---------------------------------------------------------------------------
# 6. Per-call timeout
# ---------------------------------------------------------------------------


def test_per_call_timeout_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)
    with pytest.raises(ocr.OptionsChainTimeoutError) as excinfo:
        ocr._fetch_chain_via_secdef_opt_params(fake_ib, "SPY", spot=500.0)
    assert excinfo.value.symbol == "SPY"
    assert excinfo.value.timeout_seconds == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 7. Retry / backoff under transient failures
# ---------------------------------------------------------------------------


def test_retry_recovers_flaky_symbol(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0.01")
    fake_ib = _FakeIB(mode="flaky", flaky_failures=1, spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]))
    assert rc == 0, "retry must recover the transient failure"

    cache = _isolated_runtime / "options_chains_cache.json"
    doc = json.loads(cache.read_text(encoding="utf-8"))
    assert "SPY" in doc["chains"]
    assert doc["chains"]["SPY"]["strikes"]
    assert "error" not in doc
    assert not (_isolated_runtime / ocr.FAILURE_ARTIFACT_NAME).is_file()
    # At least one backoff sleep happened between the failed and good attempt.
    assert any(s == 0.01 for s in fake_ib.sleep_calls)


# ---------------------------------------------------------------------------
# 8. Failure-artifact emission (schema unchanged) when all attempts fail
# ---------------------------------------------------------------------------


def test_failure_artifact_emitted_on_total_timeout(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "2")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")
    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]))
    assert rc != 0

    failure = _isolated_runtime / ocr.FAILURE_ARTIFACT_NAME
    assert failure.is_file()
    payload = json.loads(failure.read_text(encoding="utf-8"))
    # Schema preserved exactly (options_chain_refresh_failure.v1) ...
    assert payload["schema_version"] == ocr.FAILURE_ARTIFACT_SCHEMA == "options_chain_refresh_failure.v1"
    assert payload["status"] == "failed"
    assert payload["error_type"] == "all_symbols_failed"
    assert payload["max_attempts"] == 2
    assert "SPY" in payload["symbol_errors"]
    assert payload["attempts"]["SPY"]
    # ... but provider / blocked_reason honestly name the new API.
    assert payload["provider"] == "ibkr_secdef_opt_params"
    assert payload["blocked_reason"] == "ibkr_secdef_opt_params_unresponsive"


# ---------------------------------------------------------------------------
# 9. Cache-write schema preservation (options_chain_cache.v2)
# ---------------------------------------------------------------------------


def test_cache_write_preserves_v2_schema_and_fields(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "1")
    fake_ib = _FakeIB(mode="success", spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]))
    assert rc == 0

    cache = _isolated_runtime / "options_chains_cache.json"
    doc = json.loads(cache.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "options_chain_cache.v2"
    assert "ts_utc" in doc and isinstance(doc["ts_utc"], str)
    assert "error" not in doc
    spy = doc["chains"]["SPY"]
    # Every field the cache schema carries must be present with v2 semantics.
    for field_name in (
        "symbol", "exchange", "expirations", "strikes",
        "spot_price", "ts_utc", "ttl_seconds",
    ):
        assert field_name in spy, f"v2 cache must carry {field_name!r}"
    assert spy["symbol"] == "SPY"
    assert spy["exchange"] == "SMART"
    assert spy["spot_price"] == 500.0
    assert spy["ttl_seconds"] == ocr.CACHE_TTL_SECONDS
    assert isinstance(spy["expirations"], list) and spy["expirations"]
    assert isinstance(spy["strikes"], list) and spy["strikes"]


# ---------------------------------------------------------------------------
# 10. Consumer compatibility — the produced chain feeds the real consumers
# ---------------------------------------------------------------------------


def test_consumer_compatibility_optionschain_and_spread_selector(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    """The per-symbol chain produced by the new path must round-trip through
    OptionsChain.from_dict (what alpha_options reads) and be selectable by
    select_vertical_spread (the downstream spread builder)."""
    from chad.options.chain_provider import OptionsChain
    from chad.options.strike_selector import select_vertical_spread

    # Produce a realistic chain: strikes 1-pt apart across the window, two
    # near expiries straddling the alpha_options 21-45 DTE target.
    today = date.today()
    e_in_window = (today + timedelta(days=33)).strftime("%Y%m%d")
    e_far = (today + timedelta(days=80)).strftime("%Y%m%d")
    spot = 500.0
    strikes = [float(s) for s in range(460, 541)]  # well inside +/-10%
    chains = [_FakeOptionChain(exchange="SMART",
                               expirations=[e_in_window, e_far],
                               strikes=strikes)]
    fake_ib = _FakeIB(mode="success", spot=spot, chains=chains)
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "1")
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]))
    assert rc == 0

    doc = json.loads(
        (_isolated_runtime / "options_chains_cache.json").read_text(encoding="utf-8")
    )
    chain_data = doc["chains"]["SPY"]

    # 1) alpha_options' load path: OptionsChain.from_dict must not raise and
    #    must preserve symbol/exchange/expirations/strikes.
    chain = OptionsChain.from_dict(chain_data)
    assert chain.symbol == "SPY"
    assert chain.exchange == "SMART"
    assert chain.expirations == sorted([e_in_window, e_far])
    assert all(isinstance(s, float) for s in chain.strikes)
    assert not chain.is_expired()

    # 2) select_vertical_spread must build a defined-risk bull-call spread
    #    from the discovered structure.
    spread = select_vertical_spread(
        chain=chain,
        current_price=spot,
        direction="bullish",
        target_dte_min=21,
        target_dte_max=45,
        otm_offset_pct=0.02,
        spread_width_pct=0.01,
    )
    assert spread is not None, "the discovered chain must yield a vertical spread"
    assert spread.spread_type == "BULL_CALL"
    assert spread.expiry == e_in_window
    assert spread.short_strike > spread.long_strike
    assert spread.max_loss_per_contract > 0


# ---------------------------------------------------------------------------
# 11. Idempotency — identical IBKR responses yield identical chain structure
# ---------------------------------------------------------------------------


def test_idempotent_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")
    fake_a = _FakeIB(mode="success", spot=500.0)
    fake_b = _FakeIB(mode="success", spot=500.0)
    _install_fake_ib(monkeypatch, fake_a)
    out_a = ocr._fetch_chain_via_secdef_opt_params(fake_a, "SPY", spot=500.0)
    _install_fake_ib(monkeypatch, fake_b)
    out_b = ocr._fetch_chain_via_secdef_opt_params(fake_b, "SPY", spot=500.0)
    assert out_a == out_b, "identical IBKR responses must produce identical output"


# ---------------------------------------------------------------------------
# 12. conId qualification failure is attributable
# ---------------------------------------------------------------------------


def test_qualify_failure_raises_attributable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ib = _FakeIB(mode="success", spot=500.0)

    # qualifyContracts returns nothing → the conId resolver must raise with a
    # message the blocked_reason classifier keys on ("qualifyContracts").
    def _empty_qualify(*_contracts: Any) -> List[Any]:
        return []

    monkeypatch.setattr(fake_ib, "qualifyContracts", _empty_qualify)
    _install_fake_ib(monkeypatch, fake_ib)
    with pytest.raises(RuntimeError) as excinfo:
        ocr._resolve_underlying_conid(fake_ib, "SPY")
    assert "qualifyContracts" in str(excinfo.value)
    assert ocr._classify_blocked_reason(
        {"SPY": f"RuntimeError: {excinfo.value}"}
    ) == "ibkr_qualify_contracts_failed"
