"""PR-04 — options-chain refresh operational remediation.

Pins the PR-04 hardening contract:

* the refresher retries each symbol up to ``CHAD_OPTIONS_CHAIN_REFRESH_ATTEMPTS``
  times with a configurable backoff between attempts before declaring failure,
* every attempt is recorded in a structured per-symbol log,
* on full failure the service writes BOTH the legacy empty-chains cache (kept
  for R17 alerting) AND a new structured
  ``runtime/options_chain_refresh_failure.json`` describing attempts,
  ``blocked_reason``, ``last_successful_ts``, and the service entrypoint,
* the structured failure artifact is removed on a fully healthy run,
* a successful retry mid-run still produces a populated cache and avoids the
  failure artifact entirely,
* alpha_options fails closed when the chain cache is missing or empty (no
  signals), and omega_momentum_options is honest about synthetic fallback
  pricing by tagging the resulting signal ``synthetic_pricing=True``.

Tests are stdlib-only and never open a real IBKR socket — they reuse the
``_FakeIB`` substitute already used by ``test_options_chain_refresh.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.market_data import options_chain_refresh as ocr


# ---------------------------------------------------------------------------
# Local copies of the FakeIB helpers
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __iadd__(self, _handler: Any) -> "_FakeEvent":
        return self


class _FakeOptionChain:
    """ib_async OptionChain stand-in (reqSecDefOptParams result)."""

    def __init__(
        self,
        *,
        exchange: str = "SMART",
        expirations: List[str] | None = None,
        strikes: List[float] | None = None,
        trading_class: str = "SPY",
        multiplier: str = "100",
    ) -> None:
        self.exchange = exchange
        self.expirations = list(expirations or [])
        self.strikes = list(strikes or [])
        self.tradingClass = trading_class
        self.multiplier = multiplier


class _FakeIB:
    """Fake IBKR client whose chain-params behavior is mode-driven.

    ``mode='timeout'`` hangs every reqSecDefOptParams fetch past any
    reasonable timeout so the caller must bound it. ``mode='success'``
    returns a valid chain. ``mode='flaky'`` fails the first
    ``flaky_failures`` attempts and then succeeds.
    """

    def __init__(
        self,
        *,
        mode: str,
        spot: float = 500.0,
        flaky_failures: int = 0,
    ) -> None:
        self.mode = mode
        self.spot = spot
        self.connected = False
        self.errorEvent = _FakeEvent()
        self.flaky_failures_remaining = int(flaky_failures)
        self.sleep_calls: List[float] = []

    def connect(
        self,
        host: str,
        port: int,
        *,
        clientId: int,
        timeout: int,
        readonly: bool,
    ) -> None:
        self.connected = True

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def qualifyContracts(self, *contracts: Any) -> List[Any]:
        return list(contracts)

    def reqMktData(self, *_args: Any, **_kwargs: Any) -> Any:
        return types.SimpleNamespace(
            last=self.spot,
            close=self.spot,
            bid=self.spot - 0.1,
            ask=self.spot + 0.1,
        )

    def cancelMktData(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(float(seconds))

    async def reqSecDefOptParamsAsync(self, **_kwargs: Any) -> List[_FakeOptionChain]:
        if self.mode == "timeout":
            await asyncio.sleep(3600)
            return []
        if self.mode == "flaky" and self.flaky_failures_remaining > 0:
            self.flaky_failures_remaining -= 1
            await asyncio.sleep(3600)
            return []
        today = date.today()
        e1 = (today + timedelta(days=25)).strftime("%Y%m%d")
        e2 = (today + timedelta(days=40)).strftime("%Y%m%d")
        strikes = [self.spot - 5, self.spot, self.spot + 5]
        return [
            _FakeOptionChain(
                exchange="SMART",
                expirations=[e1, e2],
                strikes=strikes,
            )
        ]

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
            self.conId = 12345

    class _Option:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_module.IB = _factory  # type: ignore[attr-defined]
    fake_module.Stock = _Stock  # type: ignore[attr-defined]
    fake_module.Option = _Option  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ib_insync", fake_module)
    monkeypatch.setitem(sys.modules, "ib_async", fake_module)


def _run_in_thread(target_fn, timeout: float = 30.0):
    """Bound the whole ocr.run call so a regression that re-hangs the
    refresher fails the test rather than wedging the suite."""
    box: Dict[str, Any] = {}

    def _go() -> None:
        try:
            box["rc"] = target_fn()
        except BaseException as exc:  # noqa: BLE001
            box["err"] = exc

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), (
        "options chain refresh hung past the test bound — "
        "retry/backoff regression"
    )
    if "err" in box:
        raise AssertionError(f"unexpected exception in ocr.run: {box['err']!r}")
    return box.get("rc")


# ---------------------------------------------------------------------------
# T1 — full timeout writes BOTH cache and failure artifact
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    monkeypatch.setattr(ocr, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(ocr, "CACHE_PATH", tmp_path / "options_chains_cache.json")
    return tmp_path


def test_full_timeout_writes_failure_artifact_and_empty_cache(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "2")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")

    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)
    assert rc != 0, "expected non-zero rc when every attempt times out"

    cache = _isolated_runtime / "options_chains_cache.json"
    assert cache.is_file()
    cache_doc = json.loads(cache.read_text(encoding="utf-8"))
    assert cache_doc.get("chains") == {}
    assert "timeout" in cache_doc.get("error", "").lower()

    failure = _isolated_runtime / ocr.FAILURE_ARTIFACT_NAME
    assert failure.is_file(), "structured failure artifact must be written"
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ocr.FAILURE_ARTIFACT_SCHEMA
    assert payload["status"] == "failed"
    # PR-05: discovery API is now reqSecDefOptParams, so the failure
    # artifact's provider / blocked_reason labels reflect that API. The
    # schema_version is unchanged (options_chain_refresh_failure.v1).
    assert payload["provider"] == "ibkr_secdef_opt_params"
    assert payload["service_entrypoint"] == ocr.SERVICE_ENTRYPOINT
    assert payload["max_attempts"] == 2
    assert payload["error_type"] == "all_symbols_failed"
    assert "SPY" in payload["symbol_errors"]
    assert payload["blocked_reason"] == "ibkr_secdef_opt_params_unresponsive"


# ---------------------------------------------------------------------------
# T2 — retry/backoff records every attempt and honors backoff
# ---------------------------------------------------------------------------


def test_attempts_log_records_each_retry(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0.01")

    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)
    assert rc != 0

    failure = _isolated_runtime / ocr.FAILURE_ARTIFACT_NAME
    payload = json.loads(failure.read_text(encoding="utf-8"))
    spy_attempts = payload["attempts"]["SPY"]
    assert len(spy_attempts) == 3, "expected one log entry per configured attempt"
    for i, entry in enumerate(spy_attempts, start=1):
        assert entry["attempt"] == i
        assert entry["result"] in ("timeout", "error")
        assert entry["ts_utc"]
    # Backoff was non-zero and we had 3 attempts → at least 2 sleeps
    # between attempts.
    assert sum(1 for s in fake_ib.sleep_calls if s == 0.01) >= 2


# ---------------------------------------------------------------------------
# T2b — a flaky symbol that succeeds on the 2nd attempt produces a populated
# cache and NO failure artifact.
# ---------------------------------------------------------------------------


def test_retry_recovers_flaky_symbol(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")

    fake_ib = _FakeIB(mode="flaky", flaky_failures=1, spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)
    assert rc == 0, "expected rc=0 when retry succeeds"

    cache = _isolated_runtime / "options_chains_cache.json"
    cache_doc = json.loads(cache.read_text(encoding="utf-8"))
    chains = cache_doc.get("chains", {})
    assert "SPY" in chains
    assert chains["SPY"]["strikes"]
    assert chains["SPY"]["expirations"]
    # No symbol_errors → no `error` field, no failure artifact.
    assert "error" not in cache_doc
    assert not (_isolated_runtime / ocr.FAILURE_ARTIFACT_NAME).is_file()


# ---------------------------------------------------------------------------
# T2c — a healthy run scrubs a stale failure artifact left from a prior run.
# ---------------------------------------------------------------------------


def test_healthy_run_clears_stale_failure_artifact(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    stale = _isolated_runtime / ocr.FAILURE_ARTIFACT_NAME
    stale.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert stale.is_file()

    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "1")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "5")

    fake_ib = _FakeIB(mode="success", spot=500.0)
    _install_fake_ib(monkeypatch, fake_ib)

    rc = _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)
    assert rc == 0
    assert not stale.is_file(), (
        "successful run should remove the prior failure artifact so the "
        "operator surface reflects current truth"
    )


# ---------------------------------------------------------------------------
# T3 — full failure does NOT write a fake healthy cache
# ---------------------------------------------------------------------------


def test_full_failure_does_not_fake_healthy_cache(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "1")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")

    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)

    _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)

    cache = _isolated_runtime / "options_chains_cache.json"
    doc = json.loads(cache.read_text(encoding="utf-8"))
    # Defining "fake healthy" precisely: chains map must be {} and the
    # error field must be present and non-empty when every symbol failed.
    assert doc.get("chains") == {}
    assert isinstance(doc.get("error"), str) and doc["error"].strip()


# ---------------------------------------------------------------------------
# T4 — env knob plumbing accepts only sane values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "abc", "0", "-1", "11"])
def test_resolve_attempts_falls_back_on_invalid_env(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, bad)
    assert ocr._resolve_refresh_attempts() == ocr.DEFAULT_REFRESH_ATTEMPTS


def test_resolve_attempts_honors_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "5")
    assert ocr._resolve_refresh_attempts() == 5


def test_resolve_backoff_falls_back_on_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "nan")
    assert (
        ocr._resolve_refresh_backoff_seconds()
        == ocr.DEFAULT_REFRESH_BACKOFF_SECONDS
    )


# ---------------------------------------------------------------------------
# T5 — failure artifact carries last_successful_ts when a prior good cache
# exists.
# ---------------------------------------------------------------------------


def test_failure_artifact_records_last_successful_ts(
    monkeypatch: pytest.MonkeyPatch, _isolated_runtime: Path
) -> None:
    # Seed a prior healthy cache so the failure run can quote it.
    prior_ts = "2026-05-22T12:30:38Z"
    (_isolated_runtime / "options_chains_cache.json").write_text(
        json.dumps(
            {
                "schema_version": "options_chain_cache.v2",
                "ts_utc": prior_ts,
                "chains": {"SPY": {"strikes": [400], "expirations": ["20260620"]}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv(ocr.OPTIONS_CHAIN_TIMEOUT_ENV, "0.05")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_ATTEMPTS_ENV, "1")
    monkeypatch.setenv(ocr.OPTIONS_CHAIN_REFRESH_BACKOFF_ENV, "0")

    fake_ib = _FakeIB(mode="timeout")
    _install_fake_ib(monkeypatch, fake_ib)

    _run_in_thread(lambda: ocr.run(["SPY"]), timeout=20.0)
    failure = _isolated_runtime / ocr.FAILURE_ARTIFACT_NAME
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["last_successful_ts"] == prior_ts


# ---------------------------------------------------------------------------
# T8 — alpha_options fails closed when the chain cache is missing or empty
# ---------------------------------------------------------------------------


def test_alpha_options_returns_no_signals_when_chain_cache_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """alpha_options' handler must yield zero signals if the chain cache is
    not on disk. This pins the fail-closed contract."""
    from chad.strategies import alpha_options as ao

    monkeypatch.setattr(
        ao, "CHAINS_CACHE_PATH", tmp_path / "options_chains_cache.json"
    )
    assert not ao.CHAINS_CACHE_PATH.exists()

    ctx = types.SimpleNamespace(
        regime="trend_up",
        prices={"SPY": 500.0},
        ticks={},
        bars={},
        bars_1m={},
        strategy_signals=[],
        last_signals=[],
        portfolio=types.SimpleNamespace(equity=100_000.0),
    )
    signals = ao.alpha_options_handler(ctx)
    assert signals == [], "alpha_options must fail closed without a chain cache"


def test_alpha_options_returns_no_signals_when_chain_cache_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache file present but with chains={} (the today-prod state) must
    likewise produce zero signals."""
    from chad.strategies import alpha_options as ao

    cache_path = tmp_path / "options_chains_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": "options_chain_cache.v2",
                "ts_utc": "2026-05-25T12:30:33Z",
                "chains": {},
                "error": "all_symbols_failed: SPY=timeout_after_30.0s",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ao, "CHAINS_CACHE_PATH", cache_path)

    ctx = types.SimpleNamespace(
        regime="trend_up",
        prices={"SPY": 500.0},
        ticks={},
        bars={},
        bars_1m={},
        strategy_signals=[],
        last_signals=[],
        portfolio=types.SimpleNamespace(equity=100_000.0),
    )
    signals = ao.alpha_options_handler(ctx)
    assert signals == []


# ---------------------------------------------------------------------------
# T9 — omega_momentum_options must mark synthetic_pricing=True when it
# falls back to estimate_contract_price (no real chain match available).
# ---------------------------------------------------------------------------


def test_omega_momentum_options_marks_synthetic_pricing_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the chain cache is unavailable the strategy is allowed to emit
    synthetic-pricing signals, but the signal meta MUST tell downstream
    consumers it is synthetic (so no one mistakes the price for real
    market data)."""
    from chad.strategies import omega_momentum_options as omo

    # 1-min bars that satisfy the momentum + volume conditions for a BUY_CALL.
    closes = [100.0] * 14 + [100.0, 100.1, 100.2, 100.3, 100.6, 101.5]
    vols = [100] * 19 + [1000]
    bars_1m: List[Dict[str, Any]] = [
        {"close": c, "volume": v} for c, v in zip(closes, vols)
    ]

    # Synthetic-only signal generator: bypass market-hours and cache by
    # invoking _evaluate_symbol directly with no chain cache.
    sig = omo._evaluate_symbol(
        symbol="SPY",
        bars_1m=bars_1m,
        spot=101.5,
        vix=18.0,
        chain_cache=None,
        now=omo._now_utc(),
    )
    assert sig is not None, "evaluate_symbol must produce a BUY_CALL signal"
    assert sig.meta.get("synthetic_pricing") is True, (
        "omega_momentum_options must label fallback pricing synthetic so "
        "downstream consumers do not treat it as real"
    )


# ---------------------------------------------------------------------------
# T10 — sentry: live posture artifacts in the working tree remain in
# paper / no-live state. This is a working-tree assertion, not a code
# change pin.
# ---------------------------------------------------------------------------


def test_live_posture_artifacts_unchanged_paper_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    live = json.loads(
        (repo_root / "runtime" / "live_readiness.json").read_text(encoding="utf-8")
    )
    hb = json.loads(
        (repo_root / "runtime" / "decision_trace_heartbeat.json").read_text(
            encoding="utf-8"
        )
    )
    assert live.get("ready_for_live") is False
    # W1B-1: both heartbeat writers now emit the top-level posture keys, but
    # both store null when /live-gate was unreachable at write time. Skip
    # (rather than red) on that transient outage so the posture assertion is
    # robust across full-suite runs; assert whenever the endpoint was up.
    if not hb.get("live_gate"):
        pytest.skip("live-gate unreachable at heartbeat write time; posture keys null")
    assert hb.get("allow_ibkr_live") is False
    assert hb.get("allow_ibkr_paper") is True
