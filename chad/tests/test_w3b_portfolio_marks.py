"""W3B-7 — flag-gated portfolio-marks loader (freshest broker-truth mark).

Locked properties:
- flag OFF (default) → maybe_build_overlay_price_loader returns None →
  overlay behavior byte-identical;
- XOV-2345 fail-closed: dead connection → no marks, even with a warm cache;
- event handler never raises into ib_async dispatch;
- freshest-stamped-age-wins composition with mark_source recording the winner;
- zero new broker requests (the module never calls req* — source-pinned).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from chad.risk import portfolio_marks as pm

NOW = datetime(2026, 7, 22, 21, 0, 0, tzinfo=timezone.utc)


class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self

    def fire(self, *args):
        for fn in self.handlers:
            fn(*args)


class _FakeIB:
    def __init__(self, connected=True):
        self.updatePortfolioEvent = _Event()
        self._connected = connected

    def isConnected(self):
        return self._connected


def _item(symbol, price):
    return SimpleNamespace(
        contract=SimpleNamespace(localSymbol=symbol, symbol=symbol),
        marketPrice=price,
    )


def _cache(ib=None, at=NOW):
    return pm.PortfolioMarkCache(ib or _FakeIB(), clock=lambda: at)


# ---------------------------------------------------------------------------
# cache behavior
# ---------------------------------------------------------------------------


def test_event_marks_are_stamped_with_receipt_time():
    ib = _FakeIB()
    cache = _cache(ib)
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    prices, meta = cache.get_marks(["UNH"])
    assert prices == {"UNH": 424.88}
    assert meta["UNH"]["source"] == "ib_portfolio_mark"
    assert meta["UNH"]["ts_utc"].startswith("2026-07-22T21:00:00")


def test_dead_connection_fails_closed_even_with_warm_cache():
    """XOV-2345 lock: cached values from before a socket drop are not truth."""
    ib = _FakeIB()
    cache = _cache(ib)
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    ib._connected = False
    assert cache.get_marks(["UNH"]) == ({}, {})


def test_handler_never_raises_into_dispatch():
    ib = _FakeIB()
    cache = _cache(ib)
    # garbage items must be swallowed, not raised
    ib.updatePortfolioEvent.fire(SimpleNamespace(contract=None, marketPrice="x"))
    ib.updatePortfolioEvent.fire(SimpleNamespace())
    ib.updatePortfolioEvent.fire(_item("", 1.0))
    ib.updatePortfolioEvent.fire(_item("BAC", float("nan")))
    ib.updatePortfolioEvent.fire(_item("BAC", -5.0))
    assert cache.get_marks(["BAC"]) == ({}, {})


def test_missing_event_attribute_leaves_failclosed_cache():
    cache = pm.PortfolioMarkCache(SimpleNamespace(), clock=lambda: NOW)  # no event, no isConnected
    assert cache.get_marks(["UNH"]) == ({}, {})


# ---------------------------------------------------------------------------
# freshest-wins composition
# ---------------------------------------------------------------------------


def _base_loader(prices, meta):
    return lambda symbols: (prices, meta)


def test_portfolio_mark_wins_when_fresher():
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW - timedelta(seconds=2))
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    stale_ts = (NOW - timedelta(seconds=55)).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = _base_loader({"UNH": 423.0}, {"UNH": {"ts_utc": stale_ts, "source": "price_cache"}})
    loader = pm.compose_freshest_loader(base, cache, clock=lambda: NOW)
    prices, meta = loader(["UNH"])
    # the PA's exact case: 55s-stale cache loses to a 2s-old broker mark
    assert prices["UNH"] == 424.88
    assert meta["UNH"]["source"] == "ib_portfolio_mark"


def test_price_cache_wins_when_fresher():
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW - timedelta(seconds=120))
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    fresh_ts = (NOW - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = _base_loader({"UNH": 423.5}, {"UNH": {"ts_utc": fresh_ts, "source": "price_cache"}})
    loader = pm.compose_freshest_loader(base, cache, clock=lambda: NOW)
    prices, meta = loader(["UNH"])
    assert prices["UNH"] == 423.5
    assert meta["UNH"]["source"] == "price_cache"


def test_symbol_only_in_portfolio_is_added():
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW - timedelta(seconds=2))
    ib.updatePortfolioEvent.fire(_item("TLT", 92.5))
    loader = pm.compose_freshest_loader(_base_loader({}, {}), cache, clock=lambda: NOW)
    prices, meta = loader(["TLT"])
    assert prices == {"TLT": 92.5}
    assert meta["TLT"]["source"] == "ib_portfolio_mark"


def test_unstamped_base_loses_to_stamped_mark():
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW - timedelta(seconds=2))
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    loader = pm.compose_freshest_loader(lambda syms: {"UNH": 423.0}, cache, clock=lambda: NOW)
    prices, meta = loader(["UNH"])
    assert prices["UNH"] == 424.88


def test_dead_connection_composition_keeps_base():
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW)
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    ib._connected = False
    stale_ts = (NOW - timedelta(seconds=55)).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = _base_loader({"UNH": 423.0}, {"UNH": {"ts_utc": stale_ts, "source": "price_cache"}})
    prices, meta = pm.compose_freshest_loader(base, cache, clock=lambda: NOW)(["UNH"])
    assert prices["UNH"] == 423.0
    assert meta["UNH"]["source"] == "price_cache"


# ---------------------------------------------------------------------------
# flag gate
# ---------------------------------------------------------------------------


def test_flag_off_returns_none():
    assert pm.maybe_build_overlay_price_loader(Path("/tmp"), _FakeIB(), env={}) is None
    for off in ("", "0", "false", "no", "off"):
        assert pm.maybe_build_overlay_price_loader(
            Path("/tmp"), _FakeIB(), env={"CHAD_OVERLAY_PORTFOLIO_MARKS": off}
        ) is None


def test_flag_on_builds_composed_loader(tmp_path):
    ib = _FakeIB()
    cache = pm.PortfolioMarkCache(ib, clock=lambda: NOW)
    loader = pm.maybe_build_overlay_price_loader(
        tmp_path, ib, env={"CHAD_OVERLAY_PORTFOLIO_MARKS": "1"}, cache=cache
    )
    assert loader is not None
    ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
    prices, meta = loader(["UNH"])  # no price_cache file in tmp -> portfolio only
    assert prices == {"UNH": 424.88}


def test_flag_on_without_ib_returns_none():
    assert pm.maybe_build_overlay_price_loader(
        Path("/tmp"), None, env={"CHAD_OVERLAY_PORTFOLIO_MARKS": "1"}
    ) is None


def test_cache_is_singleton_per_ib_no_handler_leak(tmp_path):
    """Live-loop rebuilds the loader every cycle: the cache must be reused
    (marks persist) and the event handler attached exactly once."""
    ib = _FakeIB()
    env = {"CHAD_OVERLAY_PORTFOLIO_MARKS": "1"}
    pm._ACTIVE_CACHE = None  # isolate from other tests
    try:
        loader1 = pm.maybe_build_overlay_price_loader(tmp_path, ib, env=env)
        ib.updatePortfolioEvent.fire(_item("UNH", 424.88))
        loader2 = pm.maybe_build_overlay_price_loader(tmp_path, ib, env=env)
        assert len(ib.updatePortfolioEvent.handlers) == 1  # no per-cycle leak
        prices, _ = loader2(["UNH"])
        assert prices == {"UNH": 424.88}  # marks survived the rebuild
        # a NEW ib instance (reconnect-with-replacement) gets a fresh cache
        ib2 = _FakeIB()
        pm.maybe_build_overlay_price_loader(tmp_path, ib2, env=env)
        assert len(ib2.updatePortfolioEvent.handlers) == 1
        assert loader1 is not None
    finally:
        pm._ACTIVE_CACHE = None


# ---------------------------------------------------------------------------
# governance: zero new broker requests
# ---------------------------------------------------------------------------


def test_module_never_issues_broker_requests():
    """The read-only-hardened clientId-99 connection must gain no new
    market-data requests from this module: events in, nothing out."""
    src = Path(pm.__file__).read_text(encoding="utf-8")
    for forbidden in ("reqMktData", "reqTickers", "reqMarketDataType",
                      "reqHistoricalData", "reqAccountUpdates"):
        assert forbidden not in src, f"{forbidden} must not appear in portfolio_marks"
